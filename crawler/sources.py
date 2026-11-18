"""Where the posts come from: Bluesky live, or a frozen sample of it on disk.

Both sources have the same interface, so the agent, the baselines and the evaluation do not
know which one they use:

* ``search(query, since, until, limit)``: one page of keyword search, newest first, like the
  Bluesky search box (every word must appear; case and accents do not matter).
* ``search_all(query, since, until, max_posts)``: every post that matches (pagination).
* ``thread(post_id)``: the whole conversation the post belongs to, root first.
* ``get(post_id)``: one post already seen.

Posts are identified by ``text.post_id(uri)``, a hash: the AT-URI contains the author's DID,
so it stays inside ``LiveSource`` and never reaches the model, the sample or the reports.
Texts are anonymised (handles, links, e-mails) as soon as they are read.

``LiveSource`` uses the account in ``BSKY_HANDLE`` / ``BSKY_APP_PASSWORD`` when there is
one (the ``atproto`` SDK; cursor pagination works). Without an account it falls back to the
public AppView, where only the first page of a search is served: ``search_all`` then splits
the date range in halves while a window is full (100 results), which reaches every post too.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from crawler.text import anonymize, matches, post_id


@dataclass
class Post:
    id: str
    text: str
    created_at: str
    thread_id: str  # id of the root post of the conversation
    parent_id: str | None = None  # id of the post it answers
    reply_count: int = 0
    lang: str | None = None
    search_terms: list[str] = field(default_factory=list)  # provenance in the frozen sample

    @property
    def is_reply(self) -> bool:
        return self.parent_id is not None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Post:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


class SourceError(RuntimeError):
    pass


class Source(Protocol):
    def search(
        self, query: str, since: str | None = None, until: str | None = None, limit: int = 100
    ) -> list[Post]: ...

    def search_all(
        self,
        query: str,
        since: str | None = None,
        until: str | None = None,
        max_posts: int = 2000,
    ) -> list[Post]: ...

    def thread(self, post_id: str) -> list[Post]: ...

    def get(self, post_id: str) -> Post | None: ...


def to_iso(value: str | datetime | None) -> str | None:
    """ "2025-01-01" or a datetime -> "2025-01-01T00:00:00Z" (what the API expects)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def thread_order(posts: Iterable[Post], root_id: str) -> list[Post]:
    """Posts of a conversation in reading order: root, then each reply followed by its own
    replies (depth first, oldest first)."""
    by_parent: dict[str | None, list[Post]] = {}
    by_id = {}
    for p in posts:
        by_id[p.id] = p
        by_parent.setdefault(p.parent_id, []).append(p)
    for children in by_parent.values():
        children.sort(key=lambda p: p.created_at)
    ordered: list[Post] = []

    def visit(p: Post) -> None:
        ordered.append(p)
        for child in by_parent.get(p.id, []):
            visit(child)

    if root_id in by_id:
        visit(by_id[root_id])
    # Replies whose parent is not in the sample (deleted, or outside it) go at the end.
    seen = {p.id for p in ordered}
    for p in sorted(by_id.values(), key=lambda p: p.created_at):
        if p.id not in seen:
            visit(p)
            seen.update(q.id for q in ordered)
    return ordered


# --------------------------------------------------------------------------- frozen sample


class FrozenSource:
    """A sample of Bluesky saved to disk (``crawler.collect``): reproducible evaluation.

    Search imitates Bluesky's: keyword match, newest first, at most ``limit`` per page.
    """

    def __init__(self, posts: Iterable[Post]):
        self.posts: dict[str, Post] = {p.id: p for p in posts}
        self.threads: dict[str, list[str]] = {}
        for p in self.posts.values():
            self.threads.setdefault(p.thread_id, []).append(p.id)
        self._newest_first = sorted(self.posts.values(), key=lambda p: p.created_at, reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> FrozenSource:
        with open(path, encoding="utf-8") as f:
            return cls(Post.from_json(json.loads(line)) for line in f if line.strip())

    def _matching(self, query: str, since: str | None, until: str | None) -> list[Post]:
        since, until = to_iso(since), to_iso(until)
        return [
            p
            for p in self._newest_first
            if (since is None or p.created_at >= since)
            and (until is None or p.created_at < until)
            and matches(p.text, query)
        ]

    def search(self, query, since=None, until=None, limit=100):
        return self._matching(query, since, until)[:limit]

    def search_all(self, query, since=None, until=None, max_posts=2000):
        return self._matching(query, since, until)[:max_posts]

    def thread(self, post_id: str) -> list[Post]:
        post = self.posts.get(post_id)
        if post is None:
            raise SourceError(f"post {post_id} not found")
        members = [self.posts[i] for i in self.threads[post.thread_id]]
        return thread_order(members, post.thread_id)

    def get(self, post_id: str) -> Post | None:
        return self.posts.get(post_id)


# --------------------------------------------------------------------------- Bluesky live


def _field(obj: Any, snake: str, camel: str | None = None, default: Any = None) -> Any:
    """Read a field from an atproto model (snake_case) or a JSON dict (camelCase)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(camel or snake, obj.get(snake, default))
    return getattr(obj, snake, default)


class LiveSource:
    """Bluesky through its API, with rate limiting, retries and anonymisation."""

    PUBLIC_API = "https://api.bsky.app/xrpc"

    def __init__(
        self,
        handle: str | None = None,
        password: str | None = None,
        lang: str | None = "es",
        min_interval: float = 0.5,
        retries: int = 4,
        client: Any = None,
        anonymous: bool = False,
    ):
        self.lang = lang
        self.min_interval, self.retries = min_interval, retries
        self.uris: dict[str, str] = {}  # post id -> AT-URI (never leaves this object)
        self.cache: dict[str, Post] = {}
        self.requests = 0
        self._last = 0.0
        self._lock = threading.Lock()
        handle = handle or os.environ.get("BSKY_HANDLE")
        password = password or os.environ.get("BSKY_APP_PASSWORD")
        self.client = client
        if self.client is None and handle and password and not anonymous:
            from atproto import Client

            self.client = Client()
            self.client.login(handle, password)
        self.authenticated = self.client is not None

    # -- low level ------------------------------------------------------------------------

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        """One API call (``app.bsky.feed.searchPosts`` …) with pacing and retries."""
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(self.retries + 1):
            with self._lock:
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.requests += 1
            try:
                if self.client is not None:
                    namespace = self.client.app.bsky.feed
                    snake = {"searchPosts": "search_posts", "getPostThread": "get_post_thread"}
                    return getattr(namespace, snake[method.rsplit(".", 1)[1]])(params=params)
                import httpx

                response = httpx.get(f"{self.PUBLIC_API}/{method}", params=params, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise SourceError(f"HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise SourceError(f"HTTP {response.status_code}: {response.text[:200]}")
                return response.json()
            except Exception as exc:  # noqa: BLE001 - retry transient errors, report the rest
                message = str(exc)
                transient = any(s in message for s in ("429", "500", "502", "503", "504", "timed"))
                if "NotFound" in message or "not found" in message.lower():
                    raise SourceError(f"not found: {message[:200]}") from exc
                if not transient or attempt == self.retries:
                    raise SourceError(message[:300]) from exc
                time.sleep(2 ** (attempt + 1))
        raise SourceError("unreachable")  # pragma: no cover

    def _to_post(self, view: Any) -> Post:
        uri = _field(view, "uri")
        record = _field(view, "record")
        reply = _field(record, "reply")
        parent_uri = _field(_field(reply, "parent"), "uri")
        root_uri = _field(_field(reply, "root"), "uri") or uri
        langs = _field(record, "langs") or []
        pid = post_id(uri)
        self.uris[pid] = uri
        self.uris.setdefault(post_id(root_uri), root_uri)
        post = Post(
            id=pid,
            text=anonymize(_field(record, "text", default="") or ""),
            created_at=to_iso(_field(record, "created_at", "createdAt")) or "",
            thread_id=post_id(root_uri),
            parent_id=post_id(parent_uri) if parent_uri else None,
            reply_count=int(_field(view, "reply_count", "replyCount", 0) or 0),
            lang=langs[0] if langs else None,
        )
        self.cache[pid] = post
        return post

    # -- Source interface -------------------------------------------------------------------

    def search_page(
        self,
        query: str,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Post], str | None]:
        data = self._call(
            "app.bsky.feed.searchPosts",
            {
                "q": query,
                "lang": self.lang,
                "sort": "latest",
                "limit": min(limit, 100),
                "since": to_iso(since),
                "until": to_iso(until),
                "cursor": cursor,
            },
        )
        posts = [self._to_post(v) for v in _field(data, "posts", default=[]) or []]
        return posts, _field(data, "cursor")

    def search(self, query, since=None, until=None, limit=100):
        return self.search_page(query, since, until, limit)[0]

    def search_all(self, query, since=None, until=None, max_posts=2000):
        found: dict[str, Post] = {}
        if self.authenticated:
            cursor = None
            while len(found) < max_posts:
                posts, cursor = self.search_page(query, since, until, 100, cursor)
                found.update((p.id, p) for p in posts)
                if not posts or not cursor:
                    break
            return list(found.values())[:max_posts]
        # Public API: no pagination. Halve the date window while it comes back full.
        start = datetime.fromisoformat(to_iso(since or "2023-01-01").replace("Z", "+00:00"))
        end = datetime.fromisoformat(to_iso(until or datetime.now(UTC)).replace("Z", "+00:00"))
        stack = [(start, end)]
        while stack and len(found) < max_posts:
            a, b = stack.pop()
            posts = self.search(query, a, b, 100)
            found.update((p.id, p) for p in posts)
            if len(posts) >= 80 and b - a > timedelta(minutes=30):
                middle = a + (b - a) / 2
                stack += [(a, middle), (middle, b)]
        return list(found.values())[:max_posts]

    def thread(self, post_id: str) -> list[Post]:
        """The whole conversation: from the root, every reply (``depth`` 20)."""
        post = self.cache.get(post_id)
        root_id = post.thread_id if post else post_id
        uri = self.uris.get(root_id) or self.uris.get(post_id)
        if uri is None:
            raise SourceError(f"unknown post {post_id}: find it with a search first")
        height = "parent_height" if self.client is not None else "parentHeight"  # SDK: snake_case
        data = self._call("app.bsky.feed.getPostThread", {"uri": uri, "depth": 20, height: 0})
        posts: list[Post] = []

        def walk(node: Any) -> None:
            view = _field(node, "post")
            if view is None:  # notFoundPost / blockedPost
                return
            posts.append(self._to_post(view))
            for child in _field(node, "replies", default=[]) or []:
                walk(child)

        walk(_field(data, "thread"))
        if not posts:
            raise SourceError(f"thread of {post_id} not available (deleted or blocked)")
        return thread_order(posts, posts[0].thread_id)

    def get(self, post_id: str) -> Post | None:
        return self.cache.get(post_id)

    def uri(self, post_id: str) -> str | None:
        """The AT-URI of a post, for the monitor's state file (never for the model)."""
        return self.uris.get(post_id)

    def remember(self, post_id: str, uri: str) -> None:
        """Teach the source a URI saved by a previous run (the monitor re-reads threads)."""
        self.uris[post_id] = uri
