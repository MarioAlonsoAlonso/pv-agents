"""The crawler's tools. The model decides; the guarantees live here, in code.

* ``buscar_bluesky`` / ``leer_hilo``: read the source. Every post shown gets an id; the
  model never sees handles, links or AT-URIs.
* ``guardar_menciones(farmaco, "id tipo; id tipo")``: one call saves
  every post the model has just seen; it gives ids, the text is copied from the
  source. It can only save a post it has actually been shown (``ctx.shown``), so it cannot
  invent a mention, and the same (post, drug) is never saved twice.
* ``proponer_termino``: a new way people name the drug ("la sertra"), for the next searches.
* ``nombres_comerciales``: brands from CIMA (official data, not the model's memory).
* ``buscar_pubmed``: literature, only in ``literatura`` mode (the analyst's tool).
* Budgets (searches, threads) are enforced by the tools: when they run out the observation
  says so and the model has to finish.

Tools are built per run (``make_registry(ctx, mode)``) as closures over a ``CrawlerContext``:
the harness runs each tool in a worker thread, so a context variable would not reach it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from agent.react_agent import final_answer
from crawler.sources import Post, Source, SourceError, to_iso
from crawler.store import ConversationPost, MentionContext, MentionStore, RawMention
from crawler.text import fold
from tool_use.registry import ToolRegistry, tool

POSTS_PER_PART = 25

SEARCH_PAGE = 20  # posts per search page: 20 snippets fit in one observation
SNIPPET_CHARS = 280


_TYPES = ("nombre", "marca", "errata", "indirecta", "jerga")
_LABELS = {"post_id", "post", "id", "tipo", "tipo_mencion", "type"}


def parse_mentions(text: str) -> list[tuple[str, str | None]]:
    """ "p1a2b3c4d5e6 nombre; pa9c9fb2de600 indirecta" -> [(id, type), ...] (type None if
    missing). Pieces are separated by ';', ',' or new lines; in each piece the first word that
    is not a type or a label ("post_id:") is the id. A plain string, not a list of objects:
    the course's server sometimes dropped a nested list from native tool calls (2026-11-19)."""
    seen, out = set(), []
    for piece in re.split(r"[;,\n]+", text or ""):
        words = re.findall(r"[\w-]+", piece)
        ids = [w for w in words if w.lower() not in _TYPES and w.lower() not in _LABELS]
        if not ids or ids[0] in seen:
            continue
        kind = next((w.lower() for w in words if w.lower() in _TYPES), None)
        seen.add(ids[0])
        out.append((ids[0], kind))
    return out


@dataclass
class CrawlerContext:
    source: Source
    store: MentionStore
    brands: dict[str, list[str]]
    drugs: tuple[str, ...]  # drugs the run may save mentions of
    since: str | None = None  # the run only sees posts in [since, until)
    until: str | None = None
    max_searches: int = 20
    max_threads: int = 15
    searches: list[str] = field(default_factory=list)
    threads_read: set[str] = field(default_factory=set)
    shown: dict[str, Post] = field(default_factory=dict)
    new_terms: list[dict] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)

    def drug_name(self, name: str) -> str | None:
        """Canonical watchlist name for what the model wrote ("Sertralína" -> "sertralina")."""
        wanted = fold(name.strip())
        for drug in self.drugs:
            if fold(drug) == wanted or wanted in {fold(b) for b in self.brands.get(drug, [])}:
                return drug
        from common.farmacos import resolve  # the catalogue: English name, CIMA brands

        active = resolve(name).active
        return active if active in self.drugs else None


def build_mention(
    source: Source,
    post: Post,
    drug: str,
    kind: str,
    found_by: str,
    reason: str = "",
) -> RawMention:
    """A mention with the post's own text and its whole chain of earlier posts."""
    if post.parent_id and source.get(post.parent_id) is None:
        try:  # live search hit: its conversation was never downloaded
            source.thread(post.id)
        except SourceError:
            pass
    chain, complete, current = [], True, post
    while current.parent_id and len(chain) < 200:
        parent_post = source.get(current.parent_id)
        if parent_post is None:
            complete = False
            break
        chain.append(parent_post)
        current = parent_post
    chain.reverse()  # oldest first
    root = source.get(post.thread_id) if post.thread_id != post.id else None
    parent = chain[-1] if chain else None
    return RawMention(
        id=RawMention.make_id(post.id, drug),
        post_id=post.id,
        hilo_id=post.thread_id,
        farmaco=drug,
        tipo_mencion=kind,
        texto=post.text,
        contexto=MentionContext(
            inicio_hilo=root.text if root else None,
            responde_a=parent.text if parent and parent is not root else None,
        ),
        conversacion=[ConversationPost(post_id=p.id, texto=p.text) for p in chain],
        conversacion_completa=complete,
        idioma=post.lang,
        fecha=post.created_at,
        encontrado_por=found_by,
        motivo=reason,
    )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "…"


def make_registry(ctx: CrawlerContext, mode: str) -> ToolRegistry:
    """The tools of one run. ``mode`` is ``vigilancia``, ``hilos``, ``casos`` or ``literatura``."""

    @tool
    def nombres_comerciales(
        farmaco: Annotated[str, Field(description="Active ingredient, e.g. 'sertralina'.")],
    ) -> str:
        """Brand names under which this active ingredient is sold in Spain (official CIMA
        data). People often write the brand instead of the active ingredient."""
        drug = ctx.drug_name(farmaco)
        brands = ctx.brands.get(drug or fold(farmaco), [])
        return ", ".join(brands) if brands else f"no brands known for '{farmaco}'"

    @tool
    def buscar_bluesky(
        consulta: Annotated[
            str,
            Field(
                description="Words that must ALL appear in the post (case and accents do not "
                "matter). Short queries work best: one or two words."
            ),
        ],
        antes_de: Annotated[
            str | None,
            Field(
                description="Only posts published before this moment, to see older results: "
                "pass the value the previous page tells you. Leave empty for the newest."
            ),
        ] = None,
    ) -> str:
        """Search Bluesky posts in Spanish, newest first, one page of at most 20. Returns one
        line per post: [id] date · replies · snippet. If there are older posts, the last line
        says how to get the next page. Use leer_hilo to read a conversation."""
        if len(ctx.searches) >= ctx.max_searches:
            raise RuntimeError(
                f"search budget exhausted ({ctx.max_searches}). Save what you found and call "
                "final_answer."
            )
        ctx.searches.append(consulta if antes_de is None else f"{consulta} <{antes_de}")
        until = to_iso(ctx.until)
        if antes_de:
            before = to_iso(antes_de)
            until = before if until is None else min(until, before)
        posts = ctx.source.search(consulta, ctx.since, until, limit=SEARCH_PAGE + 1)
        if not posts:
            return f"0 posts for '{consulta}'" + (" before that date." if antes_de else ".")
        more, posts = len(posts) > SEARCH_PAGE, posts[:SEARCH_PAGE]
        lines = [f"{len(posts)} posts for '{consulta}':"]
        for p in posts:
            ctx.shown[p.id] = p
            kind = "reply" if p.is_reply else "post"
            meta = f"{p.created_at[:10]} · {kind} · {p.reply_count} replies"
            lines.append(f"[{p.id}] {meta} · {_snippet(p.text)}")
        if more:
            lines.append(
                f"… there are older posts: buscar_bluesky(consulta='{consulta}', "
                f"antes_de='{posts[-1].created_at}')"
            )
        return "\n".join(lines)

    @tool
    def leer_hilo(
        post_id: Annotated[str, Field(description="Id of any post of the conversation.")],
        parte: Annotated[int, Field(ge=1, description="Part to read in long threads.")] = 1,
    ) -> str:
        """Read the whole conversation a post belongs to, root first. Each line is
        [id] (reply to [id]) text. Replies often talk about the drug without naming it."""
        try:
            posts = ctx.source.thread(post_id)
        except SourceError as exc:
            raise RuntimeError(str(exc)) from exc
        thread = posts[0].thread_id
        if thread not in ctx.threads_read and len(ctx.threads_read) >= ctx.max_threads:
            raise RuntimeError(
                f"thread budget exhausted ({ctx.max_threads}). Save what you found and call "
                "final_answer."
            )
        ctx.threads_read.add(thread)
        since, until = to_iso(ctx.since), to_iso(ctx.until)
        start = (parte - 1) * POSTS_PER_PART
        page = posts[start : start + POSTS_PER_PART]
        if not page:
            return f"the thread has {len(posts)} posts; part {parte} is empty."
        lines = [f"thread {thread}, {len(posts)} posts (part {parte}):"]
        for p in page:
            ctx.shown[p.id] = p
            where = f"reply to [{p.parent_id}]" if p.parent_id else "root"
            late = until is not None and p.created_at >= until
            early = since is not None and p.created_at < since
            flag = " (outside the period)" if late or early else ""
            lines.append(f"[{p.id}] {p.created_at[:10]} ({where}){flag} {' '.join(p.text.split())}")
        remaining = len(posts) - start - len(page)
        if remaining > 0:
            lines.append(f"… {remaining} more posts: leer_hilo(post_id, parte={parte + 1})")
        return "\n".join(lines)

    @tool
    def guardar_menciones(
        farmaco: Annotated[str, Field(description="Active ingredient the posts talk about.")],
        menciones: Annotated[
            str,
            Field(
                description="The posts to save, all at once, as 'post_id type' pairs separated "
                "by ';'. Type is one of nombre (active ingredient), marca (brand), errata "
                "(misspelt or shortened), indirecta (does not name it; you know it from the "
                "thread), jerga (describes it without naming it). Example: "
                "'p1a2b3c4d5e6f nombre; pa9c9fb2de600 indirecta'."
            ),
        ],
    ) -> str:
        """Save the posts that talk about the drug, so that triage reads them. Save ALL the
        relevant posts you have just seen in ONE call. Do not judge them: jokes, questions
        and opinions about the drug are saved too. The text is copied from each post; you
        only give its id and how it refers to the drug."""
        drug = ctx.drug_name(farmaco)
        if drug is None:
            raise RuntimeError(f"'{farmaco}' is not a drug of this task: {list(ctx.drugs)}")
        items = parse_mentions(menciones)  # all of them: no silent cut (2026-11-19, v3)
        if not items:
            raise RuntimeError(
                "no post ids found in 'menciones'. Write 'post_id type' pairs separated by ';', "
                "e.g. 'p1a2b3c4d5e6f nombre; pa9c9fb2de600 indirecta'."
            )
        lines, saved = [], 0
        for post_id, kind in items:
            post = ctx.shown.get(post_id)
            if post is None:
                lines.append(
                    f"[{post_id}] error: unknown post id; copy it exactly as it appeared in "
                    "buscar_bluesky or leer_hilo"
                )
            elif kind is None:
                lines.append(f"[{post_id}] error: missing type (nombre, marca, errata, ...)")
            elif ctx.store.add(build_mention(ctx.source, post, drug, kind, "agente")):
                saved += 1
                lines.append(f"[{post_id}] saved")
            else:
                lines.append(f"[{post_id}] already saved")
        return f"{saved} saved of {len(items)}:" + chr(10) + chr(10).join(lines)

    @tool
    def proponer_termino(
        termino: Annotated[str, Field(description="The new word or expression, as written.")],
        farmaco: Annotated[str, Field(description="Drug it refers to.")],
        motivo: Annotated[str, Field(description="Where you saw it.")],
    ) -> str:
        """Propose a new way people name the drug (a misspelling, a nickname, a foreign
        brand) that the brand list does not have. It will be added to future searches."""
        drug = ctx.drug_name(farmaco) or farmaco
        known = {fold(t) for t in (drug, *ctx.brands.get(drug, []))}
        if fold(termino) in known or any(
            fold(termino) == fold(t["termino"]) for t in ctx.new_terms
        ):
            return "already known."
        ctx.new_terms.append({"termino": termino, "farmaco": drug, "motivo": motivo})
        return "noted."

    registry = ToolRegistry().register(final_answer)
    if mode == "hilos":
        registry.register(leer_hilo, guardar_menciones, proponer_termino)
    elif mode == "vigilancia":
        registry.register(
            nombres_comerciales, buscar_bluesky, leer_hilo, guardar_menciones, proponer_termino
        )
    elif mode == "casos":  # Bluesky only, for the analyst: no new terms for the watch list
        registry.register(nombres_comerciales, buscar_bluesky, leer_hilo, guardar_menciones)
    elif mode != "literatura":
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "literatura":  # PubMed only

        @tool
        def buscar_pubmed(
            consulta: Annotated[
                str, Field(description="English query, e.g. 'sertraline falls elderly'.")
            ],
            max_resultados: Annotated[int, Field(ge=1, le=10)] = 5,
        ) -> list[dict]:
            """Search the medical literature (PubMed): case reports and studies. Returns PMID,
            title, journal, year and abstract. Cite PMIDs in your answer."""
            from crawler.pubmed import search_pubmed

            articles = search_pubmed(consulta, max_resultados)
            ctx.articles.extend(articles)
            return articles

        registry.register(buscar_pubmed)
    return registry
