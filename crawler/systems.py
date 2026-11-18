"""The four ways of collecting mentions that the evaluation compares.

1. ``fija``: search the active ingredient and every CIMA brand; save every hit. No model.
2. ``fija_hilos``: the same, plus every post of every thread where there was a hit. No
   model; maximum recall of indirect mentions, but everything in those threads goes to
   triage, related or not.
3. ``hibrido``: ``fija`` plus the agent reading those threads (``revisar_hilos``): it
   saves only the replies that talk about the drug, and proposes new terms.
4. ``agente``: the agent alone (``vigilar``), deciding what to search and read.

Each returns a ``SystemResult``: the mentions, and what it cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from crawler.sources import Source
from crawler.store import MentionStore
from crawler.text import matches
from crawler.tools import build_mention

SYSTEMS = ("fija", "fija_hilos", "hibrido", "agente")


@dataclass
class SystemResult:
    system: str
    drug: str
    store: MentionStore
    tokens: int = 0
    steps: int = 0
    searches: int = 0
    threads_read: int = 0
    seconds: float = 0.0
    new_terms: list[dict] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)


def search_terms(drug: str, brands: dict[str, list[str]], extra: list[str] = ()) -> list[tuple]:
    """(term, mention type) pairs of the fixed search."""
    return (
        [(drug, "nombre")]
        + [(b, "marca") for b in brands.get(drug, [])]
        + [(t, "errata") for t in extra]
    )


def fixed_search(
    source: Source,
    store: MentionStore,
    drug: str,
    brands: dict[str, list[str]],
    since: str | None = None,
    until: str | None = None,
    extra_terms: list[str] = (),
) -> tuple[list, int]:
    """Save every post that contains the name, a brand or an extra term. Returns the posts
    and the number of searches."""
    hits = {}
    terms = search_terms(drug, brands, list(extra_terms))
    for term, kind in terms:
        for post in source.search_all(term, since, until):
            if post.id not in hits:
                hits[post.id] = post
                store.add(
                    build_mention(source, post, drug, kind, "busqueda_fija", f"contiene '{term}'")
                )
    return list(hits.values()), len(terms)


def threads_of(posts) -> list[str]:
    """One post id per conversation that has more than the post itself."""
    seen, ids = set(), []
    for p in posts:
        if (p.reply_count > 0 or p.is_reply) and p.thread_id not in seen:
            seen.add(p.thread_id)
            ids.append(p.id)
    return ids


def run_system(
    system: str,
    source: Source,
    drug: str,
    brands: dict[str, list[str]],
    since: str | None = None,
    until: str | None = None,
    brain: str = "27b",
    only_threads: set[str] | None = None,
) -> SystemResult:
    """Run one system for one drug. ``only_threads`` (hybrid): the agent reads only those
    threads. Evaluation passes the annotated ones: the others cannot change recall or
    precision, and reading them would cost hours of the shared server (cost is then partial).
    """
    started = time.perf_counter()
    store = MentionStore()
    result = SystemResult(system=system, drug=drug, store=store)
    if system == "agente":
        from crawler.agent import vigilar

        run = vigilar(source, store, brands, drug, since, until, brain=brain)
        s = run.summary
        result.tokens, result.steps = s["tokens"], s["steps"]
        result.searches, result.threads_read = s["searches"], s["threads_read"]
        result.new_terms, result.stop_reasons = s["new_terms"], [s["stop_reason"]]
    else:
        hits, n_searches = fixed_search(source, store, drug, brands, since, until)
        result.searches = n_searches
        threads = threads_of(hits)
        if system == "fija_hilos":
            for pid in threads:
                for post in source.thread(pid):
                    kind = "nombre" if matches(post.text, drug) else "indirecta"
                    store.add(
                        build_mention(source, post, drug, kind, "busqueda_fija", "en un hilo")
                    )
            result.threads_read = len(threads)
        elif system == "hibrido":
            from crawler.agent import revisar_hilos

            if only_threads is not None:
                threads = [p for p in threads if source.get(p).thread_id in only_threads]
            for run in revisar_hilos(source, store, brands, drug, threads, brain=brain):
                s = run.summary
                result.tokens += s["tokens"]
                result.steps += s["steps"]
                result.threads_read += s["threads_read"]
                result.new_terms += s["new_terms"]
                result.stop_reasons.append(s["stop_reason"])
        elif system != "fija":
            raise ValueError(f"unknown system {system!r}; use one of {SYSTEMS}")
    result.seconds = time.perf_counter() - started
    return result
