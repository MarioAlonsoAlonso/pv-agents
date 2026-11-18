"""The crawler's tools. The model decides; the guarantees live here, in code.

* ``buscar_bluesky`` / ``leer_hilo``: read the source. Every post shown gets an id; the
  model never sees handles, links or AT-URIs.
* ``guardar_mencion(post_id, …)``: the model gives an id, the text is copied from the
  source. It can only save a post it has actually been shown (``ctx.shown``), so it cannot
  invent a mention, and the same (post, drug) is never saved twice.
* ``proponer_termino``: a new way people name the drug ("la sertra"), for the next searches.
* ``nombres_comerciales``: brands from CIMA (official data, not the model's memory).
* ``buscar_pubmed``: literature, only in directed mode.
* Budgets (searches, threads) are enforced by the tools: when they run out the observation
  says so and the model has to finish.

Tools are built per run (``make_registry(ctx, mode)``) as closures over a ``CrawlerContext``:
the harness runs each tool in a worker thread, so a context variable would not reach it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import Field

from agent.react_agent import final_answer
from crawler.sources import Post, Source, SourceError, to_iso
from crawler.store import MentionContext, MentionStore, RawMention
from crawler.text import fold
from tool_use.registry import ToolRegistry, tool

POSTS_PER_PART = 25
SNIPPET_CHARS = 280


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
        return None


def build_mention(
    source: Source,
    post: Post,
    drug: str,
    kind: str,
    found_by: str,
    reason: str = "",
) -> RawMention:
    """A mention with the post's own text and its thread context."""
    root = source.get(post.thread_id) if post.thread_id != post.id else None
    parent = source.get(post.parent_id) if post.parent_id else None
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
        fecha=post.created_at,
        encontrado_por=found_by,
        motivo=reason,
    )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "…"


def make_registry(ctx: CrawlerContext, mode: str) -> ToolRegistry:
    """The tools of one run. ``mode`` is ``vigilancia``, ``hilos`` or ``dirigido``."""

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
    ) -> str:
        """Search recent Bluesky posts in Spanish, newest first (at most 100). Returns one
        line per post: [id] date · replies · snippet. Use leer_hilo to read a conversation."""
        if len(ctx.searches) >= ctx.max_searches:
            raise RuntimeError(
                f"search budget exhausted ({ctx.max_searches}). Save what you found and call "
                "final_answer."
            )
        ctx.searches.append(consulta)
        posts = ctx.source.search(consulta, ctx.since, ctx.until, limit=100)
        if not posts:
            return f"0 posts for '{consulta}'."
        lines = [f"{len(posts)} posts for '{consulta}':"]
        for p in posts:
            ctx.shown[p.id] = p
            kind = "reply" if p.is_reply else "post"
            meta = f"{p.created_at[:10]} · {kind} · {p.reply_count} replies"
            lines.append(f"[{p.id}] {meta} · {_snippet(p.text)}")
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
    def guardar_mencion(
        post_id: Annotated[str, Field(description="Id of a post you have been shown.")],
        farmaco: Annotated[str, Field(description="Active ingredient the post talks about.")],
        tipo_mencion: Annotated[
            Literal["nombre", "marca", "errata", "indirecta", "jerga"],
            Field(
                description="How the post refers to the drug: nombre (active ingredient), "
                "marca (brand), errata (misspelt or shortened), indirecta (does not name it; "
                "you know it from the thread), jerga (describes it without naming it)."
            ),
        ],
        motivo: Annotated[str, Field(description="One short sentence: why this post.")],
    ) -> str:
        """Save a post that talks about the drug, so that triage reads it. Do not judge it:
        jokes, questions and opinions about the drug are saved too. The text is copied from
        the post; you only give its id."""
        post = ctx.shown.get(post_id)
        if post is None:
            raise RuntimeError(
                f"unknown post id '{post_id}': copy an id exactly as it appeared in "
                "buscar_bluesky or leer_hilo."
            )
        drug = ctx.drug_name(farmaco)
        if drug is None:
            raise RuntimeError(f"'{farmaco}' is not a drug of this task: {list(ctx.drugs)}")
        mention = build_mention(ctx.source, post, drug, tipo_mencion, "agente", motivo)
        if not ctx.store.add(mention):
            return f"already saved ({mention.id})."
        return f"saved as {mention.id}."

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
        registry.register(leer_hilo, guardar_mencion, proponer_termino)
    else:
        registry.register(
            nombres_comerciales, buscar_bluesky, leer_hilo, guardar_mencion, proponer_termino
        )
    if mode == "dirigido":

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
