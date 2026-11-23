"""The crawler agent: the phase-4 ReAct harness with the crawler's tools, prompts and brains.

Nothing of the loop is rewritten: one action per step, timeouts, errors as observations,
loop detection and memory come from ``agent.react_agent.ReActAgent``. What changes:

* Tools: ``crawler.tools.make_registry`` (built per run around a ``CrawlerContext``).
* Prompts: ``crawler.prompts`` (one per mode) and longer observations (a thread page).
* Brains: ``27b`` is the course's Qwen3.8-27B-FP8 behind the DGX inference server
  (``common.llm.RemoteChatModel``), ``27b-thinking`` the same with reasoning on, and ``base``
  the local Qwen3-1.7B, to measure whether rastrear needs a big model.

Entry points, one per mode: ``vigilar`` (the agent alone), ``revisar_hilos`` (the hybrid: the
fixed search already saved the explicit mentions), and two for the analyst, also exposed as
its tools: ``investigar_literatura`` (PubMed, tool ``buscar_literatura``) and
``buscar_casos`` (more patients on Bluesky, tool ``buscar_mas_casos``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from agent.react_agent import Brain, ReActAgent
from api.schemas import AgentResponse
from crawler import prompts
from crawler.sources import Source
from crawler.store import MentionStore
from crawler.tools import CrawlerContext, make_registry
from tool_use.registry import tool

BRAINS = ("27b", "27b-thinking", "base")


def make_brain(name: str) -> Brain:
    from common.llm import get_chat_model, get_remote_chat_model

    if name == "27b":
        return Brain(llm=get_remote_chat_model(), thinking=False)
    if name == "27b-thinking":
        return Brain(llm=get_remote_chat_model(), thinking=True)
    if name == "base":
        return Brain(llm=get_chat_model(os.environ.get("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-1.7B")))
    raise ValueError(f"unknown brain {name!r}; use one of {BRAINS}")


@dataclass
class CrawlerAgent(ReActAgent):
    mode: str = "vigilancia"
    max_observation_chars: int = 8000
    token_budget: int = 24000  # the 27B has 65k tokens of context
    retry_empty_final_answer: bool = True
    brains: dict[str, Brain] = field(default_factory=dict)
    # Native tool calling with remote brains (ARCA_REMOTE_NATIVE_TOOLS=0 to turn it off): the
    # course's server drops <tool_call> text when the request does not declare the tools.
    native_tools: bool = field(
        default_factory=lambda: os.environ.get("ARCA_REMOTE_NATIVE_TOOLS", "1") != "0"
    )
    _native_now: bool = False

    def brain_for(self, name: str) -> Brain:
        if name not in self.brains:
            self.brains[name] = make_brain(name)
        return self.brains[name]

    def uses_native_tools(self, brain: Brain) -> bool:
        from common.llm import RemoteChatModel

        return self.native_tools and isinstance(brain.llm, RemoteChatModel)

    def run(self, task: str, max_steps: int = 10, brain: str = "27b", **kwargs):
        # The system prompt depends on how the brain writes actions; decide before the loop.
        self._native_now = self.uses_native_tools(self.brain_for(brain))
        return super().run(task, max_steps=max_steps, brain=brain, **kwargs)

    def system_prompt(self, max_steps: int) -> str:
        if self._native_now:  # the definitions travel in the request: names are enough here
            action, tools = prompts.NATIVE_ACTION_FORMAT, ", ".join(self.registry.tools)
        else:
            action, tools = prompts.TEXT_ACTION_FORMAT, self.registry.render_for_prompt()
        return prompts.SYSTEM_PROMPTS[self.mode].format(
            max_steps=max_steps, tools_description=tools, action_format=action
        )

    def generate(self, messages: list[dict], brain: Brain) -> tuple[str, int]:
        if not self.uses_native_tools(brain):
            return super().generate(messages, brain)
        raw, generated = brain.llm.generate(
            messages,
            # Room for a long list of ids plus the reasoning: with 768 the answer was cut
            # before the tool call and the run ended without saving (2026-11-19, v3).
            # literatura: a summary citing 15 articles did not fit in 2048 (2026-11-23).
            max_new_tokens=4096 if brain.thinking or self.mode == "literatura" else 2048,
            enable_thinking=brain.thinking,
            tools=self.registry.definitions(),
        )
        return raw, brain.llm.last_prompt_tokens + generated

    def format_result(self, result) -> str:
        return result if isinstance(result, str) else super().format_result(result)


@dataclass
class CrawlerRun:
    """What one run produced: the agent's trace plus everything the tools recorded."""

    response: AgentResponse
    context: CrawlerContext

    @property
    def summary(self) -> dict:
        r = self.response
        return {
            "stop_reason": r.stop_reason,
            "steps": r.n_steps,
            "tokens": r.tokens_used,
            "searches": len(self.context.searches),
            "threads_read": len(self.context.threads_read),
            "saved_by_agent": sum(
                1 for m in self.context.store.mentions.values() if m.encontrado_por == "agente"
            ),
            "new_terms": self.context.new_terms,
        }


_SHARED_BRAINS: dict[str, Brain] = {}


def run(
    ctx: CrawlerContext,
    mode: str,
    task: str,
    brain: str = "27b",
    max_steps: int = 30,
) -> CrawlerRun:
    agent = CrawlerAgent(registry=make_registry(ctx, mode), mode=mode, brains=_SHARED_BRAINS)
    if mode == "literatura":  # 5 whole abstracts per search do not fit in 8000 characters
        agent.max_observation_chars = 15000
    response = agent.run(task, max_steps=max_steps, brain=brain)
    return CrawlerRun(response=response, context=ctx)


def vigilar(
    source: Source,
    store: MentionStore,
    brands: dict[str, list[str]],
    drug: str,
    since: str | None = None,
    until: str | None = None,
    brain: str = "27b",
    max_steps: int = 40,
    max_searches: int = 20,
    max_threads: int = 15,
) -> CrawlerRun:
    """The agent alone: find every post about ``drug`` published in [since, until)."""
    ctx = CrawlerContext(
        source=source,
        store=store,
        brands=brands,
        drugs=(drug,),
        since=since,
        until=until,
        max_searches=max_searches,
        max_threads=max_threads,
    )
    return run(ctx, "vigilancia", prompts.VIGILANCIA_TASK.format(farmaco=drug), brain, max_steps)


def revisar_hilos(
    source: Source,
    store: MentionStore,
    brands: dict[str, list[str]],
    drug: str,
    thread_post_ids: list[str],
    brain: str = "27b",
    new_reply_ids: list[str] | None = None,
    batch: int = 6,
    max_steps_per_thread: int = 4,
    retries: int = 2,
) -> list[CrawlerRun]:
    """The hybrid's second half: read the threads where the drug is named and save the
    replies that talk about it without naming it. Threads go in batches so the context does
    not fill up; ``new_reply_ids`` (monitor) tells the agent which replies are new."""
    runs = []
    for start in range(0, len(thread_post_ids), batch):
        chunk = thread_post_ids[start : start + batch]
        ctx = CrawlerContext(
            source=source,
            store=store,
            brands=brands,
            drugs=(drug,),
            max_threads=len(chunk),
            max_searches=0,
        )
        news = ""
        if new_reply_ids:
            news = "Respuestas nuevas desde la última revisión: " + ", ".join(
                f"[{i}]" for i in new_reply_ids
            )
        task = prompts.HILOS_TASK.format(
            farmaco=drug,
            marcas=", ".join(brands.get(drug, [])) or "-",
            hilos=", ".join(f"[{i}]" for i in chunk),
            nuevos=news,
        )
        max_steps = max_steps_per_thread * len(chunk) + 4
        runs.append(run(ctx, "hilos", task, brain, max_steps))
        # Guarantee in code, not in the prompt: in the 2026-11-19 evaluation the model
        # sometimes answered "I read the threads" after reading 1 or 2 of the 6. Threads it
        # did not open go back to it, in a run of their own (at most ``retries`` times).
        unread = [p for p in chunk if _thread_of(source, p) not in ctx.threads_read]
        if unread and retries > 0:
            runs += revisar_hilos(
                source,
                store,
                brands,
                drug,
                unread,
                brain,
                new_reply_ids,
                batch,
                max_steps_per_thread,
                retries - 1,
            )
    return runs


def _thread_of(source: Source, post_id: str) -> str:
    post = source.get(post_id)
    return post.thread_id if post else post_id


def investigar_literatura(
    question: str, drug: str, brain: str = "27b", max_steps: int = 15
) -> CrawlerRun:
    """``literatura`` mode: what PubMed says about ``drug`` and the reaction in ``question``."""
    from crawler.sources import FrozenSource

    # PubMed only: no Bluesky source, no mentions. (Until 2026-11-23 one tool searched both:
    # the agent spent its steps on Bluesky, where lab reactions never appear, and searched
    # PubMed once per question; see crawler/README.md.)
    ctx = CrawlerContext(
        source=FrozenSource([]), store=MentionStore(), brands={}, drugs=(drug,), max_searches=0
    )
    task = prompts.LITERATURA_TASK.format(farmaco=drug, pregunta=question)
    return run(ctx, "literatura", task, brain, max_steps)


def buscar_casos(
    drug: str,
    reaction: str,
    source: Source,
    brands: dict[str, list[str]],
    brain: str = "27b",
    max_steps: int = 25,
    max_searches: int = 10,
    max_threads: int = 6,
) -> CrawlerRun:
    """``casos`` mode: posts where someone tells ``reaction`` with ``drug`` (saved in the
    run's store, as mentions)."""
    ctx = CrawlerContext(
        source=source,
        store=MentionStore(),
        brands=brands,
        drugs=(drug,),
        max_searches=max_searches,
        max_threads=max_threads,
    )
    task = prompts.CASOS_TASK.format(farmaco=drug, reaccion=reaction)
    return run(ctx, "casos", task, brain, max_steps)


def _brain() -> str:
    return os.environ.get("ARCA_CRAWLER_BRAIN", "27b")


@tool
def buscar_literatura(
    farmaco: Annotated[str, Field(description="Active ingredient, e.g. 'sertralina'.")],
    pregunta: Annotated[
        str,
        Field(description="What you need to know, e.g. '¿hay casos de hiponatremia en ancianos?'."),
    ],
) -> dict:
    """Ask a sub-agent what the medical literature (PubMed) says about a drug and a reaction:
    whether it is described, the kind of evidence (case reports, studies), who is at risk and
    the mechanism. Returns a summary with PMIDs and the articles it saw. Use it when the
    technical sheet does not list the reaction, or to judge plausibility. PMIDs listed in
    'referencias_sin_respaldo' were not returned by any search: do not cite them."""
    from crawler.citas import check_citations

    result = investigar_literatura(pregunta, farmaco, brain=_brain())
    ctx, answer = result.context, result.response.final_answer or ""
    check = check_citations(answer, ctx.articles, ())
    articles = {a["pmid"]: a for a in ctx.articles}  # several searches can return the same one
    return {
        "resumen": answer,
        "articulos": [
            {"pmid": a["pmid"], "titulo": a["titulo"], "año": a["año"]} for a in articles.values()
        ],
        "referencias_sin_respaldo": check["pmids_sin_respaldo"],
    }


@tool
def buscar_mas_casos(
    farmaco: Annotated[str, Field(description="Active ingredient, e.g. 'sertralina'.")],
    reaccion: Annotated[str, Field(description="The reaction, in Spanish, e.g. 'bruxismo'.")],
) -> dict:
    """Ask a sub-agent to look for more people telling this reaction with this drug on
    Bluesky (Spanish posts, any date). Returns the posts it saved (real text, anonymised) and a
    summary. Use it when a signal comes from social media and the reaction is something
    patients notice and tell (not lab values). Slow: a full agent run."""
    from crawler.cima import load_brands
    from crawler.sources import LiveSource

    result = buscar_casos(farmaco, reaccion, LiveSource(), load_brands(), brain=_brain())
    ctx = result.context
    return {
        "resumen": result.response.final_answer or "",
        "casos": [
            {"post_id": m.post_id, "hilo_id": m.hilo_id, "fecha": m.fecha, "texto": m.texto}
            for m in ctx.store.mentions.values()
        ],
    }
