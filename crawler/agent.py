"""The crawler agent: the phase-4 ReAct harness with the crawler's tools, prompts and brains.

Nothing of the loop is rewritten: one action per step, timeouts, errors as observations,
loop detection and memory come from ``agent.react_agent.ReActAgent``. What changes:

* Tools: ``crawler.tools.make_registry`` (built per run around a ``CrawlerContext``).
* Prompts: ``crawler.prompts`` (one per mode) and longer observations (a thread page).
* Brains: ``27b`` is the course's Qwen3.8-27B-FP8 behind the DGX inference server
  (``common.llm.RemoteChatModel``), ``27b-thinking`` the same with reasoning on, and ``base``
  the local Qwen3-1.7B, to measure whether rastrear needs a big model.

Three entry points, one per mode: ``vigilar`` (the agent alone), ``revisar_hilos`` (the
hybrid: the fixed search already saved the explicit mentions), and ``investigar`` (directed
search for the analyst, also exposed as the tool ``buscar_evidencia``).
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
    brains: dict[str, Brain] = field(default_factory=dict)

    def brain_for(self, name: str) -> Brain:
        if name not in self.brains:
            self.brains[name] = make_brain(name)
        return self.brains[name]

    def system_prompt(self, max_steps: int) -> str:
        return prompts.SYSTEM_PROMPTS[self.mode].format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )

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
    return runs


def investigar(
    question: str,
    drug: str,
    source: Source,
    brands: dict[str, list[str]],
    brain: str = "27b",
    max_steps: int = 25,
) -> CrawlerRun:
    """Directed mode: evidence about ``drug`` for the analyst, from Bluesky and PubMed."""
    ctx = CrawlerContext(
        source=source,
        store=MentionStore(),
        brands=brands,
        drugs=(drug,),
        max_searches=10,
        max_threads=8,
    )
    task = prompts.DIRIGIDO_TASK.format(farmaco=drug, pregunta=question)
    return run(ctx, "dirigido", task, brain, max_steps)


@tool
def buscar_evidencia(
    farmaco: Annotated[str, Field(description="Active ingredient, e.g. 'sertralina'.")],
    pregunta: Annotated[
        str,
        Field(description="What you need to know, e.g. 'caídas en ancianos con sertralina'."),
    ],
) -> dict:
    """Ask the crawler sub-agent to look for evidence in social media (Bluesky, Spanish
    patients' posts) and in the literature (PubMed). Returns its summary with post ids and
    PMIDs. Slow (a full agent run): use it only when the technical sheet and FAERS are not
    enough."""
    from crawler.cima import load_brands
    from crawler.sources import LiveSource

    result = investigar(
        pregunta,
        farmaco,
        LiveSource(),
        load_brands(),
        brain=os.environ.get("ARCA_CRAWLER_BRAIN", "27b"),
    )
    return {
        "resumen": result.response.final_answer,
        "posts": [
            {"id": m.post_id, "texto": m.texto} for m in result.context.store.mentions.values()
        ],
        "articulos": [{"pmid": a["pmid"], "titulo": a["titulo"]} for a in result.context.articles],
    }
