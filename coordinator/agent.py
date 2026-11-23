"""The coordinator agent: the crawler's harness (ReAct loop + the 27B's native tool calling)
with the coordinator's prompt and tools.

Nothing of the loop is rewritten (``agent/react_agent.py``: one action per step, timeouts,
errors as observations, loop detection, memory, human confirmation for the ICSR). What changes:

* the tools (``coordinator/tools.py``): every agent and component of the system;
* longer timeouts for the slow ones (the reasoning model loads on first use; the crawler's
  sub-agent is a whole agent run);
* a larger memory budget: the 27B has 65k tokens of context.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from api.schemas import AgentResponse, ToolCallRecord
from coordinator.cases import CaseStore
from coordinator.prompts import COORDINATOR_SYSTEM
from coordinator.tools import make_registry, rlm_coder
from crawler.agent import CrawlerAgent
from crawler.prompts import NATIVE_ACTION_FORMAT, TEXT_ACTION_FORMAT

TIMEOUTS = {
    "codificar_meddra": 600.0,  # the first call loads Qwen3-1.7B + the adapter
    "codificar_notificacion": 600.0,
    "search_knowledge_base": 300.0,  # the first call loads the embedder and the index
    "faers_reaction_counts": 90.0,
    "buscar_literatura": 1200.0,  # a whole crawler run (PubMed)
}
DEFAULT_TIMEOUT = 30.0


@dataclass
class CoordinatorAgent(CrawlerAgent):
    mode: str = "coordinador"
    token_budget: int = 30000
    max_observation_chars: int = 6000
    timeouts: dict[str, float] = field(default_factory=lambda: dict(TIMEOUTS))

    def system_prompt(self, max_steps: int) -> str:
        if self._native_now:
            action, tools = NATIVE_ACTION_FORMAT, ", ".join(self.registry.tools)
        else:
            action, tools = TEXT_ACTION_FORMAT, self.registry.render_for_prompt()
        return COORDINATOR_SYSTEM.format(
            max_steps=max_steps, tools_description=tools, action_format=action
        )

    def observe(self, name: str, arguments: dict, allow_actions: bool, pending: list) -> str:
        """Before an ICSR draft is queued for a person, check the rules of a valid ICSR (the
        harness only checks the argument types): a draft that could never be registered
        (patient not identifiable, terms outside the dictionary) must not reach the queue as
        "pending". The two serious cases of the sample have neither age nor sex."""
        if name == "create_icsr_report" and not allow_actions:
            from tool_use.tools import icsr_problem

            try:
                args = self.registry.tools[name].validate(arguments)
            except Exception as exc:  # noqa: BLE001 - validation errors are observations
                return f"error: invalid arguments: {exc}"
            problem = icsr_problem(
                args["reactions"], args["patient_age"], args["patient_sex"], args["suspect_drug"]
            )
            if problem:
                return f"error: this ICSR could not be registered: {problem}"
        return super().observe(name, arguments, allow_actions, pending)

    def execute(self, name: str, arguments: dict) -> ToolCallRecord:
        timeout = self.timeouts.get(name, DEFAULT_TIMEOUT)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.registry.call, name, arguments)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"timed out after {timeout:.0f} s",
                latency_ms=timeout * 1000,
            )
        finally:
            pool.shutdown(wait=False)


def warm_up(variant: str) -> None:
    """Load the RAG index and the reasoning model before the first question, so that their
    first call is not eaten by a timeout (needs a GPU session in the DGX)."""
    if variant != "sin_rag":
        import agent.react_agent as ra
        from rag.generate import RagPipeline

        if ra._RAG is None:
            pipeline = RagPipeline()
            pipeline.load(load_generator=False)
            ra._RAG = pipeline
    if variant != "sin_meddra":
        import agent.react_agent as ra
        from rlm.inference import ReasoningModel

        if ra._RLM is None:
            model = ReasoningModel.from_env()
            model.load()
            ra._RLM = model


def coordinar(
    question: str,
    store: CaseStore,
    brain: str = "27b",
    variant: str = "completo",
    max_steps: int = 20,
    allow_actions: bool = False,
    coder: Callable[[str], str] = rlm_coder,
    brains: dict | None = None,
) -> AgentResponse:
    """Answer one request of the analyst. ``allow_actions`` stays False: the ICSR is only
    drafted and queued (``pending_actions``) for a person to confirm."""
    agent = CoordinatorAgent(
        registry=make_registry(store, variant, coder), brains=brains if brains is not None else {}
    )
    return agent.run(question, max_steps=max_steps, brain=brain, allow_actions=allow_actions)
