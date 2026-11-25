"""The coordinator agent: the crawler's harness (ReAct loop + the 27B's native tool calling)
with the coordinator's prompt and tools.

Nothing of the loop is rewritten (``agent/react_agent.py``: one action per step, timeouts,
errors as observations, loop detection, memory, human confirmation for the ICSR). What changes:

* the tools (``coordinator/tools.py``): every agent and component of the system;
* longer timeouts for the slow ones (the reasoning model loads on first use; the crawler's
  sub-agent is a whole agent run);
* a larger memory budget: the 27B has 65k tokens of context;
* two checks in code: an ICSR draft is only queued if it could be registered, and a drug name
  the technician gave that is not recognised cannot be silently answered with another drug's
  data (``guard``, ``final_answer_problem``).
"""

from __future__ import annotations

import re
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
from crawler.text import fold as fold_text

TIMEOUTS = {
    "codificar_meddra": 600.0,  # the first call loads Qwen3-1.7B + the adapter
    "codificar_notificacion": 600.0,
    "search_knowledge_base": 300.0,  # the first call loads the embedder and the index
    "faers_reaction_counts": 90.0,
    "buscar_literatura": 1200.0,  # a whole crawler run (PubMed)
}
DEFAULT_TIMEOUT = 30.0
# The drug argument of each tool that takes one: what the substitution guard watches.
DRUG_ARGS = {
    "faers_reaction_counts": "drug",
    "buscar_literatura": "farmaco",
    "listar_casos": "farmaco",
    "identificar_farmaco": "nombre",
    "comparar_senal": "farmacos",
}
ACKNOWLEDGES = re.compile(
    r"(?i)no (lo )?(reconozco|identifico|encuentro|he encontrado|existe|consta|figura|aparece)|"
    r"desconocid|no reconocid|te refieres|quieres decir|¿es posible que|asumo|supongo|"
    r"entiendo que|interpreto"
)


def names_close_to_a_drug(text: str) -> set[str]:
    """Words of the request that look like one of our drugs but are not one ("tramadolex",
    "sertralna"). The 27B corrects them in its head before calling any tool, so the guard
    never saw them (development of benchmark v3, 2026-11-25): it answered "tramadolex" with
    tramadol's signal. Marked as not recognised from the start, the answer has to say which drug
    it assumed, or ask."""
    import difflib
    import re as _re

    from coordinator.drugs import _tables, resolve

    spanish, english, _ = _tables()
    actives = list(spanish) + list(english)  # not brands: "seguro" looks like "Segluromet"
    out = set()
    for word in _re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{6,}", text):
        if resolve(word).active is None and difflib.get_close_matches(
            fold_text(word), actives, n=1, cutoff=0.75
        ):
            out.add(word)
    return out


@dataclass
class CoordinatorAgent(CrawlerAgent):
    mode: str = "coordinador"
    token_budget: int = 30000
    max_observation_chars: int = 6000
    timeouts: dict[str, float] = field(default_factory=lambda: dict(TIMEOUTS))
    # Guard on substituted drugs (2026-11-25): asked about "xelofrina", which does not exist, the
    # coordinator answered with celecoxib's FAERS signal as if it were xelofrina's.
    guard_drug_names: bool = True
    _unresolved: set = field(default_factory=set)
    _substituted: dict = field(default_factory=dict)
    _bounced: bool = False

    def run(self, task: str, max_steps: int = 10, brain: str = "27b", **kwargs):
        self._unresolved, self._substituted, self._bounced = set(), {}, False
        if self.guard_drug_names:
            self._unresolved = names_close_to_a_drug(task)
        return super().run(task, max_steps=max_steps, brain=brain, **kwargs)

    def _names(self, name: str, arguments: dict) -> list[str]:
        value = arguments.get(DRUG_ARGS.get(name, ""), None)
        if not value:
            return []
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]

    def guard(self, name: str, arguments: dict, observation: str) -> str:
        """Watch drug names, not drug choices: the coordinator may consult any drug (a
        comparison needs that), but once a name the technician gave is not recognised, data of
        another drug is labelled as such, and the final answer must say the name was not found
        (``final_answer_problem``)."""
        from coordinator.drugs import resolve

        for drug in self._names(name, arguments):
            r = resolve(drug)
            failed = observation.startswith("error") or '"principio_activo": null' in observation
            if r.active is None and failed:
                self._unresolved.add(drug)
            elif r.active and self._unresolved and fold_text(drug) not in self._unresolved_folded():
                self._substituted[r.active] = sorted(self._unresolved)
                observation += (
                    f"\nAVISO: estos datos son de '{r.active}'. El técnico preguntó por "
                    f"{', '.join(repr(u) for u in sorted(self._unresolved))}, que no se ha "
                    "reconocido: no atribuyas estos datos a ese nombre. Si crees que se refiere a "
                    f"'{r.active}', pregúntaselo."
                )
        return observation

    def _unresolved_folded(self) -> set[str]:
        return {fold_text(u) for u in self._unresolved}

    def final_answer_problem(self, answer: str) -> str | None:
        if not self.guard_drug_names or not self._substituted or self._bounced:
            return None
        if ACKNOWLEDGES.search(answer):
            return None
        self._bounced = True  # once: the model gets one chance to fix it
        names = sorted({u for us in self._substituted.values() for u in us})
        return (
            f"tu respuesta usa datos de {sorted(self._substituted)} pero no dice que "
            f"{names} no se ha reconocido. Díselo al técnico; no presentes los datos de otro "
            "fármaco como si fueran de ese nombre (puedes preguntarle si se refiere a otro)."
        )

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
        observation = super().observe(name, arguments, allow_actions, pending)
        if self.guard_drug_names and name in DRUG_ARGS:
            observation = self.guard(name, arguments, observation)
        return observation

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
        registry=make_registry(store, variant, coder),
        brains=brains if brains is not None else {},
        guard_drug_names=variant != "sin_nombres",
    )
    return agent.run(question, max_steps=max_steps, brain=brain, allow_actions=allow_actions)
