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
  data (``guard``, ``final_answer_problem``);
* the earlier stages (2026-11-26): ``rastrear`` and ``triar`` run at once when the job is small
  and are queued for a person when it is big (``coordinator/etapas.py``: more than a month of
  Bluesky or more than 100 mentions), with their estimated cost;
* ``Conversacion``: several turns with the same memory, and the confirmation of what was queued
  (the ICSR, a big search), which until then could only be shown.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from api.schemas import AgentAction, AgentResponse, ToolCallRecord
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
    "rastrear": 1800.0,  # small jobs only (a month at most): bigger ones wait for a person
    "triar": 1800.0,
}
DEFAULT_TIMEOUT = 30.0
# The drug argument of each tool that takes one: what the substitution guard watches.
DRUG_ARGS = {
    "faers_reaction_counts": "drug",
    "buscar_literatura": "farmaco",
    "listar_casos": "farmaco",
    "resumen_casos": "farmaco",
    "identificar_farmaco": "nombre",
    "comparar_senal": "farmacos",
}
ACKNOWLEDGES = re.compile(
    r"(?i)no (lo )?(reconozco|reconoce|identifico|encuentro|he encontrado|existe|consta|figura|"
    r"aparece)|"
    r"desconocid|no reconocid|te refieres|quieres decir|¿es posible que|asumo|supongo|"
    r"entiendo que|interpreto"
)
# The id of a fragment of the knowledge base, as search_knowledge_base gives it.
FRAGMENT_ID = re.compile(r"\b[0-9a-f]{10}-s?\d{4}\b")


def names_close_to_a_drug(text: str) -> set[str]:
    """Words of the request that look like one of our drugs but are not one ("tramadolex",
    "sertralna"). The 27B corrects them in its head before calling any tool, so the guard
    never saw them (development of benchmark v3, 2026-11-25): it answered "tramadolex" with
    tramadol's signal. Marked as not recognised from the start, the answer has to say which drug
    it assumed, or ask."""
    import difflib
    import re as _re

    from common.farmacos import _tables, resolve

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
    # Guard on citations (benchmark v4, development, 2026-11-25): asked general questions on a
    # technical sheet, the coordinator read the fragments and answered "según la ficha, sección
    # 4.7" without their ids (2 of 5), so nobody could check where the answer came from.
    _fragments: list = field(default_factory=list)
    _cite_bounced: bool = False
    # The case store, for the plan of rastrear / triar (is it a big job?).
    store: CaseStore | None = None
    # Benchmarks: rastrear / triar are always queued, never run (a live crawl in the middle of an
    # exam would change the data and spend the shared server).
    hold_stages: bool = False

    def run(self, task: str, max_steps: int = 10, brain: str = "27b", **kwargs):
        self._unresolved, self._substituted, self._bounced = set(), {}, False
        self._fragments, self._cite_bounced = [], False
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
        """Checked in code before an answer is accepted; each problem is sent back once."""
        return self._name_problem(answer) or self._citation_problem(answer)

    def _citation_problem(self, answer: str) -> str | None:
        if not self._fragments or self._cite_bounced or FRAGMENT_ID.search(answer):
            return None
        self._cite_bounced = True
        return (
            "has consultado las fichas técnicas o las guías, pero tu respuesta no cita ningún "
            f"fragmento. Pon entre corchetes el id de los fragmentos en que te basas (p. ej. "
            f"[{self._fragments[0]}]); si no te has basado en ninguno, dilo."
        )

    def _name_problem(self, answer: str) -> str | None:
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
        if name in ("rastrear", "triar") and not allow_actions and self.store is not None:
            held = self.hold_if_big(name, arguments, pending)
            if held:
                return held
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
        if name == "search_knowledge_base":
            self._fragments += [
                f for f in FRAGMENT_ID.findall(observation) if f not in self._fragments
            ]
        if self.guard_drug_names and name in DRUG_ARGS:
            observation = self.guard(name, arguments, observation)
        return observation

    def hold_if_big(self, name: str, arguments: dict, pending: list) -> str | None:
        """A big crawl or triage is not run by the model's decision: it is queued, with its
        cost, for the technician (the shared 27B server: a month of Bluesky is millions of
        tokens). Errors (a drug that is not watched, a bad date, something already running)
        come back at once. None: small, run it now."""
        from coordinator import etapas

        try:
            args = self.registry.tools[name].validate(arguments)
        except Exception as exc:  # noqa: BLE001 - validation errors are observations
            return f"error: invalid arguments: {exc}"
        plan = etapas.plan(name, args, self.store)
        if plan["error"]:
            return f"error: {plan['error']}"
        if not plan["grande"] and not self.hold_stages:
            return None
        pending.append(AgentAction(name=name, arguments=arguments))
        return (
            "no ejecutado todavía: es un trabajo grande y necesita la confirmación del técnico. "
            f"{plan['estimacion']} Queda pendiente: díselo al técnico con esta estimación y "
            "pregúntale si lo lanzas (puede pedir un periodo más corto)."
        )

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
    hold_stages: bool = False,
) -> AgentResponse:
    """Answer one request of the analyst. ``allow_actions`` stays False: the ICSR is only
    drafted and queued (``pending_actions``) for a person to confirm."""
    agent = CoordinatorAgent(
        registry=make_registry(store, variant, coder),
        brains=brains if brains is not None else {},
        guard_drug_names=variant != "sin_nombres",
        store=store,
        hold_stages=hold_stages,
    )
    return agent.run(question, max_steps=max_steps, brain=brain, allow_actions=allow_actions)


def describe(action: AgentAction, store: CaseStore | None = None) -> str:
    """What a queued action will do, in words, for the technician who has to confirm it."""
    a = action.arguments
    if action.name == "create_icsr_report":
        return (
            f"Registrar el ICSR de {a.get('suspect_drug')} con {', '.join(a.get('reactions', []))}"
            " (borrador ya validado)."
        )
    if action.name in ("rastrear", "triar") and store is not None:
        from coordinator import etapas

        what = (
            f"Rastrear {a.get('farmaco')} en Bluesky desde {a.get('desde')}"
            + (f" hasta {a['hasta']}" if a.get("hasta") else " hasta hoy")
            + ("" if a.get("con_agente", True) else " (solo búsqueda fija)")
            if action.name == "rastrear"
            else f"Triar las menciones de {a.get('farmaco')} que esperan"
        )
        return f"{what}. {etapas.plan(action.name, a, store)['estimacion']}"
    return f"{action.name}({a})"


@dataclass
class Conversacion:
    """A conversation with the coordinator: the same memory across turns (the harness's
    ``session_id``) and the actions it queued, which a person confirms or rejects. A confirmed
    action is run by code, outside the model's loop and without its timeout (a crawl of months
    can take an hour), and its result goes back to the coordinator as the next turn."""

    store: CaseStore
    brain: str = "27b"
    variant: str = "completo"
    max_steps: int = 20
    coder: Callable[[str], str] = rlm_coder
    brains: dict | None = None
    pendientes: list = field(default_factory=list)
    session_id: str | None = None

    def __post_init__(self):
        self.agent = CoordinatorAgent(
            registry=make_registry(self.store, self.variant, self.coder),
            brains=self.brains if self.brains is not None else {},
            guard_drug_names=self.variant != "sin_nombres",
            store=self.store,
        )

    def preguntar(self, texto: str) -> AgentResponse:
        response = self.agent.run(
            texto, max_steps=self.max_steps, brain=self.brain, session_id=self.session_id
        )
        self.session_id = response.session_id
        self.pendientes = list(response.pending_actions)
        return response

    def confirmar(self, i: int = 0) -> AgentResponse:
        """Run the i-th queued action and tell the coordinator what came out."""
        action = self.pendientes.pop(i)
        record = self.agent.registry.call(action.name, action.arguments)
        result = self.agent.format_result(record.result) if record.ok else f"error: {record.error}"
        return self.preguntar(
            f"[Sistema] El técnico ha confirmado {action.name}({action.arguments}). "
            f"Resultado: {result[:5000]}\nDíselo al técnico y sigue con lo que pidió, si quedaba "
            "algo."
        )

    def rechazar(self, i: int = 0) -> AgentResponse:
        action = self.pendientes.pop(i)
        return self.preguntar(
            f"[Sistema] El técnico NO ha confirmado {action.name}({action.arguments}): no se ha "
            "hecho. Respóndele con lo que tengas sin ello, o propón una alternativa más pequeña."
        )
