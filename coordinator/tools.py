"""The coordinator's tools: each agent or component of the system, as a tool of the 27B.

| Tool | What is behind it |
|---|---|
| ``listar_casos``, ``ver_caso`` | The crawler + triage output (``coordinator/cases.py``) |
| ``codificar_meddra`` | The phase-1 reasoning model (Qwen3-1.7B + LoRA, SFT + GRPO) |
| ``agrupar_por_meddra`` | Equivalent cases: how many cases share each Preferred Term |
| ``search_knowledge_base`` | The phase-3 RAG over the CIMA technical sheets and EMA guidelines |
| ``faers_reaction_counts``, ``disproportionality_prr`` | Phase-2 tools: FAERS and the signal |
| ``buscar_literatura`` | The crawler's literature sub-agent (PubMed): is it described, is it new |
| ``create_icsr_report`` | The ICSR draft, held for human confirmation |

Naranjo (``naranjo_causality``) is left out on purpose: its ten questions (rechallenge,
placebo, drug levels…) cannot be answered from a social-media post, and the model would fill
them in. Causality here is what the proposal asked of that agent: whether there are equivalent
cases, whether the reaction is known (technical sheet) and whether there is a signal (FAERS).

``variant`` removes components for the ablations: ``sin_rag``, ``sin_meddra``,
``sin_literatura`` (``sin_evidencia`` is its old name, from when the sub-agent searched Bluesky
and PubMed at once). The crawler's other analyst tool, ``buscar_mas_casos``, is parked (Mario,
2026-11-23): for the watched drugs the cases are already in the case store, so "how many
patients tell X" is a query on it, not a new search.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from coordinator.cases import CaseStore
from tool_use.registry import ToolRegistry, tool

VARIANTS = ("completo", "sin_rag", "sin_meddra", "sin_literatura", "sin_evidencia")


class CodingFailed(RuntimeError):
    """The reasoning model did not reach an answer (it kept reasoning until the token limit)."""


def rlm_coder(text: str, attempts: int = 2) -> str:
    """The phase-1 reasoning model (``ARCA_RLM_ADAPTER``): 'PT; PT'.

    The model sometimes reasons until the token limit without writing ``<answer>`` (phase 1:
    19 of 200 test answers). ``split_thinking`` then falls back to the raw text, and on
    2026-11-23 its reasoning ("<think> Okay, let's tackle…") was stored as MedDRA terms. Now a
    completion without ``<answer>`` is retried once (the model samples) and then reported as
    a failure, never as terms."""
    import agent.react_agent as ra
    from rlm.rewards import extract_answer

    if ra._RLM is None:
        from rlm.inference import ReasoningModel

        model = ReasoningModel.from_env()
        model.load()
        ra._RLM = model
    for _ in range(attempts):
        answer = extract_answer(ra._RLM.answer(text).raw)
        if answer and "<think>" not in answer:
            return answer
    raise CodingFailed(
        f"the reasoning model did not finish its answer in {attempts} attempts (it kept "
        "reasoning until the token limit). Say that this case could not be coded automatically."
    )


def is_term(name: str) -> bool:
    """A MedDRA term is a short name, never a sentence or a tag."""
    return 0 < len(name) <= 60 and "<" not in name and len(name.split()) <= 6


def parse_terms(answer: str) -> tuple[list[str], list[str]]:
    """'Somnolence; Nausea' -> Preferred Terms of our dictionary, and the other proposed terms.
    Anything that is not term-like (a sentence, a tag) is dropped."""
    from rlm.meddra import MedDRADictionary

    dictionary = MedDRADictionary.default()
    pts, unknown = [], []
    for raw in answer.replace("\n", ";").split(";"):
        name = raw.strip(" .")
        if not is_term(name):
            continue
        pt = dictionary.to_pt(name)
        if pt and pt not in pts:
            pts.append(pt)
        elif not pt:
            unknown.append(name)
    return pts, unknown


def make_registry(
    store: CaseStore,
    variant: str = "completo",
    coder: Callable[[str], str] = rlm_coder,
) -> ToolRegistry:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; use one of {VARIANTS}")

    @tool
    def listar_casos(
        farmaco: Annotated[
            str, Field(description="Active ingredient in Spanish, e.g. 'sertralina'.")
        ],
        solo_graves: Annotated[bool, Field(description="Only the serious cases.")] = False,
        buscar: Annotated[
            str | None,
            Field(
                description="Words to look for in the reactions as the patients wrote them, "
                "comma-separated; a case matches if it contains ANY of them (case and accents do "
                "not matter). Use several synonyms and word stems: 'calor, sudor, sofoc'."
            ),
        ] = None,
        sexo: Annotated[
            Literal["mujer", "hombre", "desconocido"] | None, Field(description="Patient's sex.")
        ] = None,
        desenlace: Annotated[
            Literal["recuperado", "recuperandose", "no_recuperado", "desconocido"] | None,
            Field(description="Outcome."),
        ] = None,
        origen: Annotated[
            Literal["nombra_el_farmaco", "respuesta_en_conversacion"] | None,
            Field(
                description="'respuesta_en_conversacion': replies that do not name the drug "
                "(found by the crawler's agent reading the thread)."
            ),
        ] = None,
        desde: Annotated[str | None, Field(description="From this date, YYYY-MM-DD.")] = None,
        hasta: Annotated[str | None, Field(description="Until this date, YYYY-MM-DD.")] = None,
        limite: Annotated[int, Field(ge=1, le=50, description="How many cases to show.")] = 15,
    ) -> dict:
        """List the adverse event cases that the crawler found in social media and the triage
        confirmed, most urgent first (serious, then most complete, then newest), optionally
        filtered: by words in the reactions, sex, outcome, origin or dates. Each case has its id
        (M-…), date, seriousness, the reactions in the patient's own words, patient, sex, age
        and outcome. 'coinciden' is how many cases match the filters (use it to count); only
        'mostrados' of them are listed. Do not say a reaction is absent from our cases without
        searching for it with 'buscar'."""
        drug = farmaco.strip().lower()
        words = [w.strip() for w in (buscar or "").split(",") if w.strip()]
        origin = {"nombra_el_farmaco": "busqueda_fija", "respuesta_en_conversacion": "agente"}
        cases = store.filter(
            drug,
            buscar=words,
            sexo=sexo,
            desenlace=desenlace,
            origen=origin.get(origen or ""),
            grave=True if solo_graves else None,
            desde=desde,
            hasta=hasta,
        )
        all_cases = store.of_drug(drug)
        out = {
            "farmaco": drug,
            "menciones_triadas": store.reviewed.get(drug, 0),
            "efectos_adversos": len(all_cases),
            "graves": sum(c.grave for c in all_cases),
            "coinciden": len(cases),
            "mostrados": min(limite, len(cases)),
            "casos": [c.summary() for c in cases[:limite]],
        }
        if len(cases) > limite:
            out["nota"] = (
                f"hay {len(cases) - limite} casos más que coinciden y no se muestran: "
                "filtra (buscar, sexo, fechas…) o sube el límite (máx. 50)"
            )
        if words and not cases:
            out["nota"] = (
                f"ningún caso tiene {words} en las reacciones: prueba sinónimos o raíces "
                "('sud' encuentra sudor, sudores, sudoración)"
            )
        return out

    @tool
    def ver_caso(
        caso_id: Annotated[str, Field(description="Case id as listar_casos gives it (M-…).")],
    ) -> dict:
        """Read one case in full: the post (anonymised), the earlier messages of its
        conversation, everything the triage extracted and its MedDRA terms if already coded."""
        case = store.get(caso_id)
        return {
            **case.summary(),
            "texto": case.texto,
            "conversacion_anterior": case.conversacion or "(es el primer post)",
            "negadas": case.triaje.get("negadas", ""),
            # Without the reasoning model (ablation), the codes it left in the cache must not
            # leak through here: in the first sin_meddra run they did ("ya estaba codificado").
            **(
                {"meddra": store.meddra.get(case.id, "sin codificar: usa codificar_meddra")}
                if variant != "sin_meddra"
                else {}
            ),
            "fuente": "Bluesky, post público anonimizado; notificante: consumidor",
        }

    @tool
    def codificar_meddra(
        caso_id: Annotated[str, Field(description="Case id (M-…).")],
    ) -> dict:
        """Code the reactions of one case into MedDRA Preferred Terms (in English) with the
        reasoning model trained for it (it applies the MedDRA coding rules: negated symptoms
        are not coded, a diagnosis absorbs its symptoms…). Use these terms as they are in
        faers_reaction_counts. Slow the first time (loads the model)."""
        case = store.get(caso_id)
        if case.id not in store.meddra:
            pts, outside = parse_terms(coder(case.report_text()))
            store.save_meddra(case.id, pts + outside)
        else:
            pts, outside = parse_terms("; ".join(store.meddra[case.id]))
        from rlm.meddra import MedDRADictionary

        dictionary = MedDRADictionary.default()
        return {
            "caso": case.id,
            "terminos_meddra": pts + outside,
            "soc": {pt: dictionary.soc(pt) for pt in pts},
            "fuera_del_subconjunto": outside,
            "nota": (
                "Todos son los términos que propone el modelo y valen para "
                "faers_reaction_counts. El repositorio solo incluye un subconjunto de 90 PT de "
                "MedDRA (el diccionario completo tiene licencia): los de fuera_del_subconjunto "
                "no tienen SOC aquí y create_icsr_report no los acepta."
            )
            if outside
            else "Preferred Terms de MedDRA (inglés), todos en el subconjunto del repositorio",
        }

    @tool
    def agrupar_por_meddra(
        farmaco: Annotated[str, Field(description="Active ingredient in Spanish.")],
    ) -> dict:
        """Equivalent cases: how many of the already coded cases of the drug share each MedDRA
        Preferred Term, with their ids. Only counts cases coded with codificar_meddra."""
        cases = store.of_drug(farmaco.strip().lower())
        coded = [c for c in cases if c.id in store.meddra]
        by_pt: dict[str, list[str]] = {}
        for c in coded:
            for pt in store.meddra[c.id]:
                by_pt.setdefault(pt, []).append(c.id)
        counts = Counter({pt: len(ids) for pt, ids in by_pt.items()})
        return {
            "casos": len(cases),
            "codificados": len(coded),
            "por_termino": [
                {"termino": pt, "casos": n, "ids": by_pt[pt]} for pt, n in counts.most_common()
            ],
        }

    from agent.react_agent import final_answer, search_knowledge_base
    from crawler.agent import buscar_literatura
    from tool_use.tools import (
        create_icsr_report,
        disproportionality_prr,
        faers_reaction_counts,
    )

    registry = ToolRegistry().register(
        listar_casos,
        ver_caso,
        faers_reaction_counts,
        disproportionality_prr,
        create_icsr_report,
        final_answer,
    )
    if variant != "sin_meddra":
        registry.register(codificar_meddra, agrupar_por_meddra)
    if variant != "sin_rag":
        registry.register(search_knowledge_base)
    if variant not in ("sin_literatura", "sin_evidencia"):
        registry.register(buscar_literatura)
    return registry
