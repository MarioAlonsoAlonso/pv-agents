"""The coordinator's tools: each agent or component of the system, as a tool of the 27B.

| Tool | What is behind it |
|---|---|
| ``listar_casos``, ``ver_caso`` | The crawler + triage output (``coordinator/cases.py``) |
| ``resumen_casos`` | The same cases at once: counts and reactions by theme (``overview``) |
| ``codificar_meddra`` | The phase-1 reasoning model (Qwen3-1.7B + LoRA, SFT + GRPO) |
| ``codificar_notificacion`` | The same model, on a text the technician pastes (not a case) |
| ``agrupar_por_meddra`` | Equivalent cases: how many cases share each Preferred Term |
| ``search_knowledge_base`` | The phase-3 RAG over the CIMA technical sheets and EMA guidelines |
| ``faers_reaction_counts``, ``disproportionality_prr`` | Phase-2 tools: FAERS and the signal |
| ``buscar_literatura`` | The crawler's literature sub-agent (PubMed): is it described, is it new |
| ``create_icsr_report`` | The ICSR draft, held for human confirmation |
| ``farmacos_disponibles`` | ``coordinator/drugs.py``: the drugs by group, and their sources |
| ``identificar_farmaco`` | ``coordinator/drugs.py``: active ingredient, English name, brand |
| ``comparar_senal`` | FAERS + PRR for several drugs at once (comparisons) |

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

VARIANTS = ("completo", "sin_rag", "sin_meddra", "sin_literatura", "sin_evidencia", "sin_nombres")
# Tools of the drug names and comparisons (2026-11-25); ``sin_nombres`` removes them and the
# guard on substituted drugs, i.e. the coordinator as it was before.
NAME_TOOLS = ("farmacos_disponibles", "identificar_farmaco", "comparar_senal")


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

    def case_drug(farmaco: str | None) -> tuple[str | None, dict | None]:
        """The active ingredient whose cases are asked for, or the answer for a name the system
        does not know. Benchmark v3, test (2026-11-25): asked "¿qué casos tenemos de la
        xelofrina?" (an invented name), the coordinator answered "no tenemos casos de
        xelofrina" as if it were a real drug; and a brand ("Besitran") or the English name
        found none of sertraline's cases. Part of the drug-name tools (not in sin_nombres)."""
        from coordinator.drugs import resolve

        drug = (farmaco or "").strip().lower() or None
        if drug is None or variant == "sin_nombres":
            return drug, None
        r = resolve(drug)
        if r.active:
            return r.active, None
        return None, {
            "farmaco": farmaco,
            "principio_activo": None,
            "reconocido_como": "no reconocido: no está entre los fármacos del sistema",
            "parecidos": list(r.similar),
            "farmacos_con_casos": dict(Counter(c.farmaco for c in store.cases.values())),
            "nota": f"'{farmaco}' no es un fármaco que el sistema reconozca (ni principio "
            "activo, ni nombre en inglés, ni marca). Díselo al técnico; si hay parecidos, "
            "pregúntale si se refiere a alguno, sin dar sus datos como si fueran de ese nombre.",
        }

    @tool
    def listar_casos(
        farmaco: Annotated[
            str | None,
            Field(
                description="Active ingredient in Spanish, e.g. 'sertralina'. Leave it empty "
                "for the cases of every watched drug ('farmacos_con_casos' says which have cases)."
            ),
        ] = None,
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
        paciente: Annotated[
            Literal["autor", "familiar_o_conocido", "desconocido"] | None,
            Field(description="Who had the reaction: the author of the post or a relative."),
        ] = None,
        origen: Annotated[
            Literal["nombra_el_farmaco", "respuesta_en_conversacion"] | None,
            Field(
                description="'respuesta_en_conversacion': replies that do not name the drug "
                "(found by the crawler's agent reading the thread)."
            ),
        ] = None,
        con_edad: Annotated[
            bool | None,
            Field(description="True: only the cases whose age is known; False: age unknown."),
        ] = None,
        desde: Annotated[str | None, Field(description="From this date, YYYY-MM-DD.")] = None,
        hasta: Annotated[str | None, Field(description="Until this date, YYYY-MM-DD.")] = None,
        limite: Annotated[int, Field(ge=1, le=50, description="How many cases to show.")] = 15,
    ) -> dict:
        """List the adverse event cases that the crawler found in social media and the triage
        confirmed, most urgent first (serious, then most complete, then newest), optionally
        filtered: by words in the reactions, sex, patient, outcome, origin, known age or dates.
        Each case has its id (M-…), date, seriousness, the reactions in the patient's own words,
        patient, sex, age and outcome. 'coinciden' is how many cases match the filters (use it to
        count); only 'mostrados' of them are listed. Do not say a reaction is absent from our
        cases without searching for it with 'buscar'. For the whole picture of many cases, use
        resumen_casos instead of paging through them."""
        drug, unknown = case_drug(farmaco)
        if unknown:
            return unknown
        words = [w.strip() for w in (buscar or "").split(",") if w.strip()]
        origin = {"nombra_el_farmaco": "busqueda_fija", "respuesta_en_conversacion": "agente"}
        cases = store.filter(
            drug,
            buscar=words,
            sexo=sexo,
            paciente=paciente,
            desenlace=desenlace,
            origen=origin.get(origen or ""),
            grave=True if solo_graves else None,
            con_edad=con_edad,
            desde=desde,
            hasta=hasta,
        )
        all_cases = store.of_drug(drug)
        out = {
            "farmaco": drug or "todos",
            # Always: which drugs have cases. Without it, a request that named no drug made the
            # coordinator try drug names one by one until it ran out of steps (2026-11-23).
            "farmacos_con_casos": dict(Counter(c.farmaco for c in store.cases.values())),
            "menciones_triadas": sum(store.reviewed.values())
            if drug is None
            else store.reviewed.get(drug, 0),
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
        if drug and not all_cases:
            # v2 exam, f07 (2026-11-23): "the overdose case of 16 g of ciprofloxacin" is a case
            # described in the technical sheet; the coordinator only looked here and gave up.
            out["nota"] = (
                f"no tenemos casos de redes sociales de '{drug}'. Si preguntan por un caso "
                "concreto (una sobredosis, un caso clínico), puede estar descrito en la ficha "
                "técnica (search_knowledge_base) o en la literatura (buscar_literatura)."
            )
        return out

    @tool
    def resumen_casos(
        farmaco: Annotated[
            str | None,
            Field(description="Active ingredient in Spanish; empty for every watched drug."),
        ] = None,
        solo_graves: Annotated[bool, Field(description="Only the serious cases.")] = False,
        desde: Annotated[str | None, Field(description="From this date, YYYY-MM-DD.")] = None,
        hasta: Annotated[str | None, Field(description="Until this date, YYYY-MM-DD.")] = None,
    ) -> dict:
        """The whole picture of our cases in one call, counted by code: how many, which are
        serious, sex, age, outcome, origin, cases per quarter, and the reactions grouped by
        theme (sleep, heat and sweating, digestive, sexual, mood…) with how many cases mention
        each and some ids. Use it for general questions ("what do patients report?", "which
        reactions are most frequent?", "which of our reactions are not in the technical
        sheet?") instead of listing the cases page by page. The themes group the patients' own
        words; for MedDRA terms use codificar_meddra on the cases."""
        from coordinator.cases import overview

        drug, unknown = case_drug(farmaco)
        if unknown:
            return unknown
        cases = store.filter(drug, grave=True if solo_graves else None, desde=desde, hasta=hasta)
        out = {
            "farmaco": drug or "todos",
            "farmacos_con_casos": dict(Counter(c.farmaco for c in store.cases.values())),
            **overview(cases, store.meddra if variant != "sin_meddra" else None),
            "nota": "temas aproximados por palabras clave sobre lo que escribieron los "
            "pacientes; para ver los casos de un tema, listar_casos con 'buscar'",
        }
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
    def codificar_notificacion(
        texto: Annotated[
            str,
            Field(description="The notification as the technician received it, in Spanish."),
        ],
    ) -> dict:
        """Code a notification that is not one of our cases (an e-mail, a phone note, a text the
        technician pastes) into MedDRA Preferred Terms with the reasoning model trained for it.
        It applies the coding rules: what is denied is not coded, a diagnosis absorbs its
        typical symptoms, death is an outcome (code the cause if known), lack of effect is
        coded, the indication and the history are not."""
        pts, outside = parse_terms(coder(texto.strip()))
        from rlm.meddra import MedDRADictionary

        dictionary = MedDRADictionary.default()
        return {
            "terminos_meddra": pts + outside,
            "soc": {pt: dictionary.soc(pt) for pt in pts},
            "fuera_del_subconjunto": outside,
            "nota": "Preferred Terms de MedDRA (inglés) propuestos por el modelo de razonamiento",
        }

    @tool
    def farmacos_disponibles(
        grupo: Annotated[
            str | None,
            Field(
                description="Therapeutic group in Spanish, e.g. 'antidepresivo', "
                "'benzodiacepina', 'AINE', 'estatina'. Empty for every group."
            ),
        ] = None,
    ) -> dict:
        """The drugs the system knows, by therapeutic group: which have a technical sheet in
        the knowledge base, their FAERS name and how many cases we have from social media. Use
        it to compare drugs ("which antidepressant gives less insomnia?"): look at the sheet of
        each one and, for the signal, comparar_senal."""
        from coordinator.drugs import available

        cases = Counter(c.farmaco for c in store.cases.values())
        groups = available(grupo, dict(cases))
        if not groups:
            groups = available(None, dict(cases))
            return {
                "nota": f"no hay ningún grupo '{grupo}'; estos son todos",
                "total_farmacos": sum(map(len, groups.values())),
                "grupos": groups,
            }
        # The count by code: asked which drugs the system watches, the 27B said 36 (35).
        return {"total_farmacos": sum(map(len, groups.values())), "grupos": groups}

    @tool
    def identificar_farmaco(
        nombre: Annotated[str, Field(description="The drug name as the technician wrote it.")],
    ) -> dict:
        """Whether a drug name is one of the system's drugs: an active ingredient, its English
        name or a Spanish brand (CIMA). If it is not recognised, similar names are listed only
        to ASK the technician ("¿te refieres a…?"): never answer with another drug's data as if
        it were this one."""
        from coordinator.drugs import resolve

        r = resolve(nombre)
        if r.active:
            return {"nombre": nombre, "principio_activo": r.active, "reconocido_como": r.how}
        return {
            "nombre": nombre,
            "principio_activo": None,
            "reconocido_como": "no reconocido: no está entre los fármacos del sistema",
            "parecidos": list(r.similar),
            "nota": "No lo sustituyas por un parecido: dile al técnico que no lo reconoces y, "
            "si hay parecidos, pregúntale si se refiere a alguno.",
        }

    @tool
    def comparar_senal(
        farmacos: Annotated[
            list[str], Field(description="Active ingredients to compare, in Spanish or English.")
        ],
        reaccion: Annotated[
            str, Field(description="MedDRA Preferred Term in English, e.g. 'Insomnia'.")
        ],
    ) -> dict:
        """The FAERS signal (PRR with its 95% CI, Evans rule) of one reaction for several drugs
        at once, highest PRR first. Counts and PRR are computed by code from openFDA."""
        from tool_use.tools import disproportionality_prr, faers_reaction_counts

        rows, errors = [], {}
        for drug in farmacos[:8]:
            try:
                counts = faers_reaction_counts(drug=drug, reaction=reaccion)
                prr = disproportionality_prr(
                    reports_with_drug_and_reaction=counts["reports_with_drug_and_reaction"],
                    reports_with_drug=counts["reports_with_drug"],
                    reports_with_reaction=counts["reports_with_reaction"],
                    total_reports=counts["total_reports"],
                )
                rows.append(
                    {
                        "farmaco": drug,
                        "nombre_faers": counts["drug_searched"],
                        "notificaciones_con_la_reaccion": counts["reports_with_drug_and_reaction"],
                        "notificaciones_del_farmaco": counts["reports_with_drug"],
                        "prr": prr["prr"],
                        "ic95": prr["ci95"],
                        "senal": prr["signal"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one drug failing must not lose the rest
                errors[drug] = str(exc)
        rows.sort(key=lambda r: -r["prr"])
        return {
            "reaccion": reaccion,
            "por_farmaco": rows,
            **({"errores": errors} if errors else {}),
            "regla": "señal si PRR >= 2, a >= 3 y límite inferior del IC > 1 (Evans)",
            "nota": "notificaciones espontáneas no validadas: el PRR compara proporciones de "
            "notificación, no riesgos reales entre fármacos",
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
        resumen_casos,
        ver_caso,
        faers_reaction_counts,
        disproportionality_prr,
        create_icsr_report,
        final_answer,
    )
    if variant != "sin_meddra":
        registry.register(codificar_meddra, codificar_notificacion, agrupar_por_meddra)
    if variant != "sin_rag":
        registry.register(search_knowledge_base)
    if variant not in ("sin_literatura", "sin_evidencia"):
        registry.register(buscar_literatura)
    if variant != "sin_nombres":
        registry.register(farmacos_disponibles, identificar_farmaco, comparar_senal)
    return registry
