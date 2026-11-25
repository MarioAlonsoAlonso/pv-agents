"""Evaluate the coordinator on the benchmark of analyst requests, with ablations.

    uv run python -m coordinator.evaluate                                   # all variants (DGX)
    uv run python -m coordinator.evaluate --variantes completo --tareas c01 c05

Each task of ``coordinator/bench/tareas.jsonl`` has the request, the tools that should be
involved and automatic checks of the final answer. Requests may name cases with placeholders
(``{caso_prioritario}``, ``{caso_con_sexo}``, ``{caso_indirecto}``), resolved from the case
store, so the benchmark has no post text and works on any data.

Two kinds of request (``tipo``): ``dirigida`` (the request names what to do: "codifica…",
"calcula el PRR…") and ``abierta`` (what a technician would actually say: "¿qué tengo pendiente
hoy?", "¿hay algo que tenga que notificar ya?"), where the coordinator has to work out the
steps itself. Results are given for both.

Check types, besides those of ``agent/evaluate.py`` (``contains``, ``regex``, ``number``,
``pending_action``):

* ``n_efectos``, ``n_triadas``, ``n_graves``: the counts of the case store are in the answer;
* ``ids_prioritarios``: at least ``min`` of the ``n`` most urgent case ids are named;
* ``meddra_del_caso``: a MedDRA term of the case (as the reasoning model coded it) is named;
* ``pt_mas_frecuente``: the most repeated term among the ``n`` most urgent coded cases;
* ``prr_live``: the PRR computed now from FAERS is in the answer;
* ``cita_fragmento``: a fragment id of the knowledge base is cited;
* ``ids_graves``: every serious case is named;
* ``alguno``: at least one of the checks in ``de`` passes;
* ``herramientas_distintas``: at least ``min`` of the tools in ``de`` were used;
* ``n_casos``: the number of cases matching ``filtro`` (``CaseStore.filter``) is in the answer;
* ``faers_live``, ``faers_top``: a FAERS count / the top reactions, computed now;
* ``meddra_notif``: every expected Preferred Term is named and no forbidden one is coded;
* ``regex_ausente``: the pattern must NOT appear (e.g. claiming an ICSR was registered);
* ``ids_filtro``: at least ``min`` of the cases matching ``filtro`` are named.

Benchmark v2 (``--bench coordinator/bench/tareas_v2.jsonl``, ``coordinator/bench/v2.py``): ~100
requests with their expected sources, split in ``--conjunto dev`` and ``test``. Besides
success, it reports the routing: whether the coordinator consulted every source the request
needed, and whether it consulted sources it did not need.

Besides success, every answer is checked for **numbers without support**: numbers that do not
appear in any observation nor in the request (invented counts, PRRs, frequencies).

Variants (the ablations): ``completo``, ``sin_rag``, ``sin_meddra``, ``sin_evidencia``. The
numbers go to ``reports/coordinator_eval.json``; the answers and traces, which may paraphrase
posts, to ``data/coordinador/eval/``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from coordinator.cases import FOLDER, CaseStore
from tool_use.evaluate import numbers_in

BENCH = Path(__file__).parent / "bench" / "tareas.jsonl"
# Fragment ids of the knowledge base: the document's 10-hex hash and the chunk number
# (rag/ingest.py), e.g. ee3f3fd9f0-0073. A looser pattern took dates ("abr-2025") for citations.
CHUNK_ID = re.compile(r"\b[0-9a-f]{10}-s?\d{4}\b")
CASE_ID = re.compile(r"M-[0-9a-f]{10}")
# A written number: thousands in groups of exactly three ('21.054.159', '4 300') or a plain
# number with decimals ('2,66'). With a looser pattern "IC95 5,49" was read as 95549.
WRITTEN_NUMBER = re.compile(r"\d{1,3}(?:[.,\s]\d{3})+(?:,\d+)?(?!\d)|\d+(?:[.,]\d+)?")
DRUG = "sertralina"


def placeholders(store: CaseStore, drug: str = DRUG) -> dict[str, str]:
    cases = store.of_drug(drug)
    if not cases:
        return {}
    with_sex = [c for c in cases if c.triaje.get("sexo") in ("mujer", "hombre")] or cases
    indirect = [c for c in cases if c.encontrado_por == "agente"]
    deduced = [c for c in indirect if "[" in c.triaje.get("reacciones", "")]
    serious = sorted((c for c in cases if c.grave), key=lambda c: c.fecha) or cases
    return {
        "caso_prioritario": cases[0].id,
        "caso_con_sexo": with_sex[0].id,
        "caso_con_sexo_2": with_sex[min(1, len(with_sex) - 1)].id,
        "caso_indirecto": (deduced or indirect or cases)[0].id,
        "caso_grave_antiguo": serious[0].id,
        "caso_grave_reciente": serious[-1].id,
    }


def fill(value, slots: dict[str, str]):
    if isinstance(value, str):
        for k, v in slots.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, list):
        return [fill(v, slots) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, slots) for k, v in value.items()}
    return value


def has_number(answer: str, target: float) -> bool:
    return any(abs(n - target) < 1e-6 for n in numbers_in(answer))


def live_prr(drug: str, reaction: str) -> float:
    from tool_use.tools import disproportionality_prr, faers_reaction_counts

    counts = faers_reaction_counts(drug=drug, reaction=reaction)
    return disproportionality_prr(
        reports_with_drug_and_reaction=counts["reports_with_drug_and_reaction"],
        reports_with_drug=counts["reports_with_drug"],
        reports_with_reaction=counts["reports_with_reaction"],
        total_reports=counts["total_reports"],
    )["prr"]


# A line that says a term is NOT coded ("Diarrhoea no se codifica: se niega", "el dolor de
# cabeza no se ha codificado por separado"): terms named there do not count as coded.
EXCLUDING = re.compile(
    r"(?i)no se (ha )?codific|no codific|no se incluy|excluy|excluid|descart|negad|se niega|"
    r"absorb|es un desenlace|omit|sin codificar|no es una reacci|antecedente|indicaci[oó]n"
)


def coded_terms_lines(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if not EXCLUDING.search(line)]


def term_in(term: str, line: str) -> bool:
    """The term on its own, not inside a longer one: "Cough" is not in "Dry cough" (2026-11-23:
    a substring check took "Dry cough" for the expected "Cough")."""
    pattern = rf"(?<![A-Za-zÀ-ÿ] )(?<![A-Za-zÀ-ÿ]){re.escape(term)}(?![A-Za-zÀ-ÿ])"
    return re.search(pattern, line, re.IGNORECASE) is not None


def check_one(check: dict, answer: str, response: dict, store: CaseStore) -> bool:
    kind = check.get("type")
    cases = store.of_drug(DRUG)
    if kind == "n_efectos":
        return has_number(answer, len(cases))
    if kind == "n_triadas":
        return has_number(answer, store.reviewed.get(DRUG, 0))
    if kind == "n_graves":
        n = sum(c.grave for c in cases)
        return has_number(answer, n) or (
            n == 0 and re.search(r"(?i)ningun|ninguno", answer) is not None
        )
    if kind == "ids_prioritarios":
        top = [c.id for c in cases[: check.get("n", 3)]]
        return sum(i in answer for i in top) >= check.get("min", len(top))
    if kind == "meddra_del_caso":
        terms = store.meddra.get(check["caso"], [])
        return any(t.lower() in answer.lower() for t in terms)
    if kind == "pt_mas_frecuente":
        coded = [c for c in cases[: check.get("n", 5)] if c.id in store.meddra]
        counts = Counter(t for c in coded for t in store.meddra[c.id])
        if not counts:
            return False
        best = max(counts.values())
        return any(t.lower() in answer.lower() for t, n in counts.items() if n == best)
    if kind == "prr_live":
        target = live_prr(check["drug"], check["reaction"])
        tolerance = check.get("tolerance_rel", 0) * target or check.get("tolerance", 0.05)
        return any(abs(n - target) <= tolerance for n in numbers_in(answer))
    if kind == "faers_live":
        from tool_use.tools import faers_reaction_counts

        target = faers_reaction_counts(drug=check["drug"], reaction=check["reaction"])[
            "reports_with_drug_and_reaction"
        ]
        tolerance = check.get("tolerance_rel", 0.02) * target
        return any(abs(n - target) <= tolerance for n in numbers_in(answer))
    if kind == "faers_top":
        from tool_use.tools import faers_reaction_counts

        top = faers_reaction_counts(drug=check["drug"], top=check.get("n", 5))["top_reactions"]
        names = [r["reaction"].lower() for r in top]
        return sum(n in answer.lower() for n in names) >= check.get("min", 3)
    if kind == "n_casos":
        return has_number(answer, len(store.filter(DRUG, **check.get("filtro", {})))) or (
            not store.filter(DRUG, **check.get("filtro", {}))
            and re.search(r"(?i)ningún|ninguno|no hay", answer) is not None
        )
    if kind == "meddra_notif":
        coded = coded_terms_lines(answer)
        if not all(any(term_in(t, line) for line in coded) for t in check["esperados"]):
            return False
        return not any(any(term_in(t, line) for line in coded) for t in check.get("ausentes", []))
    if kind == "ids_filtro":
        ids = [c.id for c in store.filter(DRUG, **check.get("filtro", {}))]
        return (
            sum(i in answer for i in ids) >= min(check.get("min", 1), len(ids))
            if ids
            else (re.search(r"(?i)ningún|ninguno|no hay", answer) is not None)
        )
    if kind == "menciona_farmacos":
        from crawler.text import fold

        low = fold(answer)
        return sum(fold(d) in low for d in check["farmacos"]) >= check.get("min", 1)
    if kind == "citas_docs":  # fragments of at least `min` different documents (sheets)
        return len({c.split("-")[0] for c in CHUNK_ID.findall(answer)}) >= check.get("min", 2)
    if kind == "prr_live_todos":
        nums = numbers_in(answer)
        for drug in check["drugs"]:
            target = live_prr(drug, check["reaction"])
            tolerance = check.get("tolerance_rel", 0.03) * target
            if not any(abs(n - target) <= tolerance for n in nums):
                return False
        return True
    if kind == "regex_ausente":
        return re.search(check["value"], answer) is None
    if kind == "cita_fragmento":
        return CHUNK_ID.search(answer) is not None
    if kind == "ids_graves":
        serious = [c.id for c in cases if c.grave]
        return bool(serious) and all(i in answer for i in serious)
    if kind == "alguno":
        return any(check_one(c, answer, response, store) for c in check["de"])
    if kind == "herramientas_distintas":
        used = {s["action"]["name"] for s in response.get("steps", []) if s.get("action")}
        return len(used & set(check["de"])) >= check.get("min", 1)
    from agent.evaluate import _check_one

    return _check_one(check, answer, response, None)


def unsupported_numbers(answer: str, task: str, response: dict) -> list[float]:
    """Numbers in the answer that no observation (nor the request) contains. Case and
    fragment ids are removed first: their digits are not claims."""
    seen = " ".join(s.get("observation") or "" for s in response.get("steps", [])) + " " + task
    # Token by token: the loose pattern of numbers_in joined a JSON list ("14.979, 16.234") into
    # one number, so a confidence interval copied from the PRR tool looked invented (2026-11-23).
    support = {
        n for token in WRITTEN_NUMBER.findall(CASE_ID.sub(" ", seen)) for n in numbers_in(token)
    }
    clean = CHUNK_ID.sub(" ", CASE_ID.sub(" ", answer))
    whole = {int(x) for x in support if x == int(x) and x < 1e8}
    # Simple arithmetic on observed numbers is not an invention: "the other 157" (159 - 2).
    derived = {a + b for a in whole for b in whole} | {a - b for a in whole for b in whole}
    # ...nor a percentage of two observed counts: "159 of 725, 21,9 %".
    percents = [100 * a / b for a in whole for b in whole if 0 < a < b]

    def backed(n: float) -> bool:
        if n in support or abs(n) <= 10 or 1900 <= n <= 2100:  # small counts, years
            return True
        if n == int(n) and int(n) in derived:
            return True
        if n < 100 and any(abs(n - pc) <= 0.05 for pc in percents):
            return True
        return any(abs(n - s) <= 0.0051 * max(abs(s), 1) for s in support)  # rounding

    out = set()
    # A written number ('2,62', '1.680') has several readings: it is backed if any of them is.
    for token in WRITTEN_NUMBER.findall(clean):
        readings = numbers_in(token)
        if readings and not any(backed(n) for n in readings):
            out.add(max(readings))
    return sorted(out)


def meddra_from_traces(store: CaseStore, items) -> None:
    """The MedDRA terms the coordinator saw, read back from its ``codificar_meddra``
    observations, for the cases the store has not coded (scoring outside the DGX, where the
    cache is). Only term-like values: a failed coding is not a coding."""
    from coordinator.tools import is_term

    for item in items:
        for step in item.get("steps", []):
            action = step.get("action") or {}
            if action.get("name") != "codificar_meddra":
                continue
            try:
                observed = json.loads(step.get("observation") or "")
            except json.JSONDecodeError:
                continue
            terms = [t for t in observed.get("terminos_meddra", []) if is_term(t)]
            if terms and observed.get("caso") and observed["caso"] not in store.meddra:
                store.meddra[observed["caso"]] = terms


def unsupported_citations(answer: str, response: dict) -> list[str]:
    """Fragment ids cited in the answer that no observation contains (invented citations)."""
    seen = " ".join(s.get("observation") or "" for s in response.get("steps", []))
    return sorted({i for i in CHUNK_ID.findall(answer) if i not in seen})


# Which source each tool consults: the routing of a request is the set of sources it used.
SOURCE_OF = {
    "listar_casos": ("casos",),
    "ver_caso": ("casos",),
    "codificar_meddra": ("meddra",),
    "codificar_notificacion": ("meddra",),
    "agrupar_por_meddra": ("meddra",),
    "search_knowledge_base": ("rag",),
    "faers_reaction_counts": ("faers",),
    "disproportionality_prr": ("faers",),
    "comparar_senal": ("faers",),
    "buscar_literatura": ("literatura",),
    "buscar_evidencia": ("literatura", "redes"),  # the old sub-agent searched both
    "buscar_mas_casos": ("redes",),
}
SOURCES = ("casos", "meddra", "rag", "faers", "literatura", "redes")


def routing(task: dict, used: list[str]) -> dict:
    """Expected sources (``fuentes``) against the sources the coordinator consulted."""
    expected = set(task.get("fuentes", []))
    seen = {src for t in used for src in SOURCE_OF.get(t, ())}
    return {
        "fuentes_esperadas": sorted(expected),
        "fuentes_usadas": sorted(seen),
        "fuentes_faltan": sorted(expected - seen),
        "fuentes_de_mas": sorted(seen - expected),
    }


def score_task(task: dict, response: dict, store: CaseStore) -> dict:
    answer = response.get("final_answer", "")
    checks = task["comprobar"] if isinstance(task["comprobar"], list) else [task["comprobar"]]
    results = []
    for c in checks:
        try:
            results.append(check_one(c, answer, response, store))
        except Exception:  # noqa: BLE001 - a live check that fails counts as not passed
            results.append(False)
    used = [s["action"]["name"] for s in response.get("steps", []) if s.get("action")]
    return {
        "id": task["id"],
        "conjunto": task.get("conjunto", ""),
        "tipo": task.get("tipo", "dirigida"),
        "categoria": task["categoria"],
        **(routing(task, used) if "fuentes" in task else {}),
        "correcta": all(results),
        "comprobaciones": results,
        "herramientas_usadas": used,
        "herramientas_requeridas": all(t in used for t in task["herramientas"]),
        "numeros_sin_respaldo": unsupported_numbers(answer, task["tarea"], response),
        "citas_sin_respaldo": unsupported_citations(answer, response),
        "pasos": response.get("n_steps", 0),
        "tokens": response.get("tokens_used", 0),
        "parada": response.get("stop_reason", ""),
    }


def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "tareas": len(rows),
        "exito": sum(r["correcta"] for r in rows) / n,
        "herramientas_requeridas": sum(r["herramientas_requeridas"] for r in rows) / n,
        "respuestas_con_numeros_sin_respaldo": sum(bool(r["numeros_sin_respaldo"]) for r in rows),
        "respuestas_con_citas_sin_respaldo": sum(bool(r["citas_sin_respaldo"]) for r in rows),
        "pasos_medios": sum(r["pasos"] for r in rows) / n,
        "tokens_medios": sum(r["tokens"] for r in rows) / n,
        "segundos_medios": sum(r.get("segundos", 0) for r in rows) / n,
        "paradas": dict(Counter(r["parada"] for r in rows)),
        "por_tipo": {
            kind: {
                "tareas": sum(r["tipo"] == kind for r in rows),
                "exito": sum(r["correcta"] for r in rows if r["tipo"] == kind)
                / max(sum(r["tipo"] == kind for r in rows), 1),
                "pasos_medios": sum(r["pasos"] for r in rows if r["tipo"] == kind)
                / max(sum(r["tipo"] == kind for r in rows), 1),
            }
            for kind in ("dirigida", "abierta")
        },
        "por_categoria": {
            cat: {
                "tareas": sum(r["categoria"] == cat for r in rows),
                "exito": sum(r["correcta"] for r in rows if r["categoria"] == cat)
                / max(sum(r["categoria"] == cat for r in rows), 1),
            }
            for cat in sorted({r["categoria"] for r in rows})
        },
        **(routing_summary(rows) if rows and "fuentes_esperadas" in rows[0] else {}),
        "por_tarea": {r["id"]: r["correcta"] for r in rows},
    }


def routing_summary(rows: list[dict]) -> dict:
    """How often the coordinator went to every source it needed, how often it consulted one it
    did not need, and, per source, needed vs used (a confusion of the routing)."""
    n = max(len(rows), 1)
    per_source = {}
    for src in SOURCES:
        needed = [r for r in rows if src in r["fuentes_esperadas"]]
        used = [r for r in rows if src in r["fuentes_usadas"]]
        per_source[src] = {
            "necesaria": len(needed),
            "usada_cuando_necesaria": sum(src in r["fuentes_usadas"] for r in needed),
            "usada_sin_necesitarla": sum(src not in r["fuentes_esperadas"] for r in used),
        }
    return {
        "rutas": {
            "todas_las_fuentes_necesarias": sum(not r["fuentes_faltan"] for r in rows) / n,
            "alguna_fuente_de_mas": sum(bool(r["fuentes_de_mas"]) for r in rows) / n,
            "por_fuente": per_source,
        }
    }


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from coordinator.agent import coordinar, warm_up
    from coordinator.tools import VARIANTS
    from crawler.text import utf8_stdout

    load_dotenv(".env")
    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variantes", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--tareas", nargs="+", help="only these task ids")
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--de-nuevo", action="store_true", help="ignore previous answers")
    parser.add_argument(
        "--solo-puntuar",
        action="store_true",
        help="only score the answers already saved (no model: works outside the DGX)",
    )
    parser.add_argument("--bench", type=Path, default=BENCH, help="tareas.jsonl or tareas_v2.jsonl")
    parser.add_argument(
        "--conjunto",
        choices=["dev", "test", "todos"],
        default="todos",
        help="v2: development (look and improve) or test (once, at the end)",
    )
    parser.add_argument("--etiqueta", default="", help="suffix to repeat a run, e.g. rep2")
    parser.add_argument("--report", type=Path, default=Path("reports/coordinator_eval.json"))
    args = parser.parse_args(argv)

    store = CaseStore.from_files()
    slots = placeholders(store)
    bench = [
        json.loads(line) for line in args.bench.read_text("utf-8").splitlines() if line.strip()
    ]
    bench = [
        fill(t, slots)
        for t in bench
        if args.conjunto == "todos" or t.get("conjunto") == args.conjunto
    ]
    # Files and report keys of the v1 benchmark keep their names; others carry the bench name.
    suffix = (
        ("" if args.bench == BENCH else f"_{args.bench.stem}")
        + (f"_{args.conjunto}" if args.conjunto != "todos" else "")
        + (f"_{args.etiqueta}" if args.etiqueta else "")
    )
    # --tareas chooses what to run; the summary always scores every task already answered
    # (running only the new tasks must not drop the others from the report).
    tasks = [t for t in bench if not args.tareas or t["id"] in args.tareas]
    print(f"{len(store.of_drug(DRUG))} casos de {DRUG} · {len(tasks)} tareas · casos: {slots}")
    report = json.loads(args.report.read_text("utf-8")) if args.report.exists() else {}
    out_dir = FOLDER / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    brains: dict = {}
    for variant in args.variantes:
        key = f"{variant}·{args.brain}" + suffix.replace("_", "·")
        traces = out_dir / f"{variant}_{args.brain}{suffix}_respuestas.jsonl"
        if args.de_nuevo:
            traces.unlink(missing_ok=True)
        done = {}
        if traces.exists():
            for line in traces.read_text("utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    done[item["id"]] = item
        print(f"\n=== {key}" + (f" (reanudando: {len(done)} hechas)" if done else ""))
        if not args.solo_puntuar:
            warm_up(variant)
        for task in tasks:
            if task["id"] in done or args.solo_puntuar:
                continue
            start = time.perf_counter()
            try:
                response = coordinar(
                    task["tarea"], store, args.brain, variant, args.max_steps, brains=brains
                ).model_dump()
            except Exception as exc:  # noqa: BLE001 - one failed task must not stop the run
                response = {"final_answer": "", "steps": [], "stop_reason": f"error: {exc}"}
            item = {"id": task["id"], "segundos": round(time.perf_counter() - start, 1), **response}
            with open(traces, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            done[task["id"]] = item
            row = score_task(task, item, store)
            print(
                f"  {task['id']} {'✓' if row['correcta'] else '✗'} {row['pasos']} pasos, "
                f"{row['tokens']} tokens, {item['segundos']:.0f} s · {row['parada']}"
                + (
                    f" · números sin respaldo: {row['numeros_sin_respaldo']}"
                    if row["numeros_sin_respaldo"]
                    else ""
                )
            )
        meddra_from_traces(store, done.values())
        rows = []
        for task in bench:
            if task["id"] in done:
                row = score_task(task, done[task["id"]], store)
                row["segundos"] = done[task["id"]].get("segundos", 0)
                rows.append(row)
                if args.solo_puntuar:
                    print(
                        f"  {row['id']} {'✓' if row['correcta'] else '✗'} {row['comprobaciones']}"
                        f" · números sin respaldo: {row['numeros_sin_respaldo']}"
                        f" · citas sin respaldo: {row['citas_sin_respaldo']}"
                    )
        report[key] = {**summarise(rows), "filas": rows}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        s = report[key]
        print(
            f"  éxito {100 * s['exito']:.0f} % · herramientas requeridas "
            f"{100 * s['herramientas_requeridas']:.0f} % · con números sin respaldo "
            f"{s['respuestas_con_numeros_sin_respaldo']} · con citas sin respaldo "
            f"{s['respuestas_con_citas_sin_respaldo']} · {s['tokens_medios']:.0f} tokens/tarea"
        )
        for kind, k in s["por_tipo"].items():
            if k["tareas"]:
                print(
                    f"    {kind}: {round(k['exito'] * k['tareas'])}/{k['tareas']} correctas, "
                    f"{k['pasos_medios']:.1f} pasos de media"
                )
        if "rutas" in s:
            print(
                "    por categoría: "
                + ", ".join(
                    f"{c} {round(v['exito'] * v['tareas'])}/{v['tareas']}"
                    for c, v in s["por_categoria"].items()
                )
            )
            r = s["rutas"]
            print(
                f"    rutas: todas las fuentes necesarias en "
                f"{100 * r['todas_las_fuentes_necesarias']:.0f} % · alguna fuente de más en "
                f"{100 * r['alguna_fuente_de_mas']:.0f} %"
            )
            for src, v in r["por_fuente"].items():
                print(
                    f"      {src:<10} necesaria {v['necesaria']:>3} · usada cuando hacía falta "
                    f"{v['usada_cuando_necesaria']:>3} · usada sin hacer falta "
                    f"{v['usada_sin_necesitarla']:>3}"
                )
    print(f"\n→ cifras en {args.report}; respuestas y trazas en {out_dir}")


if __name__ == "__main__":
    main()
