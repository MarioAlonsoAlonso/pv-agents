"""Evaluate the coordinator on the benchmark of analyst requests, with ablations.

    uv run python -m coordinator.evaluate                                   # all variants (DGX)
    uv run python -m coordinator.evaluate --variantes completo --tareas c01 c05

Each task of ``coordinator/bench/tareas.jsonl`` has the request, the tools that should be
involved and automatic checks of the final answer. Requests may name cases with placeholders
(``{caso_prioritario}``, ``{caso_con_sexo}``, ``{caso_indirecto}``), resolved from the case
store, so the benchmark has no post text and works on any data.

Check types, besides those of ``agent/evaluate.py`` (``contains``, ``regex``, ``number``,
``pending_action``):

* ``n_efectos``, ``n_triadas``, ``n_graves``: the counts of the case store are in the answer;
* ``ids_prioritarios``: at least ``min`` of the ``n`` most urgent case ids are named;
* ``meddra_del_caso``: a MedDRA term of the case (as the reasoning model coded it) is named;
* ``pt_mas_frecuente``: the most repeated term among the ``n`` most urgent coded cases;
* ``prr_live``: the PRR computed now from FAERS is in the answer;
* ``cita_fragmento``: a fragment id of the knowledge base is cited.

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
from tool_use.evaluate import _NUMBER, numbers_in

BENCH = Path(__file__).parent / "bench" / "tareas.jsonl"
CHUNK_ID = re.compile(r"\b[A-Za-z0-9_.]{3,}-s?\d{4}\b")
CASE_ID = re.compile(r"M-[0-9a-f]{10}")
DRUG = "sertralina"


def placeholders(store: CaseStore, drug: str = DRUG) -> dict[str, str]:
    cases = store.of_drug(drug)
    if not cases:
        return {}
    with_sex = [c for c in cases if c.triaje.get("sexo") in ("mujer", "hombre")]
    indirect = [c for c in cases if c.encontrado_por == "agente"]
    deduced = [c for c in indirect if "[" in c.triaje.get("reacciones", "")]
    return {
        "caso_prioritario": cases[0].id,
        "caso_con_sexo": (with_sex or cases)[0].id,
        "caso_indirecto": (deduced or indirect or cases)[0].id,
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
        return any(abs(n - target) <= check.get("tolerance", 0.05) for n in numbers_in(answer))
    if kind == "cita_fragmento":
        return CHUNK_ID.search(answer) is not None
    from agent.evaluate import _check_one

    return _check_one(check, answer, response, None)


def unsupported_numbers(answer: str, task: str, response: dict) -> list[float]:
    """Numbers in the answer that no observation (nor the request) contains. Case and
    fragment ids are removed first: their digits are not claims."""
    seen = " ".join(s.get("observation") or "" for s in response.get("steps", [])) + " " + task
    support = set(numbers_in(CASE_ID.sub(" ", seen)))
    clean = CHUNK_ID.sub(" ", CASE_ID.sub(" ", answer))

    def backed(n: float) -> bool:
        if n in support or abs(n) <= 10 or 1900 <= n <= 2100:  # small counts, years
            return True
        return any(abs(n - s) <= 0.0051 * max(abs(s), 1) for s in support)  # rounding

    out = set()
    # A written number ('2,62', '1.680') has several readings: it is backed if any of them is.
    for token in _NUMBER.findall(clean):
        readings = numbers_in(token)
        if readings and not any(backed(n) for n in readings):
            out.add(max(readings))
    return sorted(out)


def unsupported_citations(answer: str, response: dict) -> list[str]:
    """Fragment ids cited in the answer that no observation contains (invented citations)."""
    seen = " ".join(s.get("observation") or "" for s in response.get("steps", []))
    return sorted({i for i in CHUNK_ID.findall(answer) if i not in seen})


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
        "categoria": task["categoria"],
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
        "por_tarea": {r["id"]: r["correcta"] for r in rows},
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
    parser.add_argument("--report", type=Path, default=Path("reports/coordinator_eval.json"))
    args = parser.parse_args(argv)

    store = CaseStore.from_files()
    slots = placeholders(store)
    tasks = [json.loads(line) for line in BENCH.read_text("utf-8").splitlines() if line.strip()]
    tasks = [fill(t, slots) for t in tasks if not args.tareas or t["id"] in args.tareas]
    print(f"{len(store.of_drug(DRUG))} casos de {DRUG} · {len(tasks)} tareas · casos: {slots}")
    report = json.loads(args.report.read_text("utf-8")) if args.report.exists() else {}
    out_dir = FOLDER / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    brains: dict = {}
    for variant in args.variantes:
        key = f"{variant}·{args.brain}"
        traces = out_dir / f"{variant}_{args.brain}_respuestas.jsonl"
        if args.de_nuevo:
            traces.unlink(missing_ok=True)
        done = {}
        if traces.exists():
            for line in traces.read_text("utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    done[item["id"]] = item
        print(f"\n=== {key}" + (f" (reanudando: {len(done)} hechas)" if done else ""))
        warm_up(variant)
        for task in tasks:
            if task["id"] in done:
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
        rows = []
        for task in tasks:
            if task["id"] in done:
                row = score_task(task, done[task["id"]], store)
                row["segundos"] = done[task["id"]].get("segundos", 0)
                rows.append(row)
        report[key] = {**summarise(rows), "filas": rows}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        s = report[key]
        print(
            f"  éxito {100 * s['exito']:.0f} % · herramientas requeridas "
            f"{100 * s['herramientas_requeridas']:.0f} % · con números sin respaldo "
            f"{s['respuestas_con_numeros_sin_respaldo']} · {s['tokens_medios']:.0f} tokens/tarea"
        )
    print(f"\n→ cifras en {args.report}; respuestas y trazas en {out_dir}")


if __name__ == "__main__":
    main()
