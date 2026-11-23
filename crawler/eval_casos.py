"""Evaluate the analyst's Bluesky tool (``buscar_casos`` / ``buscar_mas_casos``).

The question: "are there more people on Bluesky telling reaction R with sertraline?". The
answer key comes from the triage answer sheet: on the frozen sertraline sample, the hybrid
crawler sent 725 mentions to triage and each one is annotated (adverse event or not, and the
reactions, quoted). For each question, the **positives** are the posts annotated as an adverse
event whose reactions match the question's ``patron_gold`` (a regex written by hand and
checked against the matches, see ``construir --ver``).

    # 1. once, where the triage answer sheet is (it is in data/, not in git):
    uv run python -m crawler.eval_casos construir --ver
    #    → data/casos/gold_casos.json (post ids only, no text); copy it to the DGX
    # 2. the baseline without a model (needs data/hibrido_27b_menciones.jsonl):
    uv run python -m crawler.eval_casos evaluar --sistemas fija
    # 3. the agent, on the DGX, over the frozen sample:
    ARCA_CRAWLER_DATA=data/bluesky_sertralina \
        uv run --extra crawler python -m crawler.eval_casos evaluar --sistemas agente --brain 27b

Benches: ``crawler/bench/casos.jsonl`` (development: 5 reactions and 1 control; look at its
errors) and ``crawler/bench/casos_test.jsonl`` (test: 5 other reactions and 1 control; only
to report). Two systems:

* ``fija``: no model. Among the mentions the crawler already collected, those whose text
  contains a word of the reaction (``palabras_fija``). What the analyst would get by
  searching the crawler's own database.
* ``agente``: the sub-agent searching the frozen sample, as the analyst calls it.

Measured per question and summed: **recall** (positives found / positives), **precision**
(positives found / found posts that the answer sheet covers), posts found outside the answer
sheet (not judged), and for the controls how many posts were saved (should be 0). Plus cost.
The report keeps numbers only; the saved posts and traces go to ``data/bluesky/casos/``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from crawler.config import DATA_DIR
from crawler.stats import fmt, rate
from crawler.text import fold

BENCH = Path(__file__).parent / "bench" / "casos.jsonl"  # development: look at its errors
TEST = Path(__file__).parent / "bench" / "casos_test.jsonl"  # final exam: do not tune
GOLD = Path("data") / "casos" / "gold_casos.json"
MENTIONS = Path("data") / "hibrido_27b_menciones.jsonl"
TRIAGE_GOLD = Path("data") / "triaje" / "gold_triaje.jsonl"
REPORT = Path("reports") / "crawler_casos_eval.json"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, fold(text).lower()) is not None


def build_gold(questions: list[dict], mentions: list[dict], triage: list[dict]) -> dict:
    """Positive post ids per question, plus every post the answer sheet covers."""
    post_of = {m["id"]: m["post_id"] for m in mentions}
    adverse = [t for t in triage if t["es_efecto_adverso"] == "si" and t["id"] in post_of]
    return {
        "construido": datetime.now(UTC).isoformat(timespec="seconds"),
        "cubiertos": sorted({post_of[t["id"]] for t in triage if t["id"] in post_of}),
        "positivos": {
            q["id"]: sorted(
                {post_of[t["id"]] for t in adverse if matches(q["patron_gold"], t["reacciones"])}
            )
            for q in questions
        },
    }


def score(question: dict, found: set[str], gold: dict) -> dict:
    positives, covered = set(gold["positivos"][question["id"]]), set(gold["cubiertos"])
    judged = found & covered
    return {
        "id": question["id"],
        "control": question["control"],
        "positivos": len(positives),
        "encontrados": len(found),
        "aciertos": len(found & positives),
        "juzgados": len(judged),
        "sin_juzgar": len(found - covered),
    }


def summarize(rows: list[dict]) -> dict:
    real = [r for r in rows if not r["control"]]
    out = {
        "preguntas": len(real),
        "recall": rate(sum(r["aciertos"] for r in real), sum(r["positivos"] for r in real)),
        "precision": rate(sum(r["aciertos"] for r in real), sum(r["juzgados"] for r in real)),
        "sin_juzgar": sum(r["sin_juzgar"] for r in real),
        "controles": {r["id"]: r["encontrados"] for r in rows if r["control"]},
        "por_pregunta": {
            r["id"]: f"{r['aciertos']}/{r['positivos']} encontrados, "
            f"{r['juzgados'] - r['aciertos']} de más, {r['sin_juzgar']} sin juzgar"
            for r in real
        },
    }
    cost = [r["coste"] for r in rows if "coste" in r]
    if cost:
        out["coste"] = {
            k: round(sum(c[k] for c in cost) / len(cost), 1)
            for k in ("tokens", "pasos", "segundos", "busquedas", "hilos")
        } | {"paradas": sorted({c["parada"] for c in cost})}
    return out


def run_fixed(question: dict, mentions: list[dict], gold: dict) -> dict:
    found = {
        m["post_id"]
        for m in mentions
        if m["farmaco"] == question["farmaco"] and matches(question["palabras_fija"], m["texto"])
    }
    return score(question, found, gold)


def run_agent(question: dict, source, brands: dict, brain: str, gold: dict, out_dir: Path) -> dict:
    from crawler.agent import buscar_casos

    start = time.perf_counter()
    result = buscar_casos(question["farmaco"], question["reaccion"], source, brands, brain=brain)
    seconds = time.perf_counter() - start
    ctx, r = result.context, result.response
    found = {m.post_id for m in ctx.store.mentions.values()}
    row = score(question, found, gold)
    row["coste"] = {
        "tokens": r.tokens_used,
        "pasos": r.n_steps,
        "segundos": round(seconds, 1),
        "busquedas": len(ctx.searches),
        "hilos": len(ctx.threads_read),
        "parada": r.stop_reason,
    }
    positives = set(gold["positivos"][question["id"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{question['id']}.json").write_text(
        json.dumps(
            {
                "pregunta": question,
                "respuesta": r.final_answer,
                "puntuacion": row,
                "guardados": [
                    m.model_dump() | {"es_positivo": m.post_id in positives}
                    for m in ctx.store.mentions.values()
                ],
                "positivos_perdidos": sorted(positives - found),
                "traza": r.model_dump(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    return row


def print_summary(system: str, s: dict) -> None:
    print(f"\n== {system} ({s['preguntas']} reacciones + {len(s['controles'])} controles)")
    print(f"  recall (casos encontrados): {fmt(s['recall'])}")
    print(f"  precisión (de lo guardado, cuántos son casos): {fmt(s['precision'])}")
    print(f"  guardados fuera de la hoja de soluciones (sin juzgar): {s['sin_juzgar']}")
    for qid, text in s["por_pregunta"].items():
        print(f"    {qid}: {text}")
    for qid, n in s["controles"].items():
        print(f"  control {qid}: {n} guardados (deberían ser 0)")
    if "coste" in s:
        print(f"  coste medio por pregunta: {s['coste']}")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from crawler.text import utf8_stdout

    utf8_stdout()
    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("construir", help="answer key from the triage answer sheet")
    c.add_argument("--menciones", type=Path, default=MENTIONS)
    c.add_argument("--gold-triaje", type=Path, default=TRIAGE_GOLD)
    c.add_argument("--salida", type=Path, default=GOLD)
    c.add_argument("--ver", action="store_true", help="print the reactions each pattern takes")
    e = sub.add_parser("evaluar")
    e.add_argument("--sistemas", nargs="+", choices=["fija", "agente"], default=["agente"])
    e.add_argument("--brain", default="27b")
    e.add_argument("--bench", type=Path, default=BENCH)
    e.add_argument("--ids", nargs="+", help="only these questions")
    e.add_argument("--gold", type=Path, default=GOLD)
    e.add_argument("--menciones", type=Path, default=MENTIONS, help="for fija")
    e.add_argument("--muestra", type=Path, default=DATA_DIR / "sample.jsonl", help="for agente")
    e.add_argument("--salida", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    if args.cmd == "construir":
        questions = load_jsonl(BENCH) + load_jsonl(TEST)
        mentions, triage = load_jsonl(args.menciones), load_jsonl(args.gold_triaje)
        gold = build_gold(questions, mentions, triage)
        if args.ver:  # to check each pattern by eye (prints real text: local only)
            post_of = {m["id"]: m["post_id"] for m in mentions}
            for q in questions:
                print(f"\n## {q['id']} ({q['reaccion']})")
                for t in triage:
                    if t["es_efecto_adverso"] == "si" and post_of.get(t["id"]) in set(
                        gold["positivos"][q["id"]]
                    ):
                        print(f"   {t['reacciones'][:110]}")
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(gold, indent=1), "utf-8")
        print(f"\ncubiertos: {len(gold['cubiertos'])} posts")
        for qid, ids in gold["positivos"].items():
            print(f"  {qid}: {len(ids)} positivos")
        print(f"→ {args.salida}")
        return

    gold = json.loads(args.gold.read_text("utf-8"))
    bench = load_jsonl(args.bench)
    if args.salida == REPORT and args.bench.stem != BENCH.stem:  # each bench, its own report
        args.salida = REPORT.with_name(f"crawler_casos_eval_{args.bench.stem}.json")
    if args.ids:
        bench = [q for q in bench if q["id"] in set(args.ids)]
        args.salida = args.salida.with_name(args.salida.stem + "_parcial.json")
    report = {"sistemas": {}}
    if args.salida.exists():  # fija runs where the mentions are, agente on the DGX: one report
        report = json.loads(args.salida.read_text("utf-8"))
    report |= {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "bench": args.bench.name,
        "gold_construido": gold["construido"],
    }
    if "fija" in args.sistemas:
        mentions = load_jsonl(args.menciones)
        rows = [run_fixed(q, mentions, gold) for q in bench]
        report["sistemas"]["fija"] = {"resumen": summarize(rows), "preguntas": rows}
    if "agente" in args.sistemas:
        from crawler.cima import load_brands
        from crawler.sources import FrozenSource

        source, brands = FrozenSource.load(args.muestra), load_brands()
        out_dir = DATA_DIR / "casos" / f"{args.bench.stem}_{args.brain}"
        rows = []
        for q in bench:
            rows.append(run_agent(q, source, brands, args.brain, gold, out_dir))
            r = rows[-1]
            print(
                f"agente · {q['id']}: {r['aciertos']}/{r['positivos']} casos, "
                f"{r['encontrados']} guardados, {r['coste']['segundos']} s, "
                f"parada {r['coste']['parada']}"
            )
        report["sistemas"][f"agente_{args.brain}"] = {"resumen": summarize(rows), "preguntas": rows}
        print(f"guardados y trazas → {out_dir}")

    for system in args.sistemas:
        name = "fija" if system == "fija" else f"agente_{args.brain}"
        print_summary(name, report["sistemas"][name]["resumen"])
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"\n→ {args.salida}")


if __name__ == "__main__":
    main()
