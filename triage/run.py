"""Run the triage on the development set (or, once, on the exam) and score it.

    uv run python -m triage.run --brain palabras                 # baseline, no model, ~1 s
    uv run python -m triage.run --version v1 --limite 5          # quick check on the DGX
    uv run python -m triage.run --version v1                     # development set (DGX)
    uv run python -m triage.run --version v1 --examen            # the exam: ONCE, at the end

Outputs, under ``data/triaje/eval/`` (out of git: they contain the posts' text):
``<name>_respuestas.jsonl`` (one answer per mention), ``<name>_coste.json`` and
``<name>_errores.json`` (the failures with their text). The numbers, without text, go to
``reports/triage_eval.json`` under ``<name>``. A run that is cut (the DGX session ends) resumes
where it stopped when launched again with the same options; ``--de-nuevo`` starts over.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from crawler.store import read_mentions
from triage.agent import BRAINS, Cost, run_triage
from triage.evaluate import errors, load_jsonl, print_report, score
from triage.prompts import PROMPTS

DATA = Path("data")
EVAL = DATA / "triaje" / "eval"
SPLITS = {
    "dev": (DATA / "hibrido_27b_menciones_dev.jsonl", DATA / "triaje" / "gold_triaje_dev.jsonl"),
    "test": (DATA / "hibrido_27b_menciones_test.jsonl", DATA / "triaje" / "gold_triaje_test.jsonl"),
}


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from crawler.text import utf8_stdout

    load_dotenv(".env")
    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", default="v1", choices=sorted(PROMPTS))
    parser.add_argument("--brain", default="27b", choices=BRAINS)
    parser.add_argument(
        "--por",
        default="conversacion",
        choices=["conversacion", "mencion"],
        help="one request per conversation (default) or per mention",
    )
    parser.add_argument("--examen", action="store_true", help="the test set: only once, at the end")
    parser.add_argument("--limite", type=int, help="only the first N conversations (a quick check)")
    parser.add_argument(
        "--hilos", type=int, default=2, help="simultaneous requests (server: 5 total)"
    )
    parser.add_argument("--de-nuevo", action="store_true", help="ignore a previous partial run")
    parser.add_argument(
        "--etiqueta", default="", help="suffix for the run name, e.g. rep2 to repeat a run"
    )
    parser.add_argument("--report", type=Path, default=Path("reports/triage_eval.json"))
    args = parser.parse_args(argv)

    split = "test" if args.examen else "dev"
    mentions_file, gold_file = SPLITS[split]
    for f in (mentions_file, gold_file):
        if not f.exists():
            raise SystemExit(
                f"No encuentro {f}. Los datos del examen no están en git: pedidlos al equipo y "
                "ponedlos en esas rutas (evaluacion_triaje/GUIA_TRIAJE.md, apartado 2)."
            )
    if args.examen:
        print(
            "EXAMEN: esto mide la versión final. No cambies nada del triaje después de mirar "
            "estos resultados (evaluacion_triaje/GUIA_TRIAJE.md, apartado 5)."
        )
    mentions = read_mentions(mentions_file)
    if args.limite:
        keep = list(dict.fromkeys(m.hilo_id for m in mentions))[: args.limite]
        mentions = [m for m in mentions if m.hilo_id in set(keep)]
    version = "palabras" if args.brain == "palabras" else args.version
    name = (
        f"{version}_{args.brain}_{args.por}_{split}"
        + (f"_lim{args.limite}" if args.limite else "")
        + (f"_{args.etiqueta}" if args.etiqueta else "")
    )
    out = EVAL / f"{name}_respuestas.jsonl"
    cost_file = EVAL / f"{name}_coste.json"
    if args.de_nuevo:
        out.unlink(missing_ok=True)
        cost_file.unlink(missing_ok=True)

    print(
        f"{len(mentions)} menciones ({split}) · cerebro {args.brain} · prompt {version}"
        f" · por {args.por}"
    )
    run = run_triage(
        mentions, brain=args.brain, version=args.version, por=args.por, out=out, workers=args.hilos
    )
    cost = Cost(**json.loads(cost_file.read_text("utf-8"))) if cost_file.exists() else Cost()
    cost.add(run.cost)  # a resumed run adds to the cost of the part already done
    cost_file.write_text(json.dumps(asdict(cost), indent=1), "utf-8")

    gold = load_jsonl(gold_file)
    gold = {i: g for i, g in gold.items() if i in {m.id for m in mentions}}
    by_id = {m.id: m for m in mentions}
    result = score(run.answers, gold, by_id)
    print_report(result, asdict(cost))
    (EVAL / f"{name}_errores.json").write_text(
        json.dumps(errors(run.answers, gold, by_id), ensure_ascii=False, indent=1), "utf-8"
    )
    report = json.loads(args.report.read_text("utf-8")) if args.report.exists() else {}
    report[name] = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "conjunto": split,
        "cerebro": args.brain,
        "prompt": version,
        "por": args.por,
        **result,
        "coste": asdict(cost),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"\n→ cifras en {args.report} ['{name}']")
    print(f"→ respuestas y fallos (con texto, fuera de git) en {EVAL}/{name}_*")


if __name__ == "__main__":
    main()
