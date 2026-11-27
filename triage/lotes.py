"""Change G on the lorazepam batches: compare triage versions as the batches allow.

    uv run python -m triage.lotes --versiones v2 v3 v3p              # development
    uv run python -m triage.lotes --versiones v2 v3p --examen        # the exam, once

Run each version first with ``triage.run --por mencion --lote lorazepam`` and ``--lote
extra_lorazepam`` (same split). Then, from the saved answers (no model):

* **detection**: the adverse events found, over BOTH batches (the extra one was chosen by
  keywords to have more of them; an adverse event told without those words is not in it);
* **precision, false alarms and how many go to a person**: on the normal batch ONLY, a random
  sample of what the crawler found (the extra batch inflates the share of "yes");
* the doubtful ones (v3p) and McNemar between versions on the same mentions.

The numbers, without text, go to ``reports/triage_lorazepam.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import rutas
from common.stats import wilson
from triage.esquema import is_doubtful
from triage.evaluate import _pred, load_jsonl, mcnemar_exact, normalise_gold

NORMAL, EXTRA = "lorazepam", "extra_lorazepam"


def load(version: str, lot: str, split: str, brain: str = "27b") -> dict[str, dict] | None:
    """The saved answers of a run of ``triage.run --por mencion --lote <lot>``."""
    from triage.run import run_name

    f = rutas.TRIAJE_EVAL / f"{run_name(version, brain, 'mencion', split, lot)}_respuestas.jsonl"
    return load_jsonl(f) if f.exists() else None


def gold_of(lot: str, split: str) -> dict[str, dict]:
    from triage.run import lot_files

    return normalise_gold(load_jsonl(lot_files(lot, split)[1]))


def said_yes(answers: dict, i: str) -> bool:
    return _pred(answers, i)["es_efecto_adverso"] == "si"


def pct(k: int, n: int) -> dict:
    interval = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "valor": round(k / n, 3) if n else None,
        "ic95": [round(x, 3) for x in interval] if interval else None,
    }


def measure(answers: dict[str, dict], gold: dict[str, dict]) -> dict:
    """``answers`` and ``gold`` hold both batches, keyed by id; ``gold`` says which is normal."""
    positives = [i for i in gold if gold[i]["es_efecto_adverso"] == "si"]
    normal = [i for i in gold if gold[i]["lote"] == NORMAL]
    flagged = [i for i in normal if said_yes(answers, i)]
    right = [i for i in flagged if gold[i]["es_efecto_adverso"] == "si"]
    return {
        "deteccion_ambos_lotes": pct(sum(said_yes(answers, i) for i in positives), len(positives)),
        "deteccion_lote_normal": pct(
            len(right), sum(gold[i]["es_efecto_adverso"] == "si" for i in normal)
        ),
        "precision_lote_normal": pct(len(right), len(flagged)),
        "falsas_alarmas_lote_normal": len(flagged) - len(right),
        "a_revision_lote_normal": {
            "menciones": len(flagged),
            "de_cada_100": round(100 * len(flagged) / len(normal), 1) if normal else None,
            "dudosos": sum(is_doubtful(answers.get(i, {})) for i in flagged),
        },
        "sin_respuesta": sum(i not in answers for i in gold),
    }


def paired(a: dict, b: dict, gold: dict, ids: list[str]) -> dict:
    """McNemar on the decisions of two versions over the same mentions (right = as the sheet)."""

    def ok(answers, i):
        return said_yes(answers, i) == (gold[i]["es_efecto_adverso"] == "si")

    only_a = sum(ok(a, i) and not ok(b, i) for i in ids)
    only_b = sum(ok(b, i) and not ok(a, i) for i in ids)
    return {"solo_a": only_a, "solo_b": only_b, "p": round(mcnemar_exact(only_a, only_b), 4)}


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--versiones", nargs="+", default=["v2", "v3", "v3p"])
    parser.add_argument("--examen", action="store_true", help="the exam: only once, at the end")
    parser.add_argument("--report", type=Path, default=Path("reports/triage_lorazepam.json"))
    args = parser.parse_args(argv)
    split = "test" if args.examen else "dev"

    gold = {}
    for lot in (NORMAL, EXTRA):
        for i, g in gold_of(lot, split).items():
            gold[i] = {**g, "lote": lot}
    runs = {}
    for v in args.versiones:
        parts = [load(v, lot, split) for lot in (NORMAL, EXTRA)]
        if None in parts:
            print(f"{v}: falta ejecutarla en algún lote ({split}); se omite")
            continue
        runs[v] = {**parts[0], **parts[1]}

    n_pos = sum(g["es_efecto_adverso"] == "si" for g in gold.values())
    print(f"lorazepam · {split} · {len(gold)} menciones, {n_pos} efectos adversos (ambos lotes)")
    out = {"conjunto": split, "menciones": len(gold), "efectos_adversos": n_pos, "versiones": {}}
    for v, answers in runs.items():
        r = out["versiones"][v] = measure(answers, gold)
        d, p = r["deteccion_ambos_lotes"], r["precision_lote_normal"]
        rev = r["a_revision_lote_normal"]
        print(
            f"  {v:5} detección {d['k']}/{d['n']} · precisión (normal) {p['k']}/{p['n']}"
            f" · falsas alarmas {r['falsas_alarmas_lote_normal']} · a revisión {rev['menciones']}"
            f" ({rev['de_cada_100']} de cada 100, {rev['dudosos']} dudosos)"
        )
    names = list(runs)
    out["comparaciones"] = {}
    for k, a in enumerate(names):
        for b in names[k + 1 :]:
            c = out["comparaciones"][f"{a} frente a {b}"] = {
                "todas": paired(runs[a], runs[b], gold, list(gold)),
                "efectos_adversos": paired(
                    runs[a],
                    runs[b],
                    gold,
                    [i for i in gold if gold[i]["es_efecto_adverso"] == "si"],
                ),
            }
            print(
                f"  {a} frente a {b}: {c['todas']} · en los efectos adversos "
                f"{c['efectos_adversos']}"
            )
    report = json.loads(args.report.read_text("utf-8")) if args.report.exists() else {}
    report[split] = out
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"→ {args.report}")


if __name__ == "__main__":
    main()
