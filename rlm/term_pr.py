"""Precision and recall per MedDRA term, from the saved phase-1 evaluations (no model, no GPU).

pass@1 asks for the exact set of Preferred Terms: one missing or extra term makes the whole
answer wrong. Per term, the question is the one of pharmacovigilance: of the terms the model
gives, how many are right (precision: false signals), and of the right terms, how many it gives
(recall: missed events). A name outside the dictionary counts as a false positive, as in
``term_f1``. "Tolerant" reads a short answer written without the ``<answer>`` tags, as
``rlm/evaluate_remote.py`` does for the 27B.

    uv run python -m rlm.term_pr      # writes reports/phase1_term_pr.json
"""

from __future__ import annotations

import json
from pathlib import Path

from rlm.meddra import MedDRADictionary

P1, P27 = "reports/phase1_eval_1.7b.json", "reports/phase1_eval_27b.json"
RUNS = [
    # (name, report, key, tolerant)
    ("1.7B base · test", "reports/phase1_eval_base_1.7b.json", "base", False),
    ("1.7B SFT · test", P1, "sft", False),
    ("1.7B SFT+GRPO · test", P1, "grpo", False),
    ("1.7B SFT+GRPO · OOD", "reports/phase1_eval_ood_grpo_1.7b.json", "grpo", False),
    ("27B sin razonar · test · estricto", P27, "test·sin_thinking", False),
    ("27B sin razonar · test · tolerante", P27, "test·sin_thinking", True),
    ("27B razonando · test · tolerante", P27, "test·thinking", True),
    ("27B sin razonar · OOD · tolerante", P27, "ood·sin_thinking", True),
    ("27B razonando · OOD · tolerante", P27, "ood·thinking", True),
]  # fmt: skip


def answer_of(row: dict, tolerant: bool) -> str | None:
    """The terms the model gave: the ``<answer>`` block, or with ``tolerant`` a short final
    line without it (``lenient_correct``)."""
    if row["predicted"] is not None or not tolerant:
        return row["predicted"]
    content = row["raw"].split("</think>")[-1].strip()
    if content and len(content) <= 300 and "\n" not in content:
        return content
    return None


def term_counts(rows: list[dict], dictionary: MedDRADictionary, tolerant: bool = False) -> dict:
    tp = fp = fn = unknown = no_answer = 0
    for row in rows:
        gold = dictionary.resolve(row["expected"]).pts
        predicted = answer_of(row, tolerant)
        if predicted is None:
            no_answer += 1
            fn += len(gold)
            continue
        got = dictionary.resolve(predicted)
        tp += len(got.pts & gold)
        fp += len(got.pts - gold) + len(got.unknown)
        fn += len(gold - got.pts)
        unknown += len(got.unknown)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "problemas": len(rows),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "verdaderos": tp,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "nombres_fuera_del_diccionario": unknown,
        "sin_respuesta": no_answer,
    }


def main() -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    dictionary = MedDRADictionary.default()
    out = {}
    print(f"{'':36} precisión  recall    F1  fuera del dicc.  sin respuesta")
    for name, report, key, tolerant in RUNS:
        if not Path(report).exists():
            continue
        rows = json.loads(Path(report).read_text("utf-8"))[key]["rows"]
        r = out[name] = term_counts(rows, dictionary, tolerant)
        print(
            f"{name:36} {r['precision']:9.1%} {r['recall']:7.1%} {r['f1']:5.1%} "
            f"{r['nombres_fuera_del_diccionario']:16d} {r['sin_respuesta']:14d}"
        )
    Path("reports/phase1_term_pr.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    main()
