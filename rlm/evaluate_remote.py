"""Phase 1 test with the course's 27B: does a large general model code MedDRA better than our
trained 1.7B?

    uv run python -m rlm.evaluate_remote                         # test (200) and OOD (100)
    uv run python -m rlm.evaluate_remote --modos sin_thinking    # only one mode

Same problems, same instructions (the R1-Zero system prompt and the coding rules), same
verifier (``MedDRAVerifier``: exact set of Preferred Terms) as ``rlm.evaluate``. The 27B runs
without and with its reasoning mode. Each row is compared with the trained 1.7B (SFT + GRPO,
``reports/phase1_eval_1.7b.json`` and ``reports/phase1_eval_ood_grpo_1.7b.json``) problem by
problem, with an exact McNemar test.

Why it matters: the coordinator ablation (2026-11-23) lost nothing without the reasoning model,
because the 27B coded the cases itself; but that benchmark has no right answer for the coding.
This test has one. If the 27B is clearly better, it is a candidate teacher to distil a new
1.7B (the current one was distilled from Qwen3-4B, which got only 51 % of its traces right).

The rows (synthetic problems, no personal data) go to ``reports/phase1_eval_27b.json``,
saved after every problem: a run that is cut resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rlm.evaluate import summarize
from rlm.meddra import term_f1
from rlm.rewards import has_valid_format, thinking_length
from rlm.verifier import MedDRAVerifier

# The same system prompt as rlm/data.py (R1_ZERO_SYSTEM_PROMPT), copied so that this script does
# not need the ``datasets`` library.
SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)
SETS = {
    "test": (Path("rlm/data/test.jsonl"), Path("reports/phase1_eval_1.7b.json")),
    "ood": (Path("rlm/data/test_ood.jsonl"), Path("reports/phase1_eval_ood_grpo_1.7b.json")),
}
MODES = {"sin_thinking": False, "thinking": True}


def load_problems(path: Path, n: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def solve(llm, problem: dict, thinking: bool, verifier: MedDRAVerifier) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem["question"]},
    ]
    start = time.perf_counter()
    raw, n_tokens = llm.generate(
        messages, max_new_tokens=6000 if thinking else 1500, enable_thinking=thinking
    )
    result = verifier.verify(raw, problem["answer"])
    return {
        "question": problem["question"],
        "expected": problem["answer"],
        "raw": raw,
        "predicted": result.predicted,
        "is_correct": result.is_correct,
        "has_valid_format": has_valid_format(raw),
        "n_tokens": n_tokens,
        "thinking_words": thinking_length(raw),
        "family": (problem.get("branches") or {}).get("family", "all"),
        "detail": result.detail,
        "f1": term_f1(result.predicted, problem["answer"], verifier.dictionary),
        "seconds": round(time.perf_counter() - start, 1),
    }


def lenient_correct(row: dict, verifier: MedDRAVerifier) -> bool:
    """Correct, reading a short answer written without the <answer> tags.

    Without reasoning the 27B usually writes the terms alone ("Diarrhoea; Drug ineffective"),
    and the strict verifier counts that as no answer (2026-11-23: 97 of its 124 errors on the
    test). A completion cut while reasoning, or a long text, stays wrong."""
    if row["predicted"] is not None:
        return row["is_correct"]
    content = row["raw"].split("</think>")[-1].strip()
    if content and len(content) <= 300 and "\n" not in content:
        return verifier.verify(f"<answer>{content}</answer>", row["expected"]).is_correct
    return False


def mcnemar(rows_a: list[dict], rows_b: list[dict], key: str = "is_correct") -> dict:
    """Paired comparison by question: how many only A gets right, only B, and the exact p."""
    b_by_q = {r["question"]: r[key] for r in rows_b}
    pairs = [(r[key], b_by_q[r["question"]]) for r in rows_a if r["question"] in b_by_q]
    only_a = sum(a and not b for a, b in pairs)
    only_b = sum(b and not a for a, b in pairs)
    n = only_a + only_b
    p = (
        1.0
        if n == 0
        else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)
    )
    return {"pares": len(pairs), "solo_27b": only_a, "solo_1.7b": only_b, "p_mcnemar": round(p, 4)}


def run(
    problems: list[dict],
    thinking: bool,
    llm_factory,
    done: dict[str, dict],
    save,
    workers: int = 2,
    progress=print,
) -> list[dict]:
    verifier = MedDRAVerifier()
    pending = [p for p in problems if p["question"] not in done]
    lock = threading.Lock()

    def work(problem):
        return solve(llm_factory(), problem, thinking, verifier)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for k, future in enumerate(as_completed([pool.submit(work, p) for p in pending]), 1):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one failed request must not stop the run
                progress(f"  error: {exc}")
                continue
            with lock:
                done[row["question"]] = row
                save()
            if k % 10 == 0 or k == len(pending):
                rows = [done[p["question"]] for p in problems if p["question"] in done]
                ok = sum(r["is_correct"] for r in rows)
                progress(f"  {len(rows)}/{len(problems)} · pass@1 hasta ahora {ok / len(rows):.3f}")
    return [done[p["question"]] for p in problems if p["question"] in done]


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from common.llm import RemoteChatModel

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conjuntos", nargs="+", default=list(SETS), choices=list(SETS))
    parser.add_argument("--modos", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--n", type=int, help="only the first N problems of each set")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reports/phase1_eval_27b.json"))
    parser.add_argument(
        "--solo-puntuar", action="store_true", help="only score the saved rows (no model)"
    )
    args = parser.parse_args(argv)

    report = json.loads(args.out.read_text("utf-8")) if args.out.exists() else {}
    local = threading.local()

    def factory():
        if not hasattr(local, "llm"):
            local.llm = RemoteChatModel()
        return local.llm

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")

    for name in args.conjuntos:
        data_path, small_path = SETS[name]
        problems = load_problems(data_path, args.n)
        small = json.loads(small_path.read_text("utf-8"))["grpo"]["rows"]
        for mode in args.modos:
            key = f"{name}·{mode}"
            entry = report.setdefault(key, {"rows": []})
            done = {r["question"]: r for r in entry["rows"]}

            def save_entry(entry=entry, done=done):
                entry["rows"] = list(done.values())
                save()

            print(f"\n=== 27B · {key}: {len(problems)} problemas ({len(done)} ya hechos)")
            if args.solo_puntuar:
                rows = [done[p["question"]] for p in problems if p["question"] in done]
            else:
                rows = run(problems, MODES[mode], factory, done, save_entry, args.hilos)
            entry["rows"] = rows
            entry["summary"] = summarize(rows)
            entry["pass@1"] = entry["summary"]["pass@1"]
            entry["frente_a_1.7b"] = mcnemar(rows, small)
            small_rows = [r for r in small if r["question"] in {x["question"] for x in rows}]
            entry["pass@1_1.7b_mismos"] = sum(r["is_correct"] for r in small_rows) / max(
                len(small_rows), 1
            )
            # The same lenient reading for both models (the 1.7B almost always uses the tags).
            verifier = MedDRAVerifier()
            for r in rows + small_rows:
                r["correcto_tolerante"] = lenient_correct(r, verifier)
            n = max(len(rows), 1)
            entry["pass@1_tolerante"] = sum(r["correcto_tolerante"] for r in rows) / n
            entry["pass@1_tolerante_1.7b_mismos"] = sum(
                r["correcto_tolerante"] for r in small_rows
            ) / max(len(small_rows), 1)
            entry["sin_respuesta_aun_tolerante"] = sum(
                r["predicted"] is None and not r["correcto_tolerante"] for r in rows
            )
            entry["frente_a_1.7b_tolerante"] = mcnemar(rows, small_rows, "correcto_tolerante")
            save()
            s = entry["summary"]
            print(
                f"  27B {mode}: estricto {s['pass@1']:.3f} · tolerante "
                f"{entry['pass@1_tolerante']:.3f} | 1.7B (SFT+GRPO) "
                f"{entry['pass@1_1.7b_mismos']:.3f} | tolerante: {entry['frente_a_1.7b_tolerante']}"
            )
            print(f"  por familia: { {f: round(v, 2) for f, v in s['pass@1_by_family'].items()} }")
            print(f"  errores: {s['error_types']}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
