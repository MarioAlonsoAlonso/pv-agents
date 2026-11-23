"""Phase 1, step 1b with a better teacher: reasoning traces from the course's 27B.

    uv run python -m rlm.distill_remote                   # 1000 train problems, 1 trace each
    uv run python -m rlm.distill_remote --n-problems 20   # quick check

Why. The first student (Qwen3-1.7B, SFT + GRPO) was distilled from Qwen3-4B, which got only
51 % of its traces right (``rlm/data/sft_traces.stats.json``). On the phase-1 test the 27B with
reasoning codes better than that student when its answer is read without the format tags
(``reports/phase1_eval_27b.json``: 67.5 % vs 56.5 % on the test, 74 % vs 38 % on terms never
seen in training). A better teacher should give a better student, especially out of
distribution, where the old one was weakest.

Same recipe as ``rlm/distill.py`` (same prompt, same canonicalisation, same verifier), so the
traces are interchangeable: the 27B's reasoning comes back from the server apart and is put
back in ``<think>``; its answer often comes without tags and ``canonicalize_trace`` reads it.
Two filters keep what the student can learn from: at least ``--min-think-words`` of reasoning
(a right answer with no reasoning teaches nothing) and at most ``--max-think-words`` (the
student generates up to 1 024 tokens in GRPO; the 27B's correct answers have a median of ~400
words, and longer traces would teach it to run out of tokens).

Output: ``rlm/data/sft_traces_27b.jsonl`` (one line per trace, the format of
``sft_traces.jsonl``) and ``.stats.json`` with the acceptance per family. A run that is cut
resumes where it stopped. Then SFT and GRPO as in ``rlm/MEDDRA.md``, with this file.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rlm.distill import acceptance_stats, canonicalize_trace, think_words
from rlm.evaluate_remote import SYSTEM, load_problems
from rlm.verifier import MedDRAVerifier

TEACHER = "qwen3.8-27b-fp8 (thinking)"


def trace_row(raw: str, problem: dict, verifier, min_words: int, max_words: int) -> dict:
    trace = canonicalize_trace(raw)
    verified, reason = False, ""
    if trace is None:
        reason = "unparseable (no </think> or no final answer)"
    elif think_words(trace) < min_words:
        reason = f"reasoning shorter than {min_words} words"
    else:
        result = verifier.verify(trace, problem["answer"])
        verified = result.is_correct
        reason = "" if verified else result.detail or "wrong answer"
        if verified and think_words(trace) > max_words:
            verified, reason = False, f"reasoning longer than {max_words} words"
    return {
        "question": problem["question"],
        "answer": problem["answer"],
        "trace": trace if trace is not None else raw,
        "verified": verified,
        "rejection": reason,
        "family": (problem.get("branches") or {}).get("family", "all"),
        "teacher": TEACHER,
    }


def distill(
    problems: list[dict],
    llm_factory,
    out: Path,
    samples: int = 1,
    max_new_tokens: int = 3000,
    min_words: int = 15,
    max_words: int = 600,
    workers: int = 3,
    progress=print,
) -> list[dict]:
    rows = []
    if out.exists():
        rows = [json.loads(line) for line in out.read_text("utf-8").splitlines() if line.strip()]
    done: dict[str, int] = {}
    for r in rows:
        done[r["question"]] = done.get(r["question"], 0) + 1
    jobs = [p for p in problems for _ in range(max(0, samples - done.get(p["question"], 0)))]
    if rows:
        progress(f"reanudando: {len(rows)} trazas hechas, {len(jobs)} pendientes")
    verifier = MedDRAVerifier()
    lock = threading.Lock()

    def work(problem):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem["question"]},
        ]
        raw, _ = llm_factory().generate(
            messages, max_new_tokens=max_new_tokens, enable_thinking=True
        )
        return trace_row(raw, problem, verifier, min_words, max_words)

    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, p) for p in jobs]
        for k, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one failed request must not stop the run
                progress(f"  error: {exc}")
                continue
            with lock:
                rows.append(row)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 20 == 0 or k == len(jobs):
                kept = sum(r["verified"] for r in rows)
                progress(f"  {k}/{len(jobs)} · {kept}/{len(rows)} trazas verificadas")
    return rows


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from common.llm import RemoteChatModel

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("rlm/data/train.jsonl"))
    parser.add_argument("--n-problems", type=int, help="only the first N problems")
    parser.add_argument("--samples", type=int, default=1, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--min-think-words", type=int, default=15)
    parser.add_argument("--max-think-words", type=int, default=600)
    parser.add_argument("--hilos", type=int, default=3, help="simultaneous requests (5 total)")
    parser.add_argument("--output", type=Path, default=Path("rlm/data/sft_traces_27b.jsonl"))
    args = parser.parse_args(argv)

    local = threading.local()

    def factory():
        if not hasattr(local, "llm"):
            local.llm = RemoteChatModel()
        return local.llm

    problems = load_problems(args.data, args.n_problems)
    print(f"{len(problems)} problemas × {args.samples} · profesor {TEACHER}")
    rows = distill(
        problems,
        factory,
        args.output,
        args.samples,
        args.max_new_tokens,
        args.min_think_words,
        args.max_think_words,
        args.hilos,
    )
    stats = {
        **acceptance_stats(rows),
        "teacher": TEACHER,
        "samples": args.samples,
        "max_think_words": args.max_think_words,
        "data": str(args.data),
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    print(f"aceptación {stats['acceptance']:.3f} · {stats['n_verified']} trazas verificadas")
    print(f"por familia: { {f: round(v, 2) for f, v in stats['acceptance_by_family'].items()} }")
    print(f"rechazos: {stats['rejection_reasons']}")
    print(f"→ {args.output} y {stats_path}")


if __name__ == "__main__":
    main()
