"""Does the MedDRA coder get it right more often if the triage tells it the reactions first?

    uv run python -m rlm.evaluate_pista                          # 27B, test (200), DGX
    uv run python -m rlm.evaluate_pista --n 20                   # quick check
    uv run python -m rlm.evaluate_pista --codificador 1.7b       # the phase-1 model (GPU)
    uv run python -m rlm.evaluate_pista --solo-puntuar           # score what is saved

In the system, the coder reads the post again from scratch (``coordinator/cases.py``,
``report_text``), although the triage has already read it and written the reactions in the
patient's words and what the patient denies. The idea (2026-11-26): give the coder that as a
hint. It may help (it points at what to code, and at what not to: the denied symptoms) or hurt
(a triage error passes on to the coding).

Measured on the phase-1 test, which has the right answer for each notification (the exact set
of Preferred Terms, the same verifier as phase 1):

1. the triage (its final configuration, v2, one request per notification, 27B) reads each of the
   200 notifications, as a notification registered by the technician (contract 1.2);
2. the coder codes each notification twice: as in phase 1, and with the triage's reactions and
   denied symptoms appended;
3. the two are compared problem by problem (McNemar), strict and lenient (an answer written
   without the ``<answer>`` tags, as ``rlm/evaluate_remote.py``).

The triage answers go to ``data/coordinador/pista/`` and the rows (synthetic problems, no
personal data) to ``reports/phase1_pista_<coder>.json``; both resume where they stopped. If the
hint helps, the coordinator can use it: ``ARCA_MEDDRA_PISTA=1`` (``coordinator/cases.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path

from common import rutas
from rlm.evaluate import summarize
from rlm.evaluate_remote import lenient_correct, load_problems, run
from rlm.verifier import MedDRAVerifier

TEST = Path("rlm/data/test.jsonl")
MARK = "Notificación:\n"


def notification_of(problem: dict) -> str:
    return problem["question"].split(MARK, 1)[-1].strip()


def hint_text(answer: dict | None) -> str:
    """The same hint the coordinator gives its coder (``coordinator.cases.triage_hint``)."""
    from coordinator.cases import triage_hint

    return triage_hint(answer)


def with_hint(problem: dict, answer: dict | None) -> dict:
    return {**problem, "question": problem["question"] + hint_text(answer)}


def triage_all(problems: list[dict], out: Path, llm_factory=None) -> dict[str, dict]:
    """The triage of every notification, as notifications registered by the technician."""
    import hashlib

    from crawler.store import RawMention
    from triage.agent import read_answers, run_triage

    mentions, key = [], {}
    for p in problems:
        text = notification_of(p)
        post = "n" + hashlib.sha1(text.encode()).hexdigest()[:12]
        drug = (p.get("params") or {}).get("drug", "")
        m = RawMention(
            id=RawMention.make_id(post, drug),
            fuente="otro",
            post_id=post,
            hilo_id=post,
            farmaco=drug,
            tipo_mencion="nombre",
            texto=text,
            fecha="2026-01-01T00:00:00Z",
            encontrado_por="tecnico",
        )
        mentions.append(m)
        key[p["question"]] = m.id
    run_triage(
        mentions,
        version="v2",
        por="mencion",
        out=out,
        llm_factory=llm_factory,
        progress=lambda s: None,
    )
    answers = read_answers(out)
    return {q: answers.get(i) for q, i in key.items()}


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
    """Problem by problem: how many only the hinted coding gets right, only the plain one, p."""
    by_q = {r["question_base"]: r[key] for r in rows_b}
    pairs = [(r[key], by_q[r["question_base"]]) for r in rows_a if r["question_base"] in by_q]
    only_a = sum(a and not b for a, b in pairs)
    only_b = sum(b and not a for a, b in pairs)
    n = only_a + only_b
    p = (
        1.0
        if n == 0
        else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)
    )
    return {
        "pares": len(pairs),
        "solo_con_pista": only_a,
        "solo_sin_pista": only_b,
        "p_mcnemar": round(p, 4),
    }


def local_solver():
    """The phase-1 reasoning model (GPU), answering like ``solve`` does for the 27B."""
    from rlm.inference import ReasoningModel
    from rlm.meddra import term_f1

    model = ReasoningModel.from_env()
    model.load()
    verifier = MedDRAVerifier()

    def solve_local(_llm, problem, _thinking, _verifier):
        r = model.answer(problem["question"], expected_answer=problem["answer"])
        result = verifier.verify(r.raw, problem["answer"])
        return {
            "question": problem["question"],
            "expected": problem["answer"],
            "raw": r.raw,
            "predicted": result.predicted,
            "is_correct": result.is_correct,
            "has_valid_format": r.has_valid_format,
            "n_tokens": r.tokens_generated,
            "thinking_words": len(r.thinking.split()),
            "family": (problem.get("branches") or {}).get("family", "all"),
            "detail": result.detail,
            "f1": term_f1(result.predicted, problem["answer"], verifier.dictionary),
        }

    return solve_local


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    import rlm.evaluate_remote as remote
    from common.llm import RemoteChatModel

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--codificador", choices=["27b", "1.7b"], default="27b")
    parser.add_argument("--thinking", action="store_true", help="27B with its reasoning mode")
    parser.add_argument("--n", type=int, help="only the first N problems")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--solo-puntuar", action="store_true")
    args = parser.parse_args(argv)

    name = args.codificador + ("_thinking" if args.thinking else "")
    out = Path("reports") / f"phase1_pista_{name}.json"
    triage_file = rutas.COORDINADOR / "pista" / "triaje_test.jsonl"
    problems = load_problems(TEST, args.n)
    report = json.loads(out.read_text("utf-8")) if out.exists() else {}
    local = threading.local()

    def factory():
        if not hasattr(local, "llm"):
            local.llm = RemoteChatModel()
        return local.llm

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")

    print(f"1. triaje de {len(problems)} notificaciones → {triage_file}")
    hints = triage_all(problems, triage_file) if not args.solo_puntuar else {}
    if args.codificador == "1.7b" and not args.solo_puntuar:
        remote.solve = local_solver()  # the same loop, another coder
    verifier = MedDRAVerifier()
    results = {}
    for condition in ("sin_pista", "con_pista"):
        entry = report.setdefault(condition, {"rows": []})
        done = {r["question"]: r for r in entry["rows"]}
        batch = (
            problems
            if condition == "sin_pista"
            else [with_hint(p, hints.get(p["question"])) for p in problems]
        )

        def save_entry(entry=entry, done=done):
            entry["rows"] = list(done.values())
            save()

        print(f"2. codificación {condition} ({len(done)} ya hechas)")
        rows = (
            [done[p["question"]] for p in batch if p["question"] in done]
            if args.solo_puntuar
            else run(
                batch,
                args.thinking,
                factory,
                done,
                save_entry,
                args.hilos if args.codificador == "27b" else 1,
            )
        )
        for r in rows:
            r["question_base"] = r["question"].split("\n\n(Ayuda del triaje", 1)[0]
            r["correcto_tolerante"] = lenient_correct(r, verifier)
        entry["rows"] = rows
        entry["summary"] = summarize(rows)
        entry["pass@1_tolerante"] = sum(r["correcto_tolerante"] for r in rows) / max(len(rows), 1)
        results[condition] = rows
        save()
    report["comparacion"] = {
        "estricto": paired(results["con_pista"], results["sin_pista"], "is_correct"),
        "tolerante": paired(results["con_pista"], results["sin_pista"], "correcto_tolerante"),
    }
    save()
    for condition in ("sin_pista", "con_pista"):
        s = report[condition]["summary"]
        print(
            f"  {condition}: estricto {s['pass@1']:.3f} · tolerante "
            f"{report[condition]['pass@1_tolerante']:.3f} · F1 {s['mean_f1']:.3f}"
        )
    print(f"  con frente a sin pista: {report['comparacion']}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
