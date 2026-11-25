"""The 27B on the phase-1 MedDRA test, with a scripted model (no network)."""

from rlm.evaluate_remote import load_problems, mcnemar, run


class _Fake:
    last_prompt_tokens = 10

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        question = messages[-1]["content"]
        answer = "Cough" if "tos" in question else "Headache"
        return f"<think>\nreglas\n</think>\n\n<answer>{answer}</answer>", 20


def test_run_scores_with_the_verifier_and_resumes(tmp_path):
    problems = [
        {
            "question": "Notificación: tuve una tos seca",
            "answer": "Cough",
            "branches": {"family": "single"},
        },
        {
            "question": "Notificación: me dolía la tripa",
            "answer": "Abdominal pain",
            "branches": {"family": "single"},
        },
    ]
    done = {}
    rows = run(
        problems[:1], False, lambda: _Fake(), done, lambda: None, workers=1, progress=lambda s: None
    )
    assert rows[0]["is_correct"] and rows[0]["has_valid_format"]
    rows = run(
        problems, False, lambda: _Fake(), done, lambda: None, workers=1, progress=lambda s: None
    )
    assert [r["is_correct"] for r in rows] == [True, False]
    small = [
        {"question": p["question"], "is_correct": ok}
        for p, ok in zip(problems, [False, True], strict=True)
    ]
    assert mcnemar(rows, small) == {"pares": 2, "solo_27b": 1, "solo_1.7b": 1, "p_mcnemar": 1.0}


def test_the_phase1_sets_load():
    test = load_problems("rlm/data/test.jsonl")
    assert len(test) == 200 and "Reglas de codificación" in test[0]["question"]
