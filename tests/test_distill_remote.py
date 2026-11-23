"""Distillation from the 27B with a scripted teacher (no network)."""

from rlm.distill_remote import distill

LONG = " ".join(["razono"] * 30)


class _Teacher:
    """Answers as the server does: reasoning apart (put back in <think>), answer without tags."""

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        question = messages[-1]["content"]
        if "tos" in question:
            return f"<think>\n{LONG}\n</think>\n\nCough", 50
        if "mareo" in question:
            return f"<think>\n{' '.join(['x'] * 700)}\n</think>\n\nDizziness", 900
        return f"<think>\n{LONG}", 3000  # cut while reasoning


def test_traces_are_verified_filtered_and_resumed(tmp_path):
    problems = [
        {"question": "tuve tos", "answer": "Cough", "branches": {"family": "single"}},
        {"question": "tuve mareo", "answer": "Dizziness", "branches": {"family": "single"}},
        {"question": "otra cosa", "answer": "Headache", "branches": {"family": "multi"}},
    ]
    out = tmp_path / "t.jsonl"
    rows = distill(problems, lambda: _Teacher(), out, workers=1, progress=lambda s: None)
    by_q = {r["question"]: r for r in rows}
    assert by_q["tuve tos"]["verified"]
    assert by_q["tuve tos"]["trace"].endswith("<answer>Cough</answer>")  # canonical format
    assert (
        not by_q["tuve mareo"]["verified"] and "longer than 600" in by_q["tuve mareo"]["rejection"]
    )
    assert "unparseable" in by_q["otra cosa"]["rejection"]
    again = distill(problems, lambda: _Teacher(), out, workers=1, progress=lambda s: None)
    assert len(again) == 3  # resumed: nothing generated twice
