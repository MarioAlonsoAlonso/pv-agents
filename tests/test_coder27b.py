"""Change A (2026-11-27): the 27B as MedDRA coder (``rlm/coder27b.py``), with a scripted model.
The same function serves the phase-1 test and the coordinator."""

from rlm.coder27b import FORMAT, code, question_of
from rlm.generate_meddra import INSTRUCTION
from rlm.meddra import MedDRADictionary
from rlm.verifier import MedDRAVerifier


class Scripted:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def generate(self, messages, max_new_tokens=1500, enable_thinking=False):
        self.seen.append(messages)
        return self.replies.pop(0), 10


def test_a_right_answer_is_taken_as_it_is():
    llm = Scripted("<think>…</think><answer>Headache; Nausea</answer>")
    c = code("Tengo dolor de cabeza y náuseas con sertralina.", llm)
    assert c.answer == "Headache; Nausea" and c.devuelto is None and len(llm.seen) == 1
    user = llm.seen[0][1]["content"]
    assert user.startswith(INSTRUCTION) and user.endswith(FORMAT)  # wrapped like the 1.7B


def test_a_short_answer_without_tags_is_read():
    c = code(question_of("me duele la cabeza"), Scripted("Headache"))
    assert c.answer == "Headache" and c.devuelto is None


def test_a_name_outside_the_dictionary_goes_back_once():
    llm = Scripted("<answer>Lack of appetite; Foo syndrome</answer>", "<answer>Nausea</answer>")
    c = code("x", llm)
    assert c.primera == "Lack of appetite; Foo syndrome" and c.answer == "Nausea"
    assert "«Foo syndrome»" in c.devuelto and "Headache" in c.devuelto  # the list of terms
    assert llm.seen[1][-1]["content"] == c.devuelto and c.fuera_del_diccionario == []
    # still outside after the second try: kept, and reported
    c = code("x", Scripted("<answer>Foo syndrome</answer>", "<answer>Bar thing</answer>"))
    assert c.answer == "Bar thing" and c.fuera_del_diccionario == ["Bar thing"]


def test_no_answer_is_asked_again_and_a_second_silence_keeps_none():
    c = code("x", Scripted("Let me think about this.\nIt is long.", "still nothing\nreally"))
    assert c.answer is None and "<answer>" in c.devuelto


def test_the_phase1_evaluation_and_the_coordinator_use_the_same_coder(monkeypatch):
    import coordinator.tools as tools
    import rlm.evaluate_remote as remote

    problem = {"question": question_of("náuseas"), "answer": "Nausea", "branches": {}}
    row = remote.solve_final(Scripted("<answer>Nausea</answer>"), problem, False, MedDRAVerifier())
    assert row["is_correct"] and row["predicted"] == "Nausea" and row["devuelto"] is None

    monkeypatch.setenv("ARCA_MEDDRA_CODER", "27b")
    monkeypatch.setattr(tools, "_REMOTE_CODER", Scripted("<answer>Somnolence</answer>"))
    assert tools.rlm_coder("me da muchísimo sueño") == "Somnolence"
    monkeypatch.setattr(tools, "_REMOTE_CODER", Scripted("nada\nde nada", "otra vez\nnada"))
    try:
        tools.rlm_coder("x")
        raise AssertionError("a coding without answer must fail")
    except tools.CodingFailed:
        pass
    assert len(MedDRADictionary.default()) <= 200  # the subset: the term list fits the prompt
