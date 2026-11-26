"""Change I: the triage tells the MedDRA coder the reactions (rlm/evaluate_pista.py and
coordinator/cases.py), with a fake triage and a fake coder: no network, no GPU."""

import json

from coordinator.cases import Case, triage_hint
from rlm.evaluate_pista import notification_of, paired, triage_all, with_hint


class _FakeTriage:
    name, last_prompt_tokens = "fake", 10

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        prompt = messages[-1]["content"]
        out = []
        for block in prompt.split("### id: ")[1:]:
            i = block.split()[0]
            out.append(
                {
                    "id": i,
                    "es_efecto_adverso": "si",
                    "reacciones": "tos seca",
                    "negadas": "diarrea" if "diarrea" in block else "",
                    "paciente": "autor",
                    "edad": "desconocido",
                    "sexo": "desconocido",
                    "grave": "no_consta",
                    "desenlace": "desconocido",
                }
            )
        return json.dumps(out, ensure_ascii=False), 5

    def count_tokens(self, text):
        return len(text.split())


PROBLEM = {
    "question": "Codifica…\n\nNotificación:\nCon lamotrigina tuve una tos seca. No tuve diarrea.",
    "answer": "Cough",
    "params": {"drug": "lamotrigina"},
}


def test_the_triage_reads_the_notification_and_the_hint_reaches_the_coder(tmp_path):
    assert notification_of(PROBLEM).startswith("Con lamotrigina")
    hints = triage_all([PROBLEM], tmp_path / "t.jsonl", llm_factory=lambda: _FakeTriage())
    hinted = with_hint(PROBLEM, hints[PROBLEM["question"]])
    assert hinted["question"].startswith(PROBLEM["question"])
    assert "tos seca" in hinted["question"] and "lo que niega: diarrea" in hinted["question"]
    # resumable: the second time nothing is triaged again (the fake would still answer)
    again = triage_all([PROBLEM], tmp_path / "t.jsonl", llm_factory=lambda: None)
    assert again == hints


def test_paired_comparison():
    a = [{"question_base": "q1", "ok": True}, {"question_base": "q2", "ok": True}]
    b = [{"question_base": "q1", "ok": False}, {"question_base": "q2", "ok": True}]
    out = paired(a, b, "ok")
    assert (out["pares"], out["solo_con_pista"], out["solo_sin_pista"]) == (2, 1, 0)


def test_the_coordinator_gives_the_hint_only_when_asked(monkeypatch):
    case = Case(
        id="M-1",
        farmaco="sertralina",
        fecha="2026-01-01",
        texto="me da muchísimo sueño, sin náuseas",
        conversacion="",
        encontrado_por="busqueda_fija",
        tipo_mencion="nombre",
        triaje={"es_efecto_adverso": "si", "reacciones": "muchísimo sueño", "negadas": "náuseas"},
    )
    assert case.report_text() == "me da muchísimo sueño, sin náuseas"  # off by default
    monkeypatch.setenv("ARCA_MEDDRA_PISTA", "1")
    assert "lo que niega: náuseas" in case.report_text()
    assert triage_hint(None) == "" and "ningún efecto" in triage_hint({"es_efecto_adverso": "no"})
