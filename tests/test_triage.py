"""Triage: parsing and repair of the model's answers, the keyword baseline, the scoring and the
command line, with a scripted model (no network, no data/)."""

import json

import pytest

from crawler.store import ConversationPost, RawMention, read_mentions
from triage.agent import fix, groups, keyword_answer, parse_answers, run_triage, triage_group
from triage.evaluate import compare, errors, mcnemar_exact, score, wilson
from triage.prompts import PROMPTS, build_prompt

EXAMPLES = "crawler/contrato/ejemplo_menciones.jsonl"


def _m(i, text, thread="h1", chain=(), origin="busqueda_fija"):
    return RawMention(
        id=f"M-{i}",
        post_id=f"p{i}",
        hilo_id=thread,
        farmaco="sertralina",
        tipo_mencion="nombre",
        texto=text,
        conversacion=[ConversationPost(post_id=f"c{k}", texto=t) for k, t in enumerate(chain)],
        fecha="2025-01-01T00:00:00Z",
        encontrado_por=origin,
    )


YES = {
    "es_efecto_adverso": "si",
    "reacciones": "náuseas",
    "paciente": "autor",
    "edad": "desconocido",
    "sexo": "mujer",
    "sexo_fuente": "deducido",
    "grave": "no_consta",
    "desenlace": "desconocido",
}
NO = {"es_efecto_adverso": "no", "motivo_no": "no_es_mencion"}


class _ScriptedLLM:
    last_prompt_tokens = 100

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, **kw):
        self.seen.append(messages)
        return self.replies.pop(0), 20


def test_every_prompt_renders_with_the_conversation():
    ms = read_mentions(EXAMPLES)
    for version in PROMPTS:
        prompt = build_prompt(version, ms[:3])
        assert all(m.id in prompt for m in ms[:3]) and "sertralina" in prompt
        assert "{" not in prompt.split("Responde SOLO")[0].replace('{"', "")  # no slot left
    reply = next(m for m in ms if m.conversacion)
    assert reply.conversacion[0].texto in build_prompt("v1", [reply])


def test_parse_keeps_valid_answers_drops_invented_ids_and_survives_a_cut_list():
    raw = (
        '<think>hmm {"id": "M-1"}</think>\n```json\n['
        + json.dumps({"id": "M-1", **YES, "es_efecto_adverso": "sí"})
        + ","
        + json.dumps({"id": "M-zz", **NO})
        + ","
        + json.dumps({"id": "M-2", "es_efecto_adverso": "no", "motivo_no": "quizá"})
        + ', {"id": "M-3", "es_efecto_adverso": "s'  # cut by the token limit
    )
    answers, problems = parse_answers(raw, {"M-1", "M-2", "M-3"})
    assert set(answers) == {"M-1"} and answers["M-1"]["es_efecto_adverso"] == "si"
    assert "motivo_no" in problems["M-2"] and problems["M-3"] == "falta la respuesta"


def test_fix_does_the_mechanical_part():
    row = fix({**YES, "es_efecto_adverso": True, "grave": "no", "criterio_gravedad": "muerte"})
    assert row["es_efecto_adverso"] == "si" and row["grave"] == "no_consta"
    assert row["criterio_gravedad"] == ""
    assert fix({**YES, "sexo": "", "sexo_fuente": "deducido"})["sexo_fuente"] == ""
    assert fix({**YES, "reacciones": ["mareo", "sueño"]})["reacciones"] == "mareo | sueño"


def test_repair_once_then_send_to_human_review():
    group = [_m(1, "La sertralina me da náuseas"), _m(2, "gracias", chain=["náuseas"])]
    llm = _ScriptedLLM(
        [
            json.dumps([{"id": "M-1", **YES, "grave": "si"}]),  # serious without criterion
            json.dumps([{"id": "M-1", **YES}]),  # repaired; M-2 still missing
        ]
    )
    answers, cost = triage_group(llm, group, "v1")
    assert answers["M-1"]["estado"] == "ok" and answers["M-1"]["sexo"] == "mujer"
    assert answers["M-2"]["estado"] == "sin_respuesta"
    assert answers["M-2"]["es_efecto_adverso"] == "si"  # conservative: a person decides
    assert cost.calls == 2 and cost.repairs == 1 and cost.no_answer == 1
    assert cost.prompt_tokens == 200 and cost.completion_tokens == 40
    repair = llm.seen[1][-1]["content"]
    assert "M-1" in repair and "M-2" in repair and "criterio" in repair


def test_groups_by_conversation_in_chunks_or_by_mention():
    ms = [_m(i, "x", thread="a" if i < 10 else "b") for i in range(12)]
    assert [len(g) for g in groups(ms, size=8)] == [8, 2, 2]
    assert len(groups(ms, por="mencion")) == 12


def test_keyword_baseline():
    assert keyword_answer(_m(1, "Con la sertralina tengo NÁUSEAS"))["es_efecto_adverso"] == "si"
    assert keyword_answer(_m(2, "Gracias 🫂"))["es_efecto_adverso"] == "no"


def test_run_resumes_from_the_answers_file(tmp_path):
    ms = [_m(1, "a", thread="h1"), _m(2, "b", thread="h2")]
    out = tmp_path / "r.jsonl"
    first = _ScriptedLLM([json.dumps([{"id": "M-1", **YES}])])
    run_triage(ms[:1], out=out, llm_factory=lambda: first, workers=1, progress=lambda s: None)
    second = _ScriptedLLM([json.dumps([{"id": "M-2", **NO}])])
    run = run_triage(ms, out=out, llm_factory=lambda: second, workers=1, progress=lambda s: None)
    assert set(run.answers) == {"M-1", "M-2"} and run.cost.calls == 1
    assert len(out.read_text("utf-8").splitlines()) == 2


def test_wilson_and_mcnemar():
    lo, hi = wilson(90, 100)
    assert lo == pytest.approx(0.826, abs=0.001) and hi == pytest.approx(0.945, abs=0.001)
    assert wilson(0, 0) is None
    assert mcnemar_exact(6, 2) == pytest.approx(0.289, abs=0.001)  # the crawler's figure
    assert mcnemar_exact(0, 0) == 1.0


def test_score_errors_and_compare():
    from evaluacion_triaje.lote import normalise

    ms = {
        m.id: m
        for m in [
            _m(1, "náuseas"),
            _m(2, "mareo", origin="agente"),
            _m(3, "gracias", origin="agente"),
            _m(4, "me va genial"),
        ]
    }
    gold = {
        "M-1": {"id": "M-1", **normalise(YES)},
        "M-2": {"id": "M-2", **normalise({**YES, "sexo": "hombre"})},
        "M-3": {"id": "M-3", **normalise(NO)},
        "M-4": {"id": "M-4", **normalise({"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"})},
    }
    answers = {
        "M-1": {"id": "M-1", **YES},  # right
        "M-3": {"id": "M-3", **YES},  # false alarm; M-2 missing: lost
        "M-4": {"id": "M-4", "es_efecto_adverso": "no", "motivo_no": "pregunta_o_consejo"},
    }
    r = score(answers, gold, ms)
    assert r["deteccion"]["k"] == 1 and r["deteccion"]["n"] == 2
    assert r["precision"]["k"] == 1 and r["precision"]["n"] == 2
    assert r["descarte_no_es_mencion"]["k"] == 0 and r["sin_respuesta"] == 1
    assert r["motivo_no"]["acierto"]["k"] == 0 and r["caso"]["sexo"]["k"] == 1
    assert r["por_origen"]["agente"]["deteccion"]["n"] == 1
    e = errors(answers, gold, ms)
    assert [x["id"] for x in e["perdidos"]] == ["M-2"]
    assert [x["id"] for x in e["falsas_alarmas"]] == ["M-3"]
    assert [x["id"] for x in e["motivo_no"]] == ["M-4"]
    c = compare(answers, {i: {"id": i, **g} for i, g in gold.items()}, gold)
    assert c["aciertos_b"] == 4 and c["solo_b_acierta"] == 2 and c["solo_a_acierta"] == 0


def test_cli_with_the_keyword_baseline(tmp_path, monkeypatch, capsys):
    import triage.run as cli

    ms = read_mentions(EXAMPLES)
    mentions = tmp_path / "m.jsonl"
    mentions.write_text("".join(m.model_dump_json() + "\n" for m in ms), "utf-8")
    gold = tmp_path / "g.jsonl"
    gold.write_text("".join(json.dumps({"id": m.id, **YES}) + "\n" for m in ms), "utf-8")
    monkeypatch.setattr(cli, "SPLITS", {"dev": (mentions, gold), "test": (mentions, gold)})
    monkeypatch.setattr(cli, "EVAL", tmp_path / "eval")
    report = tmp_path / "report.json"
    cli.main(["--brain", "palabras", "--report", str(report)])
    saved = json.loads(report.read_text("utf-8"))["palabras_palabras_conversacion_dev"]
    assert saved["menciones"] == len(ms) and saved["deteccion"]["n"] == len(ms)
    assert "texto" not in json.dumps(saved)  # numbers only in reports/
    assert (tmp_path / "eval" / "palabras_palabras_conversacion_dev_errores.json").exists()
    assert "detección" in capsys.readouterr().out
