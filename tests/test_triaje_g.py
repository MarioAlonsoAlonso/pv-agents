"""Change G (2026-11-27): the permissive triage (v3p, "duda"), prompt v3 and the comparison on
the lorazepam batches. No model, no data/."""

import json

from triage.agent import parse_answers
from triage.esquema import is_doubtful
from triage.lotes import measure
from triage.prompts import PROMPTS, build_prompt


def _raw(*items):
    return json.dumps(list(items), ensure_ascii=False)


def test_a_doubtful_no_goes_to_a_person_and_old_answers_do_not_change():
    no = {"id": "a", "es_efecto_adverso": "no", "motivo_no": "sin_reaccion"}
    answers, problems = parse_answers(_raw({**no, "duda": "si"}), {"a"})
    a = answers["a"]
    assert not problems and a["es_efecto_adverso"] == "si" and is_doubtful(a)
    assert "sin_reaccion" in a["notas"] and a["reacciones"] == "[no especificada]"
    # without "duda" (v0-v3) or with "duda": "no", exactly as before
    for item in (no, {**no, "duda": "no"}):
        answers, _ = parse_answers(_raw(item), {"a"})
        assert answers["a"]["es_efecto_adverso"] == "no" and answers["a"]["notas"] == ""
    yes = {"id": "b", "es_efecto_adverso": "si", "reacciones": "mareo", "duda": "si"}
    answers, _ = parse_answers(_raw(yes), {"b"})
    assert answers["b"]["reacciones"] == "mareo" and is_doubtful(answers["b"])


def test_v3_adds_the_rules_and_v3p_the_doubt():
    from crawler.store import RawMention

    m = RawMention(
        id="M-1", fuente="correo", post_id="n1", hilo_id="n1", farmaco="lorazepam",
        tipo_mencion="nombre", texto="Paciente con lorazepam y mucha somnolencia por la mañana",
        fecha="2026-11-27T10:00:00Z", encontrado_por="tecnico", motivo="", guardado="",
    )  # fmt: skip
    v2, v3, v3p = (build_prompt(v, [m]) for v in ("v2", "v3", "v3p"))
    assert "Origen: notificación recibida por correo" in v2  # a notification says so
    for rule in ("no me tumba", "al día siguiente", "reacción paradójica", "las benzos"):
        assert rule not in v2 and rule in v3
    assert '"duda"' in v3p and '"duda"' not in v3
    assert set(PROMPTS) >= {"v0", "v1", "v2", "v3", "v3p"}


def test_the_batches_are_measured_as_they_allow():
    def g(lot, ae):
        return {"es_efecto_adverso": "si" if ae else "no", "lote": lot}

    gold = {
        "n1": g("lorazepam", True),
        "n2": g("lorazepam", False),
        "n3": g("lorazepam", False),
        "e1": g("extra_lorazepam", True),
        "e2": g("extra_lorazepam", False),
    }
    yes = {"es_efecto_adverso": "si", "reacciones": "x", "notas": ""}
    answers = {
        "n1": yes,
        "n2": {**yes, "notas": "dudoso: el triaje dudó"},
        "n3": {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"},
        "e1": {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"},
        "e2": yes,  # a false alarm in the extra batch: not counted in precision
    }
    r = measure(answers, gold)
    assert r["deteccion_ambos_lotes"]["k"] == 1 and r["deteccion_ambos_lotes"]["n"] == 2
    assert r["precision_lote_normal"]["k"] == 1 and r["precision_lote_normal"]["n"] == 2
    assert r["falsas_alarmas_lote_normal"] == 1
    assert r["a_revision_lote_normal"] == {"menciones": 2, "de_cada_100": 66.7, "dudosos": 1}
