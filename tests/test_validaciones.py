"""The two validations of 2026-11-26 that only read saved answers: MedDRA precision and recall
per term, and the keyword review rule of the triage."""

from types import SimpleNamespace

from rlm.meddra import MedDRADictionary
from rlm.term_pr import term_counts
from triage.regla_revision import measure


def test_term_precision_and_recall():
    rows = [
        # right terms
        {"expected": "Nausea; Headache", "predicted": "Headache; Nausea", "raw": ""},
        # one missing, one extra, one outside the dictionary
        {"expected": "Nausea; Headache", "predicted": "Nausea; Insomnia; Foo bar", "raw": ""},
        # no <answer> block: tolerant reads the short final line
        {"expected": "Insomnia", "predicted": None, "raw": "<think>…</think>Insomnia"},
    ]
    d = MedDRADictionary.default()
    strict = term_counts(rows, d)
    assert (strict["verdaderos"], strict["falsos_positivos"], strict["falsos_negativos"]) == (
        3,
        2,
        2,
    )
    assert strict["nombres_fuera_del_diccionario"] == 1 and strict["sin_respuesta"] == 1
    tolerant = term_counts(rows, d, tolerant=True)
    assert tolerant["verdaderos"] == 4 and tolerant["sin_respuesta"] == 0


def test_the_review_rule_counts_what_it_recovers():
    m = lambda i, text: SimpleNamespace(id=i, texto=text)  # noqa: E731
    mentions = {
        "a": m("a", "me da muchísimo sueño"),  # AE, triage said yes
        "b": m("b", "tengo náuseas desde que la tomo"),  # AE missed, has a symptom
        "c": m("c", "la tomo y me va genial"),  # AE missed, no symptom word
        "d": m("d", "jaja qué sueño tengo hoy"),  # not an AE, symptom word
    }
    gold = {i: {"es_efecto_adverso": "si" if i in "abc" else "no"} for i in mentions}
    answers = {
        "a": {"es_efecto_adverso": "si"},
        "b": {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"},
        "c": {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion"},
        "d": {"es_efecto_adverso": "no", "motivo_no": "broma_opinion_noticia"},
    }
    r = measure(mentions, gold, answers)
    assert r["v2 tal cual"]["deteccion"] == "1/3"
    assert r["«no» con síntoma → revisión"] == {
        "deteccion": "2/3",
        "precision": 0.667,
        "a_revision": 2,
        "efectos_adversos_en_revision": 1,
    }
    assert r["perdidos_por_v2"]["con_un_sintoma_de_la_lista"] == 1


def test_a_call_to_a_tool_the_variant_lacks_is_not_a_use():
    from coordinator.evaluate import used_tools

    def step(name, observation):
        return {"action": {"name": name, "arguments": {}}, "observation": observation}

    response = {
        "steps": [
            step("listar_menciones", "error: unknown tool 'listar_menciones'. Available: [...]"),
            step("ver_mencion", '{"id": "M-1"}'),
            step("faers_reaction_counts", "error: openFDA 500"),  # failed, but it exists
        ]
    }
    assert used_tools(response) == {"ver_mencion", "ver_caso", "faers_reaction_counts"}
