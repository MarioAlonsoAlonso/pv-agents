"""The triage answer sheet: values, validation and agreement, without network or models."""

from crawler.store import ConversationPost, RawMention
from evaluacion_triaje.lote import agreement, normalise, read_sheet, validate, write_sheet


def _m(i, text, chain=()):
    return RawMention(
        id=f"M-{i}",
        post_id=f"p{i}",
        hilo_id="p0",
        farmaco="sertralina",
        tipo_mencion="nombre",
        texto=text,
        conversacion=[ConversationPost(post_id=f"c{k}", texto=t) for k, t in enumerate(chain)],
        fecha="2025-01-01T00:00:00Z",
        encontrado_por="busqueda_fija",
    )


YES = {
    "es_efecto_adverso": "sí",
    "reacciones": "me da sueño",
    "paciente": "autor",
    "edad": "",
    "sexo": "mujer",
    "sexo_fuente": "deducido",
    "grave": "no_consta",
    "desenlace": "desconocido",
}
NO = {"es_efecto_adverso": "no", "motivo_no": "broma_opinion_noticia", "reacciones": "ignored"}


def test_sheet_roundtrip_and_validation(tmp_path):
    ms = [_m(1, "La sertralina me da sueño", ["¿Qué tal te va?"]), _m(2, "Shrek > sertralina")]
    path = tmp_path / "lote.csv"
    write_sheet(ms, path, {"M-1": YES, "M-2": NO})
    rows = read_sheet(path)
    assert rows[0]["conversacion"] == "[1] ¿Qué tal te va?"
    assert validate(rows[0]) is None and validate(rows[1]) is None
    yes, no = normalise(rows[0]), normalise(rows[1])
    assert yes["es_efecto_adverso"] == "si" and yes["edad"] == "desconocido"
    assert no["reacciones"] == "" and no["motivo_no"] == "broma_opinion_noticia"
    assert validate({**YES, "reacciones": ""}) is not None  # an adverse event needs a reaction
    assert validate({**YES, "grave": "si"}) is not None  # serious needs a criterion
    assert validate({"es_efecto_adverso": "quizá"}) is not None


def test_agreement_counts_decisions_and_disagreements():
    a = [{"id": "M-1", **YES}, {"id": "M-2", **NO}, {"id": "M-3", **NO}]
    b = [{"id": "M-1", **YES}, {"id": "M-2", **YES}, {"id": "M-3", **NO}]
    r = agreement(a, b)
    assert r["menciones_comunes"] == 3
    assert r["efecto_adverso"]["si_a"] == 1 and r["efecto_adverso"]["si_b"] == 2
    assert [d["id"] for d in r["desacuerdos"]] == ["M-2"]
    assert r["sexo"]["acuerdo"] == 1.0
