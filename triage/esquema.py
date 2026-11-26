"""The triage's answer: its fields, their valid values and how an answer is checked.

The fields are the ones of the answer sheet (``evaluacion_triaje/ANOTACION_TRIAJE.md``), so an
answer of the triage and a line of the sheet can be compared field by field. They live here,
with the triage, because the system needs them to work (``triage/agent.py`` validates every
answer of the 27B with them); the evaluation tool imports them from here, not the other way
round (until 2026-11-26 they were defined in ``evaluacion_triaje/lote.py``).
"""

from __future__ import annotations

LABELS = [
    "es_efecto_adverso",
    "motivo_no",
    "reacciones",
    "negadas",
    "paciente",
    "edad",
    "sexo",
    "sexo_fuente",
    "grave",
    "criterio_gravedad",
    "desenlace",
    "notas",
]
VALUES = {
    "es_efecto_adverso": {"si", "no"},
    "motivo_no": {"no_es_mencion", "pregunta_o_consejo", "broma_opinion_noticia", "sin_reaccion"},
    "paciente": {"autor", "familiar_o_conocido", "desconocido"},
    "sexo": {"mujer", "hombre", "desconocido"},
    "sexo_fuente": {"dicho", "deducido", ""},
    "grave": {"si", "no_consta"},
    "criterio_gravedad": {
        "muerte",
        "riesgo_vital",
        "hospitalizacion",
        "discapacidad",
        "otro_importante",
        "",
    },
    "desenlace": {"recuperado", "recuperandose", "no_recuperado", "desconocido"},
}
CASE_FIELDS = ["paciente", "sexo", "sexo_fuente", "grave", "criterio_gravedad", "desenlace"]


def normalise(row: dict) -> dict:
    out = {}
    for field in LABELS:
        value = (row.get(field) or "").strip()
        if field in VALUES or field == "edad":
            value = value.lower().replace("í", "i").replace("ó", "o")
        out[field] = value
    if out["es_efecto_adverso"] != "si":  # case fields only make sense for adverse events
        for field in ["reacciones", "negadas", *CASE_FIELDS, "edad"]:
            out[field] = ""
    else:
        out["motivo_no"] = ""
        out["edad"] = out["edad"] or "desconocido"
    return out


def validate(row: dict) -> str | None:
    lab = normalise(row)
    if lab["es_efecto_adverso"] not in VALUES["es_efecto_adverso"]:
        return "es_efecto_adverso debe ser si o no"
    if lab["es_efecto_adverso"] == "no":
        if lab["motivo_no"] not in VALUES["motivo_no"]:
            return f"motivo_no no válido: '{lab['motivo_no']}'"
        return None
    if not lab["reacciones"]:
        return "un efecto adverso necesita al menos una reacción"
    for field in CASE_FIELDS:
        if lab[field] not in VALUES[field]:
            return f"{field} no válido: '{lab[field]}'"
    if lab["edad"] != "desconocido" and not lab["edad"].isdigit():
        return f"edad no válida: '{lab['edad']}'"
    if lab["grave"] == "si" and not lab["criterio_gravedad"]:
        return "grave = si necesita criterio_gravedad"
    return None
