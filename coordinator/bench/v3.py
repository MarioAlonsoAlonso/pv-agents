# ruff: noqa: E501  (tables of requests: one per line reads better than wrapped)
"""Benchmark v3 of the coordinator: comparisons between drugs and drug names (22 requests).

    uv run python -m coordinator.bench.v3        # writes coordinator/bench/tareas_v3.jsonl

Written on 2026-11-25, after the v2 exam, for two things v2 did not measure:

* **Comparisons** ("¿qué antidepresivo da menos insomnio?"): the coordinator has to find the
  drugs of a group, read the technical sheet of each and compare; or compare the FAERS signal
  of several drugs. Checks: the answer names at least some of the group, cites fragments of
  several sheets (``citas_docs``), or gives the live PRR of every drug (``prr_live_todos``).
* **Names**: invented drugs (the v2 failure: "xelofrina" answered with celecoxib's signal),
  invented names close to a real one ("pantoxazol"), typos ("sertralna"), CIMA brands
  ("Besitran"), English names and a real drug outside the system (duloxetine). An unknown name
  must be said to be unknown; a brand or an English name must be resolved.

None of these requests was seen when the name tools and the guard were written. Split as v2:
about 40 % development, 60 % test.
"""

from __future__ import annotations

import json
from pathlib import Path

from coordinator.bench.v2 import rx, split

OUT = Path(__file__).parent / "tareas_v3.jsonl"
UNKNOWN = rx(
    r"(?i)no (lo )?(reconozco|reconoce|identifico|identifica|encuentro|he encontrado|existe|consta|"
    r"figura|aparece|tengo)|desconocid|no reconocid|no est[aá] (entre|en)|no dispongo"
)
# "Sí, hay señal", not "para ver si hay señal" (a dorfelina answer that asked for the real name,
# development, 2026-11-25).
NO_SIGNAL_CLAIM = {
    "type": "regex_ausente",
    "value": r"(?i)(^|[.:;!?]\s*)s[ií],? hay (una )?señal|\bsí,? hay (una )?señal|señal positiva|"
    r"cumple (el criterio|la regla)",
}
ASSUMPTION = rx(
    r"(?i)entiendo que|supongo|asumo|interpreto|te refieres|quieres decir|no (lo )?(reconozco|encuentro)"
)


def mentions(drugs, n):
    return {"type": "menciona_farmacos", "farmacos": drugs, "min": n}


def docs(n):
    return {"type": "citas_docs", "min": n}


AD = ["sertralina", "escitalopram", "venlafaxina"]
BZD = ["alprazolam", "lorazepam", "lormetazepam", "zolpidem"]
AINE = ["ibuprofeno", "naproxeno", "diclofenaco", "celecoxib"]
EST = ["atorvastatina", "rosuvastatina", "simvastatina"]
AE = ["lamotrigina", "levetiracetam", "topiramato", "ácido valproico", "valproico"]

# (request, sources, checks, kind)
COMPARATIVAS = [
    ("¿Qué antidepresivo da menos insomnio?", ["rag"], [mentions(AD, 3), docs(3)], "abierta"),
    (
        "Entre sertralina, escitalopram y venlafaxina, ¿cuál tiene más señal de hiponatremia (Hyponatraemia) en FAERS?",
        ["faers"],
        [
            {
                "type": "prr_live_todos",
                "drugs": ["sertraline", "escitalopram", "venlafaxine"],
                "reaction": "Hyponatraemia",
                "tolerance_rel": 0.03,
            }
        ],
        "dirigida",
    ),
    (
        "¿Qué antiinflamatorio tiene menos riesgo de sangrado digestivo según las fichas técnicas?",
        ["rag"],
        [mentions(AINE, 3), docs(3)],
        "abierta",
    ),
    (
        "¿Qué benzodiacepina es más adecuada para una persona mayor?",
        ["rag"],
        [mentions(BZD, 3), rx(r"(?i)ancian|mayor|edad avanzada"), docs(3)],
        "abierta",
    ),
    (
        "¿Cuál de las estatinas interacciona con la claritromicina?",
        ["rag"],
        [mentions(EST, 3), docs(3)],
        "abierta",
    ),
    (
        "Compara la señal de caídas (Fall) entre zolpidem, lorazepam y alprazolam.",
        ["faers"],
        [
            {
                "type": "prr_live_todos",
                "drugs": ["zolpidem", "lorazepam", "alprazolam"],
                "reaction": "Fall",
                "tolerance_rel": 0.03,
            }
        ],
        "dirigida",
    ),
    (
        "¿Qué antiepiléptico es más seguro en el embarazo según las fichas?",
        ["rag"],
        [mentions(AE, 3), rx(r"(?i)embaraz"), docs(3)],
        "abierta",
    ),
    (
        "¿La sertralina da más náuseas que el escitalopram?",
        ["rag"],
        [mentions(["sertralina", "escitalopram"], 2), docs(2)],
        "abierta",
    ),
    (
        "¿Qué inhibidor de la bomba de protones da menos problemas de interacciones?",
        ["rag"],
        [mentions(["omeprazol", "pantoprazol"], 2), docs(2)],
        "abierta",
    ),
    (
        "¿Hay algún antidepresivo con menos problemas sexuales?",
        ["rag"],
        [mentions(AD, 3), docs(3)],
        "abierta",
    ),
    (
        "Entre metformina y empagliflozina, ¿cuál tiene más señal de cetoacidosis (Diabetic ketoacidosis)?",
        ["faers"],
        [
            {
                "type": "prr_live_todos",
                "drugs": ["metformin", "empagliflozin"],
                "reaction": "Diabetic ketoacidosis",
                "tolerance_rel": 0.03,
            }
        ],
        "dirigida",
    ),
    (
        "Para el dolor de un paciente que toma sertralina, ¿qué analgésico es más seguro?",
        ["rag"],
        [
            mentions(["paracetamol", "tramadol", "ibuprofeno", "naproxeno"], 2),
            rx(r"(?i)serotonin|sangr|hemorrag"),
            docs(2),
        ],
        "abierta",
    ),
]

# (request, sources, checks, kind)
NOMBRES = [
    (
        "¿Hay señal de hepatotoxicidad con la dorfelina en FAERS?",
        [],
        [UNKNOWN, NO_SIGNAL_CLAIM],
        "dirigida",
    ),
    ("¿Qué dice la ficha técnica del pantoxazol sobre el embarazo?", [], [UNKNOWN], "dirigida"),
    ("¿La sertralna da insomnio?", ["rag"], [rx(r"(?i)sertralina"), ASSUMPTION], "abierta"),
    (
        "¿El Besitran puede dar insomnio?",
        ["rag"],
        [rx(r"(?i)sertralina"), {"type": "cita_fragmento"}],
        "abierta",
    ),
    (
        "Is there a hyponatraemia signal with sertraline in FAERS? Give me the PRR.",
        ["faers"],
        [
            {
                "type": "prr_live",
                "drug": "sertraline",
                "reaction": "Hyponatraemia",
                "tolerance_rel": 0.03,
            }
        ],
        "dirigida",
    ),
    ("¿Hay señal de convulsiones con el tramadolex?", [], [UNKNOWN, NO_SIGNAL_CLAIM], "dirigida"),
    (
        "Calcula el PRR de la cefalea (Headache) con el zolpimed.",
        ["faers"],
        [rx(r"(?i)zolpidem"), ASSUMPTION],
        "dirigida",
    ),
    ("¿Qué dice la ficha técnica de la duloxetina sobre el insomnio?", [], [UNKNOWN], "dirigida"),
    (
        "¿El Orfidal da somnolencia?",
        ["rag"],
        [rx(r"(?i)lorazepam"), {"type": "cita_fragmento"}],
        "abierta",
    ),
    ("¿Qué casos tenemos de la xelofrina?", ["casos"], [UNKNOWN], "dirigida"),
]


def build() -> list[dict]:
    tasks = []
    for prefix, items, category in (("v", COMPARATIVAS, "comparativa"), ("w", NOMBRES, "nombres")):
        for i, (request, sources, checks, kind) in enumerate(items):
            tasks.append(
                {
                    "id": f"{prefix}{i + 1:02d}",
                    "conjunto": split(i),
                    "tipo": kind,
                    "categoria": category,
                    "fuentes": sources,
                    "tarea": request,
                    "herramientas": [],
                    "comprobar": checks,
                }
            )
    return tasks


def main() -> None:
    from collections import Counter

    tasks = build()
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), "utf-8")
    print(f"{len(tasks)} peticiones → {OUT}")
    print(
        "por categoría y conjunto:", dict(Counter((t["categoria"], t["conjunto"]) for t in tasks))
    )


if __name__ == "__main__":
    main()
