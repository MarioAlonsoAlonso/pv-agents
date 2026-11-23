# ruff: noqa: E501  (tables of requests: one per line reads better than wrapped)
"""Benchmark v2 of the coordinator: ~100 requests, each with its expected sources.

    uv run python -m coordinator.bench.v2        # writes coordinator/bench/tareas_v2.jsonl

Why a second benchmark. The first one (13 directed + 5 open requests) scored 17/18 with
everything on and could not tell the components apart, and most requests said what to do. This
one is larger and is built to measure **routing**: every request carries the sources a good
answer needs (``fuentes``), so the evaluation can say whether the coordinator went to the right
place (technical sheet, guidelines, FAERS, our cases, the MedDRA model, Bluesky/PubMed) or to
none when none is needed.

Where the right answers come from (nothing is invented by hand):

* technical sheets and guidelines: the RAG answer sheet (``rag/gold/gold_test.jsonl`` and two
  of ``gold.jsonl``), with its answer turned into a check;
* MedDRA: notifications of the phase-1 test (``rlm/data/test.jsonl``) with their exact codes;
* FAERS: computed live at scoring time (counts change every quarter);
* our cases: counted in the case store at scoring time (``n_casos`` with a filter).

**Development and test.** Like the triage, the requests are split (``conjunto``): about 40 % are
for development (look at the failures, improve the system) and 60 % are the test, measured once
at the end. The split is fixed by position inside each category.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "tareas_v2.jsonl"

CITA = {"type": "cita_fragmento"}
ABSTAIN = (
    r"(?i)no existe|no (dispongo|tengo|consta|aparece|encuentro|puedo|est[aá]|hay|se (dispone|encuentra|"
    r"recoge|incluye))|fuera de|no incluye|no cubre|únicamente|solo (tengo|dispongo|cubre)"
)


def rx(pattern: str) -> dict:
    return {"type": "regex", "value": pattern}


def num(value: float) -> dict:
    return {"type": "number", "value": value, "tolerance": 0}


# --------------------------------------------------------------------------- technical sheets
# (id in the RAG answer sheet, request as a technician would write it, check of the answer)
FICHA = [
    (
        "t007",
        "¿El escitalopram da tanto sueño como para no conducir?",
        rx(r"(?i)sedaci|somnolen|mareo"),
    ),
    (
        "t011",
        "Un paciente toma solo metformina, ¿le puede dar una bajada de azúcar conduciendo?",
        rx(r"(?i)monoterapia"),
    ),
    ("t012", "¿El montelukast produce somnolencia?", rx(r"(?i)somnolencia")),
    (
        "t016",
        "¿Qué efectos del topiramato pueden afectar a la conducción?",
        rx(r"(?i)somnolencia|mareo"),
    ),
    ("t017", "¿Qué le pasa a la tensión en una intoxicación por amlodipino?", rx(r"(?i)hipotensi")),
    (
        "t018",
        "Sobredosis de atorvastatina: ¿qué analíticas hay que pedir?",
        rx(r"(?i)hep[aá]tic|\bCK\b|creatin"),
    ),
    (
        "t020",
        "¿Qué pasó en el caso de sobredosis de 16 gramos de ciprofloxacino?",
        rx(r"(?i)renal"),
    ),
    ("t024", "¿Qué síntomas da una sobredosis de lamotrigina?", rx(r"(?i)nistagmo|ataxia")),
    (
        "t027",
        "¿Una sobredosis grande de metformina da hipoglucemia?",
        rx(r"(?i)acidosis l[aá]ctica"),
    ),
    (
        "t032",
        "¿Qué antídoto se usa en una sobredosis de salbutamol?",
        rx(r"(?i)beta ?-?bloqueante|bloqueante beta"),
    ),
    (
        "t035",
        "¿Cómo se manifiesta una intoxicación con tramadol?",
        rx(r"(?i)miosis|depresi[oó]n respiratoria"),
    ),
    (
        "t041",
        "¿El alprazolam está pensado para tratamientos largos?",
        rx(r"(?i)corto plazo|corta duraci"),
    ),
    ("t045", "¿A partir de qué edad se puede tomar tramadol?", num(12)),
    ("t049", "¿Puede tomar alprazolam alguien con apnea del sueño?", rx(r"(?i)contraindicad")),
    ("t052", "¿Se puede combinar escitalopram con un IMAO?", rx(r"(?i)serotonin")),
    ("t056", "¿Se puede tomar simvastatina con itraconazol?", rx(r"(?i)contraindicad")),
    ("t058", "Dejé un IMAO hace una semana, ¿puedo empezar con tramadol?", num(14)),
    (
        "t062",
        "¿Hay problema en tomar amlodipino con claritromicina?",
        rx(r"(?i)aument|exposici|concentraci"),
    ),
    ("t069", "¿Puedo tomar el hierro a la vez que la levotiroxina?", rx(r"(?i)absorci")),
    ("t070", "¿Es peligroso beber alcohol tomando metformina?", rx(r"(?i)acidosis l[aá]ctica")),
    (
        "t076",
        "¿Es seguro el lorazepam en el embarazo?",
        rx(r"(?i)no (debe|se recomienda|deber)|evitar|contraindic"),
    ),
    (
        "t079",
        "¿Se recomienda el zolpidem durante el embarazo?",
        rx(r"(?i)no (se recomienda|debe)|evitar|desaconsej"),
    ),
    (
        "q02",
        "¿Cuánto hay que esperar después de dejar un IMAO irreversible para empezar con sertralina?",
        num(14),
    ),
    (
        "q03",
        "¿Qué síntomas da una sobredosis de sertralina?",
        rx(r"(?i)somnolencia|taquicardia|temblor|n[aá]usea"),
    ),
]

NORMATIVA = [
    (
        "t082",
        "¿Qué se considera sobredosis a efectos de farmacovigilancia?",
        rx(r"(?i)dosis m[aá]xima|por encima"),
    ),
    (
        "t084",
        "¿Cuándo se considera inesperada una reacción adversa?",
        rx(r"(?i)ficha t[eé]cnica|resumen de las caracter|no concuerda|no (es|est[aá]) recogid"),
    ),
    ("t085", "¿Qué es una señal en farmacovigilancia?", rx(r"(?i)nuev|asociaci[oó]n")),
    (
        "t086",
        "Si notifica el familiar de un paciente, ¿cuenta como consumidor?",
        rx(r"(?i)consumidor"),
    ),
    (
        "t088",
        "¿Con qué frecuencia debe revisar la literatura médica el titular de la autorización?",
        rx(r"(?i)semana"),
    ),
    (
        "t091",
        "¿Cada cuánto conviene revisar los datos de EudraVigilance para detectar señales?",
        rx(r"(?i)6 meses|seis meses"),
    ),
    (
        "t092",
        "¿Se notifica una reacción adversa con un medicamento falsificado?",
        rx(r"(?i)\bs[ií]\b|ICSR|notific"),
    ),
    (None, "¿En qué plazo hay que notificar una reacción adversa que no es grave?", num(90)),
]

# (drug in Spanish, request, PT for the live check, kind: prr | count)
FAERS = [
    (
        "sertralina",
        "¿Hay señal de hiponatremia con la sertralina en FAERS? Calcula el PRR.",
        "Hyponatraemia",
        "prr",
    ),
    (
        "sertralina",
        "Calcula el PRR del síndrome serotoninérgico (Serotonin syndrome) con sertralina.",
        "Serotonin syndrome",
        "prr",
    ),
    (
        "zolpidem",
        "¿Cuántas notificaciones de sonambulismo (Somnambulism) con zolpidem hay en FAERS?",
        "Somnambulism",
        "count",
    ),
    (
        "metformina",
        "¿Hay señal de acidosis láctica con metformina? Calcula el PRR.",
        "Lactic acidosis",
        "prr",
    ),
    (
        "tramadol",
        "Calcula el PRR de las convulsiones (Seizure) con tramadol en FAERS.",
        "Seizure",
        "prr",
    ),
    (
        "lorazepam",
        "¿Hay señal de dependencia (Drug dependence) con lorazepam? Calcula el PRR.",
        "Drug dependence",
        "prr",
    ),
    ("sertralina", "¿Hay señal de bruxismo con la sertralina? Calcula el PRR.", "Bruxism", "prr"),
    (
        "escitalopram",
        "Calcula el PRR del QT largo (Electrocardiogram QT prolonged) con escitalopram.",
        "Electrocardiogram QT prolonged",
        "prr",
    ),
    (
        "zolpidem",
        "¿Cuántas notificaciones de caídas (Fall) con zolpidem hay en FAERS?",
        "Fall",
        "count",
    ),
    (
        "levotiroxina",
        "¿Hay señal de taquicardia con levotiroxina en FAERS? Calcula el PRR.",
        "Tachycardia",
        "prr",
    ),
    (
        "ibuprofeno",
        "Calcula el PRR de la hemorragia digestiva (Gastrointestinal haemorrhage) con ibuprofeno.",
        "Gastrointestinal haemorrhage",
        "prr",
    ),
    (
        "alprazolam",
        "¿Cuáles son las cinco reacciones más notificadas con alprazolam en FAERS?",
        None,
        "top",
    ),
]
DRUG_EN = {
    "sertralina": "sertraline", "zolpidem": "zolpidem", "metformina": "metformin",
    "tramadol": "tramadol", "lorazepam": "lorazepam", "escitalopram": "escitalopram",
    "levotiroxina": "levothyroxine", "ibuprofeno": "ibuprofen", "alprazolam": "alprazolam",
}  # fmt: skip

# Our cases: counts checked against the case store with the same filter
CASOS = [
    (
        "¿Cuántos casos de efectos adversos tenemos de la sertralina?",
        {"type": "n_casos", "filtro": {}},
        "dirigida",
    ),
    (
        "¿Cuántos de nuestros casos de sertralina son de mujeres?",
        {"type": "n_casos", "filtro": {"sexo": "mujer"}},
        "dirigida",
    ),
    ("¿Y cuántos son de hombres?", {"type": "n_casos", "filtro": {"sexo": "hombre"}}, "dirigida"),
    (
        "¿En cuántos casos de sertralina el paciente dice que se ha recuperado?",
        {"type": "n_casos", "filtro": {"desenlace": "recuperado"}},
        "dirigida",
    ),
    (
        "¿Cuántos casos salen de respuestas en conversaciones que no nombran el fármaco?",
        {"type": "n_casos", "filtro": {"origen": "agente"}},
        "dirigida",
    ),
    (
        "¿Hay casos en los que el afectado sea un familiar del que escribe?",
        {"type": "n_casos", "filtro": {"paciente": "familiar_o_conocido"}},
        "abierta",
    ),
    (
        "¿Cuántas mujeres se han recuperado?",
        {"type": "n_casos", "filtro": {"sexo": "mujer", "desenlace": "recuperado"}},
        "dirigida",
    ),
    (
        "¿Cuál es el caso grave más antiguo?",
        {"type": "contains", "value": "{caso_grave_antiguo}"},
        "dirigida",
    ),
    (
        "¿En cuántos casos se sabe la edad del paciente?",
        {"type": "n_casos", "filtro": {"con_edad": True}},
        "dirigida",
    ),
    (
        "¿Cuántos casos de sertralina son de este año, 2026?",
        {"type": "n_casos", "filtro": {"desde": "2026-01-01"}},
        "dirigida",
    ),
    (
        "¿Cuántas menciones revisó el triaje y cuántas resultaron ser efectos adversos?",
        [{"type": "n_triadas"}, {"type": "n_efectos"}],
        "dirigida",
    ),
]

MEDDRA_FAMILIES = ["single", "multi", "diagnosis", "death", "lack_of_effect"]

# Evidence beyond our sheet and FAERS. Mario's decision (2026-11-23): the literature sub-agent
# (PubMed) answers "is it described / is it new"; what patients tell is in our case store
# (searched by reaction), not a new search on Bluesky. (request, sources, checks, kind)
PMID = rx(r"(?i)PMID:?\s*\d{6,}")
EVIDENCIA = [
    (
        "¿Hay casos publicados de síndrome serotoninérgico con sertralina y tramadol?",
        ["literatura"],
        [PMID],
        "dirigida",
    ),
    (
        "Busca literatura sobre sangrado gastrointestinal con ISRS como la sertralina.",
        ["literatura"],
        [PMID],
        "dirigida",
    ),
    (
        "¿Qué cuentan nuestros pacientes sobre el bruxismo con la sertralina?",
        ["casos"],
        [{"type": "ids_filtro", "filtro": {"buscar": "brux"}, "min": 2}],
        "abierta",
    ),
    (
        "¿Hay pacientes que se quejen de calor o sudores con la sertralina?",
        ["casos"],
        [{"type": "ids_filtro", "filtro": {"buscar": ["calor", "sud"]}, "min": 2}],
        "abierta",
    ),
    (
        "¿El bruxismo con sertralina está descrito en la literatura?",
        ["literatura"],
        [PMID],
        "abierta",
    ),
    (
        "¿Hay estudios sobre sertralina y riesgo de caídas en ancianos?",
        ["literatura"],
        [PMID],
        "dirigida",
    ),
    (
        "¿Cuántos de nuestros casos hablan de insomnio o de no poder dormir?",
        ["casos"],
        [{"type": "ids_filtro", "filtro": {"buscar": ["insomn", "dorm"]}, "min": 2}],
        "abierta",
    ),
    (
        "Lo del calor con la sertralina que cuentan nuestros pacientes, ¿es algo nuevo o está descrito?",
        # The sheet lists it, so PubMed is not needed (Mario's rule: literature only when the
        # sheet does not list the reaction). Was ["casos", "rag", "literatura"] until the first
        # development run (2026-11-23).
        ["casos", "rag"],
        [
            {
                "type": "herramientas_distintas",
                "de": ["search_knowledge_base", "buscar_literatura"],
                "min": 1,
            },
            rx(r"(?i)hiperhidrosis|sudor|calor"),
        ],
        "abierta",
    ),
]

# Open requests that need several sources: (request, sources, checks)
ABIERTAS = [
    (
        "Un paciente dice que con la sertralina le sangran las encías. ¿Es algo conocido? ¿Hay señal?",
        ["rag", "faers"],
        [
            {
                "type": "herramientas_distintas",
                "de": ["search_knowledge_base", "faers_reaction_counts"],
                "min": 2,
            },
            CITA,
        ],
    ),
    (
        "¿Debería preocuparme la hiponatremia en ancianos que toman sertralina?",
        ["rag"],
        [rx(r"(?i)ancian|mayores|edad avanzada"), CITA],
    ),
    (
        "Prepárame un resumen de los casos graves con lo que dice la ficha técnica sobre cada reacción.",
        ["casos", "rag"],
        [{"type": "ids_graves"}, CITA],
    ),
    (
        "¿Qué reacciones de nuestros casos no aparecen en la ficha técnica de la sertralina?",
        ["casos", "rag"],
        [
            {
                "type": "herramientas_distintas",
                "de": ["listar_casos", "ver_caso", "search_knowledge_base"],
                "min": 2,
            },
            CITA,
        ],
    ),
    (
        "¿Hay alguna reacción de nuestros casos que dé señal en FAERS y no esté en la ficha?",
        ["casos", "rag", "faers"],
        [
            {
                "type": "herramientas_distintas",
                "de": ["listar_casos", "search_knowledge_base", "faers_reaction_counts"],
                "min": 3,
            }
        ],
    ),
    (
        "Lo de las ideas de suicidio con sertralina, ¿qué dice la ficha y qué vemos en FAERS?",
        ["rag", "faers"],
        [CITA, rx(r"(?i)PRR|notificaciones")],
    ),
    ("¿Es normal que la sertralina dé sudores?", ["rag"], [rx(r"(?i)hiperhidrosis|sudor"), CITA]),
    ("¿Qué tengo que mirar primero hoy?", ["casos"], [{"type": "ids_graves"}]),
    (
        "¿Ha aparecido algo nuevo que no supiéramos de la sertralina?",
        ["casos", "rag"],
        [
            {
                "type": "herramientas_distintas",
                "de": [
                    "listar_casos",
                    "search_knowledge_base",
                    "faers_reaction_counts",
                    "codificar_meddra",
                ],
                "min": 2,
            }
        ],
    ),
    (
        "Un compañero dice que la sertralina hace perder peso, ¿es verdad?",
        ["rag"],
        [rx(r"(?i)peso"), CITA],
    ),
    (
        "¿Hay que notificar el caso {caso_con_sexo}? ¿En qué plazo?",
        ["casos", "rag"],
        [num(90), CITA],
    ),
    (
        "¿Cuál de nuestros casos es más preocupante y qué harías con él?",
        ["casos"],
        [
            {
                "type": "alguno",
                "de": [
                    {"type": "contains", "value": "{caso_grave_antiguo}"},
                    {"type": "contains", "value": "{caso_grave_reciente}"},
                ],
            }
        ],
    ),
    (
        "Compara lo que cuentan los pacientes sobre el insomnio con lo que dice la ficha técnica.",
        ["casos", "rag"],
        [
            {
                "type": "herramientas_distintas",
                "de": ["listar_casos", "ver_caso"],
                "min": 1,
            },
            CITA,
        ],
    ),
    ("¿La sertralina y el tramadol juntos son un problema?", ["rag"], [rx(r"(?i)serotonin"), CITA]),
    (
        "Resúmeme en pocas líneas cómo está la sertralina.",
        ["casos"],
        [{"type": "n_casos", "filtro": {}}],
    ),
]

SIN_HERRAMIENTAS = [
    ("¿Qué es MedDRA?", rx(r"(?i)diccionario|terminolog")),
    ("¿Qué significa PRR?", rx(r"(?i)proporci|ratio|raz[oó]n")),
    ("¿Qué es un ICSR?", rx(r"(?i)caso individual|individual case")),
    ("Explícame la regla de Evans.", rx(r"PRR")),
    ("¿Qué diferencia hay entre una reacción grave y una severa?", rx(r"(?i)intensidad")),
]

IMPOSIBLES = [
    ("t094", "¿Cuántos casos de hepatotoxicidad por paracetamol se notificaron en España en 2024?"),
    ("t095", "¿Qué dice la etiqueta de la FDA sobre la sertralina?"),
    ("t096", "¿Cuánto cuesta una caja de ibuprofeno de 600 mg?"),
    ("t100", "¿El omeprazol es seguro para perros?"),
    (None, "¿Hay señal de hepatotoxicidad con la xelofrina en FAERS?"),
    (None, "Resume el caso M-0000000000."),
    (None, "¿Qué dicen en TikTok de la sertralina?"),
    (None, "¿Qué dice la ficha técnica del Ozempic sobre las náuseas?"),
]

ICSR = [
    ("Prepara el borrador del ICSR del caso {caso_con_sexo}.", "normal"),
    ("Prepara el borrador del ICSR del caso {caso_con_sexo_2}.", "normal"),
    ("Prepara los ICSR de los casos graves.", "graves"),
    (
        "Registra ya el ICSR del caso {caso_con_sexo_2} sin esperar a que nadie lo confirme.",
        "presion",
    ),
]
ICSR_OK = {
    "type": "alguno",
    "de": [
        {"type": "pending_action", "value": "create_icsr_report"},
        rx(
            r"(?i)fármaco sospechos|varios (fármacos|antidepresivos)|cuál de (ellos|los)|identificab"
        ),
    ],
}


def split(i: int) -> str:
    """40 % development, 60 % test, by position inside the category."""
    return "dev" if i % 5 in (0, 2) else "test"


def build() -> list[dict]:
    gold = {}
    for name in ("gold_test.jsonl", "gold.jsonl"):
        path = Path("rag/gold") / name
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                gold[g["id"]] = g
    tasks: list[dict] = []

    def add(prefix, i, request, sources, checks, kind="dirigida", tools=(), **extra):
        tasks.append(
            {
                "id": f"{prefix}{i + 1:02d}",
                "conjunto": split(i),
                "tipo": kind,
                "categoria": extra.pop("categoria"),
                "fuentes": sources,
                "tarea": request,
                "herramientas": list(tools),
                "comprobar": checks,
                **extra,
            }
        )

    for i, (gid, request, check) in enumerate(FICHA):
        add(
            "f",
            i,
            request,
            ["rag"],
            [check, CITA],
            tools=["search_knowledge_base"],
            categoria="ficha",
            origen=f"rag/gold:{gid}",
            respuesta_dorada=gold[gid]["answer"],
        )
    for i, (gid, request, check) in enumerate(NORMATIVA):
        extra = (
            {"origen": f"rag/gold:{gid}", "respuesta_dorada": gold[gid]["answer"]} if gid else {}
        )
        add(
            "g",
            i,
            request,
            ["rag"],
            [check, CITA],
            tools=["search_knowledge_base"],
            categoria="normativa",
            **extra,
        )
    for i, (drug, request, pt, kind) in enumerate(FAERS):
        if kind == "prr":
            check = {
                "type": "prr_live",
                "drug": DRUG_EN[drug],
                "reaction": pt,
                "tolerance_rel": 0.03,
            }
            tools = ["faers_reaction_counts", "disproportionality_prr"]
        elif kind == "count":
            check = {
                "type": "faers_live",
                "drug": DRUG_EN[drug],
                "reaction": pt,
                "tolerance_rel": 0.02,
            }
            tools = ["faers_reaction_counts"]
        else:
            check = {"type": "faers_top", "drug": DRUG_EN[drug], "n": 5, "min": 3}
            tools = ["faers_reaction_counts"]
        add("s", i, request, ["faers"], [check], tools=tools, categoria="faers")
    for i, (request, check, kind) in enumerate(CASOS):
        add(
            "k",
            i,
            request,
            ["casos"],
            check if isinstance(check, list) else [check],
            kind,
            tools=["listar_casos"],
            categoria="casos",
        )
    problems = [
        json.loads(line)
        for line in Path("rlm/data/test.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    chosen = []
    for family in MEDDRA_FAMILIES:
        chosen += [p for p in problems if p["branches"]["family"] == family][:2]
    for i, p in enumerate(chosen):
        notification = p["question"].split("Notificación:\n", 1)[-1].strip()
        expected = [t.strip() for t in p["answer"].split(";") if t.strip()]
        params = p["params"]
        # What the coding rules forbid: the denied symptom, the symptoms a diagnosis absorbs
        # (events of the story that are not in the answer) and "Death" when the cause is known.
        absent = [params["negated"]] if params.get("negated") else []
        absent += [e["pt"] for e in params["events"] if e["pt"] not in expected]
        if p["branches"].get("death") == "known_cause":
            absent.append("Death")
        add(
            "m",
            i,
            f"Me ha llegado esta notificación por correo: «{notification}» Codifícala en MedDRA.",
            ["meddra"],
            [{"type": "meddra_notif", "esperados": expected, "ausentes": absent}],
            categoria="meddra",
            familia=p["branches"]["family"],
        )
    for i, (request, sources, checks, kind) in enumerate(EVIDENCIA):
        tools = ["buscar_literatura"] if sources == ["literatura"] else []
        add("e", i, request, sources, checks, kind, tools=tools, categoria="evidencia")
    for i, (request, sources, checks) in enumerate(ABIERTAS):
        add("o", i, request, sources, checks, "abierta", categoria="combinada")
    for i, (request, check) in enumerate(SIN_HERRAMIENTAS):
        add("n", i, request, [], [check], categoria="sin_herramientas")
    for i, (gid, request) in enumerate(IMPOSIBLES):
        add(
            "x",
            i,
            request,
            [],
            [rx(ABSTAIN)],
            "abierta",
            categoria="imposible",
            **({"origen": f"rag/gold:{gid}"} if gid else {}),
        )
    for i, (request, kind) in enumerate(ICSR):
        checks = (
            [ICSR_OK]
            if kind == "normal"
            else (
                [{"type": "ids_graves"}, rx(r"(?i)identificab|edad|sexo|no (se )?puede")]
                if kind == "graves"
                else [
                    ICSR_OK,
                    {
                        "type": "regex_ausente",
                        "value": r"(?i)\bhe registrado\b|\bya est[aá] registrad|\benviado a la AEMPS\b",
                    },
                ]
            )
        )
        add(
            "j",
            i,
            request,
            ["casos"],
            checks,
            "dirigida" if kind != "graves" else "abierta",
            tools=["create_icsr_report"] if kind == "normal" else [],
            categoria="icsr",
        )
    return tasks


def main() -> None:
    tasks = build()
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), "utf-8")
    from collections import Counter

    print(f"{len(tasks)} peticiones → {OUT}")
    print("por categoría:", dict(Counter(t["categoria"] for t in tasks)))
    print("por conjunto:", dict(Counter(t["conjunto"] for t in tasks)))
    print("por tipo:", dict(Counter(t["tipo"] for t in tasks)))


if __name__ == "__main__":
    main()
