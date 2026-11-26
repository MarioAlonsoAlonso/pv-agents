# ruff: noqa: E501  (tables of requests: one per line reads better than wrapped)
"""Benchmark v5 of the coordinator: the earlier stages (24 requests).

    uv run python -m coordinator.bench.v5        # writes coordinator/bench/tareas_v5.jsonl
    uv run python -m coordinator.evaluate --bench coordinator/bench/tareas_v5.jsonl --variantes completo sin_etapas

Written on 2026-11-26 with the tools of the earlier stages (``coordinator/etapas.py``). Until
then the coordinator only saw the cases: it could not say what the crawler found, what the
triage discarded, or that a drug had never been crawled. Five kinds:

* ``menciones``: what the crawler found, i.e. what it hands to the triage;
* ``descartes``: what the triage said was not an adverse event, and why;
* ``estado``: where each stage is (crawled until when, waiting for triage);
* ``falta_etapa``: a question about the end (cases, MedDRA) for a watched drug that was never
  crawled: the right answer says what is missing and offers to crawl, never "there are no
  cases" as if they had been looked for;
* ``lanzar``: the technician asks to crawl. In the benchmark ``rastrear`` / ``triar`` are
  always queued, never run (``hold_stages``), so the check is the queued action; a drug that is
  not watched must not be queued.

The counts come from the frozen sample (sertraline: 725 mentions, 159 cases), the same store
the other benchmarks use. ``sin_etapas`` (the coordinator without these tools) is the baseline.
Split as v2: about 40 % development, 60 % test.
"""

from __future__ import annotations

import json
from pathlib import Path

from coordinator.bench.v2 import rx, split

OUT = Path(__file__).parent / "tareas_v5.jsonl"
MENCIONES = ["listar_menciones", "estado_del_sistema"]
NO_RASTREADO = rx(
    r"(?i)no se ha rastreado|nunca se ha rastreado|sin rastrear|no (est[aá]|hay) rastread|rastrea"
)


def uses(tools, n=1):
    return {"type": "herramientas_distintas", "de": tools, "min": n}


def n_menciones(**filtro):
    return {"type": "n_menciones", "filtro": filtro}


def n_descartes(motivo=None):
    return {"type": "n_descartes", "motivo": motivo}


PENDIENTE = {"type": "pending_action", "value": "rastrear"}

# (request, sources, checks)
TASKS = {
    "menciones": [
        (
            "¿Cuántas menciones de sertralina encontró el rastreador en Bluesky?",
            ["menciones"],
            [n_menciones(), uses(MENCIONES)],
        ),
        (
            "De las menciones de sertralina, ¿cuántas las encontró el agente del rastreador leyendo conversaciones?",
            ["menciones"],
            [n_menciones(origen="agente"), uses(MENCIONES)],
        ),
        (
            "¿Cuántas menciones de sertralina encontró la búsqueda fija (las que nombran el fármaco)?",
            ["menciones"],
            [n_menciones(origen="busqueda_fija"), uses(MENCIONES)],
        ),
        (
            "Enséñame algunas de las menciones de sertralina que encontró el rastreador en marzo de 2026.",
            ["menciones"],
            [uses(["listar_menciones"]), rx(r"M-[0-9a-f]{10}")],
        ),
        (
            "¿Qué es exactamente lo que el rastreador le pasa al triaje? Ponme un par de ejemplos reales, resumidos.",
            ["menciones"],
            [uses(["listar_menciones"]), rx(r"M-[0-9a-f]{10}")],
        ),
    ],
    "descartes": [
        (
            "¿Cuántas menciones de sertralina descartó el triaje por no ser un efecto adverso?",
            ["triaje"],
            [n_descartes(), uses(["listar_descartes", "estado_del_sistema"])],
        ),
        (
            "¿Por qué motivos descartó el triaje las menciones de sertralina?",
            ["triaje"],
            [
                uses(["listar_descartes"]),
                rx(r"(?i)broma|opini[oó]n"),
                rx(r"(?i)pregunta|no habla|sin (ning[uú]n )?efecto|no cuenta"),
            ],
        ),
        (
            "¿Cuántas menciones de sertralina eran bromas, opiniones o noticias?",
            ["triaje"],
            [n_descartes("broma_opinion_noticia"), uses(["listar_descartes"])],
        ),
        (
            "¿Cuántas menciones guardó el rastreador que en realidad no hablaban de la sertralina?",
            ["triaje"],
            [n_descartes("no_es_mencion"), uses(["listar_descartes"])],
        ),
        (
            "¿Cuánta gente dice que toma sertralina y no cuenta ningún efecto adverso?",
            ["triaje"],
            [n_descartes("sin_reaccion"), uses(["listar_descartes"])],
        ),
    ],
    "estado": [
        (
            "¿En qué punto está el sistema con la sertralina: qué se ha rastreado, triado y codificado?",
            ["estado"],
            [uses(["estado_del_sistema"]), n_menciones()],
        ),
        (
            "¿Qué falta por hacer con el lorazepam?",
            ["estado"],
            [uses(["estado_del_sistema"]), NO_RASTREADO],
        ),
        (
            "¿Hay menciones esperando a pasar por el triaje?",
            ["estado"],
            [
                uses(["estado_del_sistema", "listar_menciones"]),
                rx(r"(?i)ningun|no hay|\b0\b|cero|todas.*triad"),
            ],
        ),
        (
            "¿Hasta qué fecha llegan las menciones de sertralina que tenemos?",
            ["estado"],
            [uses(MENCIONES), rx(r"(?i)2026-11|noviembre de 2026")],
        ),
    ],
    "falta_etapa": [
        (
            "¿Hay casos graves de tramadol?",
            ["estado"],
            [uses(["estado_del_sistema"]), NO_RASTREADO],
        ),
        (
            "¿Qué efectos adversos cuentan los pacientes del zolpidem en redes sociales?",
            ["estado"],
            [uses(["estado_del_sistema"]), NO_RASTREADO],
        ),
        (
            "¿Cuántos casos de metformina tenemos de este año?",
            ["estado"],
            [uses(["estado_del_sistema"]), NO_RASTREADO],
        ),
        (
            "Codifica en MedDRA el caso más grave de alprazolam.",
            ["estado"],
            [uses(["estado_del_sistema"]), NO_RASTREADO],
        ),
        (
            "¿Qué se ha publicado en Bluesky sobre la sertralina después del 20 de noviembre de 2026?",
            ["estado"],
            [uses(["estado_del_sistema", "listar_menciones"]), NO_RASTREADO],
        ),
    ],
    "lanzar": [
        (
            "Busca en Bluesky todo lo del lorazepam desde el 1 de enero de 2026.",
            ["rastreo"],
            [PENDIENTE],
        ),
        (
            "Rastrea el tramadol desde junio de 2025 y pásalo por el triaje.",
            ["rastreo"],
            [PENDIENTE],
        ),
        (
            "Mira qué se ha dicho del escitalopram en Bluesky la última semana.",
            ["rastreo"],
            [PENDIENTE],
        ),
        (
            "Busca en Bluesky lo del omeprazol desde enero de 2026.",
            ["rastreo"],
            [
                {"type": "sin_pendientes"},
                rx(
                    r"(?i)vigilad|no (est[aá]|es) (uno de|entre)|solo (puedo|s[eé]) (buscar|rastrear)"
                ),
            ],
        ),
        (
            "¿Puedes buscar más casos de bruxismo con sertralina en Bluesky desde 2025?",
            ["rastreo"],
            [{"type": "alguno", "de": [PENDIENTE, rx(r"(?i)bruxismo")]}],
        ),
    ],
}


def build() -> list[dict]:
    tasks = []
    for kind, rows in TASKS.items():
        for i, (request, sources, checks) in enumerate(rows):
            tasks.append(
                {
                    "id": f"e{len(tasks) + 1:02d}",
                    "conjunto": split(i),
                    "tipo": "abierta",
                    "categoria": kind,
                    "fuentes": sources,
                    "tarea": request,
                    "herramientas": [],
                    "comprobar": checks,
                }
            )
    return tasks


def main() -> None:
    tasks = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(tasks)} tareas -> {OUT}")


if __name__ == "__main__":
    main()
