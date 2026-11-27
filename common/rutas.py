"""Where the system keeps its data, in one place (everything under ``data/`` is out of git).

Until 2026-11-26 each piece wrote its own paths: the triage (its development and exam files),
the coordinator (the same files, to build the cases), the answer-sheet tool and the crawler.
They now read them from here, so that moving a file means changing one line.

    data/
      bluesky/                         the crawler (ARCA_CRAWLER_DATA changes it; the frozen
        monitor/                         sample of the evaluation is data/bluesky_sertralina)
          menciones.jsonl                the live store: every mention the monitor found
          triaje_respuestas.jsonl        what the triage said of each one
          estado.json, en_marcha.lock
      hibrido_27b_menciones_{dev,test}.jsonl   the frozen sample's 725 mentions (sertraline)
      triaje/
        gold_triaje_{dev,test}.jsonl           the answer sheet (only to evaluate the triage)
        eval/v2_27b_mencion_{dev,test}_respuestas.jsonl   the triage's answers to the sample
      notificaciones/                  notifications registered from the coordinator (e-mail,
        menciones.jsonl                  phone): as mentions (contract 1.2), and the triage's
        triaje_respuestas.jsonl          answers, like the monitor's store
      coordinador/                     MedDRA codes already done, chat logs, evaluation traces
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path("data")
CRAWLER_DATA = Path(os.environ.get("ARCA_CRAWLER_DATA", DATA / "bluesky"))
MONITOR = CRAWLER_DATA / "monitor"

SPLITS = ("dev", "test")
MENCIONES = {s: DATA / f"hibrido_27b_menciones_{s}.jsonl" for s in SPLITS}
TRIAJE = DATA / "triaje"
TRIAJE_EVAL = TRIAJE / "eval"
GOLD_TRIAJE = {s: TRIAJE / f"gold_triaje_{s}.jsonl" for s in SPLITS}
# The triage's final configuration (v2, one post per request, 27B) on the sample.
RESPUESTAS_TRIAJE = {s: TRIAJE_EVAL / f"v2_27b_mencion_{s}_respuestas.jsonl" for s in SPLITS}
# Other annotated batches for the triage (change G, 2026-11-27): lorazepam, split by conversation
# into development and exam. "lorazepam" is a random sample of what the crawler found (the real
# share of adverse events); "extra_lorazepam" was chosen by keywords to have more adverse events
# (detection only: its share of "yes" is not the real one).
LORAZEPAM = DATA / "lorazepam"
LOTES = {
    "lorazepam": {
        s: (LORAZEPAM / f"lote_lorazepam_{s}.jsonl", LORAZEPAM / f"gold_lorazepam_{s}.jsonl")
        for s in SPLITS
    },
    "extra_lorazepam": {
        s: (LORAZEPAM / f"extra_lorazepam_{s}.jsonl", LORAZEPAM / f"gold_extra_lorazepam_{s}.jsonl")
        for s in SPLITS
    },
}
COORDINADOR = DATA / "coordinador"
# Notifications the technician registers from the coordinator (an e-mail, a phone call): the
# same two files as the monitor, so they go through the same triage and become the same cases.
NOTIFICACIONES = DATA / "notificaciones"
NOTIFICACIONES_MENCIONES = NOTIFICACIONES / "menciones.jsonl"
NOTIFICACIONES_TRIAJE = NOTIFICACIONES / "triaje_respuestas.jsonl"
