"""What the crawler watches and where it keeps its data.

Drugs: eight of the 34 in the RAG corpus (so the analyst can read their technical sheet),
chosen by their volume of Spanish posts in Bluesky (measured on 2026-11-17): psychiatric
drugs, which people talk about in first person, plus tramadol, metformin and levothyroxine
for variety. Paracetamol and ibuprofen (7 301 and 5 364 posts) are almost all noise
("tómate un ibuprofeno"); ibuprofen enters only as a small noise sample.

Data: everything under ``data/bluesky`` (``ARCA_CRAWLER_DATA``), which is in ``.gitignore``:
the posts are public but they are health data about people, and the repository is public.
"""

from __future__ import annotations

import os
from pathlib import Path

WATCHLIST: tuple[str, ...] = (
    "sertralina",
    "escitalopram",
    "lorazepam",
    "alprazolam",
    "zolpidem",
    "tramadol",
    "metformina",
    "levotiroxina",
)

# Searches that do not name a drug: they bring posts and threads where a drug may be talked
# about without its name, so the frozen sample is not built only from the queries that the
# fixed-search baseline will use (that would make the evaluation circular).
BROAD_TERMS: tuple[str, ...] = (
    "antidepresivo",
    "antidepresivos",
    "ansiolítico",
    "ansiolíticos",
    "pastillas para dormir",
    "me han recetado",
    "efectos secundarios",
    "psiquiatra medicación",
)

# A small sample of a very noisy drug, to measure precision.
NOISE_TERMS: tuple[str, ...] = ("ibuprofeno",)

DEFAULT_SINCE = "2025-01-01"
LANG = "es"

DATA_DIR = Path(os.environ.get("ARCA_CRAWLER_DATA", "data/bluesky"))
PACKAGE_DIR = Path(__file__).parent
BRANDS_FILE = PACKAGE_DIR / "data" / "marcas.json"  # public CIMA data, versioned

MENTION_TYPES = ("nombre", "marca", "errata", "indirecta", "jerga")
