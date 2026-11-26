"""The drugs the system knows: their names, brands, therapeutic group and which sources have them.

Two uses:

* **Comparisons** ("¿qué antidepresivo da menos insomnio?"): the coordinator needs to know which
  drugs of a group have a technical sheet in the knowledge base, which are watched in social
  media and what they are called in FAERS. ``farmacos_disponibles`` lists them.
* **Names** ("xelofrina"): on 2026-11-23 the coordinator met a drug that does not exist, decided
  it was celecoxib and answered with celecoxib's signal. ``resolve`` says whether a name is an
  active ingredient, an English name or a CIMA brand of one of our drugs, and otherwise that it
  is not recognised, with similar names to *ask* the technician about (never to use instead).

Source: the system's drug catalogue (``common/farmacos.py``: the 35 drugs with their FAERS name,
group, technical sheet and whether they are watched, plus the CIMA brands of the watched ones),
where ``resolve`` lives too; this module keeps what only the coordinator needs.
"""

from __future__ import annotations

from common.farmacos import NO_SHEET, Resolution, grupos, nombre_faers, resolve  # noqa: F401
from crawler.text import fold

GROUPS = grupos()


def group_of(drug: str) -> str | None:
    return next((g for g, drugs in GROUPS.items() if drug in drugs), None)


def available(group: str | None = None, cases_by_drug: dict[str, int] | None = None) -> dict:
    """The drugs of a group (or all), with what each source has about them."""
    cases_by_drug = cases_by_drug or {}
    key = fold(group or "")
    out = {}
    for g, drugs in GROUPS.items():
        if key and key not in fold(g):
            continue
        out[g] = [
            {
                "farmaco": d,
                "nombre_faers": nombre_faers(d),
                "ficha_tecnica": d not in NO_SHEET,
                "casos_en_redes": cases_by_drug.get(d, 0),
            }
            for d in drugs
        ]
    return out
