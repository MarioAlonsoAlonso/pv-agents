"""The drugs the system knows: their names, brands, therapeutic group and which sources have them.

Two uses:

* **Comparisons** ("¿qué antidepresivo da menos insomnio?"): the coordinator needs to know which
  drugs of a group have a technical sheet in the knowledge base, which are watched in social
  media and what they are called in FAERS. ``farmacos_disponibles`` lists them.
* **Names** ("xelofrina"): on 2026-11-23 the coordinator met a drug that does not exist, decided
  it was celecoxib and answered with celecoxib's signal. ``resolve`` says whether a name is an
  active ingredient, an English name or a CIMA brand of one of our drugs, and otherwise that it
  is not recognised, with similar names to *ask* the technician about (never to use instead).

Sources: the Spanish-English table of the FAERS tool (``tool_use.tools.DRUG_NAMES_EN``, the 35
drugs of the knowledge base), the CIMA brands of the watched drugs (``crawler/data/marcas.json``)
and a therapeutic group per drug written here.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from crawler.text import fold

GROUPS = {
    "antidepresivo": ["sertralina", "escitalopram", "venlafaxina"],
    "benzodiacepina o hipnótico": ["alprazolam", "lorazepam", "lormetazepam", "zolpidem"],
    "antiinflamatorio (AINE)": ["ibuprofeno", "naproxeno", "diclofenaco", "celecoxib"],
    "analgésico": ["paracetamol", "tramadol"],
    "estatina": ["atorvastatina", "rosuvastatina", "simvastatina"],
    "antiepiléptico": ["lamotrigina", "levetiracetam", "topiramato", "acido valproico"],
    "inhibidor de la bomba de protones": ["omeprazol", "pantoprazol"],
    "antihipertensivo": ["amlodipino", "enalapril", "losartan"],
    "antidiabético": ["metformina", "empagliflozina", "insulina glargina"],
    "antibiótico": ["ciprofloxacino", "fosfomicina"],
    "otros": [
        "lactulosa", "levotiroxina", "montelukast", "salbutamol", "sumatriptan",
    ],
}  # fmt: skip
NO_SHEET = {"insulina glargina"}  # CIMA has no sectioned sheet for it (rag/CORPUS.md)
BRANDS_FILE = Path("crawler/data/marcas.json")


@dataclass(frozen=True)
class Resolution:
    name: str  # as written
    active: str | None  # active ingredient in Spanish, or None if not recognised
    how: str  # "principio activo", "nombre en inglés", "marca (CIMA)" or "no reconocido"
    similar: tuple[str, ...] = ()


@cache
def _tables() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    from tool_use.tools import DRUG_NAMES_EN

    group_of = {drug: g for g, drugs in GROUPS.items() for drug in drugs}
    spanish = {fold(es): es for es in group_of}
    english = {fold(en): es for es, en in DRUG_NAMES_EN.items() if fold(es) in spanish}
    brands: dict[str, str] = {}
    if BRANDS_FILE.exists():
        for drug, names in json.loads(BRANDS_FILE.read_text("utf-8")).items():
            for brand in names:
                brands[fold(brand)] = drug
    return spanish, english, brands


def group_of(drug: str) -> str | None:
    return next((g for g, drugs in GROUPS.items() if drug in drugs), None)


def resolve(name: str) -> Resolution:
    """What ``name`` is among our drugs. Exact matches only (case and accents aside): a close
    name is a suggestion, never a resolution ("pantoxazol" is not pantoprazol)."""
    spanish, english, brands = _tables()
    key = fold(name.strip())
    if key in spanish:
        return Resolution(name, spanish[key], "principio activo")
    if key in english:
        return Resolution(name, english[key], "nombre en inglés")
    if key in brands:
        return Resolution(name, brands[key], "marca (CIMA)")
    candidates = list(spanish) + list(english) + list(brands)
    close = difflib.get_close_matches(key, candidates, n=3, cutoff=0.75)
    similar = tuple(dict.fromkeys(spanish.get(c) or english.get(c) or brands[c] for c in close))
    return Resolution(name, None, "no reconocido", similar)


def available(group: str | None = None, cases_by_drug: dict[str, int] | None = None) -> dict:
    """The drugs of a group (or all), with what each source has about them."""
    from tool_use.tools import DRUG_NAMES_EN

    cases_by_drug = cases_by_drug or {}
    key = fold(group or "")
    out = {}
    for g, drugs in GROUPS.items():
        if key and key not in fold(g):
            continue
        out[g] = [
            {
                "farmaco": d,
                "nombre_faers": DRUG_NAMES_EN.get(d, d),
                "ficha_tecnica": d not in NO_SHEET,
                "casos_en_redes": cases_by_drug.get(d, 0),
            }
            for d in drugs
        ]
    return out
