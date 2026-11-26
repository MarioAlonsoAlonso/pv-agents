"""The drugs of the system, in one place: every piece reads them from here.

Until 2026-11-26 the same 35 drugs were written in four places that did not know about each
other: the watch list of the crawler (``crawler/config.py``), the Spanish-English names of the
FAERS tool (``tool_use/tools.py``), the therapeutic groups of the coordinator
(``coordinator/drugs.py``) and the indications of the phase-1 dataset
(``rlm/generate_meddra.py``, which also decides which technical sheets the RAG downloads). A
test checks that the phase-1 list is still the same set of drugs as this one.

Each drug: its name in Spanish (the active ingredient, without accents, as the system writes it),
the name FAERS indexes (the US adopted name: paracetamol is "acetaminophen"), its therapeutic
group, whether the crawler watches it in Bluesky and whether the knowledge base has its
technical sheet. The brands come from CIMA (``crawler/data/marcas.json``, built by
``crawler.cima`` for the watched drugs).

``resolve`` says what a name is: an active ingredient, its English name or a Spanish brand,
or nothing (with similar names to *ask* about, never to use instead: "xelofrina" is not
celecoxib).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from crawler.text import fold


@dataclass(frozen=True)
class Farmaco:
    es: str
    en: str  # the name FAERS indexes
    grupo: str
    vigilado: bool = False  # watched by the crawler in Bluesky
    ficha: bool = True  # technical sheet in the knowledge base (RAG)


# The eight watched drugs, in the crawler's order (its reports and benchmarks use it).
VIGILADOS: tuple[str, ...] = (
    "sertralina",
    "escitalopram",
    "lorazepam",
    "alprazolam",
    "zolpidem",
    "tramadol",
    "metformina",
    "levotiroxina",
)

_TABLE = [
    ("sertralina", "sertraline", "antidepresivo"),
    ("escitalopram", "escitalopram", "antidepresivo"),
    ("venlafaxina", "venlafaxine", "antidepresivo"),
    ("alprazolam", "alprazolam", "benzodiacepina o hipnótico"),
    ("lorazepam", "lorazepam", "benzodiacepina o hipnótico"),
    ("lormetazepam", "lormetazepam", "benzodiacepina o hipnótico"),
    ("zolpidem", "zolpidem", "benzodiacepina o hipnótico"),
    ("ibuprofeno", "ibuprofen", "antiinflamatorio (AINE)"),
    ("naproxeno", "naproxen", "antiinflamatorio (AINE)"),
    ("diclofenaco", "diclofenac", "antiinflamatorio (AINE)"),
    ("celecoxib", "celecoxib", "antiinflamatorio (AINE)"),
    ("paracetamol", "acetaminophen", "analgésico"),
    ("tramadol", "tramadol", "analgésico"),
    ("atorvastatina", "atorvastatin", "estatina"),
    ("rosuvastatina", "rosuvastatin", "estatina"),
    ("simvastatina", "simvastatin", "estatina"),
    ("lamotrigina", "lamotrigine", "antiepiléptico"),
    ("levetiracetam", "levetiracetam", "antiepiléptico"),
    ("topiramato", "topiramate", "antiepiléptico"),
    ("acido valproico", "valproic acid", "antiepiléptico"),
    ("omeprazol", "omeprazole", "inhibidor de la bomba de protones"),
    ("pantoprazol", "pantoprazole", "inhibidor de la bomba de protones"),
    ("amlodipino", "amlodipine", "antihipertensivo"),
    ("enalapril", "enalapril", "antihipertensivo"),
    ("losartan", "losartan", "antihipertensivo"),
    ("metformina", "metformin", "antidiabético"),
    ("empagliflozina", "empagliflozin", "antidiabético"),
    ("insulina glargina", "insulin glargine", "antidiabético"),
    ("ciprofloxacino", "ciprofloxacin", "antibiótico"),
    ("fosfomicina", "fosfomycin", "antibiótico"),
    ("lactulosa", "lactulose", "otros"),
    ("levotiroxina", "levothyroxine", "otros"),
    ("montelukast", "montelukast", "otros"),
    ("salbutamol", "albuterol", "otros"),
    ("sumatriptan", "sumatriptan", "otros"),
]
NO_SHEET = {"insulina glargina"}  # CIMA has no sectioned sheet for it (rag/CORPUS.md)

CATALOGO: tuple[Farmaco, ...] = tuple(
    Farmaco(es, en, grupo, vigilado=es in VIGILADOS, ficha=es not in NO_SHEET)
    for es, en, grupo in _TABLE
)
POR_NOMBRE = {f.es: f for f in CATALOGO}
BRANDS_FILE = Path(__file__).resolve().parent.parent / "crawler" / "data" / "marcas.json"


def grupos() -> dict[str, list[str]]:
    """Therapeutic group -> its drugs, in the catalogue's order."""
    out: dict[str, list[str]] = {}
    for f in CATALOGO:
        out.setdefault(f.grupo, []).append(f.es)
    return out


def nombre_faers(drug: str) -> str:
    f = POR_NOMBRE.get(drug)
    return f.en if f else drug


@dataclass(frozen=True)
class Resolution:
    name: str  # as written
    active: str | None  # active ingredient in Spanish, or None if not recognised
    how: str  # "principio activo", "nombre en inglés", "marca (CIMA)" or "no reconocido"
    similar: tuple[str, ...] = ()


@cache
def _tables() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    spanish = {fold(f.es): f.es for f in CATALOGO}
    english = {fold(f.en): f.es for f in CATALOGO}
    brands: dict[str, str] = {}
    if BRANDS_FILE.exists():
        for drug, names in json.loads(BRANDS_FILE.read_text("utf-8")).items():
            for brand in names:
                brands[fold(brand)] = drug
    return spanish, english, brands


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
