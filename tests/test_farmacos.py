"""The drug catalogue (common/farmacos.py): one list for the whole system."""

from common.farmacos import CATALOGO, VIGILADOS, grupos, nombre_faers, resolve
from crawler.text import fold


def test_the_catalogue_is_the_one_every_piece_reads():
    from coordinator.drugs import GROUPS
    from crawler.config import WATCHLIST
    from tool_use.tools import DRUG_NAMES_EN

    assert WATCHLIST == VIGILADOS and len(VIGILADOS) == 8
    assert all(DRUG_NAMES_EN[f.es] == f.en for f in CATALOGO)
    assert GROUPS == grupos() and sum(map(len, GROUPS.values())) == len(CATALOGO) == 35
    assert nombre_faers("paracetamol") == "acetaminophen" and nombre_faers("xyz") == "xyz"


def test_the_phase1_drugs_are_the_catalogue():
    """The phase-1 indications decide which technical sheets the RAG downloads: if a drug is
    added there and not here (or the other way round), this fails."""
    from rlm.generate_meddra import INDICATIONS

    phase1 = {fold(d) for _, _, drugs in INDICATIONS for d in drugs}
    assert phase1 == {fold(f.es) for f in CATALOGO}


def test_resolve_names_brands_and_unknown():
    assert resolve("Sertralína").active == "sertralina"
    assert resolve("sertraline").how == "nombre en inglés"
    assert resolve("Besitran").active == "sertralina"
    unknown = resolve("pantoxazol")
    assert unknown.active is None and "pantoprazol" in unknown.similar
