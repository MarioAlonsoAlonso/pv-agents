"""Are the references in a sub-agent's answer backed by what its tools returned?

The analyst only gets the crawler's summary, so a PMID or a post id the model made up would
reach the report looking like evidence. ``check_citations`` compares the references in the
final answer with what the tools actually showed in this run: PubMed articles returned by
``buscar_pubmed`` and Bluesky posts shown by ``buscar_bluesky``/``leer_hilo``. Used by
``buscar_literatura`` (the analyst sees the unbacked ones) and by ``crawler.eval_literatura``.
"""

from __future__ import annotations

import re

# "PMID: 12345678", "PMID 123", "(PMIDs 111, 222)" and bare 7-8 digit numbers (current PMIDs;
# years have 4 digits and doses rarely 7).
_PMID_TAGGED = re.compile(r"PMIDs?[\s:#]*((?:\d{1,9}(?:\s*(?:,|y|and|;|/)\s*)?)+)", re.I)
_PMID_BARE = re.compile(r"(?<![\w.])\d{7,8}(?!\w|\.\d)")  # "…11965835." yes, "1234567.5" no
_POST = re.compile(r"\bp[0-9a-f]{12}\b")


def cited_pmids(text: str) -> list[str]:
    found = []
    for group in _PMID_TAGGED.findall(text):
        found += re.findall(r"\d{1,9}", group)
    found += _PMID_BARE.findall(text)
    return list(dict.fromkeys(found))  # unique, in order


def cited_posts(text: str) -> list[str]:
    return list(dict.fromkeys(_POST.findall(text)))


def check_citations(answer: str, articles: list[dict], shown_posts) -> dict:
    """References in ``answer`` split into backed (returned by a tool) and unbacked."""
    returned = {a["pmid"] for a in articles}
    shown = set(shown_posts)
    pmids, posts = cited_pmids(answer), cited_posts(answer)
    return {
        "pmids_citados": pmids,
        "pmids_sin_respaldo": [p for p in pmids if p not in returned],
        "posts_citados": posts,
        "posts_sin_respaldo": [p for p in posts if p not in shown],
    }
