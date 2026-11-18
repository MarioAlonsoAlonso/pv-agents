"""PubMed through NCBI's E-utilities (public API, no key needed at ≤ 3 requests/s).

Used by the crawler in directed mode, when the analyst asks for evidence: published case
reports and studies are the literature channel of pharmacovigilance.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "arca-crawler"  # NCBI asks clients to identify the tool (no personal e-mail sent)


def search_pubmed(query: str, max_results: int = 5, client=None) -> list[dict]:
    """Most relevant articles for ``query``: PMID, title, journal, year and abstract."""
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        ids = (
            client.get(
                f"{EUTILS}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                    "tool": TOOL,
                },
            )
            .raise_for_status()
            .json()["esearchresult"]["idlist"]
        )
        if not ids:
            return []
        time.sleep(0.4)
        xml = (
            client.get(
                f"{EUTILS}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": TOOL},
            )
            .raise_for_status()
            .text
        )
    finally:
        if own:
            client.close()
    return parse_articles(xml)


def parse_articles(xml: str) -> list[dict]:
    articles = []
    for node in ET.fromstring(xml).iter("PubmedArticle"):
        abstract = " ".join(
            "".join(part.itertext()).strip() for part in node.iter("AbstractText")
        ).strip()
        articles.append(
            {
                "pmid": node.findtext(".//PMID", ""),
                "titulo": "".join(node.find(".//ArticleTitle").itertext()).strip()
                if node.find(".//ArticleTitle") is not None
                else "",
                "revista": node.findtext(".//Journal/Title", ""),
                "año": node.findtext(".//JournalIssue/PubDate/Year", "")
                or node.findtext(".//JournalIssue/PubDate/MedlineDate", "")[:4],
                "resumen": abstract[:900] + ("…" if len(abstract) > 900 else ""),
            }
        )
    return articles
