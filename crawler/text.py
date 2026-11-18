"""Text helpers: accent folding, term matching, anonymisation and stable post ids."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD = re.compile(r"\w+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Bluesky handles: @name.bsky.social, @domain.com, and plain @name.
_HANDLE = re.compile(r"@[\w-]+(?:\.[\w-]+)*")


def fold(text: str) -> str:
    """Lowercase without accents: "Sertralína" -> "sertralina"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    return _WORD.findall(fold(text))


def matches(text: str, query: str) -> bool:
    """Keyword match like Bluesky's search: every word of the query appears in the text
    (case- and accent-insensitive, whole words)."""
    words = set(tokens(text))
    return all(t in words for t in tokens(query))


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if matches(text, t)]


def anonymize(text: str) -> str:
    """Drop what identifies people beyond the text itself: e-mails, links and @handles."""
    text = _EMAIL.sub("[correo]", text)
    text = _URL.sub("[enlace]", text)
    return _HANDLE.sub("@usuario", text)


def post_id(uri: str) -> str:
    """Stable pseudonymous id of a post. The AT-URI contains the author's DID, so it never
    reaches the sample, the model or the reports; only this hash does."""
    return "p" + hashlib.sha1(uri.encode()).hexdigest()[:12]


def utf8_stdout() -> None:
    """Print accents and arrows on any console (Windows defaults to cp1252)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
