"""What the crawler hands to triage: one ``RawMention`` per (post, drug).

This is the contract with the triage component: whatever channel a mention comes from,
triage receives this. The text is the post itself (copied from the source, never written
by the model), with the minimum context needed to understand a reply ("a mí también me
pasó" only makes sense next to the post it answers).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MentionType = Literal["nombre", "marca", "errata", "indirecta", "jerga"]


SCHEMA_VERSION = "1.1"  # 1.1: conversacion, conversacion_completa


class MentionContext(BaseModel):
    """The minimum of the conversation needed to understand the post."""

    inicio_hilo: str | None = Field(
        None,
        description="Text of the first post of the conversation (anonymised). None when the "
        "mention is itself the first post, or the first post was not available.",
    )
    responde_a: str | None = Field(
        None,
        description="Text of the post this one directly answers (anonymised). None when the "
        "mention is not a reply, or when the post it answers is the first post (then it is in "
        "inicio_hilo).",
    )


class ConversationPost(BaseModel):
    """One earlier post of the conversation."""

    post_id: str = Field(description="Pseudonymous id of the post.")
    texto: str = Field(description="Text of the post (anonymised).")


class RawMention(BaseModel):
    """One post that talks about one watched drug, as the crawler hands it to triage.

    Contract between the crawler and triage (``crawler/CONTRATO_TRIAJE.md``). Not judged: it
    may or may not describe an adverse reaction; deciding that is triage's job.
    """

    version: str = Field(SCHEMA_VERSION, description="Version of this format.")
    id: str = Field(
        description="Stable id of the mention, 'M-' + 10 hex chars, hash of (post_id, farmaco). "
        "The same post and drug always give the same id: use it to drop duplicates."
    )
    fuente: Literal["bluesky"] = Field("bluesky", description="Social network the post comes from.")
    post_id: str = Field(
        description="Pseudonymous stable id of the post ('p' + 12 hex chars). It is a hash: it "
        "does not identify the author and cannot be turned back into a link."
    )
    hilo_id: str = Field(
        description="post_id of the first post of the conversation. Mentions with the same "
        "hilo_id belong to the same conversation (they may describe the same case)."
    )
    farmaco: str = Field(
        description="Watched active ingredient the post talks about, lower case, as in "
        "crawler/config.py WATCHLIST (e.g. 'sertralina'). A post about two drugs gives two "
        "mentions."
    )
    tipo_mencion: MentionType = Field(
        description="How the post refers to the drug: nombre (active ingredient), marca (brand), "
        "errata (misspelt or shortened), indirecta (does not name it; known from the "
        "conversation), jerga (describes it without naming it)."
    )
    texto: str = Field(
        description="Full text of the post, copied from the source (never written by a model) "
        "and anonymised: @handles -> '@usuario', links -> '[enlace]', e-mails -> '[correo]'."
    )
    contexto: MentionContext = Field(default_factory=MentionContext)
    conversacion: list[ConversationPost] = Field(
        default_factory=list,
        description="Every post from the first one of the conversation down to the one this "
        "post answers, oldest first (the post itself is not included: it is in texto). Empty if "
        "the post is the first one. Lets triage read deep reply chains whole (added in 1.1).",
    )
    conversacion_completa: bool = Field(
        True,
        description="False if a post of the chain was not available (deleted, blocked or outside "
        "the sample): then conversacion starts at the oldest available post.",
    )
    idioma: str | None = Field(
        None, description="Language declared by the author in Bluesky (ISO 639-1, e.g. 'es')."
    )
    fecha: str = Field(description="When the post was published, ISO 8601 in UTC.")
    encontrado_por: Literal["busqueda_fija", "agente"] = Field(
        description="busqueda_fija: the post contains the name, a brand or a known misspelling "
        "(almost always right). agente: the crawler agent read the conversation and decided it "
        "talks about the drug (indirect mentions; about 1 in 4 is not really about it)."
    )
    motivo: str = Field(
        "", description="Why it was saved (e.g. \"contiene 'sertralina'\"); may be empty."
    )
    guardado: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"),
        description="When the crawler saved it, ISO 8601 in UTC.",
    )

    @staticmethod
    def make_id(post_id: str, drug: str) -> str:
        return "M-" + hashlib.sha1(f"{post_id}|{drug}".encode()).hexdigest()[:10]


class MentionStore:
    """Mentions found in one run, without duplicates (same post and drug)."""

    def __init__(self):
        self.mentions: dict[str, RawMention] = {}

    def add(self, mention: RawMention) -> bool:
        """False if it was already there (the first finder is kept)."""
        if mention.id in self.mentions:
            return False
        self.mentions[mention.id] = mention
        return True

    def __len__(self) -> int:
        return len(self.mentions)

    def post_ids(self, drug: str | None = None) -> set[str]:
        return {m.post_id for m in self.mentions.values() if drug is None or m.farmaco == drug}

    def save(self, path: str | Path, append: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if append else "w", encoding="utf-8") as f:
            for m in self.mentions.values():
                f.write(m.model_dump_json() + "\n")

    @staticmethod
    def load_ids(path: str | Path) -> set[str]:
        """Ids already written to a file (the monitor appends run after run)."""
        try:
            with open(path, encoding="utf-8") as f:
                return {json.loads(line)["id"] for line in f if line.strip()}
        except FileNotFoundError:
            return set()


def read_mentions(path: str | Path) -> list[RawMention]:
    """Read a mentions file (one JSON object per line), validating every line against the
    contract. Lines written by an older version without the new optional fields still load."""
    with open(path, encoding="utf-8") as f:
        return [RawMention.model_validate_json(line) for line in f if line.strip()]


def write_schema(path: str | Path) -> None:
    """Write the JSON Schema of RawMention (for teammates who do not use Python)."""
    schema = RawMention.model_json_schema()
    Path(path).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", "utf-8")
