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


class MentionContext(BaseModel):
    inicio_hilo: str | None = Field(None, description="Text of the first post of the thread.")
    responde_a: str | None = Field(None, description="Text of the post this one answers.")


class RawMention(BaseModel):
    id: str
    fuente: Literal["bluesky"] = "bluesky"
    post_id: str
    hilo_id: str
    farmaco: str
    tipo_mencion: MentionType
    texto: str
    contexto: MentionContext = MentionContext()
    fecha: str
    encontrado_por: Literal["busqueda_fija", "agente"]
    motivo: str = ""
    guardado: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

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
