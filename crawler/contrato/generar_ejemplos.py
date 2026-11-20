"""Regenerate the contract files: JSON Schema and a sample of INVENTED mentions.

    uv run python crawler/contrato/generar_ejemplos.py

The sample texts are made up (no real Bluesky post), so they can live in the public repo and
triage can be developed and tested without access to the crawler's data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from crawler.store import MentionContext, RawMention, write_schema

HERE = Path(__file__).parent


def fake_post_id(text: str) -> str:
    return "p" + hashlib.sha1(text.encode()).hexdigest()[:12]


def mention(text, drug, kind, found_by, date, root=None, parent=None, motivo=""):
    post = fake_post_id(text)
    return RawMention(
        id=RawMention.make_id(post, drug),
        post_id=post,
        hilo_id=fake_post_id(root) if root else post,
        farmaco=drug,
        tipo_mencion=kind,
        texto=text,
        contexto=MentionContext(inicio_hilo=root, responde_a=parent),
        idioma="es",
        fecha=date,
        encontrado_por=found_by,
        motivo=motivo,
        guardado="2026-11-20T09:00:00+00:00",
    )


ROOT = "Llevo tres semanas con sertralina y me paso el día con náuseas, ¿a alguien más le pasa?"
REPLY = "A mí también me pasó al principio, a las dos semanas se me quitó"
EXAMPLES = [
    # 1. Names the drug: found by the fixed search. A possible adverse reaction (nausea).
    mention(
        ROOT,
        "sertralina",
        "nombre",
        "busqueda_fija",
        "2026-03-02T10:15:00Z",
        motivo="contiene 'sertralina'",
    ),
    # 2. Reply that does not name it: found by the agent reading the conversation.
    mention(REPLY, "sertralina", "indirecta", "agente", "2026-03-02T11:40:00Z", root=ROOT),
    # 3. Reply to a reply: responde_a carries the post it answers, inicio_hilo the first one.
    mention(
        "Yo la dejé por eso y me mareaba muchísimo al bajar la dosis",
        "sertralina",
        "indirecta",
        "agente",
        "2026-03-02T12:05:00Z",
        root=ROOT,
        parent=REPLY,
    ),
    # 4. A brand instead of the active ingredient.
    mention(
        "Desde que tomo Besitran no pego ojo, son las 4 y aquí sigo",
        "sertralina",
        "marca",
        "busqueda_fija",
        "2026-04-11T02:03:00Z",
        motivo="contiene 'Besitran'",
    ),
    # 5. Mentions the drug but describes no reaction (a joke): triage should discard it.
    mention(
        "Ver a mi gato dormir me hace más efecto que la sertralina",
        "sertralina",
        "nombre",
        "busqueda_fija",
        "2026-05-20T18:30:00Z",
        motivo="contiene 'sertralina'",
    ),
    # 6. An agent mistake (about 1 in 4 agent mentions): a polite reply, not about the drug.
    mention(
        "Gracias, mucho ánimo 🫂",
        "sertralina",
        "indirecta",
        "agente",
        "2026-03-02T13:00:00Z",
        root=ROOT,
    ),
    # 7. Another watched drug, misspelt.
    mention(
        "Me han subido el lorazepan a 1 mg y estoy todo el día atontada",
        "lorazepam",
        "errata",
        "agente",
        "2026-06-01T08:20:00Z",
    ),
]

if __name__ == "__main__":
    write_schema(HERE / "raw_mention.schema.json")
    with open(HERE / "ejemplo_menciones.jsonl", "w", encoding="utf-8") as f:
        for m in EXAMPLES:
            f.write(m.model_dump_json() + "\n")
    print(f"{len(EXAMPLES)} ejemplos y el esquema en {HERE}")
