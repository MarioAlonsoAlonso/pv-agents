"""Change G, the cheapest form: a rule in the code that sends to human review the mentions the
triage said "no" to but whose post names a symptom (the keyword baseline's list, not tuned on
any set). Measured on saved answers only: no model, no GPU.

    uv run python -m triage.regla_revision          # development set (287 mentions)

Development, 2026-11-26: of the 7 adverse events v2 misses, only 1 has a word of the list, so
the rule recovers 1 and sends 15-20 more mentions to review. The missed cases are told in words
no list foresees; a more permissive triage has to come from the model (a "doubtful" answer or
its confidence), measured on a new annotated batch, not from keywords.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable

from common import rutas
from crawler.store import read_mentions
from crawler.text import fold
from triage.agent import SYMPTOMS
from triage.evaluate import load_jsonl, normalise_gold

RULES: dict[str, Callable[[bool, str], bool]] = {
    # (the post names a symptom of the list, the triage's reason for "no") -> to review?
    "v2 tal cual": lambda hit, motivo: False,
    "«no» con síntoma → revisión": lambda hit, motivo: hit,
    "«no» con síntoma, salvo «no habla del fármaco»": lambda hit, motivo: (
        hit and motivo != "no_es_mencion"
    ),
    "«no» con síntoma, solo sin_reaccion o pregunta": lambda hit, motivo: (
        hit and motivo in ("sin_reaccion", "pregunta_o_consejo")
    ),
}


def measure(mentions: dict, gold: dict, answers: dict) -> dict:
    ids = [i for i in gold if i in mentions]
    is_ae = {i: gold[i].get("es_efecto_adverso") == "si" for i in ids}
    said_yes = {i: (answers.get(i) or {}).get("es_efecto_adverso") == "si" for i in ids}
    motivo = {i: (answers.get(i) or {}).get("motivo_no", "") for i in ids}
    hit = {i: any(s in fold(mentions[i].texto) for s in SYMPTOMS) for i in ids}
    n_ae = sum(is_ae.values())
    out = {}
    for name, rule in RULES.items():
        review = [i for i in ids if not said_yes[i] and rule(hit[i], motivo[i])]
        flagged = [i for i in ids if said_yes[i]] + review
        tp = sum(is_ae[i] for i in flagged)
        out[name] = {
            "deteccion": f"{tp}/{n_ae}",
            "precision": round(tp / len(flagged), 3) if flagged else None,
            "a_revision": len(review),
            "efectos_adversos_en_revision": sum(is_ae[i] for i in review),
        }
    missed = [i for i in ids if is_ae[i] and not said_yes[i]]
    out["perdidos_por_v2"] = {
        "total": len(missed),
        "por_motivo": dict(Counter(motivo[i] for i in missed)),
        "con_un_sintoma_de_la_lista": sum(hit[i] for i in missed),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conjunto", choices=["dev"], default="dev", help="only development: the exam is used"
    )
    args = parser.parse_args(argv)
    s = args.conjunto
    mentions = {m.id: m for m in read_mentions(rutas.MENCIONES[s])}
    gold = normalise_gold(load_jsonl(rutas.GOLD_TRIAJE[s]))
    answers = load_jsonl(rutas.TRIAJE_EVAL / f"v2_27b_mencion_{s}_respuestas.jsonl")
    result = measure(mentions, gold, answers)
    for name, r in result.items():
        print(f"{name:50} {r}")


if __name__ == "__main__":
    main()
