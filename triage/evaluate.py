"""Score the triage answers against the answer sheet (``evaluacion_triaje/GUIA_TRIAJE.md``, §6).

    uv run python -m triage.evaluate data/triaje/eval/v1_27b_conversacion_dev_respuestas.jsonl
    uv run python -m triage.evaluate comparar A_respuestas.jsonl B_respuestas.jsonl

``python -m triage.run`` already calls this after each run. Measures, matched by ``id``:

* detection (recall): of the sheet's adverse events, how many the triage marks ``si``. Losing a
  real case is the serious error in pharmacovigilance;
* precision: of what it marks ``si``, how many are ``si`` in the sheet (false alarms are work
  for the human reviewer);
* discarding the crawler's errors: of the sheet's ``no_es_mencion``, how many it marks ``no``;
* the case: in the adverse events both mark ``si``, agreement on ``paciente``, ``edad``,
  ``sexo``, ``grave`` and ``desenlace``;
* all of it also by ``encontrado_por`` (fixed search or the crawler's agent), and the cost.

Every share comes with its 95 % Wilson interval. The numbers go to ``reports/`` (no text); the
failures, with the posts' text, to ``data/`` (never to git).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from crawler.annotate import kappa
from crawler.store import RawMention, read_mentions
from evaluacion_triaje.lote import conversation_text, normalise

CASE = ["paciente", "edad", "sexo", "grave", "desenlace"]
MOTIVES = ["no_es_mencion", "pregunta_o_consejo", "broma_opinion_noticia", "sin_reaccion"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def share(k: int, n: int) -> dict:
    interval = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "valor": k / n if n else None,
        "ic95": [round(x, 3) for x in interval] if interval else None,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test from the discordant pairs (b: only A right, c: only B)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_jsonl(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def _pred(answers: dict[str, dict], i: str) -> dict:
    """The normalised answer; a missing one counts as ``no`` (the case is lost)."""
    a = answers.get(i)
    return normalise(a) if a else normalise({"es_efecto_adverso": "no", "motivo_no": ""})


def normalise_gold(gold: dict[str, dict]) -> dict[str, dict]:
    return {i: normalise(g) for i, g in gold.items()}


def detection(ids: list[str], answers: dict, gold: dict) -> dict:
    pos = [i for i in ids if gold[i]["es_efecto_adverso"] == "si"]
    said_yes = [i for i in ids if _pred(answers, i)["es_efecto_adverso"] == "si"]
    tp = len(set(pos) & set(said_yes))
    return {
        "menciones": len(ids),
        "efectos_adversos": len(pos),
        "deteccion": share(tp, len(pos)),
        "precision": share(tp, len(said_yes)),
        "f1": round(2 * tp / (len(pos) + len(said_yes)), 3) if pos or said_yes else None,
    }


def score(answers: dict[str, dict], gold: dict[str, dict], mentions: dict[str, RawMention]):
    gold = normalise_gold(gold)
    ids = sorted(gold)
    pred = {i: _pred(answers, i) for i in ids}
    out = {
        **detection(ids, answers, gold),
        "sin_respuesta": sum(i not in answers for i in ids),
        "a_revision_humana": sum(answers.get(i, {}).get("estado") == "sin_respuesta" for i in ids),
        "acierto_si_no": share(
            sum(pred[i]["es_efecto_adverso"] == gold[i]["es_efecto_adverso"] for i in ids), len(ids)
        ),
        "kappa_si_no": round(
            kappa(
                [gold[i]["es_efecto_adverso"] for i in ids],
                [pred[i]["es_efecto_adverso"] for i in ids],
            ),
            3,
        ),
    }
    not_mention = [i for i in ids if gold[i]["motivo_no"] == "no_es_mencion"]
    out["descarte_no_es_mencion"] = share(
        sum(pred[i]["es_efecto_adverso"] == "no" for i in not_mention), len(not_mention)
    )
    both_no = [
        i for i in ids if gold[i]["es_efecto_adverso"] == pred[i]["es_efecto_adverso"] == "no"
    ]
    out["motivo_no"] = {
        "acierto": share(
            sum(gold[i]["motivo_no"] == pred[i]["motivo_no"] for i in both_no), len(both_no)
        ),
        "confusion": {
            g: dict(Counter(pred[i]["motivo_no"] for i in both_no if gold[i]["motivo_no"] == g))
            for g in MOTIVES
        },
    }
    both_yes = [
        i for i in ids if gold[i]["es_efecto_adverso"] == pred[i]["es_efecto_adverso"] == "si"
    ]
    out["caso"] = {
        f: {
            **share(sum(gold[i][f] == pred[i][f] for i in both_yes), len(both_yes)),
            "valores_hoja": dict(Counter(gold[i][f] for i in both_yes)),
        }
        for f in CASE
    }
    out["por_origen"] = {
        origin: detection(
            [i for i in ids if i in mentions and mentions[i].encontrado_por == origin],
            answers,
            gold,
        )
        for origin in ("busqueda_fija", "agente")
    }
    return out


def errors(answers: dict, gold: dict, mentions: dict[str, RawMention]) -> dict:
    """The failures with their text, for the analysis (contains posts: goes to ``data/``)."""

    def row(i: str, **extra) -> dict:
        m = mentions.get(i)
        a = answers.get(i, {})
        return {
            "id": i,
            "encontrado_por": m.encontrado_por if m else None,
            "conversacion": conversation_text(m) if m else None,
            "texto": m.texto if m else None,
            "hoja": {k: v for k, v in gold[i].items() if v and k != "id"},
            "triaje": {k: v for k, v in a.items() if v and k != "id"},
            **extra,
        }

    out: dict[str, list] = {
        "perdidos": [],
        "falsas_alarmas": [],
        "datos_del_caso": [],
        "motivo_no": [],
        "reacciones": [],  # free text: compared by hand, side by side
    }
    gold = normalise_gold(gold)
    for i, g in sorted(gold.items()):
        p = _pred(answers, i)
        if g["es_efecto_adverso"] == "si" and p["es_efecto_adverso"] != "si":
            out["perdidos"].append(row(i))
        elif g["es_efecto_adverso"] == "no" and p["es_efecto_adverso"] == "si":
            out["falsas_alarmas"].append(row(i))
        elif g["es_efecto_adverso"] == "si":
            wrong = [f for f in CASE if g[f] != p[f]]
            if wrong:
                out["datos_del_caso"].append(row(i, campos=wrong))
            out["reacciones"].append({"id": i, "hoja": g["reacciones"], "triaje": p["reacciones"]})
        elif g["motivo_no"] != p["motivo_no"]:
            out["motivo_no"].append(row(i))
    return out


def compare(a: dict, b: dict, gold: dict) -> dict:
    """Paired comparison of two runs on the yes/no decision (exact McNemar)."""
    gold = normalise_gold(gold)
    ids = sorted(gold)
    right_a = {i: _pred(a, i)["es_efecto_adverso"] == gold[i]["es_efecto_adverso"] for i in ids}
    right_b = {i: _pred(b, i)["es_efecto_adverso"] == gold[i]["es_efecto_adverso"] for i in ids}
    only_a = sum(right_a[i] and not right_b[i] for i in ids)
    only_b = sum(right_b[i] and not right_a[i] for i in ids)
    pos = [i for i in ids if gold[i]["es_efecto_adverso"] == "si"]
    det_a = sum(_pred(a, i)["es_efecto_adverso"] == "si" for i in pos)
    det_b = sum(_pred(b, i)["es_efecto_adverso"] == "si" for i in pos)
    return {
        "aciertos_a": sum(right_a.values()),
        "aciertos_b": sum(right_b.values()),
        "solo_a_acierta": only_a,
        "solo_b_acierta": only_b,
        "p_mcnemar": round(mcnemar_exact(only_a, only_b), 4),
        "deteccion_a": det_a,
        "deteccion_b": det_b,
        "efectos_adversos": len(pos),
    }


def pct(s: dict | None) -> str:
    if not s or s.get("valor") is None:
        return "-"
    lo, hi = s["ic95"]
    return f"{100 * s['valor']:.0f} % [{100 * lo:.0f}–{100 * hi:.0f}] ({s['k']}/{s['n']})"


def print_report(r: dict, cost: dict | None = None) -> None:
    print(f"\nmenciones: {r['menciones']} · efectos adversos en la hoja: {r['efectos_adversos']}")
    print(f"  detección (recall)          {pct(r['deteccion'])}")
    print(f"  precisión (falsas alarmas)  {pct(r['precision'])}")
    print(f"  F1                          {r['f1']}")
    print(f"  acierto sí/no               {pct(r['acierto_si_no'])} · kappa {r['kappa_si_no']}")
    print(f"  descarta 'no_es_mencion'    {pct(r['descarte_no_es_mencion'])}")
    print(f"  motivo del no (si ambos no) {pct(r['motivo_no']['acierto'])}")
    for origin, d in r["por_origen"].items():
        print(f"  [{origin}] detección {pct(d['deteccion'])} · precisión {pct(d['precision'])}")
    print("  datos del caso (en los efectos adversos que ambos marcan):")
    for f, s in r["caso"].items():
        print(f"    {f:<10} {pct(s)}")
    if r["sin_respuesta"] or r["a_revision_humana"]:
        print(
            f"  sin respuesta: {r['sin_respuesta']} · a revisión humana: {r['a_revision_humana']}"
        )
    if cost:
        n = max(r["menciones"], 1)
        tokens = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
        print(
            f"  coste: {tokens} tokens ({tokens / n:.0f} por mención), {cost.get('calls', 0)} "
            f"peticiones, {cost.get('repairs', 0)} reparaciones, {cost.get('seconds', 0):.0f} s"
        )


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("respuestas", type=Path, nargs="+", help="one file, or 'comparar A B'")
    parser.add_argument("--gold", type=Path, default=Path("data/triaje/gold_triaje_dev.jsonl"))
    parser.add_argument(
        "--menciones", type=Path, default=Path("data/hibrido_27b_menciones_dev.jsonl")
    )
    args = parser.parse_args(argv)
    gold = load_jsonl(args.gold)
    if str(args.respuestas[0]) == "comparar":
        a, b = (load_jsonl(p) for p in args.respuestas[1:3])
        print(json.dumps(compare(a, b, gold), ensure_ascii=False, indent=1))
        return
    mentions = {m.id: m for m in read_mentions(args.menciones)}
    answers = load_jsonl(args.respuestas[0])
    cost_file = Path(str(args.respuestas[0]).replace("_respuestas.jsonl", "_coste.json"))
    cost = json.loads(cost_file.read_text("utf-8")) if cost_file.exists() else None
    print_report(score(answers, gold, mentions), cost)


if __name__ == "__main__":
    main()
