"""The answer sheet for evaluating the triage agent (guide: ANOTACION_TRIAJE.md).

    python -m evaluacion_triaje.lote crear --menciones data/hibrido_27b_menciones.jsonl
    python -m evaluacion_triaje.lote prenotar --menciones data/hibrido_27b_menciones.jsonl
    python -m evaluacion_triaje.lote acuerdo CLAUDE.csv PRENOTADO_27B.csv   (in data/triaje/)
    python -m evaluacion_triaje.lote consolidar data/triaje/lote_triaje_final.csv

The input is exactly what the crawler hands to triage (RawMention 1.1). The sheet is a CSV with
``;`` and UTF-8 BOM (opens directly in a Spanish Excel): the reading columns (conversation and
text) and the answer columns of the guide. Everything under ``data/`` stays out of git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from crawler.annotate import kappa
from crawler.store import RawMention, read_mentions

FOLDER = Path("data/triaje")
READ_COLUMNS = ["n", "id", "hilo_id", "encontrado_por", "idioma", "conversacion", "texto"]
LABELS = [
    "es_efecto_adverso",
    "motivo_no",
    "reacciones",
    "negadas",
    "paciente",
    "edad",
    "sexo",
    "sexo_fuente",
    "grave",
    "criterio_gravedad",
    "desenlace",
    "notas",
]
VALUES = {
    "es_efecto_adverso": {"si", "no"},
    "motivo_no": {"no_es_mencion", "pregunta_o_consejo", "broma_opinion_noticia", "sin_reaccion"},
    "paciente": {"autor", "familiar_o_conocido", "desconocido"},
    "sexo": {"mujer", "hombre", "desconocido"},
    "sexo_fuente": {"dicho", "deducido", ""},
    "grave": {"si", "no_consta"},
    "criterio_gravedad": {
        "muerte",
        "riesgo_vital",
        "hospitalizacion",
        "discapacidad",
        "otro_importante",
        "",
    },
    "desenlace": {"recuperado", "recuperandose", "no_recuperado", "desconocido"},
}
CASE_FIELDS = ["paciente", "sexo", "sexo_fuente", "grave", "criterio_gravedad", "desenlace"]


# --------------------------------------------------------------------------- sheet


def conversation_text(m: RawMention) -> str:
    """The earlier posts, numbered, one line each (the sheet's reading column)."""
    return "\n".join(f"[{i}] {' '.join(c.texto.split())}" for i, c in enumerate(m.conversacion, 1))


def write_sheet(mentions: list[RawMention], path: Path, labels: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, READ_COLUMNS + LABELS, delimiter=";")
        writer.writeheader()
        for n, m in enumerate(mentions):
            row = {
                "n": n,
                "id": m.id,
                "hilo_id": m.hilo_id,
                "encontrado_por": m.encontrado_por,
                "idioma": m.idioma or "",
                "conversacion": conversation_text(m),
                "texto": " ".join(m.texto.split()),
            }
            row.update((labels or {}).get(m.id, {}))
            writer.writerow(row)


def read_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def normalise(row: dict) -> dict:
    out = {}
    for field in LABELS:
        value = (row.get(field) or "").strip()
        if field in VALUES or field == "edad":
            value = value.lower().replace("í", "i").replace("ó", "o")
        out[field] = value
    if out["es_efecto_adverso"] != "si":  # case fields only make sense for adverse events
        for field in ["reacciones", "negadas", *CASE_FIELDS, "edad"]:
            out[field] = ""
    else:
        out["motivo_no"] = ""
        out["edad"] = out["edad"] or "desconocido"
    return out


def validate(row: dict) -> str | None:
    lab = normalise(row)
    if lab["es_efecto_adverso"] not in VALUES["es_efecto_adverso"]:
        return "es_efecto_adverso debe ser si o no"
    if lab["es_efecto_adverso"] == "no":
        if lab["motivo_no"] not in VALUES["motivo_no"]:
            return f"motivo_no no válido: '{lab['motivo_no']}'"
        return None
    if not lab["reacciones"]:
        return "un efecto adverso necesita al menos una reacción"
    for field in CASE_FIELDS:
        if lab[field] not in VALUES[field]:
            return f"{field} no válido: '{lab[field]}'"
    if lab["edad"] != "desconocido" and not lab["edad"].isdigit():
        return f"edad no válida: '{lab['edad']}'"
    if lab["grave"] == "si" and not lab["criterio_gravedad"]:
        return "grave = si necesita criterio_gravedad"
    return None


# --------------------------------------------------------------------------- 27B

PRELABEL_PROMPT = """Eres anotador de farmacovigilancia. Para cada mención de abajo (un post de
Bluesky que habla de {farmaco}, con los mensajes anteriores de su conversación), decide lo
siguiente.
Anota SOLO lo que dice el post, leído con su conversación; no supongas nada.

- es_efecto_adverso: "si" si el autor cuenta que a él o a alguien concreto le pasó algo malo con
  el fármaco (síntomas, molestias, síntomas al dejarlo o bajar la dosis, o que no le hace efecto),
  aunque dude de la causa. "no" en otro caso.
- motivo_no (solo si "no"): "no_es_mencion" (no habla del fármaco: un "gracias", otro fármaco,
  otra cosa), "pregunta_o_consejo", "broma_opinion_noticia" o "sin_reaccion" (lo toma o le va
  bien, sin nada malo).
- reacciones (solo si "si"): cada reacción con las palabras del autor, separadas por " | ". Si
  solo se sabe por la conversación ("a mí también me pasó"), escríbela entre corchetes: [náuseas].
- negadas: lo que el autor dice que NO tuvo (vacío si nada).
- paciente: "autor", "familiar_o_conocido" o "desconocido".
- edad: el número si se dice; si no, "desconocido".
- sexo: "mujer", "hombre" o "desconocido"; sexo_fuente: "dicho", "deducido" (por el género
  gramatical: "estoy cansada") o "" si desconocido.
- grave: "si" solo si dice muerte, riesgo vital, hospitalización (ingreso), discapacidad u otro
  evento médicamente importante (p. ej. ideación suicida, manía, convulsiones); si no, "no_consta".
- criterio_gravedad: "muerte", "riesgo_vital", "hospitalizacion", "discapacidad",
  "otro_importante" o "".
- desenlace: "recuperado", "recuperandose", "no_recuperado" o "desconocido".

Menciones:
{menciones}

Responde SOLO con una lista JSON, un objeto por mención, con "id" y los campos de arriba:
[{{"id": "...", "es_efecto_adverso": "...", "motivo_no": "...", "reacciones": "...", "negadas": "",
  "paciente": "...", "edad": "...", "sexo": "...", "sexo_fuente": "...", "grave": "...",
  "criterio_gravedad": "", "desenlace": "..."}}]"""


def prelabel(mentions: list[RawMention], brain: str = "27b") -> dict[str, dict]:
    """The 27B fills the sheet, one request per conversation (the mentions of a thread share
    their context)."""
    from crawler.agent import make_brain

    mind = make_brain(brain)
    by_thread: dict[str, list[RawMention]] = {}
    for m in mentions:
        by_thread.setdefault(m.hilo_id, []).append(m)
    labels: dict[str, dict] = {}
    for k, group in enumerate(by_thread.values(), 1):
        blocks = []
        for m in group:
            chain = conversation_text(m) or "(es el primer post de la conversación)"
            blocks.append(f"### id: {m.id}\nConversación anterior:\n{chain}\nPost: {m.texto}")
        prompt = PRELABEL_PROMPT.format(farmaco=group[0].farmaco, menciones="\n\n".join(blocks))
        raw, _ = mind.llm.generate(
            [{"role": "user", "content": prompt}], max_new_tokens=300 + 220 * len(group)
        )
        match = re.search(r"\[.*\]", raw.split("</think>")[-1], re.DOTALL)
        try:
            items = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            items = []
        ids = {m.id for m in group}
        for item in items:
            if isinstance(item, dict) and item.get("id") in ids:
                row = {f: str(item.get(f, "") or "") for f in LABELS if f != "notas"}
                row["notas"] = f"prenotado {brain}"
                labels[item["id"]] = row
        done = sum(m.id in labels for m in group)
        print(f"  conversación {k}/{len(by_thread)}: {done}/{len(group)}")
    return labels


# --------------------------------------------------------------------------- agreement


def agreement(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Field by field: Cohen's kappa on the decisions, and the disagreements to review."""
    a = {r["id"]: normalise(r) for r in rows_a if (r.get("es_efecto_adverso") or "").strip()}
    b = {r["id"]: normalise(r) for r in rows_b if (r.get("es_efecto_adverso") or "").strip()}
    common = sorted(set(a) & set(b))
    both_yes = [i for i in common if a[i]["es_efecto_adverso"] == b[i]["es_efecto_adverso"] == "si"]
    both_no = [i for i in common if a[i]["es_efecto_adverso"] == b[i]["es_efecto_adverso"] == "no"]
    result = {
        "menciones_comunes": len(common),
        "efecto_adverso": {
            "kappa": kappa(
                [a[i]["es_efecto_adverso"] for i in common],
                [b[i]["es_efecto_adverso"] for i in common],
            ),
            "si_a": sum(a[i]["es_efecto_adverso"] == "si" for i in common),
            "si_b": sum(b[i]["es_efecto_adverso"] == "si" for i in common),
        },
        "motivo_no": {
            "kappa": kappa(
                [a[i]["motivo_no"] for i in both_no], [b[i]["motivo_no"] for i in both_no]
            ),
            "n": len(both_no),
        },
    }
    for field in ["paciente", "sexo", "grave", "desenlace"]:
        result[field] = {
            "kappa": kappa([a[i][field] for i in both_yes], [b[i][field] for i in both_yes]),
            "acuerdo": sum(a[i][field] == b[i][field] for i in both_yes) / max(len(both_yes), 1),
            "n": len(both_yes),
        }
    result["desacuerdos"] = [
        {
            "id": i,
            "campos": {
                f: [a[i][f], b[i][f]]
                for f in ["es_efecto_adverso", "motivo_no", *CASE_FIELDS]
                if a[i][f] != b[i][f]
            },
        }
        for i in common
        if any(a[i][f] != b[i][f] for f in ["es_efecto_adverso", "motivo_no", "grave"])
    ]
    return result


# --------------------------------------------------------------------------- human review

REVIEW_COLUMNS = [
    "n", "id", "conversacion", "texto",
    "a_efecto_adverso", "a_detalle", "a_grave",
    "b_efecto_adverso", "b_detalle", "b_grave",
    "razon_a", "decision", "notas_revision",
]  # fmt: skip


def _detail(lab: dict) -> str:
    if lab["es_efecto_adverso"] == "si":
        return lab["reacciones"]
    return lab["motivo_no"]


def review_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """Only the disagreements that change the answer: adverse event yes/no, and seriousness
    when both say yes. The motive of a "no" is left out: its categories overlap (kappa 0.47)
    and it does not change what triage must do."""
    by_id_b = {r["id"]: r for r in rows_b if (r.get("es_efecto_adverso") or "").strip()}
    out = []
    for ra in rows_a:
        rb = by_id_b.get(ra["id"])
        if rb is None:
            continue
        a, b = normalise(ra), normalise(rb)
        differs = a["es_efecto_adverso"] != b["es_efecto_adverso"] or (
            a["es_efecto_adverso"] == "si" and a["grave"] != b["grave"]
        )
        if differs:
            out.append(
                {
                    "n": ra["n"],
                    "id": ra["id"],
                    "conversacion": ra["conversacion"],
                    "texto": ra["texto"],
                    "a_efecto_adverso": a["es_efecto_adverso"],
                    "a_detalle": _detail(a),
                    "a_grave": f"{a['grave']} {a['criterio_gravedad']}".strip(),
                    "b_efecto_adverso": b["es_efecto_adverso"],
                    "b_detalle": _detail(b),
                    "b_grave": f"{b['grave']} {b['criterio_gravedad']}".strip(),
                    "razon_a": (ra.get("notas") or "").replace("prenotado claude", "").strip("; "),
                    "decision": "",
                    "notas_revision": "",
                }
            )
    return out


def apply_review(rows_a: list[dict], rows_b: list[dict], review: list[dict]) -> list[dict]:
    """The final sheet: A, except where the reviewer chose ``b`` (take B's answer for that
    mention) or ``descartar`` (drop the mention: no clear answer). ``a`` or empty keeps A."""
    by_id_b = {r["id"]: r for r in rows_b}
    decisions = {r["id"]: (r.get("decision") or "").strip().lower() for r in review}
    unknown = {d for d in decisions.values() if d not in {"", "a", "b", "descartar"}}
    if unknown:
        raise ValueError(f"decision must be a, b or descartar (got {sorted(unknown)})")
    final = []
    for ra in rows_a:
        d = decisions.get(ra["id"], "")
        if d == "descartar":
            continue
        row = dict(by_id_b[ra["id"]]) if d == "b" else dict(ra)
        if d:
            row["notas"] = f"revisado: {d}; " + (row.get("notas") or "")
        final.append(row)
    return final


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from crawler.text import utf8_stdout

    load_dotenv(".env")
    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crear", help="blank sheet from a mentions file")
    c.add_argument("--menciones", type=Path, required=True)
    c.add_argument("--out", type=Path, default=FOLDER / "lote_triaje.csv")
    p = sub.add_parser("prenotar", help="the 27B fills a copy of the sheet (DGX)")
    p.add_argument("--menciones", type=Path, required=True)
    p.add_argument("--brain", default="27b")
    p.add_argument("--out", type=Path, default=FOLDER / "lote_triaje_prenotado_27b.csv")
    a = sub.add_parser("acuerdo", help="agreement between two filled sheets")
    a.add_argument("a", type=Path)
    a.add_argument("b", type=Path)
    r = sub.add_parser("revisar", help="sheet with only the disagreements, for a person")
    r.add_argument("a", type=Path)
    r.add_argument("b", type=Path)
    r.add_argument("--out", type=Path, default=FOLDER / "revision.csv")
    ap = sub.add_parser("aplicar", help="apply the reviewer's decisions -> final sheet")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("revision", type=Path)
    ap.add_argument("--out", type=Path, default=FOLDER / "lote_triaje_final.csv")
    pa = sub.add_parser("partir", help="gold -> development and test files, by conversation")
    pa.add_argument("gold", type=Path)
    pa.add_argument("--menciones", type=Path, required=True)
    pa.add_argument("--test", type=float, default=0.6, help="share of conversations for test")
    pa.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("consolidar", help="validated sheet -> gold_triaje.jsonl")
    k.add_argument("lote", type=Path)
    k.add_argument("--out", type=Path, default=FOLDER / "gold_triaje.jsonl")
    args = parser.parse_args(argv)

    if args.cmd == "crear":
        mentions = read_mentions(args.menciones)
        write_sheet(mentions, args.out)
        print(f"{len(mentions)} menciones → {args.out}")
    elif args.cmd == "prenotar":
        mentions = read_mentions(args.menciones)
        labels = prelabel(mentions, args.brain)
        write_sheet(mentions, args.out, labels)
        print(f"{len(labels)}/{len(mentions)} menciones prenotadas → {args.out}")
    elif args.cmd == "acuerdo":
        r = agreement(read_sheet(args.a), read_sheet(args.b))
        print(f"menciones comunes: {r['menciones_comunes']}")
        ea = r["efecto_adverso"]
        print(f"¿efecto adverso?: kappa {ea['kappa']:.2f} (sí: {ea['si_a']} frente a {ea['si_b']})")
        print(f"motivo del no: kappa {r['motivo_no']['kappa']:.2f} (en {r['motivo_no']['n']})")
        for f in ["paciente", "sexo", "grave", "desenlace"]:
            share = 100 * r[f]["acuerdo"]
            print(f"{f}: kappa {r[f]['kappa']:.2f}, acuerdo {share:.0f} % (en {r[f]['n']})")
        out = args.a.with_name(f"desacuerdos_{args.a.stem}_{args.b.stem}.json")
        out.write_text(json.dumps(r["desacuerdos"], ensure_ascii=False, indent=1), "utf-8")
        print(f"{len(r['desacuerdos'])} desacuerdos en las decisiones → {out}")
    elif args.cmd == "revisar":
        rows = review_rows(read_sheet(args.a), read_sheet(args.b))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, REVIEW_COLUMNS, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} desacuerdos para revisar → {args.out}")
        print("Rellenad la columna 'decision' con a (primera hoja), b (segunda) o descartar.")
    elif args.cmd == "aplicar":
        final = apply_review(read_sheet(args.a), read_sheet(args.b), read_sheet(args.revision))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, READ_COLUMNS + LABELS, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final)
        print(f"{len(final)} menciones en la hoja final → {args.out}")
    elif args.cmd == "partir":
        import random

        thread = {m.id: m.hilo_id for m in read_mentions(args.menciones)}
        with open(args.gold, encoding="utf-8") as f:
            gold = [json.loads(line) for line in f if line.strip()]
        threads = sorted({thread[g["id"]] for g in gold})
        random.Random(args.seed).shuffle(threads)
        test = set(threads[: round(len(threads) * args.test)])
        for name, keep in (("dev", False), ("test", True)):
            part = [g for g in gold if (thread[g["id"]] in test) == keep]
            out = args.gold.with_name(f"{args.gold.stem}_{name}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for g in part:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            n_ae = sum(g["es_efecto_adverso"] == "si" for g in part)
            print(f"{name}: {len(part)} menciones, {n_ae} efectos adversos → {out}")
    elif args.cmd == "consolidar":
        rows = read_sheet(args.lote)
        gold, errors = [], []
        for r in rows:
            error = validate(r)
            if error:
                errors.append(f"n={r['n']} ({r['id']}): {error}")
            else:
                gold.append({"id": r["id"], **normalise(r)})
        for e in errors[:30]:
            print("  !", e)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for g in gold:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"{len(gold)} menciones válidas → {args.out} ({len(errors)} con problemas)")
        print("efectos adversos:", Counter(g["es_efecto_adverso"] for g in gold))


if __name__ == "__main__":
    main()
