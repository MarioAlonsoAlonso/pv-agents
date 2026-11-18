"""Annotation of the frozen sample: the ground truth the crawler is evaluated against.

    python -m crawler.annotate seleccionar            # lote.csv: whole threads to annotate
    python -m crawler.annotate prenotar --brain 27b   # optional: the 27B fills a copy (DGX)
    python -m crawler.annotate acuerdo a.csv b.csv    # Cohen's kappa between two annotators
    python -m crawler.annotate consolidar lote.csv    # gold.jsonl for crawler.evaluate

The guide is ``crawler/ANOTACION.md``. The unit is the post, but whole threads are selected,
because a reply can only be judged next to what it answers. The CSV uses ``;`` and UTF-8
with BOM so that a Spanish Excel opens it directly.

Selection is stratified: for each drug, conversations (threads with replies, where indirect
mentions live) and single posts; threads found only by the broad terms (no drug named in
the search); and a few noise threads (ibuprofen).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from crawler.cima import load_brands
from crawler.collect import plan_terms
from crawler.config import DATA_DIR, MENTION_TYPES, WATCHLIST
from crawler.sources import FrozenSource, Post

COLUMNS = [
    "post_id",
    "hilo",
    "posicion",
    "responde_a",
    "texto_responde_a",
    "texto",
    "farmaco",
    "tipo_mencion",
    "es_experiencia",
    "notas",
]
NONE_LABEL, OTHER_LABEL = "ninguno", "otro"


# --------------------------------------------------------------------------- select


def thread_groups(source: FrozenSource, drugs) -> dict[str, str]:
    """Each thread's stratum: the first drug whose terms found it, else amplio / ruido."""
    group_of_term = {t: g for t, g, _ in plan_terms(drugs, load_brands())}
    groups: dict[str, set[str]] = {}
    for p in source.posts.values():
        for term in p.search_terms:
            groups.setdefault(p.thread_id, set()).add(group_of_term.get(term, "amplio"))
    out = {}
    for thread, gs in groups.items():
        named = [d for d in drugs if d in gs]
        out[thread] = named[0] if named else ("amplio" if "amplio" in gs else "ruido")
    return out


def select(
    source: FrozenSource,
    drugs=WATCHLIST,
    conversations: int = 10,
    singles: int = 5,
    broad: int = 12,
    noise: int = 6,
    max_posts_per_thread: int = 40,
    seed: int = 0,
) -> list[Post]:
    rng = random.Random(seed)
    groups = thread_groups(source, drugs)
    by_group: dict[str, dict[str, list[str]]] = {}
    for thread, group in sorted(groups.items()):
        kind = "conv" if len(source.threads[thread]) > 1 else "single"
        by_group.setdefault(group, {"conv": [], "single": []})[kind].append(thread)
    chosen: list[str] = []

    def take(group: str, kind: str, n: int) -> None:
        pool = by_group.get(group, {}).get(kind, [])
        chosen.extend(rng.sample(pool, min(n, len(pool))))

    for drug in drugs:
        take(drug, "conv", conversations)
        take(drug, "single", singles)
    take("amplio", "conv", broad)
    take("ruido", "conv", noise // 2)
    take("ruido", "single", noise - noise // 2)
    posts: list[Post] = []
    for thread in chosen:
        posts += source.thread(source.threads[thread][0])[:max_posts_per_thread]
    return posts


def write_csv(posts: list[Post], source: FrozenSource, path: Path, labels: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    position: Counter = Counter()
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter=";")
        writer.writeheader()
        for p in posts:
            position[p.thread_id] += 1
            parent = source.get(p.parent_id) if p.parent_id else None
            row = {
                "post_id": p.id,
                "hilo": p.thread_id,
                "posicion": position[p.thread_id],
                "responde_a": p.parent_id or "",
                "texto_responde_a": " ".join(parent.text.split()) if parent else "",
                "texto": " ".join(p.text.split()),
            }
            row.update((labels or {}).get(p.id, {}))
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


# --------------------------------------------------------------------------- labels


def normalise(row: dict) -> dict:
    """Canonical labels: lowercase, "sí"/"si"/"1" -> si, empty -> ""."""
    drug = (row.get("farmaco") or "").strip().lower()
    kind = (row.get("tipo_mencion") or "").strip().lower()
    exp = (row.get("es_experiencia") or "").strip().lower().replace("í", "i")
    exp = {"1": "si", "true": "si", "0": "no", "false": "no"}.get(exp, exp)
    return {"farmaco": drug, "tipo_mencion": kind, "es_experiencia": exp}


def validate(row: dict, drugs=WATCHLIST) -> str | None:
    lab = normalise(row)
    if not lab["farmaco"]:
        return "sin anotar"
    if lab["farmaco"] not in (*drugs, NONE_LABEL, OTHER_LABEL):
        return f"farmaco '{lab['farmaco']}' no válido (usa {', '.join(drugs)}, ninguno u otro)"
    if lab["farmaco"] in drugs:
        if lab["tipo_mencion"] not in MENTION_TYPES:
            return f"tipo_mencion '{lab['tipo_mencion']}' no válido ({', '.join(MENTION_TYPES)})"
        if lab["es_experiencia"] not in ("si", "no"):
            return "es_experiencia debe ser si o no"
    return None


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa: agreement corrected for the agreement expected by chance."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(rows_a: list[dict], rows_b: list[dict], drugs=WATCHLIST) -> dict:
    a = {r["post_id"]: normalise(r) for r in rows_a if normalise(r)["farmaco"]}
    b = {r["post_id"]: normalise(r) for r in rows_b if normalise(r)["farmaco"]}
    common = sorted(set(a) & set(b))
    mentions = lambda lab: "si" if lab["farmaco"] in drugs else "no"  # noqa: E731
    both_pos = [
        i for i in common if a[i]["farmaco"] in drugs and a[i]["farmaco"] == b[i]["farmaco"]
    ]
    result = {
        "posts_comunes": len(common),
        "kappa_menciona": kappa([mentions(a[i]) for i in common], [mentions(b[i]) for i in common]),
        "kappa_farmaco": kappa(
            [a[i]["farmaco"] for i in common], [b[i]["farmaco"] for i in common]
        ),
        "kappa_tipo": kappa(
            [a[i]["tipo_mencion"] for i in both_pos], [b[i]["tipo_mencion"] for i in both_pos]
        ),
        "kappa_experiencia": kappa(
            [a[i]["es_experiencia"] for i in both_pos], [b[i]["es_experiencia"] for i in both_pos]
        ),
        "desacuerdos": [
            {"post_id": i, "a": a[i], "b": b[i]}
            for i in common
            if (a[i]["farmaco"], a[i]["tipo_mencion"]) != (b[i]["farmaco"], b[i]["tipo_mencion"])
        ],
    }
    return result


def consolidate(rows: list[dict], drugs=WATCHLIST) -> tuple[list[dict], list[str]]:
    gold, errors = [], []
    for r in rows:
        error = validate(r, drugs)
        if error:
            errors.append(f"{r['post_id']}: {error}")
            continue
        lab = normalise(r)
        positive = lab["farmaco"] in drugs
        gold.append(
            {
                "post_id": r["post_id"],
                "hilo": r["hilo"],
                "farmaco": lab["farmaco"] if positive else None,
                "otro_farmaco": lab["farmaco"] == OTHER_LABEL,
                "tipo_mencion": lab["tipo_mencion"] if positive else None,
                "es_experiencia": lab["es_experiencia"] == "si" if positive else None,
                "notas": r.get("notas", ""),
            }
        )
    return gold, errors


# --------------------------------------------------------------------------- pre-annotation

PRELABEL_PROMPT = """Anota los posts de esta conversación de Bluesky para un sistema de
farmacovigilancia. Fármacos vigilados: {drugs} (marcas: {brands}).

Para cada post decide:
- farmaco: el fármaco vigilado del que habla el post, "otro" si habla de otro medicamento,
  o "ninguno". Un post habla del fármaco aunque no lo nombre si, por la conversación, se sabe
  que se refiere a él ("a mí también me dejó KO" respondiendo a alguien que toma sertralina).
- tipo_mencion (solo si es un fármaco vigilado): nombre (principio activo), marca, errata
  (mal escrito o abreviado), indirecta (no lo nombra; se sabe por la conversación) o jerga
  (lo describe sin nombrarlo).
- es_experiencia (solo si es un fármaco vigilado): "si" si alguien cuenta lo que le pasó a
  él o a alguien cercano con el fármaco; "no" para preguntas, opiniones, bromas, noticias.

Conversación (id, a qué responde, texto):
{thread}

Responde SOLO con una lista JSON, un objeto por post, en el mismo orden:
[{{"post_id": "...", "farmaco": "...", "tipo_mencion": "...", "es_experiencia": "..."}}]"""


def prelabel(posts: list[Post], source: FrozenSource, brain: str = "27b", drugs=WATCHLIST) -> dict:
    from crawler.agent import make_brain

    mind = make_brain(brain)
    brands = load_brands()
    labels: dict[str, dict] = {}
    threads: dict[str, list[Post]] = {}
    for p in posts:
        threads.setdefault(p.thread_id, []).append(p)
    for n, members in enumerate(threads.values(), 1):
        text = "\n".join(
            f"[{p.id}] (responde a [{p.parent_id}]) {' '.join(p.text.split())}"
            if p.parent_id
            else f"[{p.id}] (inicio) {' '.join(p.text.split())}"
            for p in members
        )
        prompt = PRELABEL_PROMPT.format(
            drugs=", ".join(drugs),
            brands="; ".join(f"{d}: {', '.join(brands.get(d, [])[:6])}" for d in drugs),
            thread=text,
        )
        raw, _ = mind.llm.generate(
            [{"role": "user", "content": prompt}], max_new_tokens=200 + 80 * len(members)
        )
        match = re.search(r"\[.*\]", raw.split("</think>")[-1], re.DOTALL)
        try:
            items = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            items = []
        for item in items:
            if isinstance(item, dict) and item.get("post_id") in {p.id for p in members}:
                labels[item["post_id"]] = {
                    "farmaco": item.get("farmaco", ""),
                    "tipo_mencion": item.get("tipo_mencion", "") or "",
                    "es_experiencia": item.get("es_experiencia", "") or "",
                    "notas": f"prenotado {brain}",
                }
        print(f"  hilo {n}/{len(threads)}: {sum(p.id in labels for p in members)}/{len(members)}")
    return labels


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    folder = DATA_DIR / "anotacion"

    s = sub.add_parser("seleccionar", help="choose the threads to annotate")
    s.add_argument("--muestra", type=Path, default=DATA_DIR / "sample.jsonl")
    s.add_argument("--conversaciones", type=int, default=10, help="threads with replies per drug")
    s.add_argument("--sueltos", type=int, default=5, help="single posts per drug")
    s.add_argument("--amplios", type=int, default=12)
    s.add_argument("--ruido", type=int, default=6)
    s.add_argument("--validacion", type=int, default=100, help="posts for the blind 2nd pass")
    s.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prenotar", help="fill a copy of the batch with a model (DGX)")
    p.add_argument("--muestra", type=Path, default=DATA_DIR / "sample.jsonl")
    p.add_argument("--lote", type=Path, default=folder / "lote.csv")
    p.add_argument("--brain", default="27b")

    a = sub.add_parser("acuerdo", help="Cohen's kappa between two annotated CSVs")
    a.add_argument("a", type=Path)
    a.add_argument("b", type=Path)

    c = sub.add_parser("consolidar", help="annotated CSV -> gold.jsonl")
    c.add_argument("lote", type=Path)
    c.add_argument("--out", type=Path, default=folder / "gold.jsonl")
    args = parser.parse_args(argv)

    if args.cmd == "seleccionar":
        source = FrozenSource.load(args.muestra)
        posts = select(
            source,
            WATCHLIST,
            args.conversaciones,
            args.sueltos,
            args.amplios,
            args.ruido,
            seed=args.seed,
        )
        write_csv(posts, source, folder / "lote.csv")
        rng = random.Random(args.seed + 1)
        blind = rng.sample(posts, min(args.validacion, len(posts)))
        order = {p.id: i for i, p in enumerate(posts)}
        write_csv(sorted(blind, key=lambda p: order[p.id]), source, folder / "validacion.csv")
        n_threads = len({p.thread_id for p in posts})
        print(f"{len(posts)} posts en {n_threads} hilos → {folder / 'lote.csv'}")
        print(f"{len(blind)} posts para la segunda anotación ciega → {folder / 'validacion.csv'}")
    elif args.cmd == "prenotar":
        source = FrozenSource.load(args.muestra)
        rows = read_csv(args.lote)
        posts = [source.get(r["post_id"]) for r in rows if source.get(r["post_id"])]
        labels = prelabel(posts, source, args.brain)
        out = args.lote.with_name(args.lote.stem + f"_prenotado_{args.brain}.csv")
        write_csv(posts, source, out, labels)
        print(f"{len(labels)}/{len(posts)} posts prenotados → {out}")
    elif args.cmd == "acuerdo":
        result = agreement(read_csv(args.a), read_csv(args.b))
        for k, v in result.items():
            if k != "desacuerdos":
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        out = args.a.with_name(f"desacuerdos_{args.a.stem}_{args.b.stem}.json")
        out.write_text(json.dumps(result["desacuerdos"], ensure_ascii=False, indent=1), "utf-8")
        print(f"{len(result['desacuerdos'])} desacuerdos → {out}")
    elif args.cmd == "consolidar":
        gold, errors = consolidate(read_csv(args.lote))
        for e in errors[:30]:
            print("  !", e)
        if errors:
            print(f"{len(errors)} filas con problemas (no entran en gold)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for g in gold:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        positives = Counter(g["tipo_mencion"] for g in gold if g["farmaco"])
        print(f"{len(gold)} posts → {args.out}; menciones por tipo: {dict(positives)}")


if __name__ == "__main__":
    main()
