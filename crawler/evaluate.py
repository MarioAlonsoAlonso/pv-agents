"""Compare the four collection systems on the frozen sample against the annotations.

    uv run python -m crawler.evaluate --systems fija fija_hilos             # no model needed
    uv run python -m crawler.evaluate --systems hibrido agente --brain 27b  # DGX

Every system runs over the whole frozen sample (``FrozenSource``: same posts, same search,
every time), but it is scored only on the annotated posts (``gold.jsonl``): a post outside
the annotation is neither right nor wrong. For each drug ``d``:

* positives ``G_d``: annotated posts whose ``farmaco`` is ``d``;
* recall = |collected_d ∩ G_d| / |G_d|, also per mention type (nombre, marca, errata,
  indirecta, jerga): the agent is only worth its cost if it finds what the fixed search
  cannot (indirect and slang mentions);
* precision = |collected_d ∩ G_d| / |collected_d ∩ annotated|: of what it sent to triage
  among the annotated posts, how much was about the drug.

Plus the cost: posts sent to triage (all, annotated or not), tokens, steps, threads read and
time. Results are micro-averaged over drugs. ``reports/crawler_eval.json`` gets the numbers
only (post ids are hashes, no text); the mentions of each system go to ``data/`` for
inspection, because they contain the posts' text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from crawler.cima import load_brands
from crawler.config import DATA_DIR, MENTION_TYPES, WATCHLIST
from crawler.sources import FrozenSource
from crawler.systems import SYSTEMS, SystemResult, run_system


def load_gold(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {g["post_id"]: g for g in map(json.loads, f) if g}


def score(results: list[SystemResult], gold: dict[str, dict]) -> dict:
    """Micro-averaged recall (overall and per type), precision and cost of one system."""
    tp = fp = 0
    found_by_type: dict[str, int] = defaultdict(int)
    total_by_type: dict[str, int] = defaultdict(int)
    per_drug = {}
    for r in results:
        collected = r.store.post_ids(r.drug)
        positives = {i for i, g in gold.items() if g["farmaco"] == r.drug}
        annotated_collected = collected & set(gold)
        d_tp = len(collected & positives)
        d_fp = len(annotated_collected - positives)
        tp, fp = tp + d_tp, fp + d_fp
        for i in positives:
            kind = gold[i]["tipo_mencion"]
            total_by_type[kind] += 1
            found_by_type[kind] += i in collected
        per_drug[r.drug] = {
            "positivos": len(positives),
            "recall": d_tp / len(positives) if positives else None,
            "precision": d_tp / (d_tp + d_fp) if d_tp + d_fp else None,
            "enviados_al_triaje": len(collected),
        }
    n_pos = sum(total_by_type.values())
    return {
        "recall": tp / n_pos if n_pos else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_por_tipo": {
            k: {
                "encontrados": found_by_type[k],
                "total": total_by_type[k],
                "recall": found_by_type[k] / total_by_type[k] if total_by_type[k] else None,
            }
            for k in MENTION_TYPES
        },
        "enviados_al_triaje": sum(len(r.store) for r in results),
        "tokens": sum(r.tokens for r in results),
        "pasos": sum(r.steps for r in results),
        "busquedas": sum(r.searches for r in results),
        "hilos_leidos": sum(r.threads_read for r in results),
        "segundos": round(sum(r.seconds for r in results), 1),
        # Without the model's "motivo": it sometimes quotes the post where it saw the term.
        "terminos_nuevos": [
            {"termino": t["termino"], "farmaco": t["farmaco"]} for r in results for t in r.new_terms
        ],
        "motivos_parada": [s for r in results for s in r.stop_reasons],
        "por_farmaco": per_drug,
    }


def errors(results: list[SystemResult], gold: dict[str, dict], source) -> dict:
    """What each system got wrong, with the text, for failure analysis: annotated mentions
    it missed (``perdidos``) and annotated posts it sent that are not about the drug
    (``sobrantes``). Contains post texts: it goes to ``data/``, never to ``reports/``."""

    def row(pid: str, extra: dict) -> dict:
        post = source.get(pid)
        parent = source.get(post.parent_id) if post and post.parent_id else None
        return {
            "post_id": pid,
            "hilo": post.thread_id if post else None,
            "texto": post.text if post else None,
            "responde_a": parent.text if parent else None,
            **extra,
        }

    out = {"perdidos": [], "sobrantes": []}
    for r in results:
        collected = r.store.post_ids(r.drug)
        for pid, g in gold.items():
            if g["farmaco"] == r.drug and pid not in collected:
                out["perdidos"].append(row(pid, {"farmaco": r.drug, "tipo": g["tipo_mencion"]}))
            elif pid in collected and g["farmaco"] != r.drug:
                out["sobrantes"].append(row(pid, {"farmaco_buscado": r.drug, "gold": g["farmaco"]}))
    return out


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    from dotenv import load_dotenv

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--muestra", type=Path, default=DATA_DIR / "sample.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA_DIR / "anotacion" / "gold.jsonl")
    parser.add_argument("--systems", nargs="+", default=["fija", "fija_hilos"], choices=SYSTEMS)
    parser.add_argument("--farmacos", nargs="+", default=list(WATCHLIST))
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--out", type=Path, default=Path("reports/crawler_eval.json"))
    parser.add_argument(
        "--ventana-dias",
        type=int,
        default=30,
        help="agente: one run per window of this many days (0 = the whole period at once)",
    )
    parser.add_argument(
        "--todos-los-hilos",
        action="store_true",
        help="hybrid: read every thread, not only the annotated ones (full cost, hours)",
    )
    args = parser.parse_args(argv)

    source = FrozenSource.load(args.muestra)
    manifest_file = args.muestra.with_name("manifest.json")
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    )
    since, until = manifest.get("since"), manifest.get("until")
    gold = load_gold(args.gold)
    brands = load_brands()
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    report.setdefault("sistemas", {})
    report.update(
        {
            "muestra": {
                "posts": len(source.posts),
                "anotados": len(gold),
                "desde": since,
                "hasta": until,
            },
            "actualizado": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    print(f"muestra: {len(source.posts)} posts, {len(gold)} anotados")
    annotated_threads = None if args.todos_los_hilos else {g["hilo"] for g in gold.values()}
    for system in args.systems:
        results = []
        for drug in args.farmacos:
            r = run_system(
                system,
                source,
                drug,
                brands,
                since,
                until,
                args.brain,
                annotated_threads,
                args.ventana_dias or None,
            )
            results.append(r)
            print(f"  {system} · {drug}: {len(r.store)} menciones, {r.tokens} tokens")
        key = system if system in ("fija", "fija_hilos") else f"{system}·{args.brain}"
        name = key.replace("·", "_")
        out_dir = DATA_DIR / "eval"
        for i, r in enumerate(results):
            r.store.save(out_dir / f"{name}_menciones.jsonl", append=i > 0)
        (out_dir / f"{name}_errores.json").write_text(
            json.dumps(errors(results, gold, source), ensure_ascii=False, indent=1), "utf-8"
        )
        traces = [t for r in results for t in r.traces]
        if traces:
            with open(out_dir / f"{name}_trazas.jsonl", "w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
        report["sistemas"][key] = score(results, gold)
        if system == "hibrido":
            report["sistemas"][key]["coste_solo_hilos_anotados"] = annotated_threads is not None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
        )  # after each system

    print(
        f"\n{'sistema':<20} {'recall':>6} {'prec.':>6} "
        + " ".join(f"{k[:6]:>6}" for k in MENTION_TYPES)
        + f" {'triaje':>7} {'tokens':>8}"
    )
    for name, s in report["sistemas"].items():
        types = " ".join(f"{fmt(s['recall_por_tipo'][k]['recall']):>6}" for k in MENTION_TYPES)
        print(
            f"{name:<20} {fmt(s['recall']):>6} {fmt(s['precision']):>6} {types} "
            f"{s['enviados_al_triaje']:>7} {s['tokens']:>8}"
        )
    print(f"→ {args.out}")
    print(f"→ menciones, errores y trazas de cada sistema en {DATA_DIR / 'eval'}")


if __name__ == "__main__":
    main()
