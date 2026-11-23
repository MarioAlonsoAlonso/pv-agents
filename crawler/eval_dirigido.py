"""Evaluate the directed mode (``investigar`` / ``buscar_evidencia``): the analyst's questions.

    uv run python -m crawler.eval_dirigido --sistemas fija                  # no model, anywhere
    uv run --extra crawler python -m crawler.eval_dirigido --brain 27b      # DGX: fija + agente
    uv run --extra crawler python -m crawler.eval_dirigido --ids caidas_zolpidem --brain 27b
    uv run --extra crawler python -m crawler.eval_dirigido --bench crawler/bench/dirigido_test.jsonl

Two benches, like a practice exam and a final one:

* ``crawler/bench/dirigido.jsonl`` (development): 12 questions with an answer in the literature
  and 2 controls where there should be none. Look at its errors and change the agent.
* ``crawler/bench/dirigido_test.jsonl`` (test, ``--bench crawler/bench/dirigido_test.jsonl``):
  10 other questions and 2 controls, written before the first change to the agent. Run it
  only to report, never to decide what to change.
Each question carries the drug and reaction terms in English and 3 reference PMIDs (found
with title-restricted PubMed searches and checked by title on 2026-11-23). Two systems:

* ``fija``: one PubMed search with a fixed English query (``consulta_fija``), top 10. No model:
  what you get without an agent.
* ``agente``: the crawler in directed mode, Bluesky + PubMed, as the analyst would call it.

What is measured, per question and summed over the bench:

* **References without backing** (the most important): PMIDs and post ids cited in the final
  answer that no tool returned in that run. Each one is an invented reference that would reach
  the analyst's report as evidence.
* **Relevance**: an article is relevant if its title or abstract contains a drug term and a
  reaction term. Computed for what the system delivers (all articles seen) and, for the
  agent, for what it cites. A proxy: crude, but automatic and the same for both systems.
* **Reference articles found** (of the 3 per question): a rough recall. Many other articles
  are equally valid, so a miss is not necessarily an error; useful to compare systems.
* **Controls**: whether the answer admits that there is no evidence (heuristic: read them).
* **Cost**: tokens, steps, seconds, PubMed and Bluesky searches.

``reports/crawler_dirigido_eval.json`` gets numbers and PMIDs only. Answers, traces and posts
go to ``data/bluesky/dirigido/<bench>_<brain>/`` (they may quote posts by real people).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from crawler.citas import check_citations
from crawler.config import DATA_DIR

BENCH = Path(__file__).parent / "bench" / "dirigido.jsonl"  # development: look at its errors
TEST = Path(__file__).parent / "bench" / "dirigido_test.jsonl"  # final exam: run, do not tune
REPORT = Path("reports") / "crawler_dirigido_eval.json"
FIXED_TOP = 10

# The answer admits there is nothing (controls). Heuristic: the report says to read them.
_NO_EVIDENCE = re.compile(
    r"no (he |se )?(encontr|hall)|no hay (evidencia|datos|casos|art[ií]culos|publicaciones)"
    r"|sin evidencia|ning[uú]n (art[ií]culo|caso|post|estudio)|no (existe|consta)",
    re.I,
)


def load_bench(path: Path = BENCH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def is_relevant(article: dict, question: dict) -> bool:
    text = f"{article.get('titulo', '')} {article.get('resumen', '')}".lower()
    return any(t in text for t in question["terminos_farmaco"]) and any(
        t in text for t in question["terminos_reaccion"]
    )


def unique(articles: list[dict]) -> list[dict]:
    return list({a["pmid"]: a for a in articles}.values())


def score_question(
    question: dict,
    articles: list[dict],
    answer: str | None = None,
    shown_posts=(),
) -> dict:
    """One question for one system. ``answer`` is None for the fixed search (no summary)."""
    seen = unique(articles)
    by_pmid = {a["pmid"]: a for a in seen}
    key = set(question["pmids_clave"])
    row = {
        "id": question["id"],
        "control": question["control"],
        "articulos": len(seen),
        "relevantes": sum(is_relevant(a, question) for a in seen),
        "clave_encontrados": len(key & set(by_pmid)),
        "clave_total": len(key),
        "pmids": sorted(by_pmid),
    }
    if answer is not None:
        check = check_citations(answer, seen, shown_posts)
        backed = [p for p in check["pmids_citados"] if p in by_pmid]
        row |= {
            "pmids_citados": len(check["pmids_citados"]),
            "pmids_sin_respaldo": check["pmids_sin_respaldo"],
            "posts_citados": len(check["posts_citados"]),
            "posts_sin_respaldo": check["posts_sin_respaldo"],
            "citados_relevantes": sum(is_relevant(by_pmid[p], question) for p in backed),
            "clave_citados": len(key & set(backed)),
            "admite_sin_evidencia": bool(_NO_EVIDENCE.search(answer)),
        }
    return row


def summarize(rows: list[dict]) -> dict:
    """Sums over the bench; controls are kept apart (their relevant articles should be 0)."""
    real = [r for r in rows if not r["control"]]
    controls = [r for r in rows if r["control"]]

    def rate(k, n):
        return {"k": k, "n": n, "valor": round(k / n, 3) if n else None, "ic95": wilson(k, n)}

    out = {
        "preguntas": len(real),
        "relevancia_entregados": rate(
            sum(r["relevantes"] for r in real), sum(r["articulos"] for r in real)
        ),
        "clave_encontrados": rate(
            sum(r["clave_encontrados"] for r in real), sum(r["clave_total"] for r in real)
        ),
        "controles": {r["id"]: {"relevantes": r["relevantes"]} for r in controls},
    }
    if rows and "pmids_citados" in rows[0]:
        cited = sum(r["pmids_citados"] for r in rows)
        bad = sum(len(r["pmids_sin_respaldo"]) for r in rows)
        posts = sum(r["posts_citados"] for r in rows)
        bad_posts = sum(len(r["posts_sin_respaldo"]) for r in rows)
        out |= {
            "pmids_sin_respaldo": rate(bad, cited),
            "posts_sin_respaldo": rate(bad_posts, posts),
            "relevancia_citados": rate(
                sum(r["citados_relevantes"] for r in real),
                sum(r["pmids_citados"] - len(r["pmids_sin_respaldo"]) for r in real),
            ),
            "clave_citados": rate(
                sum(r["clave_citados"] for r in real), sum(r["clave_total"] for r in real)
            ),
            "preguntas_sin_citas": sum(1 for r in real if r["pmids_citados"] == 0),
        }
        for r in controls:
            out["controles"][r["id"]] |= {
                "admite_sin_evidencia": r["admite_sin_evidencia"],
                "pmids_citados": r["pmids_citados"],
            }
        cost = [r["coste"] for r in rows if "coste" in r]
        if cost:
            out["coste"] = {
                k: round(sum(c[k] for c in cost) / len(cost), 1)
                for k in ("tokens", "pasos", "segundos", "busquedas_pubmed", "busquedas_bluesky")
            } | {"paradas": sorted({c["parada"] for c in cost})}
    return out


def run_fixed(question: dict) -> dict:
    from crawler.pubmed import search_pubmed

    return score_question(question, search_pubmed(question["consulta_fija"], FIXED_TOP))


def run_agent(question: dict, source, brands: dict, brain: str, out_dir: Path) -> dict:
    from crawler.agent import investigar

    start = time.perf_counter()
    result = investigar(question["pregunta"], question["farmaco"], source, brands, brain=brain)
    seconds = time.perf_counter() - start
    ctx, r = result.context, result.response
    answer = r.final_answer or ""
    row = score_question(question, ctx.articles, answer, ctx.shown)
    row["coste"] = {
        "tokens": r.tokens_used,
        "pasos": r.n_steps,
        "segundos": round(seconds, 1),
        "busquedas_pubmed": sum(
            1 for s in r.steps if s.action and s.action.name == "buscar_pubmed"
        ),
        "busquedas_bluesky": len(ctx.searches),
        "parada": r.stop_reason,
    }
    row["posts_guardados"] = len(ctx.store)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{question['id']}.json").write_text(
        json.dumps(
            {
                "pregunta": question,
                "respuesta": answer,
                "puntuacion": row,
                "traza": r.model_dump(),
                "menciones": [m.model_dump() for m in ctx.store.mentions.values()],
                "articulos": unique(ctx.articles),
            },
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    return row


def _fmt(rate: dict | None) -> str:
    if not rate or rate["valor"] is None:
        return "—"
    lo, hi = rate["ic95"]
    return f"{rate['k']}/{rate['n']} = {rate['valor']:.0%} [{lo:.0%}-{hi:.0%}]"


def print_summary(system: str, s: dict) -> None:
    print(f"\n== {system} ({s['preguntas']} preguntas + {len(s['controles'])} controles)")
    print(f"  artículos relevantes entregados: {_fmt(s['relevancia_entregados'])}")
    print(f"  artículos de referencia encontrados: {_fmt(s['clave_encontrados'])}")
    if "pmids_sin_respaldo" in s:
        print(f"  PMID citados SIN respaldo (inventados): {_fmt(s['pmids_sin_respaldo'])}")
        print(f"  posts citados SIN respaldo: {_fmt(s['posts_sin_respaldo'])}")
        print(f"  citados relevantes: {_fmt(s['relevancia_citados'])}")
        print(f"  referencia citados: {_fmt(s['clave_citados'])}")
        print(f"  preguntas sin ningún PMID citado: {s['preguntas_sin_citas']}")
    for cid, c in s["controles"].items():
        print(f"  control {cid}: {c}")
    if "coste" in s:
        print(f"  coste medio por pregunta: {s['coste']}")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from crawler.text import utf8_stdout

    utf8_stdout()
    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sistemas", nargs="+", choices=["fija", "agente"], default=["fija", "agente"]
    )
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--ids", nargs="+", help="only these questions")
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--muestra", type=Path, help="Bluesky from a frozen sample, not live")
    parser.add_argument("--salida", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    bench = load_bench(args.bench)
    if args.salida == REPORT and args.bench != BENCH:  # each bench, its own report
        args.salida = REPORT.with_name(f"crawler_dirigido_eval_{args.bench.stem}.json")
    if args.ids:
        bench = [q for q in bench if q["id"] in set(args.ids)]
        if args.salida == REPORT:  # a few questions must not overwrite the full report
            args.salida = REPORT.with_name("crawler_dirigido_eval_parcial.json")
    report = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "bench": str(args.bench),
        "bluesky": str(args.muestra) if args.muestra else "en vivo",
        "sistemas": {},
    }
    if "fija" in args.sistemas:
        rows = []
        for q in bench:
            rows.append(run_fixed(q))
            print(f"fija · {q['id']}: {rows[-1]['relevantes']}/{rows[-1]['articulos']} relevantes")
            time.sleep(0.5)  # NCBI: at most 3 requests per second without a key
        report["sistemas"]["fija"] = {"resumen": summarize(rows), "preguntas": rows}
    if "agente" in args.sistemas:
        from crawler.cima import load_brands
        from crawler.sources import FrozenSource, LiveSource

        source = FrozenSource.load(args.muestra) if args.muestra else LiveSource()
        brands = load_brands()
        out_dir = DATA_DIR / "dirigido" / f"{args.bench.stem}_{args.brain}"
        rows = []
        for q in bench:
            rows.append(run_agent(q, source, brands, args.brain, out_dir))
            r = rows[-1]
            print(
                f"agente · {q['id']}: {r['pmids_citados']} PMID citados "
                f"({len(r['pmids_sin_respaldo'])} sin respaldo), "
                f"{r['relevantes']}/{r['articulos']} relevantes, "
                f"{r['coste']['segundos']} s, parada {r['coste']['parada']}"
            )
        report["sistemas"][f"agente_{args.brain}"] = {"resumen": summarize(rows), "preguntas": rows}
        print(f"respuestas y trazas → {out_dir}")

    for system, data in report["sistemas"].items():
        print_summary(system, data["resumen"])
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"\n→ {args.salida}")


if __name__ == "__main__":
    main()
