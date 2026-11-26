"""Try the crawler agent by hand and read its trace, step by step.

    # the agent alone, on live Bluesky, last 7 days
    uv run --extra crawler python -m crawler.demo vigilar sertralina --dias 7
    # the hybrid: fixed search + the agent reading the threads (live, last 14 days)
    uv run --extra crawler python -m crawler.demo hilos lorazepam --dias 14 --max-hilos 6
    # the analyst's tools: what PubMed says, and more patients on Bluesky
    uv run --extra crawler python -m crawler.demo literatura sertralina "¿hiponatremia en ancianos?"
    # the same, but on the frozen sample instead of live Bluesky
    uv run python -m crawler.demo vigilar sertralina --muestra data/bluesky/sample.jsonl

``--brain`` chooses the brain (``27b``, ``27b-thinking``, ``base``). The trace (thought,
action, real observation) is printed and saved, with the mentions, to
``data/bluesky/demo/<mode>_<drug>_<brain>.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crawler.cima import load_brands
from crawler.config import DATA_DIR
from crawler.sources import FrozenSource, LiveSource, to_iso
from crawler.store import MentionStore


def show(run) -> None:
    r = run.response
    for i, step in enumerate(r.steps, 1):
        print(f"\n── paso {i} " + "─" * 60)
        if step.thought:
            print(f"Pensamiento: {step.thought.strip()[:800]}")
        if step.action:
            arguments = json.dumps(step.action.arguments, ensure_ascii=False)
            print(f"Acción: {step.action.name}({arguments})")
        if step.observation:
            obs = step.observation.strip()
            print(f"Observación: {obs[:1200]}{' …' if len(obs) > 1200 else ''}")
    print("\n══ respuesta final " + "═" * 52)
    print(r.final_answer)
    print(f"\nparada: {r.stop_reason} · pasos: {r.n_steps} · tokens: {r.tokens_used}")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from crawler.text import utf8_stdout

    utf8_stdout()
    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("modo", choices=["vigilar", "hilos", "literatura"])
    parser.add_argument("farmaco")
    parser.add_argument("pregunta", nargs="?", default="", help="literatura: the question")
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--dias", type=int, default=7, help="live: how far back to look")
    parser.add_argument("--max-hilos", type=int, default=6, help="hilos: threads to read")
    parser.add_argument("--muestra", type=Path, default=None, help="use a frozen sample")
    args = parser.parse_args(argv)

    from crawler.agent import investigar_literatura, revisar_hilos, vigilar
    from crawler.systems import fixed_search, threads_of

    brands = load_brands()
    if args.muestra:
        source = FrozenSource.load(args.muestra)
        manifest = args.muestra.with_name("manifest.json")
        period = json.loads(manifest.read_text("utf-8")) if manifest.exists() else {}
        since, until = period.get("since"), period.get("until")
    else:
        source = LiveSource()
        until = to_iso(datetime.now(UTC))
        since = to_iso(datetime.now(UTC) - timedelta(days=args.dias))
    print(f"fuente: {'muestra ' + str(args.muestra) if args.muestra else 'Bluesky en vivo'}")
    if args.modo == "literatura":  # the coordinator's tool: no period
        print(f"periodo: sin límite (herramienta del analista) · cerebro: {args.brain}")
    else:
        print(f"periodo: {since} → {until} · cerebro: {args.brain}")

    store = MentionStore()
    if args.modo == "vigilar":
        runs = [vigilar(source, store, brands, args.farmaco, since, until, brain=args.brain)]
    elif args.modo == "hilos":
        hits, _ = fixed_search(source, store, args.farmaco, brands, since, until)
        threads = threads_of(hits)[: args.max_hilos]
        print(f"búsqueda fija: {len(hits)} posts que nombran el fármaco, {len(threads)} hilos")
        runs = revisar_hilos(source, store, brands, args.farmaco, threads, brain=args.brain)
    else:
        if not args.pregunta:
            parser.error("literatura necesita una pregunta")
        runs = [investigar_literatura(args.pregunta, args.farmaco, brain=args.brain)]
        store = runs[0].context.store

    for run in runs:
        show(run)
    by_agent = [m for m in store.mentions.values() if m.encontrado_por == "agente"]
    print(f"\nmenciones: {len(store)} en total, {len(by_agent)} guardadas por el agente")
    for m in by_agent:
        print(f"  [{m.post_id}] {m.tipo_mencion}: {' '.join(m.texto.split())[:150]}")
    terms = [t for run in runs for t in run.context.new_terms]
    if terms:
        print(f"términos propuestos: {terms}")

    out = DATA_DIR / "demo" / f"{args.modo}_{args.farmaco}_{args.brain}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "trazas": [run.response.model_dump() for run in runs],
                "menciones": [m.model_dump() for m in store.mentions.values()],
                "articulos": [a for run in runs for a in run.context.articles],
            },
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
