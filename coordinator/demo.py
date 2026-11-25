"""Ask the coordinator something and read its trace, step by step (DGX: the 27B, and a GPU
session for the reasoning model and the RAG).

    uv run python -m coordinator.demo "¿Qué casos de sertralina debería revisar primero?"
    uv run python -m coordinator.demo "Prepara el ICSR del caso M-…" --variante completo
    uv run python -m coordinator.demo --casos       # only the summary of the case store

The trace (thought, action, real observation) is printed and saved, with the pending ICSR
drafts, in ``data/coordinador/demo/`` (it may paraphrase posts: out of git).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from coordinator.cases import FOLDER, CaseStore


def show(response) -> None:
    for i, step in enumerate(response.steps, 1):
        print(f"\n── paso {i} " + "─" * 60)
        if step.thought:
            print(f"Pensamiento: {step.thought.strip()[:800]}")
        if step.action:
            arguments = json.dumps(step.action.arguments, ensure_ascii=False)
            print(f"Acción: {step.action.name}({arguments[:400]})")
        if step.observation:
            obs = step.observation
            print(f"Observación: {obs[:1200]}{' …' if len(obs) > 1200 else ''}")
    print("\n══ respuesta " + "═" * 58)
    print(response.final_answer)
    for action in response.pending_actions:
        print(f"\n⏸ PENDIENTE DE CONFIRMACIÓN HUMANA: {action.name}")
        print(json.dumps(action.arguments, ensure_ascii=False, indent=1))
    print(
        f"\nparada: {response.stop_reason} · pasos: {response.n_steps}"
        f" · tokens: {response.tokens_used}"
    )


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from coordinator.agent import coordinar, warm_up
    from coordinator.tools import VARIANTS
    from crawler.text import utf8_stdout

    load_dotenv(".env")
    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pregunta", nargs="?")
    parser.add_argument("--variante", default="completo", choices=VARIANTS)
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--casos", action="store_true", help="only summarise the case store")
    args = parser.parse_args(argv)

    store = CaseStore.from_files()
    if args.casos or not args.pregunta:
        from coordinator.cases import main as summary

        summary()
        top = store.of_drug("sertralina")[:5]
        print("\nlos cinco más prioritarios:")
        for c in top:
            print(f"  {json.dumps(c.summary(), ensure_ascii=False)}")
        return
    print("cargando el RAG y el modelo de razonamiento…")
    warm_up(args.variante)
    response = coordinar(args.pregunta, store, args.brain, args.variante, args.max_steps)
    show(response)
    out = FOLDER / "demo" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"pregunta": args.pregunta, "variante": args.variante, **response.model_dump()},
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        "utf-8",
    )
    print(f"→ traza en {Path(out)}")


if __name__ == "__main__":
    main()
