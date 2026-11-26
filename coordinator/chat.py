"""Talk to the system: a conversation with the coordinator in the terminal (DGX: the 27B; a GPU
session for the reasoning model and the RAG).

    uv run --extra crawler python -m coordinator.chat
    uv run --extra crawler python -m coordinator.chat --traza        # also every step
    uv run --extra crawler python -m coordinator.chat --sin-vigilante  # only the frozen sample

Anything you write is a request for the coordinator, about any stage: what the crawler found,
what the triage discarded, the cases, MedDRA, the technical sheets, FAERS, the literature. When
the coordinator queues an action (an ICSR, a search of more than a month, a big triage), it is
shown with what it will do and its cost, and you answer:

    sí / confirmo      run it (the first pending one; "sí 2" for the second)
    no                 do not run it
    pendientes         show what is waiting
    salir

The conversation is saved in ``data/coordinador/chat/`` (it may paraphrase posts: out of git).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

YES = {"si", "sí", "confirmo", "adelante", "vale", "ok", "hazlo"}
NO = {"no", "cancela", "cancelar"}


def show_pending(conv) -> None:
    from coordinator.agent import describe

    for i, action in enumerate(conv.pendientes, 1):
        print(f"  ⏸ [{i}] {describe(action, conv.store)}")
    if conv.pendientes:
        print("  → «sí» para lanzarlo, «no» para descartarlo")


def show(response, trace: bool) -> None:
    if trace:
        for step in response.steps:
            if step.action:
                args = json.dumps(step.action.arguments, ensure_ascii=False)[:200]
                print(f"    · {step.action.name}({args})")
    print(f"\n{response.final_answer}\n")
    print(f"  ({response.n_steps} pasos, {response.tokens_used} tokens)")


def command(text: str) -> tuple[str, int]:
    """'sí 2' -> ('si', 1); anything else is a request."""
    words = text.strip().lower().split()
    if not words:
        return "", 0
    head = words[0].strip(".,!¡")
    index = int(words[1]) - 1 if len(words) > 1 and words[1].isdigit() else 0
    if head in YES and len(words) <= 2:
        return "si", index
    if head in NO and len(words) <= 2:
        return "no", index
    if head in ("salir", "adios", "adiós", "exit"):
        return "salir", 0
    if head == "pendientes":
        return "pendientes", 0
    return "pregunta", 0


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from coordinator.agent import Conversacion, warm_up
    from coordinator.cases import FOLDER, CaseStore
    from coordinator.tools import VARIANTS
    from crawler.text import utf8_stdout

    load_dotenv(".env")
    utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variante", default="completo", choices=VARIANTS)
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--traza", action="store_true", help="show every tool call")
    parser.add_argument("--sin-vigilante", action="store_true", help="only the frozen sample")
    args = parser.parse_args(argv)

    store = CaseStore.from_files(include_monitor=not args.sin_vigilante, allow_empty=True)
    print(
        f"{len(store.mentions)} menciones, {sum(store.reviewed.values())} triadas, "
        f"{len(store.cases)} casos. Cargando el RAG y el modelo de razonamiento…"
    )
    if not store.mentions:
        print(
            "  (todavía no hay datos: ni la muestra de evaluación ni el vigilante. Puedes pedirle "
            "que rastree un fármaco, o copiar la muestra a data/: coordinator/README.md)"
        )
    try:
        warm_up(args.variante)
    except Exception as exc:  # noqa: BLE001 - no RAG index or no GPU: the rest still works
        print(
            f"  aviso: no se pudo cargar el RAG o el modelo MedDRA ({type(exc).__name__}: {exc}).\n"
            "  Rastrear, triar, casos, FAERS y literatura funcionan; las fichas técnicas y la "
            "codificación MedDRA darán error (en la DGX: una sesión con GPU y make rag-corpus)."
        )
    conv = Conversacion(store, brain=args.brain, variant=args.variante, max_steps=args.max_steps)
    log = FOLDER / "chat" / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Escribe tu pregunta («salir» para terminar).\n")
    while True:
        try:
            text = input("técnico> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        kind, index = command(text)
        if kind == "salir":
            break
        if kind == "":
            continue
        if kind == "pendientes":
            show_pending(conv) if conv.pendientes else print("  nada pendiente")
            continue
        if kind in ("si", "no") and not conv.pendientes:
            kind = "pregunta"  # a plain "sí" with nothing queued is part of the conversation
        if kind in ("si", "no") and index >= len(conv.pendientes):
            print(f"  no hay un pendiente {index + 1}")
            continue
        if kind == "si":
            print("  lanzado; puede tardar…")
            response = conv.confirmar(index)
        elif kind == "no":
            response = conv.rechazar(index)
        else:
            response = conv.preguntar(text)
        show(response, args.traza)
        show_pending(conv)
        with open(log, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"tecnico": text, **response.model_dump()}, ensure_ascii=False, default=str
                )
                + "\n"
            )
    print(f"conversación guardada en {log}")


if __name__ == "__main__":
    main()
