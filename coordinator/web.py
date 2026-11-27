"""The conversation with the coordinator in a web page (presentation and technician).

    uv run --extra agent --extra crawler python -m coordinator.web              # http://127.0.0.1:7860
    uv run --extra agent --extra crawler python -m coordinator.web --puerto 7861

The same ``Conversacion`` as the terminal chat (``coordinator/chat.py``): memory across turns,
the actions the coordinator queues (an ICSR, a big search or triage) with what they will do and
their cost, confirmed or rejected with a button, and the searches and triages that run in the
background, checked every few seconds. On the right, the tools the coordinator used for its last
answer, the trace that shows where each piece of the answer came from.

On the DGX the page listens on the session's machine only. To open it from your computer: with
VS Code (Remote-SSH) the Ports panel forwards 7860; with ssh, ``ssh -L 7860:localhost:7860
<user>@<dgx>`` and http://localhost:7860; with JupyterHub, ``--root-path /user/<u>/proxy/7860``
and the hub's ``/user/<u>/proxy/7860/`` address.

The page logic is in ``Panel`` (plain Python, tested without a browser); ``build_app`` only wires
it to Gradio.
"""

from __future__ import annotations

import argparse
import html
import json

POLL_SECONDS = 5


def _card(title: str, body: str, kind: str) -> str:
    colours = {
        "action": ("#ffedd5", "#9a3412"),
        "observation": ("#dcfce7", "#166534"),
        "pending": ("#fef9c3", "#854d0e"),
    }
    bg, fg = colours[kind]
    return (
        f'<div style="background:{bg};color:{fg};border-left:5px solid {fg};padding:6px 10px;'
        f'margin:6px 0;border-radius:6px;font-size:0.9em"><b>{html.escape(title)}</b>'
        f'<div style="white-space:pre-wrap">{html.escape(body)}</div></div>'
    )


class Panel:
    """What the page shows and does, independent of Gradio."""

    def __init__(self, conv):
        self.conv = conv
        self.history: list[dict] = []  # chat messages: {"role": ..., "content": ...}
        self.trace = ""

    # ------------------------------------------------------------------ what is shown
    def pending_choices(self) -> list[str]:
        from coordinator.agent import describe

        return [
            f"{i}. {describe(a, self.conv.store)}" for i, a in enumerate(self.conv.pendientes, 1)
        ]

    def pending_text(self) -> str:
        choices = self.pending_choices()
        if not choices:
            return "Nada pendiente de confirmar."
        return "**Pendiente de tu confirmación:**\n\n" + "\n\n".join(f"⏸ {c}" for c in choices)

    def jobs_text(self) -> str:
        if not self.conv.trabajos:
            return "Ningún trabajo en segundo plano."
        return "**Trabajos:**\n\n" + "\n\n".join(f"· {j.describir()}" for j in self.conv.trabajos)

    def store_text(self) -> str:
        s = self.conv.store
        return (
            f"{len(s.mentions)} menciones · {sum(s.reviewed.values())} triadas · "
            f"{len(s.cases)} casos"
        )

    def _show(self, response, label: str | None = None) -> None:
        self.history.append({"role": "assistant", "content": response.final_answer})
        cards = []
        for step in response.steps:
            if step.action:
                args = json.dumps(step.action.arguments, ensure_ascii=False)
                cards.append(_card(step.action.name, args[:300], "action"))
            if step.observation:
                obs = step.observation.strip()
                cards.append(
                    _card("resultado", obs[:500] + ("…" if len(obs) > 500 else ""), "observation")
                )
        for action in response.pending_actions:
            cards.append(_card("pendiente de confirmación", action.name, "pending"))
        title = html.escape(label or "Última respuesta")
        head = f"<p><b>{title}</b> · {response.n_steps} pasos · {response.tokens_used} tokens</p>"
        self.trace = head + "".join(cards)

    # ------------------------------------------------------------------ what can be done
    def send(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.history.append({"role": "user", "content": text})
        self._show(self.conv.preguntar(text))

    def _index(self, choice: str | None) -> int | None:
        if not self.conv.pendientes:
            return None
        if not choice:
            return 0
        try:
            return int(choice.split(".", 1)[0]) - 1
        except ValueError:
            return 0

    def confirm(self, choice: str | None) -> None:
        from coordinator.agent import Trabajo

        i = self._index(choice)
        if i is None or i >= len(self.conv.pendientes):
            return
        self.history.append({"role": "user", "content": "✔ Confirmo: " + self.pending_choices()[i]})
        out = self.conv.confirmar(i)
        if isinstance(out, Trabajo):
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"Lanzado en segundo plano: {out.describir()}. Puedes seguir "
                    "preguntando; te aviso aquí cuando termine.",
                }
            )
        else:
            self._show(out)

    def reject(self, choice: str | None) -> None:
        i = self._index(choice)
        if i is None or i >= len(self.conv.pendientes):
            return
        self.history.append({"role": "user", "content": "✘ No: " + self.pending_choices()[i]})
        self._show(self.conv.rechazar(i))

    def check_jobs(self) -> bool:
        """Tell the coordinator about finished background jobs; True if something changed."""
        changed = False
        for job in self.conv.terminados():
            self.history.append({"role": "user", "content": f"[terminado: {job.describir()}]"})
            self._show(self.conv.informar(job), "Trabajo terminado")
            changed = True
        return changed


def build_app(panel: Panel):
    import gradio as gr

    def view():
        choices = panel.pending_choices()
        return (
            panel.history,
            gr.update(choices=choices, value=choices[0] if choices else None),
            panel.pending_text(),
            panel.jobs_text(),
            panel.store_text(),
            panel.trace,
        )

    with gr.Blocks(title="Farmacovigilancia · coordinador") as app:
        gr.Markdown(
            "# Farmacovigilancia en redes sociales\n"
            "Pregunta al coordinador por cualquier etapa: lo que encontró el rastreador en "
            "Bluesky, lo que descartó el triaje, los casos, MedDRA, las fichas técnicas, FAERS o "
            "la literatura. Lo que tenga consecuencias (un ICSR, una búsqueda larga) espera tu "
            "confirmación."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversación", height=520)
                with gr.Row():
                    box = gr.Textbox(
                        placeholder="Escribe tu petición y pulsa Enter…", show_label=False, scale=5
                    )
                    send = gr.Button("Enviar", variant="primary", scale=1)
            with gr.Column(scale=2):
                datos = gr.Markdown(panel.store_text())
                pendiente = gr.Markdown(panel.pending_text())
                eleccion = gr.Dropdown(label="Acción pendiente", choices=[], interactive=True)
                with gr.Row():
                    si = gr.Button("Confirmar", variant="primary")
                    no = gr.Button("Rechazar")
                trabajos = gr.Markdown(panel.jobs_text())
                traza = gr.HTML(label="Qué ha consultado")
        outputs = [chat, eleccion, pendiente, trabajos, datos, traza]

        def on_send(text):
            panel.send(text)
            return ("", *view())

        def on_confirm(choice):
            panel.confirm(choice)
            return view()

        def on_reject(choice):
            panel.reject(choice)
            return view()

        def on_tick():
            panel.check_jobs()
            return view()

        box.submit(on_send, [box], [box, *outputs])
        send.click(on_send, [box], [box, *outputs])
        si.click(on_confirm, [eleccion], outputs)
        no.click(on_reject, [eleccion], outputs)
        gr.Timer(POLL_SECONDS).tick(on_tick, None, outputs)
    return app


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from coordinator.agent import Conversacion, warm_up
    from coordinator.cases import CaseStore
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
    parser.add_argument("--puerto", type=int, default=7860)
    parser.add_argument(
        "--root-path",
        default="",
        help="behind a proxy, the path the page is served at (Jupyter: /user/<u>/proxy/7860)",
    )
    parser.add_argument("--sin-vigilante", action="store_true", help="only the frozen sample")
    args = parser.parse_args(argv)

    store = CaseStore.from_files(include_monitor=not args.sin_vigilante, allow_empty=True)
    try:
        warm_up(args.variante)
    except Exception as exc:  # noqa: BLE001 - no RAG index or no GPU: the rest still works
        print(f"aviso: no se pudo cargar el RAG o el modelo MedDRA ({exc}); el resto funciona")
    conv = Conversacion(store, brain=args.brain, variant=args.variante, max_steps=args.max_steps)
    build_app(Panel(conv)).launch(
        server_name="127.0.0.1", server_port=args.puerto, root_path=args.root_path or None
    )


if __name__ == "__main__":
    main()
