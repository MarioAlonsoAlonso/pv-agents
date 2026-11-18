"""Check how the course's 27B server answers the crawler, before running the agent.

    uv run python scripts/check_remote_llm.py
    uv run python scripts/check_remote_llm.py --thinking

Three requests, each printed with the fields of the server's answer:

1. The teachers' example ("Reply with the word READY."): is the server answering at all?
2. The agent's first step in **text mode** (tool format written in the prompt, no ``tools``
   in the request), with ``logprobs`` and ``skip_special_tokens=false``: the tokens the model
   really generated, even if the server drops them from ``content``.
3. The same step in **native mode** (``tools`` in the request): do the calls come back in
   ``tool_calls``? This is what the agent uses with the 27B by default
   (``ARCA_REMOTE_NATIVE_TOOLS=1``).
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv


def show(title: str, llm, text: str) -> None:
    from tool_use.parser import parse_tool_calls

    choice = llm.last_response["choices"][0]
    print(f"\n==================== {title}")
    for key, value in choice["message"].items():
        if value in (None, "", []):
            continue
        shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        print(f"  {key}: {shown[:1200]}")
    usage = llm.last_response.get("usage")
    print(f"  finish_reason: {choice.get('finish_reason')} · usage: {usage}")
    logprobs = (choice.get("logprobs") or {}).get("content") or []
    if logprobs:
        generated = "".join(t.get("token", "") for t in logprobs)
        print(f"  tokens generados: {generated[:1500]!r}")
    calls = [c.name for c in parse_tool_calls(text).tool_calls]
    print(f"  → lo que recibe el agente: {text[:600]!r}")
    print(f"  → llamadas a herramientas detectadas: {calls}")


def main() -> None:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--farmaco", default="sertralina")
    args = parser.parse_args()

    from common.llm import RemoteChatModel
    from crawler import prompts
    from crawler.agent import CrawlerAgent
    from crawler.cima import load_brands
    from crawler.store import MentionStore
    from crawler.text import utf8_stdout
    from crawler.tools import CrawlerContext, make_registry

    utf8_stdout()
    llm = RemoteChatModel()
    key_set = llm.api_key not in ("", "EMPTY")
    print(f"servidor: {llm.base_url} · modelo: {llm.model_id} · thinking: {args.thinking}")
    print(f"clave: {'puesta' if key_set else 'NO: falta ARCA_REMOTE_LLM_API_KEY en .env'}")

    text, _ = llm.generate(
        [{"role": "user", "content": "Reply with the word READY."}],
        max_new_tokens=64,
        enable_thinking=args.thinking,
    )
    show("1. Ejemplo de los profesores", llm, text)

    class _NoSource:  # the tools are only described, never run
        pass

    ctx = CrawlerContext(_NoSource(), MentionStore(), load_brands(), (args.farmaco,))
    agent = CrawlerAgent(registry=make_registry(ctx, "vigilancia"), mode="vigilancia")
    task = {"role": "user", "content": prompts.VIGILANCIA_TASK.format(farmaco=args.farmaco)}

    agent._native_now = False
    text, _ = llm.generate(
        [{"role": "system", "content": agent.system_prompt(40)}, task],
        max_new_tokens=1024,
        enable_thinking=args.thinking,
        extra_body={"logprobs": True, "skip_special_tokens": False},
    )
    show("2. Primer paso del agente, modo texto", llm, text)

    agent._native_now = True
    text, _ = llm.generate(
        [{"role": "system", "content": agent.system_prompt(40)}, task],
        max_new_tokens=1024,
        enable_thinking=args.thinking,
        tools=agent.registry.definitions(),
    )
    show("3. Primer paso del agente, modo nativo (el que usa el agente)", llm, text)


if __name__ == "__main__":
    main()
