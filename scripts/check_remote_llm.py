"""Send the crawler's first step to the 27B and show the raw answer of the server.

    uv run python scripts/check_remote_llm.py            # without thinking
    uv run python scripts/check_remote_llm.py --thinking

Prints, for one request with the real system prompt and task of ``crawler.demo vigilar``:
the fields of the answer (``content``, ``reasoning``/``reasoning_content``, ``tool_calls``),
``finish_reason``, the token usage, and the text the agent's parser gets from it. Use it when
the agent gets empty answers: it shows where the server put what the model wrote.
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv


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
    from tool_use.parser import parse_tool_calls

    utf8_stdout()

    class _NoSource:  # the tools are only described, never run
        pass

    ctx = CrawlerContext(_NoSource(), MentionStore(), load_brands(), (args.farmaco,))
    agent = CrawlerAgent(registry=make_registry(ctx, "vigilancia"), mode="vigilancia")
    messages = [
        {"role": "system", "content": agent.system_prompt(40)},
        {"role": "user", "content": prompts.VIGILANCIA_TASK.format(farmaco=args.farmaco)},
    ]
    llm = RemoteChatModel()
    print(f"servidor: {llm.base_url} · modelo: {llm.model_id} · thinking: {args.thinking}")
    print(
        f"clave: {'puesta' if llm.api_key not in ('', 'EMPTY') else 'NO (ARCA_REMOTE_LLM_API_KEY)'}"
    )
    text, generated = llm.generate(messages, max_new_tokens=1024, enable_thinking=args.thinking)

    choice = llm.last_response["choices"][0]
    message = choice["message"]
    print("\n== respuesta del servidor (campos del mensaje) ==")
    for key, value in message.items():
        if value in (None, "", []):
            print(f"{key}: (vacío)")
        else:
            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            print(f"{key}: {shown[:1500]}")
    print(f"\nfinish_reason: {choice.get('finish_reason')}")
    print(f"usage: {llm.last_response.get('usage')}")

    parsed = parse_tool_calls(text)
    print("\n== lo que recibe el agente ==")
    print(text[:2000] or "(texto vacío)")
    print(f"\nllamadas a herramientas detectadas: {[c.name for c in parsed.tool_calls]}")


if __name__ == "__main__":
    main()
