"""One chat model shared by phases 2, 3 and 4.

The API serves every phase from a single process on a 16 GB GPU, so the generator of
``/rag``, the model behind ``/tools`` and the "base" brain of ``/agent`` are the same
instance: ``get_chat_model`` caches one ``ChatModel`` per (model id, adapter) pair.

Configuration: ``ARCA_LLM_MODEL`` (default ``Qwen/Qwen3-1.7B``). Qwen3 is a hybrid model:
``enable_thinking=False`` makes it answer directly, which is what RAG wants (the reasoning
is in the retrieved text, not in a long chain of thought); the agent can turn it on.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

DEFAULT_LLM = "Qwen/Qwen3-1.7B"
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a ``<think>…</think>`` block (and an unterminated one) from a completion."""
    text = _THINK.sub("", text)
    return text.split("<think>")[0].strip() if "<think>" in text else text.strip()


def cut_at_stop(text: str, stop: list[str] | None) -> str:
    """Text up to the first stop string, ignoring stop strings inside a ``<think>`` block."""
    if not stop:
        return text
    start = text.find("</think>")
    if start >= 0:
        start += len("</think>")
    elif "<think>" in text:
        return text  # still thinking
    else:
        start = 0
    cuts = [i for s in stop if (i := text.find(s, start)) >= 0]
    return text[: min(cuts)] if cuts else text


class ChatModel:
    """A Hugging Face causal LM (plus optional LoRA adapter) behind a chat interface."""

    def __init__(self, model_id: str = DEFAULT_LLM, adapter: str | None = None):
        self.model_id = model_id
        self.adapter = adapter
        self.model = None
        self.tokenizer = None
        self.last_prompt_tokens = 0
        self._lock = threading.Lock()  # one generation at a time on the shared GPU

    @property
    def name(self) -> str:
        return (
            f"{self.model_id}+{os.path.basename(self.adapter)}" if self.adapter else self.model_id
        )

    def load(self) -> ChatModel:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        self.model = model.eval()
        return self

    def count_tokens(self, text: str) -> int:
        """Tokens of a text with this model's tokenizer (the agent's memory budget)."""
        if self.tokenizer is None:
            self.load()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """The prompt string the model actually sees (useful to show in the README)."""
        kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        """One completion for the conversation. Returns ``(raw text, generated tokens)``.

        ``temperature=0`` is greedy decoding, so answers are reproducible for the grader.
        ``stop`` ends the generation at the first of those strings, which is cut from the
        text (the code agent stops at "Observation:" so the model does not invent one).
        """
        import torch

        if self.model is None:
            self.load()
        prompt = self.render(messages, tools=tools, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])  # cost accounting (phase 4)
        sampling = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            if temperature > 0
            else {"do_sample": False}
        )
        if stop:
            # Not transformers' stop_strings: those would also fire inside <think>.
            sampling["stopping_criteria"] = [self._stop_criterion(stop, inputs)]
        with self._lock, torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        # Keep <think>/<tool_call> markers (they are not special tokens in Qwen3) but drop
        # <|im_end|> and padding.
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return cut_at_stop(text, stop), int(new_tokens.shape[0])

    def _stop_criterion(self, stop: list[str], inputs):
        """Stop once a stop string appears after the thinking block (one sequence)."""
        import torch
        from transformers import StoppingCriteria

        tokenizer, prompt_len = self.tokenizer, int(inputs["input_ids"].shape[1])

        class StopAfterThinking(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                done = cut_at_stop(text, stop) != text
                return torch.full((input_ids.shape[0],), done, device=input_ids.device)

        return StopAfterThinking()


class RemoteChatModel:
    """A chat model served behind an OpenAI-compatible API (the course's vLLM server).

    Same interface as ``ChatModel`` (``generate``, ``count_tokens``, ``last_prompt_tokens``,
    ``name``), so the ReAct harness can use it as a brain without changes. In the DGX the
    teachers serve ``qwen3.8-27b-fp8``, reachable only from inside a DGX session. Since
    November 2026 it listens at ``http://0.0.0.0:8001/v1`` and needs an API key (teachers' e-mail;
    ``docs/dgx-instructions.pdf`` still shows the old address). Configuration:
    ``ARCA_REMOTE_LLM_URL``, ``ARCA_REMOTE_LLM_API_KEY`` (only in ``.env``, never in git) and
    ``ARCA_REMOTE_LLM_MODEL``.

    Details that matter:

    * The harness keeps tool results as ``role="tool"`` messages without ``tool_call_id``,
      which the OpenAI schema does not accept. They are sent as user messages wrapped in
      ``<tool_response>`` tags, which is how Qwen's own chat template renders them.
    * vLLM may return the reasoning separately (``reasoning_content``); it is put back as a
      ``<think>`` block so the trace shows it, exactly as with the local model.
    * The server is shared by every team (at most 5 sequences at a time): at most
      ``ARCA_REMOTE_LLM_CONCURRENCY`` requests leave this process at once, and 429/5xx
      answers are retried with backoff.
    """

    _semaphore: threading.Semaphore | None = None

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        retries: int = 4,
        transport=None,
    ):
        self.model_id = model or os.environ.get("ARCA_REMOTE_LLM_MODEL", "qwen3.8-27b-fp8")
        self.base_url = (
            base_url or os.environ.get("ARCA_REMOTE_LLM_URL", "http://0.0.0.0:8001/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("ARCA_REMOTE_LLM_API_KEY", "EMPTY")
        self.timeout, self.retries, self.transport = timeout, retries, transport
        self.last_prompt_tokens = 0
        self.tokenizer = None
        if RemoteChatModel._semaphore is None:
            RemoteChatModel._semaphore = threading.Semaphore(
                int(os.environ.get("ARCA_REMOTE_LLM_CONCURRENCY", "2"))
            )

    @property
    def name(self) -> str:
        return f"{self.model_id} (remoto)"

    @staticmethod
    def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Harness messages -> OpenAI chat messages (only ``role`` and ``content``)."""
        out: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{m['content']}\n</tool_response>",
                    }
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def count_tokens(self, text: str) -> int:
        """Approximate count for the memory budget: the Qwen3 tokenizer if it is available
        locally, otherwise ~3.5 characters per token (Spanish and JSON)."""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    os.environ.get("ARCA_REMOTE_LLM_TOKENIZER", "Qwen/Qwen3-1.7B")
                )
            except Exception:  # noqa: BLE001 - no transformers, no cache: approximate
                self.tokenizer = False
        if self.tokenizer:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return max(1, int(len(text) / 3.5))

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        """One completion. Returns ``(raw text, generated tokens)`` like ``ChatModel``."""
        import time

        import httpx

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self.to_openai_messages(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if temperature == 0:
            # No explicit temperature: the model card's recommended sampling for each mode
            # (huggingface.co/Qwen/Qwen3.8-27B-FP8); greedy decoding makes Qwen loop.
            payload.update(
                {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
                if enable_thinking
                else {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                }
            )
        if stop:
            payload["stop"] = stop
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with (
                    self._semaphore,
                    httpx.Client(timeout=self.timeout, transport=self.transport) as client,
                ):
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                last_error = exc
                if attempt == self.retries:
                    raise RuntimeError(f"remote LLM unavailable: {exc}") from exc
                time.sleep(2**attempt)
        else:  # pragma: no cover - the loop always breaks or raises
            raise RuntimeError(f"remote LLM unavailable: {last_error}")
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        text = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
        usage = data.get("usage") or {}
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
        return text, int(usage.get("completion_tokens", 0))


_REMOTE: dict[str, RemoteChatModel] = {}


def get_remote_chat_model(model: str | None = None) -> RemoteChatModel:
    """The process-wide client for a remote model (no weights to load: it is just a client)."""
    key = model or os.environ.get("ARCA_REMOTE_LLM_MODEL", "qwen3.8-27b-fp8")
    with _MODELS_LOCK:
        if key not in _REMOTE:
            _REMOTE[key] = RemoteChatModel(model=key)
        return _REMOTE[key]


_MODELS: dict[tuple[str, str | None], ChatModel] = {}
_MODELS_LOCK = threading.Lock()


def get_chat_model(model_id: str | None = None, adapter: str | None = None) -> ChatModel:
    """The process-wide instance for ``(model_id, adapter)``, loaded on first use."""
    key = (model_id or os.environ.get("ARCA_LLM_MODEL") or DEFAULT_LLM, adapter)
    with _MODELS_LOCK:
        if key not in _MODELS:
            _MODELS[key] = ChatModel(*key).load()
        return _MODELS[key]
