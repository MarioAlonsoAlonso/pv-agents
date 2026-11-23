"""The triage agent: for each mention, is it an adverse event of the drug and, if so, the case.

The design follows what the crawler learned: what can be guaranteed in code is not asked of the
model. The model only reads and decides; the code

* groups the mentions by conversation (one request reads the whole thread, so the replies of
  a thread share their context) and splits long threads into chunks;
* keeps only answers for ids it sent (an invented id is dropped), with the exact values of the
  answer sheet (``evaluacion_triaje.lote.validate``), and fixes what is mechanical (``true`` →
  ``si``, a criterion without ``grave = si``, a source of sex when the sex is unknown);
* asks again, once, for the mentions whose answer is missing or invalid, saying what was wrong;
* and if there is still no valid answer, does NOT drop the mention: it goes on as a possible
  adverse event for a person to review (``estado = "sin_respuesta"``). In pharmacovigilance,
  losing a real case is worse than one more case to review.

Brains: ``27b`` (the course's Qwen3.8-27B, DGX), ``27b-thinking`` (the same with reasoning) and
``palabras``, a keyword baseline without a model, to measure whether triage needs an LLM at all.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crawler.store import RawMention
from crawler.text import fold
from evaluacion_triaje.lote import LABELS, normalise, validate
from triage.prompts import REPAIR, build_prompt

FIELDS = [f for f in LABELS if f != "notas"]
BRAINS = ("27b", "27b-thinking", "palabras")


@dataclass
class Cost:
    calls: int = 0
    repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    no_answer: int = 0

    def add(self, other: Cost) -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)


# --------------------------------------------------------------------------- parsing


def fix(item: dict) -> dict:
    """Mechanical fixes before validating: what has one obvious meaning is not an error."""
    row = {f: item.get(f, "") for f in FIELDS}
    for f, v in row.items():
        if isinstance(v, bool):
            v = "si" if v else "no"
        elif v is None:
            v = ""
        elif isinstance(v, list):
            v = " | ".join(str(x) for x in v)
        row[f] = str(v).strip()
    if row["es_efecto_adverso"].lower() in {"sí", "yes", "true"}:
        row["es_efecto_adverso"] = "si"
    if row["es_efecto_adverso"].lower() in {"false"}:
        row["es_efecto_adverso"] = "no"
    if row["grave"].lower() in {"no", "no consta", "no_consta", ""}:
        row["grave"] = "no_consta"
        row["criterio_gravedad"] = ""
    if row["sexo"].lower() in {"", "desconocido"}:
        row["sexo"], row["sexo_fuente"] = "desconocido", ""
    if not row["paciente"]:
        row["paciente"] = "desconocido"
    if not row["desenlace"]:
        row["desenlace"] = "desconocido"
    return row


def parse_answers(raw: str, ids: set[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """The valid answers in a completion, by id, and what was wrong with the others.

    Reads every flat JSON object in the text after the reasoning, so a list cut by the token
    limit still yields the objects before the cut."""
    text = raw.split("</think>")[-1]
    answers: dict[str, dict] = {}
    problems: dict[str, str] = {}
    for chunk in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            item = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        i = item.get("id") if isinstance(item, dict) else None
        if i not in ids:
            continue  # an id we did not send: ignored
        row = fix(item)
        error = validate(row)
        if error:
            problems[i] = error
        else:
            answers[i] = normalise({**row, "notas": ""})
            problems.pop(i, None)
    for i in ids - set(answers) - set(problems):
        problems[i] = "falta la respuesta"
    return answers, problems


def to_review(mention: RawMention, reason: str) -> dict:
    """No valid answer after the repair: a possible adverse event, for a person to review."""
    return {
        **{f: "" for f in FIELDS},
        "es_efecto_adverso": "si",
        "reacciones": "[sin determinar]",
        "paciente": "desconocido",
        "edad": "desconocido",
        "sexo": "desconocido",
        "grave": "no_consta",
        "desenlace": "desconocido",
        "notas": f"sin respuesta válida del modelo ({reason}): revisión humana",
    }


# --------------------------------------------------------------------------- keyword baseline

# Common adverse reactions of antidepressants and anxiolytics, in the words people use. Not
# tuned on the development set: it is the "is an LLM needed?" baseline.
SYMPTOMS = [
    "nausea", "vomit", "mareo", "mareada", "mareado", "sueno", "somnolencia", "insomnio",
    "no duermo", "no pego ojo", "dolor de cabeza", "jaqueca", "migrana", "diarrea", "temblor",
    "sudor", "ansiedad", "cansancio", "cansada", "cansado", "agotad", "apetito", "engorde",
    "engorda", "he engordado", "libido", "sexual", "boca seca", "efectos secundarios",
    "efecto secundario", "me sienta mal", "me sento mal", "me dejo ko", "abstinencia",
    "al dejarla", "no me hace nada", "no me hace efecto", "taquicardia", "palpitaciones",
    "zumbidos", "bostez", "pesadillas", "irritable", "apatia", "anorgasmia",
]  # fmt: skip


def keyword_answer(m: RawMention) -> dict:
    text = fold(m.texto)
    hits = [s for s in SYMPTOMS if s in text]
    if not hits:
        return normalise(
            {"es_efecto_adverso": "no", "motivo_no": "sin_reaccion", "notas": "palabras"}
        )
    return normalise(
        {
            "es_efecto_adverso": "si",
            "reacciones": " | ".join(hits),
            "paciente": "autor",
            "sexo": "desconocido",
            "grave": "no_consta",
            "desenlace": "desconocido",
            "notas": "palabras",
        }
    )


# --------------------------------------------------------------------------- the agent


def groups(mentions: list[RawMention], por: str = "conversacion", size: int = 8):
    """The requests: one per conversation (chunks of ``size``), or one per mention."""
    if por == "mencion":
        return [[m] for m in mentions]
    if por != "conversacion":
        raise ValueError("por must be 'conversacion' or 'mencion'")
    by_thread: dict[str, list[RawMention]] = {}
    for m in mentions:
        by_thread.setdefault(m.hilo_id, []).append(m)
    out = []
    for group in by_thread.values():
        out.extend(group[i : i + size] for i in range(0, len(group), size))
    return out


def triage_group(
    llm, group: list[RawMention], version: str, thinking: bool = False
) -> tuple[dict[str, dict], Cost]:
    """One conversation: request, validation, one repair, and review for what is left."""
    cost = Cost()
    ids = {m.id for m in group}
    budget = 300 + 250 * len(group) + (4000 if thinking else 0)
    messages = [{"role": "user", "content": build_prompt(version, group)}]

    def ask() -> str:
        start = time.perf_counter()
        raw, generated = llm.generate(messages, max_new_tokens=budget, enable_thinking=thinking)
        cost.calls += 1
        cost.prompt_tokens += int(getattr(llm, "last_prompt_tokens", 0) or 0)
        cost.completion_tokens += int(generated or 0)
        cost.seconds += time.perf_counter() - start
        return raw

    raw = ask()
    answers, problems = parse_answers(raw, ids)
    if problems:
        cost.repairs += 1
        listing = "\n".join(f"- {i}: {p}" for i, p in sorted(problems.items()))
        messages += [
            {"role": "assistant", "content": raw.split("</think>")[-1].strip()},
            {"role": "user", "content": REPAIR.format(problemas=listing, ids=", ".join(problems))},
        ]
        more, problems = parse_answers(ask(), set(problems))
        answers.update(more)
    for m in group:
        if m.id in answers:
            answers[m.id]["estado"] = "ok"
        else:
            answers[m.id] = {**to_review(m, problems.get(m.id, "")), "estado": "sin_respuesta"}
            cost.no_answer += 1
    return answers, cost


def remote_llm_factory(brain: str) -> tuple[Callable[[], object], bool]:
    """One client per worker thread (the token counts of a shared client would mix requests);
    the concurrency limit of the shared server is global anyway (``RemoteChatModel``)."""
    from common.llm import RemoteChatModel

    if brain not in ("27b", "27b-thinking"):
        raise ValueError(f"unknown brain {brain!r}; use one of {BRAINS}")
    local = threading.local()

    def factory():
        if not hasattr(local, "llm"):
            local.llm = RemoteChatModel()
        return local.llm

    return factory, brain == "27b-thinking"


@dataclass
class TriageRun:
    answers: dict[str, dict] = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)


def read_answers(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {a["id"]: a for a in map(json.loads, filter(str.strip, f))}


def run_triage(
    mentions: list[RawMention],
    brain: str = "27b",
    version: str = "v1",
    por: str = "conversacion",
    out: Path | None = None,
    workers: int = 2,
    llm_factory: Callable[[], object] | None = None,
    thinking: bool = False,
    progress: Callable[[str], None] = print,
) -> TriageRun:
    """Triage every mention. With ``out``, each answer is appended as soon as its conversation
    is done, and a run that was cut resumes where it stopped (ids already in ``out``)."""
    run = TriageRun(answers=read_answers(out) if out else {})
    pending = [m for m in mentions if m.id not in run.answers]
    if run.answers:
        progress(f"reanudando: {len(run.answers)} ya hechas, {len(pending)} pendientes")
    lock = threading.Lock()

    def save(batch: dict[str, dict]) -> None:
        with lock:
            run.answers.update(batch)
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "a", encoding="utf-8") as f:
                    for i, a in batch.items():
                        f.write(json.dumps({"id": i, **a}, ensure_ascii=False) + "\n")

    if brain == "palabras":
        start = time.perf_counter()
        save({m.id: {**keyword_answer(m), "estado": "ok"} for m in pending})
        run.cost.seconds += time.perf_counter() - start
        return run

    if llm_factory is None:
        llm_factory, thinking = remote_llm_factory(brain)
    batches = groups(pending, por)
    done = 0

    def work(group):
        return triage_group(llm_factory(), group, version, thinking)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, g) for g in batches]
        for future in as_completed(futures):
            answers, cost = future.result()
            save(answers)
            with lock:
                run.cost.add(cost)
                done += 1
            yes = sum(a["es_efecto_adverso"] == "si" for a in answers.values())
            progress(
                f"  {done}/{len(batches)}: {len(answers)} menciones, {yes} efecto adverso"
                f"{'s' if yes != 1 else ''}, {cost.prompt_tokens + cost.completion_tokens} tokens"
                + (f", {cost.no_answer} a revisión" if cost.no_answer else "")
            )
    return run
