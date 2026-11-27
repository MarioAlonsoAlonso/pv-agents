"""Change A (2026-11-27): the course's 27B as MedDRA coder, in ONE place.

What is measured is what the system uses: the phase-1 test (``rlm.evaluate_remote
--codificador final``), the triage's help (``rlm.evaluate_pista --codificador final``, change I)
and the coordinator (``codificar_meddra`` and ``codificar_notificacion`` with
``ARCA_MEDDRA_CODER=27b``) all call ``code``. The trained 1.7B stays the default and the local
fallback (no network, verified by phase 1).

Why these steps (from the measured errors of the 27B on the phase-1 test, 2026-11-23/26):

1. **The format.** Without reasoning, 97 of its 124 strict errors were right terms written
   without ``<answer>``. The prompt asks for the tags at the end, with an example, and the code
   also reads a short final line without them (``lenient_correct``).
2. **Names outside the dictionary.** Reasoning, it gave 157 names that are not in the 90-PT
   subset of the repository (a valid MedDRA term we do not ship, or a synonym). Every name is
   checked with the dictionary; if one is not there, the answer goes back ONCE with the error
   and the list of valid terms, as the coordinator does with the citations.

The problems are the phase-1 questions (instruction, coding rules, notification): a bare
notification is wrapped with ``build_question``, as the 1.7B's ``ReasoningModel`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm.generate_meddra import INSTRUCTION, build_question
from rlm.meddra import MedDRADictionary
from rlm.rewards import extract_answer

# The phase-1 system prompt (R1-Zero, ``rlm/data.py``), the same the 1.7B was trained with.
SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)
FORMAT = (
    "\n\nTermina SIEMPRE con los Preferred Terms en inglés dentro de <answer></answer>, "
    "separados por punto y coma, por ejemplo: <answer>Headache; Nausea</answer>"
)
# Lists the valid terms when a name is not in the dictionary, if the dictionary is small (the
# repository's subset); with the official MedDRA (``ARCA_MEDDRA_DIR``) it only names the error.
LIST_LIMIT = 200


def question_of(text: str) -> str:
    """A phase-1 question as it is, a bare notification wrapped like the 1.7B does."""
    text = text.strip()
    return text if text.startswith(INSTRUCTION) else build_question(text)


def read_terms(raw: str) -> str | None:
    """The ``<answer>`` block or, without it, a short final line (``lenient_correct``)."""
    answer = extract_answer(raw)
    if answer is not None:
        return answer
    content = raw.split("</think>")[-1].strip()
    if content and len(content) <= 300 and "\n" not in content:
        return content
    return None


def problem_with(answer: str | None, dictionary: MedDRADictionary) -> str | None:
    """What to send back: no answer, or names that are not in the dictionary."""
    if not answer:
        return (
            "No has dado la respuesta. Escribe solo los Preferred Terms dentro de "
            "<answer></answer>, separados por punto y coma."
        )
    unknown = dictionary.resolve(answer).unknown
    if not unknown:
        return None
    names = ", ".join(f"«{u}»" for u in unknown)
    listing = (
        f" Elige el que corresponda de esta lista: {'; '.join(sorted(dictionary.pt_soc))}."
        if len(dictionary) <= LIST_LIMIT
        else ""
    )
    return (
        f"Estos términos no están en nuestro diccionario MedDRA: {names}.{listing} Si ninguno "
        "corresponde, deja fuera ese acontecimiento. Responde de nuevo solo con "
        "<answer>PT; PT</answer>."
    )


@dataclass
class Coding:
    answer: str | None  # "PT; PT", or None if the model gave no answer
    raw: str  # the completion (the second one if the answer was sent back)
    n_tokens: int
    devuelto: str | None = None  # why it was sent back, if it was
    primera: str | None = None  # the first answer, before sending it back
    fuera_del_diccionario: list[str] = field(default_factory=list)  # left after the second try


def code(
    text: str,
    llm,
    thinking: bool = False,
    dictionary: MedDRADictionary | None = None,
    send_back: bool = True,
) -> Coding:
    """Code a notification (or a phase-1 question) with the 27B. ``llm`` is a chat model with
    ``generate(messages, max_new_tokens, enable_thinking) -> (text, n_tokens)``."""
    dictionary = dictionary or MedDRADictionary.default()
    max_tokens = 6000 if thinking else 1500
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question_of(text) + FORMAT},
    ]
    raw, n_tokens = llm.generate(messages, max_new_tokens=max_tokens, enable_thinking=thinking)
    answer = read_terms(raw)
    problem = problem_with(answer, dictionary) if send_back else None
    if problem is None:
        return Coding(answer, raw, n_tokens, fuera_del_diccionario=_unknown(answer, dictionary))
    messages += [
        {"role": "assistant", "content": raw.split("</think>")[-1].strip() or "(sin respuesta)"},
        {"role": "user", "content": problem},
    ]
    raw2, n2 = llm.generate(messages, max_new_tokens=max_tokens, enable_thinking=thinking)
    second = read_terms(raw2)
    final = second if second else answer  # a second try without answer keeps the first one
    return Coding(
        final,
        raw2 if second else raw,
        n_tokens + n2,
        devuelto=problem,
        primera=answer,
        fuera_del_diccionario=_unknown(final, dictionary),
    )


def _unknown(answer: str | None, dictionary: MedDRADictionary) -> list[str]:
    return list(dictionary.resolve(answer).unknown) if answer else []
