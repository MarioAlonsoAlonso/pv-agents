"""The cases the coordinator works on: the crawler's mentions joined with the triage's answers.

What can be done by code is done before the coordinator starts (the lesson of the crawler and
the triage): the crawler finds the posts, the triage decides which are adverse events and
extracts the case, and this module joins both, keeps the adverse events and sorts them by
priority. The coordinator (the 27B) then reads cases, not hundreds of posts.

    uv run python -m coordinator.cases          # summary of the cases built from data/

Priority, as a pharmacovigilance unit would review them: serious cases first (they have a
15-day legal deadline instead of 90), then cases with a known patient and outcome (closer to a
valid ICSR), then the most recent.

The MedDRA codes are added later, one case at a time, by the reasoning model
(``codificar_meddra``); they are kept in ``data/coordinador/meddra.json`` so that a case is
coded only once.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from crawler.store import RawMention, read_mentions
from evaluacion_triaje.lote import conversation_text

DATA = Path("data")
FOLDER = DATA / "coordinador"
# The mentions and the triage's answers (the final configuration, v2 per post), development
# and exam together: the triage is finished, so both are just data for the coordinator.
SOURCES = [
    (
        DATA / "hibrido_27b_menciones_dev.jsonl",
        DATA / "triaje" / "eval" / "v2_27b_mencion_dev_respuestas.jsonl",
    ),
    (
        DATA / "hibrido_27b_menciones_test.jsonl",
        DATA / "triaje" / "eval" / "v2_27b_mencion_test_respuestas.jsonl",
    ),
]


@dataclass
class Case:
    id: str
    farmaco: str
    fecha: str
    texto: str
    conversacion: str
    encontrado_por: str
    tipo_mencion: str
    triaje: dict

    @property
    def grave(self) -> bool:
        return self.triaje.get("grave") == "si"

    @property
    def es_efecto_adverso(self) -> bool:
        return self.triaje.get("es_efecto_adverso") == "si"

    def priority(self) -> tuple:
        """Sort key: serious first, then the most complete case, then the newest."""
        t = self.triaje
        complete = (t.get("sexo") not in ("", "desconocido")) + (
            t.get("desenlace") not in ("", "desconocido")
        )
        return (not self.grave, -complete, _neg_date(self.fecha))

    def summary(self) -> dict:
        t = self.triaje
        return {
            "id": self.id,
            "fecha": self.fecha[:10],
            "grave": f"si ({t.get('criterio_gravedad')})" if self.grave else "no consta",
            "reacciones": t.get("reacciones", ""),
            "paciente": t.get("paciente", ""),
            "sexo": t.get("sexo", ""),
            "edad": t.get("edad", ""),
            "desenlace": t.get("desenlace", ""),
            "origen": "respuesta en una conversación (agente)"
            if self.encontrado_por == "agente"
            else "nombra el fármaco",
        }

    def report_text(self) -> str:
        """The case as a notification, for the MedDRA coder: the post itself, plus the reaction
        the triage deduced from the conversation when the post does not say it."""
        deduced = [
            r.strip(" []")
            for r in self.triaje.get("reacciones", "").split("|")
            if r.strip().startswith("[")
        ]
        text = self.texto.strip()
        if deduced:
            text += f" (Según la conversación, la reacción es: {', '.join(deduced)}.)"
        return text


def _neg_date(iso: str) -> str:
    """Newest first with an ascending sort: invert the digits of the date."""
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in iso)


@dataclass
class CaseStore:
    cases: dict[str, Case] = field(default_factory=dict)
    reviewed: dict[str, int] = field(default_factory=dict)  # mentions triaged, per drug
    meddra: dict[str, list[str]] = field(default_factory=dict)
    meddra_file: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def build(
        cls, pairs: list[tuple[list[RawMention], dict[str, dict]]], meddra_file: Path | None = None
    ) -> CaseStore:
        """From (mentions, triage answers by id) pairs. Only the adverse events become cases;
        the rest is counted (``reviewed``) so the coordinator can say "N of M"."""
        store = cls(meddra_file=meddra_file)
        for mentions, answers in pairs:
            for m in mentions:
                answer = answers.get(m.id)
                if answer is None:
                    continue
                store.reviewed[m.farmaco] = store.reviewed.get(m.farmaco, 0) + 1
                case = Case(
                    id=m.id,
                    farmaco=m.farmaco,
                    fecha=m.fecha,
                    texto=m.texto,
                    conversacion=conversation_text(m),
                    encontrado_por=m.encontrado_por,
                    tipo_mencion=m.tipo_mencion,
                    triaje={k: v for k, v in answer.items() if k != "id"},
                )
                if case.es_efecto_adverso:
                    store.cases[m.id] = case
        if meddra_file and meddra_file.exists():
            store.meddra = json.loads(meddra_file.read_text("utf-8"))
        return store

    @classmethod
    def from_files(
        cls, sources=None, meddra_file: Path | None = FOLDER / "meddra.json"
    ) -> CaseStore:
        pairs = []
        for mentions_file, answers_file in sources or SOURCES:
            if not (Path(mentions_file).exists() and Path(answers_file).exists()):
                continue
            with open(answers_file, encoding="utf-8") as f:
                answers = {a["id"]: a for a in map(json.loads, filter(str.strip, f))}
            pairs.append((read_mentions(mentions_file), answers))
        if not pairs:
            raise FileNotFoundError(
                "No hay casos: faltan las menciones del rastreador o las respuestas del triaje "
                f"en data/ ({', '.join(str(a) for a, _ in sources or SOURCES)}). Ver "
                "coordinator/README.md."
            )
        return cls.build(pairs, meddra_file)

    def of_drug(self, drug: str | None = None, serious_only: bool = False) -> list[Case]:
        out = [
            c
            for c in self.cases.values()
            if (drug is None or c.farmaco == drug) and (c.grave or not serious_only)
        ]
        return sorted(out, key=Case.priority)

    def get(self, case_id: str) -> Case:
        case_id = case_id.strip().strip("[]")
        if case_id not in self.cases:
            raise KeyError(
                f"no case with id '{case_id}': use the ids that listar_casos returns (M-…)"
            )
        return self.cases[case_id]

    def save_meddra(self, case_id: str, terms: list[str]) -> None:
        with self._lock:
            self.meddra[case_id] = terms
            if self.meddra_file:
                self.meddra_file.parent.mkdir(parents=True, exist_ok=True)
                self.meddra_file.write_text(
                    json.dumps(self.meddra, ensure_ascii=False, indent=1), "utf-8"
                )


def main() -> None:
    from collections import Counter

    from crawler.text import utf8_stdout

    utf8_stdout()
    store = CaseStore.from_files()
    for drug, n in store.reviewed.items():
        cases = store.of_drug(drug)
        print(f"{drug}: {n} menciones triadas → {len(cases)} efectos adversos")
        print(f"  graves: {sum(c.grave for c in cases)}")
        print(f"  origen: {dict(Counter(c.encontrado_por for c in cases))}")
        print(f"  codificados en MedDRA: {sum(c.id in store.meddra for c in cases)}")


if __name__ == "__main__":
    main()
