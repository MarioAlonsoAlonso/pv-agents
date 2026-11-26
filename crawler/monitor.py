"""Periodic surveillance: new mentions of each drug since the last run.

    uv run --extra crawler python -m crawler.monitor --sin-agente          # one run, no model
    uv run --extra crawler python -m crawler.monitor --brain 27b           # with the agent (DGX)
    uv run --extra crawler python -m crawler.monitor --cada 6 --bucle      # every 6 hours
    uv run --extra crawler python -m crawler.monitor --farmacos lorazepam --desde 2026-01-01
    uv run --extra crawler python -m crawler.monitor --brain 27b --triar   # + triage of the new

The scheduler is code, not an agent: the same steps every time. Each run, for each drug:

1. **Fixed search** of the name, the CIMA brands and the terms the agent discovered in
   earlier runs, only for posts published since the last run. Every hit is a mention.
2. **Threads under watch.** A thread with a hit is watched for ``--dias-seguimiento`` days:
   it is downloaded again to find new replies. Those replies do not name the drug, so no
   search would ever find them; this is what keeps indirect mentions from being lost.
3. **Agent (optional).** It reads the threads that changed and saves the replies that talk
   about the drug (``revisar_hilos``). New ways of naming the drug it proposes are added to
   the search terms of the next run.

``--desde`` searches everything since a date in this run (for a drug added to the list, or to
fill a gap), whatever the last run was; the next runs carry on from now. Every thread found is
new, so the agent reads all of them: with months of posts, expect many agent runs (tokens).
Each search term brings at most 2000 posts.

``--triar`` passes the new mentions through the triage (its final configuration: v2, one post
per request) as soon as the run ends, so that they become cases for the coordinator.

Files, in ``data/bluesky/monitor/`` (the system's store of live data; the coordinator reads them
together with the frozen evaluation sample):

* ``estado.json``: last run per drug, watched threads (with their AT-URIs, needed to download
  them again) and discovered terms;
* ``menciones.jsonl``: every mention found, appended run after run (what triage reads);
* ``triaje_respuestas.jsonl``: the triage's answer for each of them (``--triar``);
* ``en_marcha.lock``: while a run or a triage is going on. Two at once would write the same
  files, so the second one stops with an error (a lock older than 12 hours is stale).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crawler.cima import load_brands
from crawler.config import DATA_DIR, WATCHLIST
from crawler.sources import LiveSource, SourceError, to_iso
from crawler.store import MentionStore
from crawler.systems import fixed_search, threads_of

STATE_DIR = DATA_DIR / "monitor"
TRIAGE_VERSION, TRIAGE_BY = "v2", "mencion"  # the triage's final configuration (triage/README)
LOCK_HOURS = 12


def files(folder: Path | None = None) -> dict[str, Path]:
    folder = Path(folder or STATE_DIR)
    return {
        "estado": folder / "estado.json",
        "menciones": folder / "menciones.jsonl",
        "triaje": folder / "triaje_respuestas.jsonl",
        "lock": folder / "en_marcha.lock",
    }


class Busy(RuntimeError):
    """Another run or triage is writing the monitor's files."""


def running(folder: Path | None = None) -> dict | None:
    """What is running now (``{"que": ..., "desde": ...}``), or None. A stale lock is ignored."""
    lock = files(folder)["lock"]
    if not lock.exists():
        return None
    try:
        info = json.loads(lock.read_text("utf-8"))
        started = datetime.fromisoformat(info["desde"].replace("Z", "+00:00"))
    except (ValueError, KeyError):
        return None
    if datetime.now(UTC) - started > timedelta(hours=LOCK_HOURS):
        return None
    return info


class exclusive:  # noqa: N801 - used as ``with exclusive(...)``
    """Only one writer of the monitor's files at a time (the periodic monitor, a search the
    coordinator launched, a triage)."""

    def __init__(self, what: str, folder: Path | None = None):
        self.what, self.lock = what, files(folder)["lock"]

    def __enter__(self):
        busy = running(self.lock.parent)
        if busy:
            raise Busy(
                f"ya hay algo en marcha desde {busy['desde']}: {busy['que']}. Espera a que "
                "termine (o borra el fichero en_marcha.lock si se cortó)."
            )
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock.write_text(
            json.dumps({"que": self.what, "desde": to_iso(datetime.now(UTC))}), "utf-8"
        )
        return self

    def __exit__(self, *exc):
        self.lock.unlink(missing_ok=True)
        return False


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"ultima_ejecucion": {}, "hilos": {}, "terminos": {}}


def run_once(
    source,
    state: dict,
    drugs=WATCHLIST,
    brain: str | None = None,
    follow_days: int = 7,
    first_lookback_hours: int = 24,
    now: datetime | None = None,
    since: str | None = None,
    until: str | None = None,
) -> tuple[MentionStore, dict]:
    """One surveillance run. Returns the new mentions and a summary; updates ``state``.
    ``since`` (ISO date) overrides where the search starts, for every drug of this run.
    ``until`` (ISO date, exclusive) closes the window: a past period ("only September"), which
    does not move the last-run date, so the periodic monitor carries on where it was."""
    now = now or datetime.now(UTC)
    brands = load_brands()
    store = MentionStore()
    summary: dict = {}
    start = since
    for drug in drugs:
        since = to_iso(start) if start else None
        since = since or state["ultima_ejecucion"].get(drug)
        since = since or to_iso(now - timedelta(hours=first_lookback_hours))
        extra = [t["termino"] for t in state["terminos"].get(drug, [])]
        before = len(store)
        end = to_iso(until) if until else to_iso(now)
        hits, _ = fixed_search(source, store, drug, brands, since, end, extra)
        explicit = len(store) - before

        # Watch the threads of the hits, and download the watched ones again.
        until = to_iso(now + timedelta(days=follow_days))
        for pid in threads_of(hits):
            post = source.get(pid)
            uri = source.uri(post.thread_id)
            entry = state["hilos"].setdefault(
                post.thread_id, {"farmaco": drug, "uri": uri, "hasta": until, "vistos": []}
            )
            entry["hasta"] = until
        changed, new_replies = [], []
        for thread_id, entry in list(state["hilos"].items()):
            if entry["farmaco"] != drug:
                continue
            if entry["hasta"] < to_iso(now):
                del state["hilos"][thread_id]  # stopped watching
                continue
            source.remember(thread_id, entry["uri"])
            try:
                posts = source.thread(thread_id)
            except SourceError:
                del state["hilos"][thread_id]  # deleted or blocked
                continue
            fresh = [p.id for p in posts if p.id not in entry["vistos"]]
            if fresh:
                changed.append(thread_id)
                new_replies += [i for i in fresh if entry["vistos"]]  # not on first sight
                entry["vistos"] += fresh

        tokens = 0
        if brain and changed:
            from crawler.agent import revisar_hilos

            runs = revisar_hilos(
                source, store, brands, drug, changed, brain=brain, new_reply_ids=new_replies
            )
            for run in runs:
                tokens += run.summary["tokens"]
                for term in run.context.new_terms:
                    known = state["terminos"].setdefault(drug, [])
                    if term["termino"].lower() not in {t["termino"].lower() for t in known}:
                        known.append({**term, "desde": to_iso(now)})
        if not until:
            state["ultima_ejecucion"][drug] = to_iso(now)
        summary[drug] = {
            "explicitas": explicit,
            "por_agente": len(store) - before - explicit,
            "hilos_vigilados": sum(1 for e in state["hilos"].values() if e["farmaco"] == drug),
            "hilos_con_novedades": len(changed),
            "tokens": tokens,
        }
    return store, summary


def crawl_and_save(
    source,
    drugs=WATCHLIST,
    brain: str | None = None,
    follow_days: int = 7,
    first_lookback_hours: int = 24,
    since: str | None = None,
    now: datetime | None = None,
    folder: Path | None = None,
    until: str | None = None,
) -> dict:
    """One run that keeps its result: the new mentions are appended to ``menciones.jsonl``
    and the state is saved. What the command line and the coordinator's ``rastrear`` call.
    With ``until``, replies the agent found in those threads after the window are left out."""
    f = files(folder)
    with exclusive(f"rastreo de {', '.join(drugs)}", f["lock"].parent):
        state = load_state(f["estado"])
        store, summary = run_once(
            source, state, drugs, brain, follow_days, first_lookback_hours, now, since, until
        )
        known = MentionStore.load_ids(f["menciones"])
        end = to_iso(until) if until else None
        new = MentionStore()
        for m in store.mentions.values():
            if m.id not in known and (end is None or to_iso(m.fecha) < end):
                new.add(m)
        f["menciones"].parent.mkdir(parents=True, exist_ok=True)
        new.save(f["menciones"], append=True)
        f["estado"].write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    by_origin = {"busqueda_fija": 0, "agente": 0}
    for m in new.mentions.values():
        by_origin[m.encontrado_por] = by_origin.get(m.encontrado_por, 0) + 1
    return {"nuevas": len(new), "por_origen": by_origin, "por_farmaco": summary}


def pending_mentions(drugs=None, already: set[str] = frozenset(), folder: Path | None = None):
    """Mentions of the monitor without a triage answer (``already``: ids triaged elsewhere,
    e.g. in the evaluation sample, which are not triaged twice)."""
    from crawler.store import read_mentions
    from triage.agent import read_answers

    f = files(folder)
    if not f["menciones"].exists():
        return []
    done = set(read_answers(f["triaje"])) | set(already)
    return [
        m
        for m in read_mentions(f["menciones"])
        if m.id not in done and (drugs is None or m.farmaco in drugs)
    ]


def triage_pending(
    drugs=None,
    brain: str = "27b",
    already: set[str] = frozenset(),
    folder: Path | None = None,
    llm_factory=None,
    progress=print,
) -> dict:
    """Pass the monitor's mentions that have no answer yet through the triage (v2, one post per
    request) and append the answers to ``triaje_respuestas.jsonl``."""
    from triage.agent import run_triage

    f = files(folder)
    with exclusive(f"triaje de {', '.join(drugs or ['todos'])}", f["lock"].parent):
        pending = pending_mentions(drugs, already, folder)
        if not pending:
            return {"triadas": 0, "efectos_adversos": 0, "graves": 0, "a_revision": 0}
        run = run_triage(
            pending,
            brain=brain,
            version=TRIAGE_VERSION,
            por=TRIAGE_BY,
            out=f["triaje"],
            llm_factory=llm_factory,
            progress=progress,
        )
    answers = [run.answers[m.id] for m in pending if m.id in run.answers]
    return {
        "triadas": len(answers),
        "efectos_adversos": sum(a.get("es_efecto_adverso") == "si" for a in answers),
        "graves": sum(a.get("grave") == "si" for a in answers),
        "a_revision": sum(a.get("estado") == "sin_respuesta" for a in answers),
        "tokens": run.cost.prompt_tokens + run.cost.completion_tokens,
    }


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    from dotenv import load_dotenv

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--farmacos", nargs="+", default=list(WATCHLIST))
    parser.add_argument("--brain", default="27b")
    parser.add_argument("--sin-agente", action="store_true", help="only the fixed search")
    parser.add_argument("--dias-seguimiento", type=int, default=7)
    parser.add_argument("--primera-vez-horas", type=int, default=24, help="look-back of run 1")
    parser.add_argument("--cada", type=float, default=6.0, help="hours between runs (--bucle)")
    parser.add_argument("--bucle", action="store_true", help="keep running every --cada hours")
    parser.add_argument(
        "--desde", default=None, help="first run: everything since this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--hasta", default=None, help="with --desde: only until this date, exclusive (YYYY-MM-DD)"
    )
    parser.add_argument("--triar", action="store_true", help="triage the new mentions too")
    args = parser.parse_args(argv)

    source = LiveSource()
    while True:
        started = datetime.now(UTC)
        result = crawl_and_save(
            source,
            args.farmacos,
            brain=None if args.sin_agente else args.brain,
            follow_days=args.dias_seguimiento,
            first_lookback_hours=args.primera_vez_horas,
            since=args.desde,
            now=started,
            until=args.hasta,
        )
        args.desde = args.hasta = None  # only the first run of a loop: then from the last run on
        print(f"[{to_iso(started)}] {result['nuevas']} menciones nuevas → {files()['menciones']}")
        for drug, s in result["por_farmaco"].items():
            print(f"  {drug}: {s}")
        if args.triar:
            print(f"  triaje: {triage_pending(args.farmacos, brain=args.brain)}")
        if not args.bucle:
            break
        time.sleep(args.cada * 3600)


if __name__ == "__main__":
    main()
