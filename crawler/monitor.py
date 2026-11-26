"""Periodic surveillance: new mentions of each drug since the last run.

    uv run --extra crawler python -m crawler.monitor --sin-agente          # one run, no model
    uv run --extra crawler python -m crawler.monitor --brain 27b           # with the agent (DGX)
    uv run --extra crawler python -m crawler.monitor --cada 6 --bucle      # every 6 hours
    uv run --extra crawler python -m crawler.monitor --farmacos lorazepam --desde 2026-01-01

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

State (``data/bluesky/monitor/estado.json``): last run per drug, watched threads (with their
AT-URIs, needed to download them again) and discovered terms. New mentions are appended to
``data/bluesky/monitor/menciones.jsonl``: that file is what triage reads.
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
) -> tuple[MentionStore, dict]:
    """One surveillance run. Returns the new mentions and a summary; updates ``state``.
    ``since`` (ISO date) overrides where the search starts, for every drug of this run."""
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
        hits, _ = fixed_search(source, store, drug, brands, since, to_iso(now), extra)
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
        state["ultima_ejecucion"][drug] = to_iso(now)
        summary[drug] = {
            "explicitas": explicit,
            "por_agente": len(store) - before - explicit,
            "hilos_vigilados": sum(1 for e in state["hilos"].values() if e["farmaco"] == drug),
            "hilos_con_novedades": len(changed),
            "tokens": tokens,
        }
    return store, summary


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
    args = parser.parse_args(argv)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file, out = STATE_DIR / "estado.json", STATE_DIR / "menciones.jsonl"
    source = LiveSource()
    while True:
        state = load_state(state_file)
        started = datetime.now(UTC)
        store, summary = run_once(
            source,
            state,
            args.farmacos,
            brain=None if args.sin_agente else args.brain,
            follow_days=args.dias_seguimiento,
            first_lookback_hours=args.primera_vez_horas,
            now=started,
            since=args.desde,
        )
        args.desde = None  # only the first run of a loop: then from the last run on
        known = MentionStore.load_ids(out)
        new = MentionStore()
        for m in store.mentions.values():
            if m.id not in known:
                new.add(m)
        new.save(out, append=True)
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
        print(f"[{to_iso(started)}] {len(new)} menciones nuevas → {out}")
        for drug, s in summary.items():
            print(f"  {drug}: {s}")
        if not args.bucle:
            break
        time.sleep(args.cada * 3600)


if __name__ == "__main__":
    main()
