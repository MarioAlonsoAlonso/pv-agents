"""Build the frozen Bluesky sample the crawler is evaluated on.

    uv run --extra crawler python -m crawler.collect                 # everything
    uv run --extra crawler python -m crawler.collect --max-hilos 500 # a quicker sample

Steps (each one resumable: a DGX session can die halfway):

1. **Seeds.** Search every term since ``--desde``: the eight drugs and their CIMA brands, the
   broad terms that do not name any drug (``config.BROAD_TERMS``) and a small noise sample
   (``ibuprofeno``). The broad terms are what keeps the evaluation from being circular: the
   sample contains posts that the fixed-search baseline cannot find by construction.
2. **Threads.** For every seed that is a reply or has replies, download the whole
   conversation. That is where indirect mentions are ("a mí también me pasó").
3. **Build.** ``sample.jsonl``: one post per line, anonymised, with hashed ids, its thread
   and parent, and the search terms that found it (``search_terms``, empty for posts that
   only came with a thread). ``manifest.json``: parameters and counts.

``raw/`` keeps the AT-URIs (they are needed to resume and to re-read threads); it is in
``.gitignore`` with the rest of ``data/`` and can be deleted once the sample is built.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from crawler.cima import load_brands
from crawler.config import BROAD_TERMS, DATA_DIR, DEFAULT_SINCE, NOISE_TERMS, WATCHLIST
from crawler.sources import LiveSource, Post, SourceError, thread_order


def plan_terms(drugs, brands) -> list[tuple[str, str, int]]:
    """(term, group, max posts): a group is a drug, ``amplio`` or ``ruido``."""
    terms = []
    for drug in drugs:
        terms += [(t, drug, 2000) for t in (drug, *brands.get(drug, []))]
    terms += [(t, "amplio", 1500) for t in BROAD_TERMS]
    terms += [(t, "ruido", 300) for t in NOISE_TERMS]
    return terms


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_seeds(source: LiveSource, terms, since, until, raw: Path) -> None:
    progress_file = raw / "progress.json"
    progress = (
        json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}
    )
    done = set(progress.get("terms_done", []))
    with open(raw / "seeds.jsonl", "a", encoding="utf-8") as out:
        for term, group, cap in terms:
            if term in done:
                continue
            posts = source.search_all(term, since, until, max_posts=cap)
            for p in posts:
                out.write(
                    json.dumps(
                        {
                            "term": term,
                            "group": group,
                            "uri": source.uri(p.id),
                            "root_uri": source.uri(p.thread_id),
                            "post": p.to_json(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out.flush()
            done.add(term)
            progress["terms_done"] = sorted(done)
            progress_file.write_text(json.dumps(progress, indent=1), "utf-8")
            print(f"  {term!r} ({group}): {len(posts)} posts")


def collect_threads(source: LiveSource, raw: Path, max_threads: int) -> None:
    seeds = _read_jsonl(raw / "seeds.jsonl")
    done = {t["thread_id"] for t in _read_jsonl(raw / "threads.jsonl")}
    todo: dict[str, dict] = {}
    for s in seeds:
        post = Post.from_json(s["post"])
        source.remember(post.id, s["uri"])
        if s.get("root_uri"):
            source.remember(post.thread_id, s["root_uri"])
        if (post.reply_count > 0 or post.is_reply) and post.thread_id not in done:
            todo.setdefault(post.thread_id, s)
    budget = max(0, max_threads - len(done))
    print(f"  hilos: {len(done)} ya descargados, {len(todo)} pendientes, presupuesto {budget}")
    with open(raw / "threads.jsonl", "a", encoding="utf-8") as out:
        for n, thread_id in enumerate(list(todo)[:budget], 1):
            try:
                posts = source.thread(thread_id)  # the root's id: the whole conversation
            except SourceError as exc:
                posts, error = [], str(exc)[:120]
            else:
                error = None
            out.write(
                json.dumps(
                    {"thread_id": thread_id, "posts": [p.to_json() for p in posts], "error": error},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if n % 50 == 0:
                out.flush()
                print(f"  … {n} hilos")


def build_sample(raw: Path, out_dir: Path, params: dict) -> dict:
    posts: dict[str, Post] = {}
    terms_of: dict[str, set[str]] = {}
    by_term: Counter = Counter()
    for s in _read_jsonl(raw / "seeds.jsonl"):
        post = Post.from_json(s["post"])
        posts.setdefault(post.id, post)
        terms_of.setdefault(post.id, set()).add(s["term"])
        by_term[s["term"]] += 1
    failed = 0
    for t in _read_jsonl(raw / "threads.jsonl"):
        failed += bool(t.get("error"))
        for data in t["posts"]:
            post = Post.from_json(data)
            posts.setdefault(post.id, post)
    for pid, post in posts.items():
        post.search_terms = sorted(terms_of.get(pid, []))
    threads: dict[str, list[Post]] = {}
    for post in posts.values():
        threads.setdefault(post.thread_id, []).append(post)
    with open(out_dir / "sample.jsonl", "w", encoding="utf-8") as f:
        for thread_id, members in sorted(threads.items()):
            for post in thread_order(members, thread_id):
                f.write(json.dumps(post.to_json(), ensure_ascii=False) + "\n")
    manifest = {
        **params,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "posts": len(posts),
        "posts_found_by_search": sum(1 for p in posts.values() if p.search_terms),
        "posts_only_from_threads": sum(1 for p in posts.values() if not p.search_terms),
        "threads": len(threads),
        "threads_with_replies": sum(1 for m in threads.values() if len(m) > 1),
        "threads_failed": failed,
        "hits_by_term": dict(by_term.most_common()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    from dotenv import load_dotenv

    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--desde", default=DEFAULT_SINCE, help="start date (YYYY-MM-DD)")
    parser.add_argument("--hasta", default=None, help="end date (default: now, fixed at start)")
    parser.add_argument("--farmacos", nargs="*", default=list(WATCHLIST))
    parser.add_argument("--max-hilos", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument("--solo-construir", action="store_true", help="only rebuild sample.jsonl")
    args = parser.parse_args(argv)

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    params_file = raw / "params.json"
    if params_file.exists():  # resuming: same period as the first run
        params = json.loads(params_file.read_text(encoding="utf-8"))
    else:
        params = {
            "since": args.desde,
            "until": args.hasta or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "drugs": args.farmacos,
            "broad_terms": list(BROAD_TERMS),
            "noise_terms": list(NOISE_TERMS),
            "max_threads": args.max_hilos,
        }
        params_file.write_text(json.dumps(params, indent=1), "utf-8")
    if not args.solo_construir:
        source = LiveSource()
        print(f"Bluesky {'con cuenta' if source.authenticated else 'sin cuenta (API pública)'}")
        brands = load_brands()
        print("1/3 búsquedas")
        collect_seeds(
            source, plan_terms(params["drugs"], brands), params["since"], params["until"], raw
        )
        print("2/3 hilos")
        collect_threads(source, raw, params["max_threads"])
        print(f"   peticiones a Bluesky: {source.requests}")
    print("3/3 muestra")
    manifest = build_sample(raw, args.out, params)
    print(json.dumps({k: v for k, v in manifest.items() if k != "hits_by_term"}, indent=1))
    print(f"→ {args.out / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
