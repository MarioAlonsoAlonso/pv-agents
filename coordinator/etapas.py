"""The stages of the system as the coordinator sees them: what each one has done and how to
launch the crawler and the triage from a conversation.

    rastreador (crawler/monitor.py) ──► menciones ──► triaje (triage/agent.py) ──► casos

The coordinator can answer about any stage (``estado``: what has been crawled, triaged, coded)
and launch the first two (``rastrear``, ``triar``). Both run the same code as the command line
(``crawler.monitor.crawl_and_save`` and ``triage_pending``), on the same files, under the same
lock, so the periodic monitor and a search asked in a conversation never write at once.

What a launch costs, and when it needs the technician's confirmation (``plan``). The 27B is the
course's shared server (5 requests at once for everybody, ~50 tokens/s), so a long job is slow
and slows everybody down. Measured with sertraline (2026-11-19/23): about one mention per day
of Bluesky in Spanish; the hybrid crawler, ~3 100 tokens per mention (2.26 M for 23 months, 11
minutes); the triage, ~2 400 tokens per mention, ~44 mentions per minute. A search of more than
``SMALL_DAYS`` days or a triage of more than ``SMALL_TRIAGE`` mentions is not run by the model's
decision alone: it is queued, with its estimate, for a person to confirm (the harness's
``requires_confirmation``, as the ICSR). Smaller jobs run at once.

Only the watched drugs (``crawler.config.WATCHLIST``): the crawler knows their brands and
misspellings, and the mention format says ``farmaco`` is one of them.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime

from crawler.config import WATCHLIST

STAGE_TOOLS = ("rastrear", "triar")
SMALL_DAYS = 31
SMALL_TRIAGE = 100
EARLIEST = date(2023, 2, 1)  # Bluesky's beta; public since 2024-02-06
# Sertraline: 725 mentions in 687 days in the frozen sample (2025-01 to 2026-11), but 128 in
# 86 days live (2026-09 to 2026-11-26, first crawl from the chat): Bluesky in Spanish grows.
MENTIONS_PER_DAY = 1.5
CRAWL_TOKENS_PER_MENTION = 3100
CRAWL_MINUTES_PER_DAY = 11 / 687
TRIAGE_TOKENS_PER_MENTION = 2400
TRIAGE_MENTIONS_PER_MINUTE = 44

# Hooks for the tests (no network, no 27B): the Bluesky source and the triage's model.
SOURCE_FACTORY = None
TRIAGE_LLM_FACTORY = None


def brain() -> str:
    return os.environ.get("ARCA_CRAWLER_BRAIN", "27b")


def watched(name: str) -> tuple[str | None, str | None]:
    """(active ingredient, error): only the watched drugs can be crawled and triaged."""
    from coordinator.drugs import resolve

    active = resolve(name or "").active or (name or "").strip().lower()
    if active in WATCHLIST:
        return active, None
    return None, (
        f"'{name}' no es uno de los fármacos vigilados en Bluesky ({', '.join(WATCHLIST)}): "
        "el rastreador solo sabe buscar esos (sus marcas y erratas)."
    )


def parse_day(value: str) -> tuple[date | None, str | None]:
    try:
        day = date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None, f"fecha '{value}' no válida: usa AAAA-MM-DD"
    if day < EARLIEST:
        return None, f"Bluesky no tiene posts públicos antes de {EARLIEST.isoformat()}"
    if day >= datetime.now(UTC).date():
        return None, "la fecha de inicio tiene que ser anterior a hoy"
    return day, None


def window(since: str, until: str | None) -> tuple[date | None, date | None, str | None]:
    """[since, until] as dates (until included; today if empty), or an error."""
    from datetime import timedelta

    day, error = parse_day(since)
    if error:
        return None, None, error
    today = datetime.now(UTC).date()
    if not until:
        return day, today, None
    try:
        last = date.fromisoformat(str(until).strip()[:10])
    except ValueError:
        return None, None, f"fecha '{until}' no válida: usa AAAA-MM-DD"
    if last < day:
        return None, None, "la fecha final es anterior a la inicial"
    return day, min(last + timedelta(days=1), today), None  # end: exclusive


def _minutes(m: float) -> str:
    return f"{max(1, round(m))} min" if m < 90 else f"{m / 60:.1f} h"


def estimate_crawl(days: int, with_agent: bool = True) -> dict:
    mentions = round(days * MENTIONS_PER_DAY)
    tokens = mentions * (CRAWL_TOKENS_PER_MENTION if with_agent else 0)
    minutes = days * CRAWL_MINUTES_PER_DAY * (1 if with_agent else 0.3)
    then = mentions * TRIAGE_TOKENS_PER_MENTION
    return {
        "dias": days,
        "menciones_aprox": mentions,
        "tokens_aprox": tokens,
        "tiempo_aprox": _minutes(minutes),
        "texto": (
            f"Rastrear {days} días: unas {mentions} menciones, "
            + (f"~{tokens / 1e6:.1f} M tokens del 27B y " if tokens else "sin el agente (gratis), ")
            + f"~{_minutes(minutes)}; triarlas después costaría ~{then / 1e6:.1f} M tokens más. "
            "Estimación con el ritmo de la sertralina: otro fármaco puede tener más o menos posts."
        ),
    }


def estimate_triage(n: int) -> dict:
    tokens = n * TRIAGE_TOKENS_PER_MENTION
    minutes = n / TRIAGE_MENTIONS_PER_MINUTE
    return {
        "menciones": n,
        "tokens_aprox": tokens,
        "tiempo_aprox": _minutes(minutes),
        "texto": f"Triar {n} menciones: ~{tokens / 1e6:.2f} M tokens del 27B y "
        f"~{_minutes(minutes)}.",
    }


def pending(store, drug: str | None = None) -> list:
    """Mentions of the monitor waiting for the triage (those triaged in the sample excluded)."""
    from crawler.monitor import pending_mentions

    return pending_mentions([drug] if drug else None, already=set(store.answers))


def plan(name: str, args: dict, store) -> dict:
    """Before running ``rastrear`` or ``triar``: an error, or whether it is a big job that needs
    confirmation, with its estimate."""
    from crawler.monitor import running

    drug, error = watched(args.get("farmaco", ""))
    if error:
        return {"error": error, "grande": False, "estimacion": ""}
    busy = running()
    if busy:
        return {
            "error": f"ya hay algo en marcha desde {busy['desde']} ({busy['que']}): espera a "
            "que termine",
            "grande": False,
            "estimacion": "",
        }
    if name == "rastrear":
        day, end, error = window(args.get("desde", ""), args.get("hasta"))
        if error:
            return {"error": error, "grande": False, "estimacion": ""}
        days = (end - day).days
        e = estimate_crawl(days, args.get("con_agente", True))
        return {"error": None, "grande": days > SMALL_DAYS, "estimacion": e["texto"]}
    n = len(pending(store, drug))
    if n == 0:
        return {"error": None, "grande": False, "estimacion": "no hay menciones esperando"}
    e = estimate_triage(n)
    return {"error": None, "grande": n > SMALL_TRIAGE, "estimacion": e["texto"]}


def state(store, drug: str | None = None) -> dict:
    """What each stage has done for each watched drug, and what is missing."""
    from crawler.monitor import files, load_state, running

    monitor = load_state(files()["estado"])
    out = {}
    for d in [drug] if drug else list(WATCHLIST):
        ms = store.mentions_of(d)
        triaged = [m for m in ms if m.id in store.answers]
        cases = store.of_drug(d)
        waiting = len(ms) - len(triaged)
        last = monitor["ultima_ejecucion"].get(d)
        sample = sum(store.origin.get(m.id) == "muestra" for m in ms)
        dates = sorted(m.fecha[:10] for m in ms)
        missing = []
        if not ms and not last:
            missing.append(
                f"nunca se ha rastreado {d} en Bluesky: rastrear(farmaco='{d}', desde=AAAA-MM-DD)"
            )
        elif not last and sample:
            missing.append(
                f"solo hay la muestra congelada de evaluación (hasta {dates[-1]}): para lo "
                f"posterior, rastrear(farmaco='{d}', desde='{dates[-1]}')"
            )
        elif last:
            missing.append(
                f"último rastreo del vigilante: {last[:10]}; lo publicado después aún no está "
                f"(rastrear(farmaco='{d}', desde='{last[:10]}'))"
            )
        # triar() takes the monitor's mentions; the sample's answers come from the triage's own
        # evaluation run (triage.run): without that file, the sample is not triaged here.
        untriaged_sample = sum(
            store.origin.get(m.id) == "muestra" and m.id not in store.answers for m in ms
        )
        if waiting - untriaged_sample:
            missing.append(
                f"{waiting - untriaged_sample} menciones del vigilante esperando al triaje: "
                f"triar(farmaco='{d}')"
            )
        if untriaged_sample:
            missing.append(
                f"faltan las respuestas del triaje para {untriaged_sample} menciones de la "
                "muestra de evaluación (data/triaje/eval/v2_27b_mencion_*_respuestas.jsonl, de "
                "triage.run en la DGX): no se pueden triar desde aquí"
            )
        if cases and not any(c.id in store.meddra for c in cases):
            missing.append("ningún caso codificado en MedDRA todavía (codificar_meddra, por caso)")
        out[d] = {
            "menciones": len(ms),
            "periodo_menciones": [dates[0], dates[-1]] if dates else None,
            "de_la_muestra_de_evaluacion": sample,
            "del_vigilante": len(ms) - sample,
            "ultimo_rastreo_del_vigilante": last[:10] if last else "nunca",
            "por_origen": {
                "nombran_el_farmaco": sum(m.encontrado_por == "busqueda_fija" for m in ms),
                "respuestas_en_conversaciones": sum(m.encontrado_por == "agente" for m in ms),
            },
            "triadas": len(triaged),
            "esperando_triaje": waiting,
            "efectos_adversos": len(cases),
            "descartadas": len(triaged) - len(cases),
            "graves": sum(c.grave for c in cases),
            "codificados_meddra": sum(c.id in store.meddra for c in cases),
            "que_falta": missing,
        }
    busy = running()
    return {
        "farmacos": out,
        "en_marcha": f"{busy['que']} (desde {busy['desde']})" if busy else "nada",
        "nota": "rastrear y triar se pueden lanzar desde aquí; los trabajos grandes (más de "
        f"{SMALL_DAYS} días o más de {SMALL_TRIAGE} menciones) esperan la confirmación del técnico",
    }


def crawl(store, drug: str, since: str, with_agent: bool = True, until: str | None = None) -> dict:
    """Run the crawler (the monitor, from ``since``; until ``until`` exclusive, or now) and
    reload the store."""
    from crawler.monitor import crawl_and_save

    if SOURCE_FACTORY is not None:
        source = SOURCE_FACTORY()
    else:
        from crawler.sources import LiveSource

        source = LiveSource()
    result = crawl_and_save(
        source, [drug], brain=brain() if with_agent else None, since=since, until=until
    )
    store.reload()
    waiting = len(pending(store, drug))
    return {
        "farmaco": drug,
        "desde": since,
        "hasta": until or "ahora",
        "menciones_nuevas": result["nuevas"],
        "por_origen": result["por_origen"],
        "esperando_triaje": waiting,
        "siguiente_paso": f"triar(farmaco='{drug}') para saber cuáles son efectos adversos"
        if waiting
        else "nada: no hay menciones nuevas que triar",
    }


def triage(store, drug: str) -> dict:
    """Triage the mentions of ``drug`` that are waiting, and reload the store."""
    from crawler.monitor import triage_pending

    done = triage_pending(
        [drug],
        brain=brain(),
        already=set(store.answers),
        llm_factory=TRIAGE_LLM_FACTORY,
        progress=lambda _: None,
    )
    store.reload()
    return {
        "farmaco": drug,
        **done,
        "casos_ahora": len(store.of_drug(drug)),
        "nota": "los efectos adversos ya son casos: listar_casos, ver_mencion, codificar_meddra…",
    }
