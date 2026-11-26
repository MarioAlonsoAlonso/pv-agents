"""Proportions with a 95 % Wilson interval (``common.stats``) for the crawler's reports."""

from __future__ import annotations


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """``common.stats.wilson`` rounded to 3 decimals, as the crawler's reports have it."""
    from common.stats import wilson as interval

    ci = interval(k, n, z)
    return None if ci is None else [round(ci[0], 3), round(ci[1], 3)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "valor": round(k / n, 3) if n else None, "ic95": wilson(k, n)}


def fmt(r: dict | None) -> str:
    if not r or r["valor"] is None:
        return "—"
    lo, hi = r["ic95"]
    return f"{r['k']}/{r['n']} = {r['valor']:.0%} [{lo:.0%}-{hi:.0%}]"
