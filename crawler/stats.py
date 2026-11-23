"""Proportions with a 95 % Wilson interval, shared by the analyst-tool evaluations."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "valor": round(k / n, 3) if n else None, "ic95": wilson(k, n)}


def fmt(r: dict | None) -> str:
    if not r or r["valor"] is None:
        return "—"
    lo, hi = r["ic95"]
    return f"{r['k']}/{r['n']} = {r['valor']:.0%} [{lo:.0%}-{hi:.0%}]"
