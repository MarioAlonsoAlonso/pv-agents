"""Statistics shared by every evaluation of the project (crawler, triage, coordinator)."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes out of n (None when n is 0)."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
