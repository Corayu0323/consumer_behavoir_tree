# -*- coding: utf-8 -*-
"""
Timing / summary helpers shared by baseline & skip detectors.
"""

from __future__ import annotations
import time
import math
from typing import List, Dict


def perf_counter() -> float:
    return time.perf_counter()


def elapsed_ms(t_start: float, t_end: float | None = None) -> float:
    if t_end is None:
        t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0


def percentile(sorted_vals: List[float], q: float) -> float:
    """q in [0,1], expects sorted list."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = int(math.ceil(q * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def summarize(values: List[float]) -> Dict[str, float]:
    """Return mean, p50, p95, max for a list."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    return {
        "mean": float(mean),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": float(vals[-1]),
    }
