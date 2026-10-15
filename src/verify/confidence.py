from __future__ import annotations

import math
from typing import Any


def bonferroni_per_comparison_delta(delta: float, num_comparisons: int) -> float:
    delta = float(delta)
    num_comparisons = int(num_comparisons)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1).")
    if num_comparisons <= 0:
        raise ValueError("num_comparisons must be positive.")
    return float(delta / num_comparisons)


def bounded_hoeffding_half_width(
    n: int,
    delta: float,
    *,
    lower: float,
    upper: float,
) -> float:
    n = int(n)
    delta = float(delta)
    lower = float(lower)
    upper = float(upper)
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1).")
    if not lower < upper:
        raise ValueError("lower must be less than upper.")
    value_range = upper - lower
    return float(value_range * math.sqrt(math.log(2.0 / delta) / (2.0 * n)))


def bounded_hoeffding_ci(
    mean: float,
    n: int,
    delta: float,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    mean = float(mean)
    lower = float(lower)
    upper = float(upper)
    half_width = bounded_hoeffding_half_width(
        n,
        delta,
        lower=lower,
        upper=upper,
    )
    return (
        float(max(lower, mean - half_width)),
        float(min(upper, mean + half_width)),
    )


def bonferroni_hoeffding_metadata(
    *,
    delta: float,
    num_agents: int,
) -> dict[str, Any]:
    per_agent_delta = bonferroni_per_comparison_delta(delta, num_agents)
    return {
        "method": "bonferroni_hoeffding",
        "delta": float(delta),
        "confidence": float(1.0 - float(delta)),
        "num_agents": int(num_agents),
        "per_agent_delta": per_agent_delta,
    }


__all__ = [
    "bonferroni_hoeffding_metadata",
    "bonferroni_per_comparison_delta",
    "bounded_hoeffding_ci",
    "bounded_hoeffding_half_width",
]
