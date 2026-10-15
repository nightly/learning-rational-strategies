"""Threshold certification of frozen-profile satisfaction estimates.

A probabilistic objective ``P_{cmp q}[phi]`` is certified from a confidence
interval ``[L, U]`` for ``Pr(phi)`` when the interval lies on the satisfying
side of ``q``: ``L >= q`` for ``>=``, ``L > q`` for ``>``, ``U <= q`` for
``<=`` and ``U < q`` for ``<``. A profile is jointly certified when every
objective passes with its simultaneous (Bonferroni) interval.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

LOWER_BOUND_COMPARATORS = (">=", ">")
UPPER_BOUND_COMPARATORS = ("<=", "<")
COMPARATORS = LOWER_BOUND_COMPARATORS + UPPER_BOUND_COMPARATORS
DEFAULT_PROBABILITY_THRESHOLD = (">=", 0.5)


def normalize_probability_threshold(threshold: Any) -> dict[str, Any]:
    """Return ``{"comparator", "value"}`` for a pair or an already-normalized mapping."""
    if isinstance(threshold, Mapping):
        comparator = threshold["comparator"]
        value = threshold["value"]
    else:
        comparator, value = threshold
    comparator = str(comparator).strip()
    value = float(value)
    if comparator not in COMPARATORS:
        raise ValueError(f"Unsupported comparator {comparator!r}; expected one of {COMPARATORS}.")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Probability thresholds must lie in [0, 1].")
    return {"comparator": comparator, "value": value}


def resolve_probability_thresholds(
    thresholds: Any,
    agent_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Resolve a uniform pair or a per-agent mapping into one threshold per agent."""
    agent_ids = [str(agent_id) for agent_id in agent_ids]
    if thresholds is None:
        thresholds = DEFAULT_PROBABILITY_THRESHOLD
    is_per_agent = isinstance(thresholds, Mapping) and not {"comparator", "value"} <= set(thresholds)
    if not is_per_agent:
        uniform = normalize_probability_threshold(thresholds)
        return {agent_id: dict(uniform) for agent_id in agent_ids}
    missing = [agent_id for agent_id in agent_ids if agent_id not in thresholds]
    if missing:
        raise KeyError(f"Missing probability thresholds for agents: {missing}")
    return {agent_id: normalize_probability_threshold(thresholds[agent_id]) for agent_id in agent_ids}


def is_lower_bound(comparator: str) -> bool:
    return str(comparator) in LOWER_BOUND_COMPARATORS


def certification_margin(ci: Sequence[float], threshold: Mapping[str, Any]) -> float:
    """Signed distance of the interval from the threshold on the satisfying side."""
    lower, upper = float(ci[0]), float(ci[1])
    value = float(threshold["value"])
    if is_lower_bound(threshold["comparator"]):
        return lower - value
    return value - upper


def interval_certifies(ci: Sequence[float], threshold: Mapping[str, Any]) -> bool:
    margin = certification_margin(ci, threshold)
    if threshold["comparator"] in (">=", "<="):
        return margin >= 0.0
    return margin > 0.0


def certify_satisfaction_run(
    run: Mapping[str, Any],
    thresholds: Any = None,
) -> dict[str, Any]:
    """Certify one frozen profile from its marginal and joint satisfaction intervals.

    ``max_uniform_lower_threshold`` is the smallest joint lower endpoint over
    agents: the largest common ``q`` at which ``P_{>=q}`` would be jointly
    certified. It is reported only when every objective is a lower bound.
    """
    per_agent_estimates = run["per_agent"]
    agent_ids = list(per_agent_estimates)
    if thresholds is None:
        thresholds = run.get("probability_thresholds")
    resolved = resolve_probability_thresholds(thresholds, agent_ids)

    per_agent: dict[str, dict[str, Any]] = {}
    for agent_id in agent_ids:
        estimate = per_agent_estimates[agent_id]
        threshold = resolved[agent_id]
        per_agent[agent_id] = {
            "marginal_certified": interval_certifies(estimate["ci"], threshold),
            "joint_certified": interval_certifies(estimate["joint_ci"], threshold),
            "marginal_margin": certification_margin(estimate["ci"], threshold),
            "joint_margin": certification_margin(estimate["joint_ci"], threshold),
        }

    all_lower = all(is_lower_bound(threshold["comparator"]) for threshold in resolved.values())
    max_uniform_lower_threshold = (
        float(min(float(per_agent_estimates[agent_id]["joint_ci"][0]) for agent_id in agent_ids))
        if all_lower and agent_ids
        else None
    )
    return {
        "thresholds": resolved,
        "per_agent": per_agent,
        "marginally_certified_all": all(entry["marginal_certified"] for entry in per_agent.values()),
        "jointly_certified": all(entry["joint_certified"] for entry in per_agent.values()),
        "joint_margin": float(min(entry["joint_margin"] for entry in per_agent.values())),
        "max_uniform_lower_threshold": max_uniform_lower_threshold,
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def summarize_certifications(certifications: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed certificates; counts, not averaged endpoints, carry the claim."""
    if not certifications:
        return {"num_runs": 0, "num_jointly_certified_runs": 0}
    agent_ids = list(certifications[0]["per_agent"])
    margins = [float(cert["joint_margin"]) for cert in certifications]
    uniform = [
        float(cert["max_uniform_lower_threshold"])
        for cert in certifications
        if cert.get("max_uniform_lower_threshold") is not None
    ]
    margin_stats = _stats(margins)
    uniform_stats = _stats(uniform)
    return {
        "num_runs": len(certifications),
        "num_jointly_certified_runs": int(sum(bool(cert["jointly_certified"]) for cert in certifications)),
        "num_marginally_certified_runs": int(
            sum(bool(cert["marginally_certified_all"]) for cert in certifications)
        ),
        "jointly_certified_run_indices": [
            index for index, cert in enumerate(certifications) if bool(cert["jointly_certified"])
        ],
        "thresholds": certifications[0]["thresholds"],
        "mean_joint_margin": margin_stats["mean"],
        "min_joint_margin": margin_stats["min"],
        "max_joint_margin": margin_stats["max"],
        "mean_max_uniform_lower_threshold": uniform_stats["mean"],
        "min_max_uniform_lower_threshold": uniform_stats["min"],
        "max_max_uniform_lower_threshold": uniform_stats["max"],
        "per_agent": {
            agent_id: {
                "num_joint_certified_runs": int(
                    sum(bool(cert["per_agent"][agent_id]["joint_certified"]) for cert in certifications)
                ),
                "num_marginal_certified_runs": int(
                    sum(bool(cert["per_agent"][agent_id]["marginal_certified"]) for cert in certifications)
                ),
            }
            for agent_id in agent_ids
        },
    }


def joint_outcome_counts(
    success_bits: Mapping[str, np.ndarray],
    agent_ids: Sequence[str],
) -> dict[str, int]:
    """Count rollouts by their satisfaction pattern, one character per agent in ``agent_ids`` order."""
    if not agent_ids:
        return {}
    matrix = np.stack([np.asarray(success_bits[agent_id], dtype=np.int8) for agent_id in agent_ids], axis=1)
    patterns, counts = np.unique(matrix, axis=0, return_counts=True)
    return {"".join(str(int(bit)) for bit in pattern): int(count) for pattern, count in zip(patterns, counts)}


def exclusivity_violations(
    success_bits: Mapping[str, np.ndarray],
    exclusive_groups: Iterable[Sequence[str]],
) -> dict[str, int]:
    """Count rollouts on which two objectives declared mutually exclusive both hold.

    Groups naming an agent absent from ``success_bits`` are skipped, so one
    declaration can cover task variants with different agent counts.
    """
    violations: dict[str, int] = {}
    for group in exclusive_groups:
        members = [str(agent_id) for agent_id in group]
        if len(members) < 2 or any(agent_id not in success_bits for agent_id in members):
            continue
        matrix = np.stack([np.asarray(success_bits[agent_id], dtype=np.int8) for agent_id in members], axis=1)
        violations["+".join(members)] = int(np.sum(matrix.sum(axis=1) >= 2))
    return violations


__all__ = [
    "COMPARATORS",
    "DEFAULT_PROBABILITY_THRESHOLD",
    "certification_margin",
    "certify_satisfaction_run",
    "exclusivity_violations",
    "interval_certifies",
    "is_lower_bound",
    "joint_outcome_counts",
    "normalize_probability_threshold",
    "resolve_probability_thresholds",
    "summarize_certifications",
]
