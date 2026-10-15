from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.verify.certification import (
    certify_satisfaction_run,
    exclusivity_violations,
    joint_outcome_counts,
    resolve_probability_thresholds,
    summarize_certifications,
)
from src.verify.confidence import (
    bonferroni_hoeffding_metadata,
    bounded_hoeffding_ci,
    bounded_hoeffding_half_width,
)
from src.verify.nash_conv.common import NashConvProgress, _to_builtin
from src.verify.nash_conv.ippo import rollout_successes_parallel_ippo
from src.verify.nash_conv.iql import rollout_successes_parallel_iql
from src.verify.nash_conv.nashq import rollout_successes_parallel_nashq
from src.verify.nash_conv.pr2_ippo import rollout_successes_parallel_pr2_ippo
from src.verify.nash_conv.pr2_iql import rollout_successes_parallel_pr2_iql

RolloutSuccessesFn = Callable[..., tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]


def hoeffding_half_width(n: int, delta: float) -> float:
    return bounded_hoeffding_half_width(
        n,
        delta,
        lower=0.0,
        upper=1.0,
    )


def hoeffding_ci(p_hat: float, n: int, delta: float) -> tuple[float, float]:
    return bounded_hoeffding_ci(
        p_hat,
        n,
        delta,
        lower=0.0,
        upper=1.0,
    )


def _agent_estimate(
    success_bits: np.ndarray,
    *,
    delta: float,
    joint_delta: float,
) -> dict[str, Any]:
    bits = np.asarray(success_bits, dtype=np.int8)
    n = int(bits.size)
    successes = int(bits.sum())
    p_hat = float(successes / n)
    half_width = hoeffding_half_width(n, delta)
    ci = hoeffding_ci(p_hat, n, delta)
    joint_half_width = hoeffding_half_width(n, joint_delta)
    joint_ci = hoeffding_ci(p_hat, n, joint_delta)
    return {
        "successes": successes,
        "p_hat": p_hat,
        "half_width": half_width,
        "ci": ci,
        "joint_half_width": joint_half_width,
        "joint_ci": joint_ci,
        "success_bits": "".join("1" if bit else "0" for bit in bits.tolist()),
    }


def estimate_final_satisfaction(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    rollout_successes_fn: RolloutSuccessesFn,
    rollout_kwargs: Mapping[str, Any] | None = None,
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    description: str = "Final satisfaction",
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    eval_episodes = int(eval_episodes)
    delta = float(delta)
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1).")

    joint_confidence = bonferroni_hoeffding_metadata(
        delta=delta,
        num_agents=len(agent_ids),
    )
    episode_seeds = [int(eval_seed0) + i for i in range(eval_episodes)]
    with NashConvProgress(
        total=eval_episodes,
        description=description,
        enabled=show_progress,
        refresh_percent=progress_refresh_percent,
    ) as progress:
        success_bits, _ = rollout_successes_fn(
            train_states=train_states,
            config=dict(config),
            make_base_env_fn=make_base_env_fn,
            agent_ids=list(agent_ids),
            episode_seeds=episode_seeds,
            success_threshold=float(success_threshold),
            progress=progress,
            **dict(rollout_kwargs or {}),
        )

    per_agent = {
        agent_id: _agent_estimate(
            success_bits[agent_id],
            delta=delta,
            joint_delta=joint_confidence["per_agent_delta"],
        )
        for agent_id in agent_ids
    }
    outcome_counts = joint_outcome_counts(success_bits, list(agent_ids))
    run = {
        "agent_ids": list(agent_ids),
        "eval_episodes": eval_episodes,
        "eval_seed0": int(eval_seed0),
        "delta": delta,
        "confidence": float(1.0 - delta),
        "joint_confidence": joint_confidence,
        "success_threshold": float(success_threshold),
        "probability_thresholds": resolve_probability_thresholds(probability_thresholds, agent_ids),
        "per_agent": per_agent,
        "joint_outcome_counts": outcome_counts,
        "all_satisfied_count": int(outcome_counts.get("1" * len(agent_ids), 0)),
        "exclusivity_violations": exclusivity_violations(success_bits, exclusive_objective_groups),
    }
    run["certification"] = certify_satisfaction_run(run)
    return run


def estimate_final_satisfaction_ippo(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    return estimate_final_satisfaction(
        train_states=train_states,
        config=config,
        make_base_env_fn=make_base_env_fn,
        agent_ids=agent_ids,
        rollout_successes_fn=rollout_successes_parallel_ippo,
        rollout_kwargs={"deterministic": False},
        eval_episodes=eval_episodes,
        eval_seed0=eval_seed0,
        delta=delta,
        success_threshold=success_threshold,
        show_progress=show_progress,
        progress_refresh_percent=progress_refresh_percent,
        probability_thresholds=probability_thresholds,
        exclusive_objective_groups=exclusive_objective_groups,
        description="IPPO final satisfaction",
    )


def estimate_final_satisfaction_pr2_ippo(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    return estimate_final_satisfaction(
        train_states=train_states,
        config=config,
        make_base_env_fn=make_base_env_fn,
        agent_ids=agent_ids,
        rollout_successes_fn=rollout_successes_parallel_pr2_ippo,
        rollout_kwargs={"deterministic": False},
        eval_episodes=eval_episodes,
        eval_seed0=eval_seed0,
        delta=delta,
        success_threshold=success_threshold,
        show_progress=show_progress,
        progress_refresh_percent=progress_refresh_percent,
        probability_thresholds=probability_thresholds,
        exclusive_objective_groups=exclusive_objective_groups,
        description="PR2-IPPO final satisfaction",
    )


def estimate_final_satisfaction_iql(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    return estimate_final_satisfaction(
        train_states=train_states,
        config=config,
        make_base_env_fn=make_base_env_fn,
        agent_ids=agent_ids,
        rollout_successes_fn=rollout_successes_parallel_iql,
        rollout_kwargs={"explore": False},
        eval_episodes=eval_episodes,
        eval_seed0=eval_seed0,
        delta=delta,
        success_threshold=success_threshold,
        show_progress=show_progress,
        progress_refresh_percent=progress_refresh_percent,
        probability_thresholds=probability_thresholds,
        exclusive_objective_groups=exclusive_objective_groups,
        description="IQL final satisfaction",
    )


def estimate_final_satisfaction_pr2_iql(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    return estimate_final_satisfaction(
        train_states=train_states,
        config=config,
        make_base_env_fn=make_base_env_fn,
        agent_ids=agent_ids,
        rollout_successes_fn=rollout_successes_parallel_pr2_iql,
        rollout_kwargs={"explore": False},
        eval_episodes=eval_episodes,
        eval_seed0=eval_seed0,
        delta=delta,
        success_threshold=success_threshold,
        show_progress=show_progress,
        progress_refresh_percent=progress_refresh_percent,
        probability_thresholds=probability_thresholds,
        exclusive_objective_groups=exclusive_objective_groups,
        description="PR2-IQL final satisfaction",
    )


def estimate_final_satisfaction_nashq(
    *,
    train_states: Mapping[str, Any],
    config: Mapping[str, Any],
    make_base_env_fn,
    agent_ids: list[str],
    eval_episodes: int = 1000,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
    probability_thresholds: Any = None,
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    return estimate_final_satisfaction(
        train_states=train_states,
        config=config,
        make_base_env_fn=make_base_env_fn,
        agent_ids=agent_ids,
        rollout_successes_fn=rollout_successes_parallel_nashq,
        eval_episodes=eval_episodes,
        eval_seed0=eval_seed0,
        delta=delta,
        success_threshold=success_threshold,
        show_progress=show_progress,
        progress_refresh_percent=progress_refresh_percent,
        probability_thresholds=probability_thresholds,
        exclusive_objective_groups=exclusive_objective_groups,
        description="Nash-Q final satisfaction",
    )


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _normalize_joint_confidence(
    run: Mapping[str, Any],
    probability_thresholds: Any = None,
) -> dict[str, Any]:
    normalized = _to_builtin(dict(run))
    per_agent = normalized["per_agent"]
    joint_confidence = bonferroni_hoeffding_metadata(
        delta=float(normalized["delta"]),
        num_agents=len(per_agent),
    )
    normalized["joint_confidence"] = joint_confidence
    n = int(normalized["eval_episodes"])
    per_agent_delta = float(joint_confidence["per_agent_delta"])
    for estimate in per_agent.values():
        estimate["joint_half_width"] = hoeffding_half_width(n, per_agent_delta)
        estimate["joint_ci"] = hoeffding_ci(
            float(estimate["p_hat"]),
            n,
            per_agent_delta,
        )
    if probability_thresholds is not None or "probability_thresholds" not in normalized:
        normalized["probability_thresholds"] = resolve_probability_thresholds(
            probability_thresholds,
            list(per_agent),
        )
    normalized["certification"] = certify_satisfaction_run(normalized)
    return normalized


def summarize_satisfaction_runs(
    satisfaction_runs: list[dict[str, Any]],
    *,
    probability_thresholds: Any = None,
) -> dict[str, Any]:
    """Summarize per-seed satisfaction runs.

    Interval endpoints are averaged only descriptively; certification is
    decided per seed profile and aggregated as counts under ``certification``.
    Runs without stored thresholds (legacy reports) use
    ``probability_thresholds`` or the default ``P_{>=0.5}``.
    """
    runs = [_normalize_joint_confidence(run, probability_thresholds) for run in satisfaction_runs]
    if not runs:
        return {
            "metrics": {"num_evaluated_runs": 0},
            "per_agent": {},
            "runs": [],
        }

    agent_ids = sorted(runs[0]["per_agent"].keys())
    per_agent = {}
    for agent_id in agent_ids:
        p_values = [float(run["per_agent"][agent_id]["p_hat"]) for run in runs]
        lower_values = [float(run["per_agent"][agent_id]["ci"][0]) for run in runs]
        upper_values = [float(run["per_agent"][agent_id]["ci"][1]) for run in runs]
        joint_lower_values = [
            float(run["per_agent"][agent_id]["joint_ci"][0]) for run in runs
        ]
        joint_upper_values = [
            float(run["per_agent"][agent_id]["joint_ci"][1]) for run in runs
        ]
        successes = [int(run["per_agent"][agent_id]["successes"]) for run in runs]
        p_stats = _summary_stats(p_values)
        per_agent[agent_id] = {
            "mean_p_hat": p_stats["mean"],
            "std_p_hat": p_stats["std"],
            "min_p_hat": p_stats["min"],
            "max_p_hat": p_stats["max"],
            "mean_ci_lower": float(np.mean(lower_values)),
            "mean_ci_upper": float(np.mean(upper_values)),
            "mean_joint_ci_lower": float(np.mean(joint_lower_values)),
            "mean_joint_ci_upper": float(np.mean(joint_upper_values)),
            "mean_successes": float(np.mean(successes)),
        }

    certification = summarize_certifications([run["certification"] for run in runs])
    for agent_id in agent_ids:
        per_agent[agent_id].update(certification["per_agent"][agent_id])
    all_satisfied = [run["all_satisfied_count"] for run in runs if "all_satisfied_count" in run]
    violation_totals: dict[str, int] = {}
    for run in runs:
        for group, count in run.get("exclusivity_violations", {}).items():
            violation_totals[group] = violation_totals.get(group, 0) + int(count)

    return {
        "metrics": {
            "num_evaluated_runs": len(runs),
            "eval_episodes": int(runs[0]["eval_episodes"]),
            "delta": float(runs[0]["delta"]),
            "confidence": float(runs[0]["confidence"]),
            "joint_confidence": runs[0]["joint_confidence"],
            "num_jointly_certified_runs": certification["num_jointly_certified_runs"],
            "mean_all_satisfied_count": float(np.mean(all_satisfied)) if all_satisfied else None,
            "exclusivity_violations": violation_totals,
        },
        "certification": certification,
        "per_agent": per_agent,
        "runs": runs,
    }


def save_satisfaction_report(
    satisfaction_runs: list[dict[str, Any]],
    path: str | Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
    probability_thresholds: Any = None,
) -> dict[str, Any]:
    report = summarize_satisfaction_runs(
        satisfaction_runs,
        probability_thresholds=probability_thresholds,
    )
    if extra_metadata is not None:
        report["metadata"] = _to_builtin(extra_metadata)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "estimate_final_satisfaction",
    "estimate_final_satisfaction_iql",
    "estimate_final_satisfaction_ippo",
    "estimate_final_satisfaction_nashq",
    "estimate_final_satisfaction_pr2_iql",
    "estimate_final_satisfaction_pr2_ippo",
    "hoeffding_ci",
    "hoeffding_half_width",
    "save_satisfaction_report",
    "summarize_satisfaction_runs",
]
