from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from tqdm.auto import tqdm

from src.verify.confidence import (
    bonferroni_hoeffding_metadata,
    bounded_hoeffding_ci,
    bounded_hoeffding_half_width,
)

MakeBaseEnv = Callable[..., Any]


class NashConvProgress:
    def __init__(
        self,
        *,
        total: int,
        description: str = "NashConv",
        enabled: bool = True,
        refresh_percent: float = 5.0,
    ) -> None:
        self.total = max(0, int(total))
        self._completed = 0
        self._displayed = 0
        self._refresh_units = max(
            1,
            int(math.ceil(self.total * max(0.0, float(refresh_percent)) / 100.0)),
        )
        self._pbar = None
        if enabled and self.total > 0:
            self._pbar = tqdm(
                total=self.total,
                desc=description,
                unit="step",
                miniters=1,
                leave=True,
            )

    def __enter__(self) -> "NashConvProgress":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def advance(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        self._completed = min(self.total, self._completed + int(amount))
        if self._pbar is None:
            return
        if self._completed == self.total or self._completed - self._displayed >= self._refresh_units:
            self._pbar.update(self._completed - self._displayed)
            self._displayed = self._completed

    def close(self) -> None:
        if self._pbar is None:
            return
        if self._completed > self._displayed:
            self._pbar.update(self._completed - self._displayed)
            self._displayed = self._completed
        self._pbar.close()
        self._pbar = None


def bernoulli_ci95(p_hat: float, n: int):
    se = np.sqrt(max(1e-12, p_hat * (1.0 - p_hat)) / max(1, n))
    return (p_hat - 1.96 * se, p_hat + 1.96 * se)


def paired_diff_ci95(diff: np.ndarray):
    n = len(diff)
    if n <= 1:
        return (np.nan, np.nan)
    se = diff.std(ddof=1) / np.sqrt(n)
    m = diff.mean()
    return (m - 1.96 * se, m + 1.96 * se)


def nashconv_joint_confidence_payload(
    delta_p: Mapping[str, float],
    *,
    eval_episodes: int,
    delta: float,
) -> dict[str, Any]:
    joint_confidence = bonferroni_hoeffding_metadata(
        delta=delta,
        num_agents=len(delta_p),
    )
    per_agent_delta = float(joint_confidence["per_agent_delta"])
    half_width = bounded_hoeffding_half_width(
        eval_episodes,
        per_agent_delta,
        lower=-1.0,
        upper=1.0,
    )
    delta_joint_half_width = {
        agent_id: half_width
        for agent_id in delta_p
    }
    delta_joint_ci = {
        agent_id: bounded_hoeffding_ci(
            estimate,
            eval_episodes,
            per_agent_delta,
            lower=-1.0,
            upper=1.0,
        )
        for agent_id, estimate in delta_p.items()
    }
    nashconv_joint_ci = (
        float(sum(max(0.0, bounds[0]) for bounds in delta_joint_ci.values())),
        float(sum(max(0.0, bounds[1]) for bounds in delta_joint_ci.values())),
    )
    return {
        "joint_confidence": joint_confidence,
        "delta_joint_half_width": delta_joint_half_width,
        "delta_joint_ci": delta_joint_ci,
        "nashconv_joint_ci": nashconv_joint_ci,
    }


def train_state_params(train_state: Any) -> Any:
    if hasattr(train_state, "params"):
        return train_state.params
    if isinstance(train_state, Mapping) and "params" in train_state:
        return train_state["params"]
    raise TypeError("Expected a TrainState-like object or checkpoint mapping with a 'params' entry.")


def safe_reset(env, seed):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def call_make_base_env(
    make_base_env_fn: MakeBaseEnv,
    *,
    seed: int,
    num_agents: int,
):
    signature = inspect.signature(make_base_env_fn)
    kwargs: dict[str, Any] = {}

    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    if "num_agents" in signature.parameters:
        kwargs["num_agents"] = num_agents

    if kwargs:
        return make_base_env_fn(**kwargs)
    if len(signature.parameters) == 1:
        return make_base_env_fn(seed)
    return make_base_env_fn()


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def rollout_episode_done(env, terminations: Mapping[str, Any] | None, truncations: Mapping[str, Any] | None) -> bool:
    """Return whether an evaluation rollout has reached the end of its episode.

    Training resets as soon as any agent reports termination or truncation, so
    evaluation must stop at the same step; relying on ``env.agents`` alone lets
    base environments with a longer native horizon keep stepping after the
    objective wrapper's horizon and re-emit terminal monitor rewards.
    """
    if not getattr(env, "agents", None):
        return True
    for flags in (terminations or {}, truncations or {}):
        if any(bool(flag) for flag in flags.values()):
            return True
    return False


def summarize_nashconv_runs(nash_runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    for run in nash_runs:
        normalized = _to_builtin(run)
        confidence_delta = float(
            normalized.get("joint_confidence", {}).get("delta", 0.05)
        )
        normalized.update(
            _to_builtin(
                nashconv_joint_confidence_payload(
                    normalized["delta_p"],
                    eval_episodes=int(normalized["eval_episodes"]),
                    delta=confidence_delta,
                )
            )
        )
        runs.append(normalized)

    if not runs:
        return {
            "metrics": {
                "num_evaluated_runs": 0,
                "mean_nashconv": None,
                "std_nashconv": None,
                "min_nashconv": None,
                "max_nashconv": None,
                "median_nashconv": None,
            },
            "per_agent": {},
            "runs": [],
        }

    agent_ids = sorted(runs[0]["delta_p"].keys())
    nashconv_values = [float(run["nashconv"]) for run in runs]
    nashconv_joint_lower_values = [
        float(run["nashconv_joint_ci"][0]) for run in runs
    ]
    nashconv_joint_upper_values = [
        float(run["nashconv_joint_ci"][1]) for run in runs
    ]
    nashconv_stats = _summary_stats(nashconv_values)

    per_agent = {}
    for aid in agent_ids:
        baseline_values = [float(run["baseline_p"][aid]) for run in runs]
        br_values = [float(run["br_p"][aid]) for run in runs]
        delta_values = [float(run["delta_p"][aid]) for run in runs]
        positive_delta_values = [max(0.0, value) for value in delta_values]
        joint_lower_values = [
            float(run["delta_joint_ci"][aid][0]) for run in runs
        ]
        joint_upper_values = [
            float(run["delta_joint_ci"][aid][1]) for run in runs
        ]

        per_agent[aid] = {
            "mean_baseline_p": float(np.mean(baseline_values)),
            "mean_br_p": float(np.mean(br_values)),
            "mean_delta_p": float(np.mean(delta_values)),
            "mean_positive_delta_p": float(np.mean(positive_delta_values)),
            "max_positive_delta_p": float(np.max(positive_delta_values)),
            "mean_delta_joint_ci_lower": float(np.mean(joint_lower_values)),
            "mean_delta_joint_ci_upper": float(np.mean(joint_upper_values)),
        }

    return {
        "metrics": {
            "num_evaluated_runs": len(runs),
            "mean_nashconv": nashconv_stats["mean"],
            "std_nashconv": nashconv_stats["std"],
            "min_nashconv": nashconv_stats["min"],
            "max_nashconv": nashconv_stats["max"],
            "median_nashconv": nashconv_stats["median"],
            "mean_nashconv_joint_ci_lower": float(
                np.mean(nashconv_joint_lower_values)
            ),
            "mean_nashconv_joint_ci_upper": float(
                np.mean(nashconv_joint_upper_values)
            ),
            "joint_confidence": runs[0]["joint_confidence"],
        },
        "per_agent": per_agent,
        "runs": runs,
    }


def save_nashconv_report(
    nash_runs: list[dict[str, Any]],
    path: str | Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = summarize_nashconv_runs(nash_runs)
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
    "MakeBaseEnv",
    "NashConvProgress",
    "bernoulli_ci95",
    "call_make_base_env",
    "close_env",
    "nashconv_joint_confidence_payload",
    "paired_diff_ci95",
    "rollout_episode_done",
    "safe_reset",
    "save_nashconv_report",
    "summarize_nashconv_runs",
    "train_state_params",
]
