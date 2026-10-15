from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
from flax import serialization
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from src.experiments.experiment_timings import (
    load_experiment_timings,
    record_timing,
    save_experiment_timings,
    summarize_timings,
    timings_filename,
)
from src.monitor.reward_monitor import monitor_state_numbering
from src.rl.ippo import make_train as make_ippo_train
from src.rl.iql import make_train as make_iql_train
from src.rl.nash_q import make_train as make_nashq_train
from src.rl.pr2_ippo import make_train as make_pr2_ippo_train
from src.rl.pr2_iql import make_train as make_pr2_iql_train
from src.verify.certification import DEFAULT_PROBABILITY_THRESHOLD
from src.verify.nash_conv import (
    approximate_nashconv_bernoulli_ippo,
    approximate_nashconv_bernoulli_iql,
    approximate_nashconv_bernoulli_nashq,
    approximate_nashconv_bernoulli_pr2_ippo,
    approximate_nashconv_bernoulli_pr2_iql,
    save_nashconv_report,
    summarize_nashconv_runs,
)
from src.verify.satisfaction import (
    estimate_final_satisfaction_ippo,
    estimate_final_satisfaction_iql,
    estimate_final_satisfaction_nashq,
    estimate_final_satisfaction_pr2_ippo,
    estimate_final_satisfaction_pr2_iql,
    save_satisfaction_report,
    summarize_satisfaction_runs,
)

HistoryByAgent = dict[str, list[dict[str, Any]]]
EnvFactory = Callable[..., Any]
ALGORITHM_ORDER = (
    "IPPO",
    "IPPO_CER",
    "PR2_IPPO",
    "PR2_IPPO_CER",
    "IQL",
    "IQL_CER",
    "PR2_IQL",
    "PR2_IQL_CER",
    "NASHQ",
    "NASHQ_CER",
    "hc_IPPO",
    "hc_PR2_IPPO",
    "hc_IQL",
    "hc_PR2_IQL",
    "hc_NASHQ",
)
HANDCRAFTED_ALGORITHMS = (
    "hc_IPPO",
    "hc_PR2_IPPO",
    "hc_IQL",
    "hc_PR2_IQL",
    "hc_NASHQ",
)
ALGORITHM_OUTPUT_DIRS = {
    "IPPO": "ippo",
    "IPPO_CER": "ippo_cer",
    "PR2_IPPO": "pr2_ippo",
    "PR2_IPPO_CER": "pr2_ippo_cer",
    "IQL": "iql",
    "IQL_CER": "iql_cer",
    "PR2_IQL": "pr2_iql",
    "PR2_IQL_CER": "pr2_iql_cer",
    "NASHQ": "nashq",
    "NASHQ_CER": "nashq_cer",
    "hc_IPPO": "hc_ippo",
    "hc_PR2_IPPO": "hc_pr2_ippo",
    "hc_IQL": "hc_iql",
    "hc_PR2_IQL": "hc_pr2_iql",
    "hc_NASHQ": "hc_nashq",
}

TEMPORAL_GAME_SHARED_CONFIG_KEYS = (
    "num_agents",
    "num_env_steps",
    "seed",
    "num_runs",
    "progress_chunks",
    "calculate_conv",
    "calculate_satisfaction",
    "calculate_conv_quick",
    "eval_episodes",
    "br_train_iters",
    "br_batch_steps",
    "satisfaction_eval_episodes",
    "satisfaction_delta",
    "nashconv_delta",
    "nashconv_auto_resume",
    "nashconv_auto_resume_threshold",
    "nashconv_auto_resume_step_fraction",
    "resume_partial_runs",
    "reward_mode",
    "save_dir",
)

TEMPORAL_GAME_SHARED_CONFIG_DEFAULTS = {
    "calculate_satisfaction": False,
    "satisfaction_eval_episodes": 1000,
    "satisfaction_delta": 0.05,
    "nashconv_delta": 0.05,
    "nashconv_auto_resume": False,
    "nashconv_auto_resume_threshold": 0.05,
    "nashconv_auto_resume_step_fraction": 0.25,
    "resume_partial_runs": False,
    "reward_mode": "objective",
}

PARTIAL_RESUME_MANIFEST_FILENAME = "partial_resume_manifest.json"
PARTIAL_RESUME_MANIFEST_VERSION = 1

SATISFACTION_FN_BY_ALGORITHM = {
    "IPPO": estimate_final_satisfaction_ippo,
    "IPPO_CER": estimate_final_satisfaction_ippo,
    "PR2_IPPO": estimate_final_satisfaction_pr2_ippo,
    "PR2_IPPO_CER": estimate_final_satisfaction_pr2_ippo,
    "IQL": estimate_final_satisfaction_iql,
    "IQL_CER": estimate_final_satisfaction_iql,
    "PR2_IQL": estimate_final_satisfaction_pr2_iql,
    "PR2_IQL_CER": estimate_final_satisfaction_pr2_iql,
    "NASHQ": estimate_final_satisfaction_nashq,
    "NASHQ_CER": estimate_final_satisfaction_nashq,
    "hc_IPPO": estimate_final_satisfaction_ippo,
    "hc_PR2_IPPO": estimate_final_satisfaction_pr2_ippo,
    "hc_IQL": estimate_final_satisfaction_iql,
    "hc_PR2_IQL": estimate_final_satisfaction_pr2_iql,
    "hc_NASHQ": estimate_final_satisfaction_nashq,
}

TEMPORAL_GAME_IPPO_BLOCK_KEYS = (
    "num_envs",
    "num_steps",
    "num_minibatches",
    "update_epochs",
    "lr",
    "anneal_lr",
    "max_grad_norm",
    "activation",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "vf_coef",
    "ent_coef",
)

TEMPORAL_GAME_IQL_BLOCK_KEYS = (
    "network_kind",
    "num_envs",
    "num_steps",
    "num_epochs",
    "lr",
    "max_grad_norm",
    "gamma",
    "hidden_size",
    "activation",
    "buffer_size",
    "buffer_batch_size",
    "replay_sequence_length",
    "burn_in_steps",
    "learning_starts",
    "target_update_interval",
    "tau",
    "eps_start",
    "eps_finish",
    "eps_decay",
    "test_during_training",
    "test_num_envs",
    "test_num_steps",
    "test_interval",
)

TEMPORAL_GAME_NASHQ_BLOCK_KEYS = (
    "num_envs",
    "num_steps",
    "gamma",
    "eps_start",
    "eps_finish",
    "eps_decay",
    "min_alpha",
    "equilibrium_selection",
    "counterfactual_max_joint_variants",
)


@dataclass(frozen=True)
class TemporalGameExperimentSpec:
    """Environment and objective metadata consumed by temporal-game runners."""

    title: str
    env_factory: EnvFactory
    success_threshold: float = 0.5
    extra_config_keys: tuple[str, ...] = ()
    # PLTLf requirement per objective: a (comparator, q) pair for every agent or
    # a mapping from agent id to such a pair. The curated tasks set per-agent
    # values from src/experiments/objective_thresholds.py.
    probability_thresholds: Any = DEFAULT_PROBABILITY_THRESHOLD
    # Agent groups whose objectives cannot hold on the same rollout; any
    # co-satisfied rollout in evaluation is reported as a consistency violation.
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = ()


def _dedupe_keys(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for key in group:
            if key not in merged:
                merged.append(key)
    return tuple(merged)


def _is_handcrafted_algorithm(algorithm: str) -> bool:
    return str(algorithm).startswith("hc_")


def _base_algorithm_name(algorithm: str) -> str:
    return (
        str(algorithm)[len("hc_") :]
        if _is_handcrafted_algorithm(algorithm)
        else str(algorithm)
    )


def _shared_config_keys(spec: TemporalGameExperimentSpec) -> tuple[str, ...]:
    return _dedupe_keys(TEMPORAL_GAME_SHARED_CONFIG_KEYS, tuple(spec.extra_config_keys))


def _ippo_config_keys(spec: TemporalGameExperimentSpec) -> tuple[str, ...]:
    return _dedupe_keys(_shared_config_keys(spec), TEMPORAL_GAME_IPPO_BLOCK_KEYS)


def _iql_config_keys(spec: TemporalGameExperimentSpec) -> tuple[str, ...]:
    return _dedupe_keys(_shared_config_keys(spec), TEMPORAL_GAME_IQL_BLOCK_KEYS)


def _nashq_config_keys(spec: TemporalGameExperimentSpec) -> tuple[str, ...]:
    return _dedupe_keys(_shared_config_keys(spec), TEMPORAL_GAME_NASHQ_BLOCK_KEYS)


def _suite_config_keys(spec: TemporalGameExperimentSpec) -> tuple[str, ...]:
    return _dedupe_keys(
        _shared_config_keys(spec),
        (
            "run_ippo",
            "run_ippo_cer",
            "run_pr2_ippo",
            "run_pr2_ippo_cer",
            "run_iql",
            "run_iql_cer",
            "run_pr2_iql",
            "run_pr2_iql_cer",
            "run_nashq",
            "run_nashq_cer",
            "run_hc_ippo",
            "run_hc_pr2_ippo",
            "run_hc_iql",
            "run_hc_pr2_iql",
            "run_hc_nashq",
            "ippo",
            "iql",
            "nashq",
        ),
    )


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


def _algorithm_artifact_metadata(
    *,
    algo_name: str,
    num_agents: int,
    save_dir: Path,
) -> dict[str, Any]:
    return {
        "algorithm": algo_name,
        "num_agents": int(num_agents),
        "save_dir": str(save_dir),
    }


def _algorithm_artifact_paths(save_dir: Path) -> dict[str, Path]:
    return {
        "runs": save_dir / "runs.npy",
        "nash": save_dir / "nash.npy",
        "nash_report": save_dir / "nash_report.json",
        "satisfaction": save_dir / "satisfaction.npy",
        "satisfaction_report": save_dir / "satisfaction_report.json",
        "partial_resume_manifest": save_dir / PARTIAL_RESUME_MANIFEST_FILENAME,
    }


def _save_algorithm_artifacts(
    *,
    algo_name: str,
    num_agents: int,
    save_dir: Path,
    runs: list[HistoryByAgent],
    nash_runs: list[dict[str, Any]],
    satisfaction_runs: list[dict[str, Any]],
) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = _algorithm_artifact_metadata(
        algo_name=algo_name,
        num_agents=num_agents,
        save_dir=save_dir,
    )

    artifact_paths = _algorithm_artifact_paths(save_dir)

    np.save(artifact_paths["runs"], np.array(runs), allow_pickle=True)
    np.save(artifact_paths["nash"], np.array(nash_runs), allow_pickle=True)
    np.save(
        artifact_paths["satisfaction"],
        np.array(satisfaction_runs),
        allow_pickle=True,
    )
    save_nashconv_report(
        nash_runs,
        artifact_paths["nash_report"],
        extra_metadata=metadata,
    )
    save_satisfaction_report(
        satisfaction_runs,
        artifact_paths["satisfaction_report"],
        extra_metadata=metadata,
    )

    return artifact_paths


def _run_label(run_idx: int, run_seed: int) -> str:
    return f"run_{int(run_idx) + 1}_seed_{int(run_seed)}"


def _should_evaluate_nash_run(
    *,
    calculate_conv: bool,
    calculate_conv_quick: bool,
    run_idx: int,
) -> bool:
    return bool(calculate_conv) and (
        (not bool(calculate_conv_quick)) or int(run_idx) == 0
    )


def _partial_resume_fingerprint(
    *,
    algo_name: str,
    config: Mapping[str, Any],
    algo_config: Mapping[str, Any],
    spec: TemporalGameExperimentSpec,
) -> str:
    fingerprint_config = dict(config)
    payload = {
        "algorithm": algo_name,
        "config": _to_builtin(fingerprint_config),
        "algo_config": _to_builtin(dict(algo_config)),
        "spec": {
            "title": spec.title,
            "success_threshold": float(spec.success_threshold),
            "extra_config_keys": list(spec.extra_config_keys),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_run_seeds(*, seed: int, num_runs: int) -> list[int]:
    return [int(seed) + run_idx for run_idx in range(int(num_runs))]


def _partial_resume_manifest_payload(
    *,
    algo_name: str,
    num_agents: int,
    save_dir: Path,
    expected_run_seeds: list[int],
    config_fingerprint: str,
    completed_run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PARTIAL_RESUME_MANIFEST_VERSION,
        "algorithm": algo_name,
        "num_agents": int(num_agents),
        "save_dir": str(save_dir),
        "expected_run_seeds": [int(run_seed) for run_seed in expected_run_seeds],
        "config_fingerprint": config_fingerprint,
        "completed_runs": completed_run_records,
    }


def _save_partial_resume_manifest(
    *,
    save_dir: Path,
    algo_name: str,
    num_agents: int,
    expected_run_seeds: list[int],
    config_fingerprint: str,
    completed_run_records: list[dict[str, Any]],
) -> Path:
    path = save_dir / PARTIAL_RESUME_MANIFEST_FILENAME
    payload = _partial_resume_manifest_payload(
        algo_name=algo_name,
        num_agents=num_agents,
        save_dir=save_dir,
        expected_run_seeds=expected_run_seeds,
        config_fingerprint=config_fingerprint,
        completed_run_records=completed_run_records,
    )
    _write_json_file(path, payload)
    return path


def _load_numpy_list(path: Path) -> list[Any] | None:
    try:
        value = np.load(path, allow_pickle=True)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return None
    loaded = value.tolist()
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def _load_checkpoint_bundle(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return bundle if isinstance(bundle, dict) else None


def _clear_known_algorithm_artifacts(save_dir: Path, checkpoint_dir: Path) -> None:
    artifact_paths = _algorithm_artifact_paths(save_dir)
    for key in (
        "runs",
        "nash",
        "nash_report",
        "satisfaction",
        "satisfaction_report",
        "partial_resume_manifest",
    ):
        artifact_paths[key].unlink(missing_ok=True)
    if checkpoint_dir.is_dir():
        for checkpoint_path in checkpoint_dir.glob("run_*_seed_*.pkl"):
            checkpoint_path.unlink(missing_ok=True)


def _trim_timings_to_completed_prefix(
    timings: Mapping[str, dict[str, Any]],
    *,
    completed_prefix: int,
) -> dict[str, dict[str, Any]]:
    trimmed: dict[str, dict[str, Any]] = {}
    for stage, entry in timings.items():
        runs = entry.get("runs", []) if isinstance(entry, Mapping) else []
        if not isinstance(runs, list):
            continue

        kept_runs = []
        for run in runs:
            if not isinstance(run, Mapping) or "run_idx" not in run:
                continue
            try:
                run_idx = int(run["run_idx"])
            except (TypeError, ValueError):
                continue
            if run_idx < int(completed_prefix):
                kept_runs.append(dict(run))

        if not kept_runs:
            continue

        new_entry = dict(entry)
        new_entry["runs"] = kept_runs
        new_entry["total_seconds"] = float(
            sum(float(run.get("seconds", 0.0)) for run in kept_runs)
        )
        trimmed[stage] = new_entry
    return trimmed


def _completed_record_is_usable(
    *,
    record: Mapping[str, Any],
    run_idx: int,
    run_seed: int,
    checkpoint_dir: Path,
    runs: list[Any],
    nash_runs: list[Any],
    satisfaction_runs: list[Any],
    nash_report: Mapping[str, Any] | None,
    satisfaction_report: Mapping[str, Any] | None,
    expected_nash_count: int,
    expected_satisfaction_count: int,
    calculate_conv: bool,
    calculate_conv_quick: bool,
    calculate_satisfaction: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if int(record.get("run_idx", -1)) != int(run_idx):
        return False, None
    if int(record.get("run_seed", -1)) != int(run_seed):
        return False, None
    if str(record.get("label", "")) != _run_label(run_idx, run_seed):
        return False, None
    if len(runs) <= run_idx:
        return False, None

    should_have_nash = _should_evaluate_nash_run(
        calculate_conv=calculate_conv,
        calculate_conv_quick=calculate_conv_quick,
        run_idx=run_idx,
    )
    if bool(record.get("nash_evaluated", False)) != should_have_nash:
        return False, None
    if should_have_nash:
        if int(record.get("nash_index", -1)) != expected_nash_count:
            return False, None
        if len(nash_runs) <= expected_nash_count or nash_report is None:
            return False, None
    else:
        if record.get("nash_index") is not None:
            return False, None

    if bool(record.get("satisfaction_evaluated", False)) != bool(
        calculate_satisfaction
    ):
        return False, None
    if calculate_satisfaction:
        if int(record.get("satisfaction_index", -1)) != expected_satisfaction_count:
            return False, None
        if (
            len(satisfaction_runs) <= expected_satisfaction_count
            or satisfaction_report is None
        ):
            return False, None
    else:
        if record.get("satisfaction_index") is not None:
            return False, None

    checkpoint_path = checkpoint_dir / f"{_run_label(run_idx, run_seed)}.pkl"
    allowed_checkpoint_paths = {
        str(checkpoint_path),
        str(checkpoint_path.relative_to(checkpoint_dir.parent)),
    }
    if str(record.get("checkpoint_path", "")) not in allowed_checkpoint_paths:
        return False, None
    bundle = _load_checkpoint_bundle(checkpoint_path)
    if bundle is None:
        return False, None
    if int(bundle.get("run_seed", -1)) != int(run_seed):
        return False, None
    if "train_states" not in bundle or "agent_ids" not in bundle:
        return False, None
    return True, bundle


def _load_partial_resume_state(
    *,
    enabled: bool,
    save_dir: Path,
    checkpoint_dir: Path,
    algo_name: str,
    num_agents: int,
    seed: int,
    num_runs: int,
    calculate_conv: bool,
    calculate_conv_quick: bool,
    calculate_satisfaction: bool,
    config_fingerprint: str,
) -> dict[str, Any]:
    empty_state = {
        "start_run_idx": 0,
        "runs": [],
        "nash_runs": [],
        "satisfaction_runs": [],
        "completed_run_records": [],
        "checkpoint_paths": [],
        "auto_resume_events": [],
        "agent_ids": None,
        "timings": {},
        "fresh_restart": False,
    }
    if not enabled:
        return empty_state

    manifest_path = save_dir / PARTIAL_RESUME_MANIFEST_FILENAME
    artifact_paths = _algorithm_artifact_paths(save_dir)
    manifest = _read_json_file(manifest_path)
    expected_seeds = _expected_run_seeds(seed=seed, num_runs=num_runs)
    if manifest is None:
        if (
            any(path.exists() for path in artifact_paths.values())
            or checkpoint_dir.exists()
        ):
            _clear_known_algorithm_artifacts(save_dir, checkpoint_dir)
            empty_state["fresh_restart"] = True
        return empty_state

    metadata_matches = (
        int(manifest.get("schema_version", -1)) == PARTIAL_RESUME_MANIFEST_VERSION
        and str(manifest.get("algorithm")) == algo_name
        and int(manifest.get("num_agents", -1)) == int(num_agents)
        and manifest.get("config_fingerprint") == config_fingerprint
        and manifest.get("expected_run_seeds") == expected_seeds
    )
    if not metadata_matches:
        _clear_known_algorithm_artifacts(save_dir, checkpoint_dir)
        empty_state["fresh_restart"] = True
        return empty_state

    raw_records = manifest.get("completed_runs", [])
    if not isinstance(raw_records, list):
        _clear_known_algorithm_artifacts(save_dir, checkpoint_dir)
        empty_state["fresh_restart"] = True
        return empty_state

    runs = _load_numpy_list(artifact_paths["runs"])
    nash_runs = _load_numpy_list(artifact_paths["nash"])
    satisfaction_runs = _load_numpy_list(artifact_paths["satisfaction"])
    nash_report = _read_json_file(artifact_paths["nash_report"])
    satisfaction_report = _read_json_file(artifact_paths["satisfaction_report"])
    if runs is None or nash_runs is None or satisfaction_runs is None:
        runs = []
        nash_runs = []
        satisfaction_runs = []

    completed_records: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    auto_resume_events: list[dict[str, Any]] = []
    agent_ids: list[str] | None = None
    nash_count = 0
    satisfaction_count = 0

    for run_idx, record in enumerate(raw_records):
        if run_idx >= num_runs or not isinstance(record, Mapping):
            break
        run_seed = seed + run_idx
        usable, bundle = _completed_record_is_usable(
            record=record,
            run_idx=run_idx,
            run_seed=run_seed,
            checkpoint_dir=checkpoint_dir,
            runs=runs,
            nash_runs=nash_runs,
            satisfaction_runs=satisfaction_runs,
            nash_report=nash_report,
            satisfaction_report=satisfaction_report,
            expected_nash_count=nash_count,
            expected_satisfaction_count=satisfaction_count,
            calculate_conv=calculate_conv,
            calculate_conv_quick=calculate_conv_quick,
            calculate_satisfaction=calculate_satisfaction,
        )
        if not usable or bundle is None:
            break

        completed_records.append(dict(record))
        checkpoint_path = checkpoint_dir / f"{_run_label(run_idx, run_seed)}.pkl"
        checkpoint_paths.append(checkpoint_path)
        agent_ids = [str(agent_id) for agent_id in bundle["agent_ids"]]
        for event in record.get("auto_resume_events", []):
            if isinstance(event, Mapping):
                auto_resume_events.append(dict(event))
        if bool(record.get("nash_evaluated", False)):
            nash_count += 1
        if bool(record.get("satisfaction_evaluated", False)):
            satisfaction_count += 1

    completed_prefix = len(completed_records)
    if completed_prefix == 0 and raw_records:
        _clear_known_algorithm_artifacts(save_dir, checkpoint_dir)
    timings = _trim_timings_to_completed_prefix(
        load_experiment_timings(save_dir, num_agents=num_agents),
        completed_prefix=completed_prefix,
    )
    return {
        "start_run_idx": completed_prefix,
        "runs": runs[:completed_prefix],
        "nash_runs": nash_runs[:nash_count],
        "satisfaction_runs": satisfaction_runs[:satisfaction_count],
        "completed_run_records": completed_records,
        "checkpoint_paths": checkpoint_paths,
        "auto_resume_events": auto_resume_events,
        "agent_ids": agent_ids,
        "timings": timings,
        "fresh_restart": completed_prefix == 0 and bool(raw_records),
    }


def _satisfaction_score(experiment: Mapping[str, Any]) -> float | None:
    summary = experiment.get("satisfaction_summary")
    if not isinstance(summary, Mapping):
        return None
    metrics = summary.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or int(metrics.get("num_evaluated_runs", 0)) <= 0
    ):
        return None
    per_agent = summary.get("per_agent")
    if not isinstance(per_agent, Mapping) or not per_agent:
        return None
    values = [
        float(agent_summary["mean_p_hat"])
        for agent_summary in per_agent.values()
        if isinstance(agent_summary, Mapping)
        and agent_summary.get("mean_p_hat") is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def _save_handcrafted_baseline_table(
    experiments: Mapping[str, Mapping[str, Any]],
    save_dir: Path,
) -> Path | None:
    if not any(_is_handcrafted_algorithm(algorithm) for algorithm in experiments):
        return None

    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHM_ORDER:
        experiment = experiments.get(algorithm)
        if experiment is None:
            continue
        score = _satisfaction_score(experiment)
        rows.append(
            {
                "algorithm": algorithm,
                "kind": "Handcrafted"
                if _is_handcrafted_algorithm(algorithm)
                else "PLTLf",
                "score": score,
                "runs": experiment.get("satisfaction_summary", {})
                .get("metrics", {})
                .get("num_evaluated_runs", 0),
            }
        )

    if not rows:
        return None

    best_by_kind: dict[str, str] = {}
    for kind in ("PLTLf", "Handcrafted"):
        candidates = [
            row for row in rows if row["kind"] == kind and row["score"] is not None
        ]
        if candidates:
            best = max(candidates, key=lambda row: float(row["score"]))
            best_by_kind[kind] = str(best["algorithm"])

    table = Table(title="Handcrafted Baseline Comparison")
    table.add_column("Kind")
    table.add_column("Algorithm")
    table.add_column("Mean final P(sat.)", justify="right")
    table.add_column("Satisfaction runs", justify="right")
    table.add_column("Best")

    for row in rows:
        is_best = best_by_kind.get(row["kind"]) == row["algorithm"]
        style = "bold green" if is_best else None
        score = "n/a" if row["score"] is None else f"{float(row['score']):.3f}"
        table.add_row(
            str(row["kind"]),
            str(row["algorithm"]),
            score,
            str(row["runs"]),
            "yes" if is_best else "",
            style=style,
        )

    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=True, color_system="truecolor", width=120
    )
    console.print(table)
    output_path = save_dir / "handcrafted_baseline_comparison.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(buffer.getvalue(), encoding="utf-8")
    return output_path


def _default_num_steps(num_env_steps: int, progress_chunks: int, num_envs: int) -> int:
    target = max(1, num_env_steps // max(1, progress_chunks) // max(1, num_envs))
    return max(1, min(256, target))


def _progress_milestones(total: int, chunks: int) -> tuple[int, ...]:
    total = max(0, int(total))
    chunks = max(1, int(chunks))
    if total == 0:
        return ()
    return tuple(
        sorted(
            {
                min(total, math.ceil(total * idx / chunks))
                for idx in range(1, chunks + 1)
            }
        )
    )


class _ChunkedProgressBar:
    def __init__(self, pbar: Any, *, total: int, chunks: int):
        self._pbar = pbar
        self._total = max(0, int(total))
        self._milestones = _progress_milestones(self._total, chunks)
        self._next_milestone_idx = 0
        self._observed = 0

    def advance(self, amount: int, *, postfix: Mapping[str, str] | None = None) -> None:
        if self._total <= 0:
            return

        self._observed = min(self._total, self._observed + max(0, int(amount)))
        while (
            self._next_milestone_idx < len(self._milestones)
            and self._observed >= self._milestones[self._next_milestone_idx]
        ):
            self._next_milestone_idx += 1

        if self._next_milestone_idx == 0:
            return

        display_n = self._milestones[self._next_milestone_idx - 1]
        delta = display_n - int(self._pbar.n)
        if delta <= 0:
            return

        if postfix is not None:
            self._pbar.set_postfix(**postfix, refresh=False)
        self._pbar.update(delta)


def _resolve_num_minibatches(batch_size: int, requested: int) -> int:
    requested = max(1, int(requested))
    upper = min(batch_size, requested)
    for candidate in range(upper, 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


def _call_env_factory(
    env_factory: EnvFactory,
    *,
    seed: int | None,
    num_agents: int,
    config: Mapping[str, Any],
):
    signature = inspect.signature(env_factory)
    kwargs: dict[str, Any] = {}
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    if "num_agents" in signature.parameters:
        kwargs["num_agents"] = num_agents
    if "config" in signature.parameters:
        kwargs["config"] = config

    if kwargs:
        return env_factory(**kwargs)
    if len(signature.parameters) == 1:
        return env_factory(seed)
    return env_factory()


def _make_base_env(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
    *,
    seed: int | None,
    num_agents: int,
):
    env = _call_env_factory(
        spec.env_factory, seed=seed, num_agents=num_agents, config=config
    )
    reward_mode = config.get("reward_mode", None)
    set_reward_mode = getattr(env, "set_reward_mode", None)
    if reward_mode is not None and callable(set_reward_mode):
        set_reward_mode(str(reward_mode))
    return env


def _apply_shared_config_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    for key, value in TEMPORAL_GAME_SHARED_CONFIG_DEFAULTS.items():
        resolved.setdefault(key, value)
    return resolved


def _merge_env_histories(histories: list[HistoryByAgent]) -> HistoryByAgent:
    merged: HistoryByAgent = {}
    agent_ids = sorted({agent_id for history in histories for agent_id in history})

    for agent_id in agent_ids:
        merged_entries: list[dict[str, Any]] = []
        for history in histories:
            for entry in history.get(agent_id, []):
                merged_entries.append(dict(entry))

        merged_entries.sort(
            key=lambda entry: (
                int(entry.get("t_end", 0)),
                int(entry.get("episode", 0)),
            )
        )
        for episode_index, entry in enumerate(merged_entries, start=1):
            entry["episode"] = episode_index
        merged[agent_id] = merged_entries

    return merged


def _offset_history_t_end(history: HistoryByAgent, offset: int) -> HistoryByAgent:
    if offset <= 0:
        return {
            agent_id: [dict(entry) for entry in entries]
            for agent_id, entries in history.items()
        }

    shifted: HistoryByAgent = {}
    for agent_id, entries in history.items():
        shifted_entries: list[dict[str, Any]] = []
        for entry in entries:
            shifted_entry = dict(entry)
            shifted_entry["t_end"] = int(shifted_entry.get("t_end", 0)) + int(offset)
            shifted_entries.append(shifted_entry)
        shifted[agent_id] = shifted_entries
    return shifted


def _auto_resume_budget(
    algo_config: Mapping[str, Any], step_fraction: float
) -> tuple[int, int]:
    batch_size = int(algo_config["NUM_ENVS"]) * int(algo_config["NUM_STEPS"])
    original_total_steps = int(algo_config["TOTAL_TIMESTEPS"])
    max_extra_steps = math.floor(original_total_steps * float(step_fraction))
    extra_updates = max(0, max_extra_steps // max(1, batch_size))
    return extra_updates, extra_updates * batch_size


def _make_resume_algo_config(
    algo_config: Mapping[str, Any],
    *,
    extra_updates: int,
    extra_steps: int,
) -> dict[str, Any]:
    batch_size = int(algo_config["NUM_ENVS"]) * int(algo_config["NUM_STEPS"])
    original_updates = int(algo_config["TOTAL_TIMESTEPS"]) // max(1, batch_size)
    resume_config = dict(algo_config)
    resume_config["TOTAL_TIMESTEPS"] = int(extra_steps)
    resume_config["SCHEDULE_UPDATE_OFFSET"] = int(original_updates)
    resume_config["SCHEDULE_TOTAL_UPDATES"] = int(original_updates + extra_updates)
    return resume_config


def _make_final_checkpoint_config(
    algo_config: Mapping[str, Any],
    *,
    extra_steps: int,
) -> dict[str, Any]:
    final_config = dict(algo_config)
    final_config["TOTAL_TIMESTEPS"] = int(algo_config["TOTAL_TIMESTEPS"]) + int(
        extra_steps
    )
    final_config["AUTO_RESUME_EXTRA_TIMESTEPS"] = int(extra_steps)
    return final_config


def _print_nashconv_report(
    *,
    algo_name: str,
    nash: Mapping[str, Any],
    agent_ids: tuple[str, ...] | list[str],
) -> None:
    print("*" * 111)
    joint_confidence = nash["joint_confidence"]
    print(
        f"{algo_name} Approx NashConv: {nash['nashconv']} "
        f"jointCI={nash['nashconv_joint_ci']} "
        f"(confidence={joint_confidence['confidence']:.3f})"
    )
    for agent_id in agent_ids:
        print(
            f"{agent_id}: p={nash['baseline_p'][agent_id]:.3f} -> "
            f"BR p={nash['br_p'][agent_id]:.3f} "
            f"delta={nash['delta_p'][agent_id]:+.3f} "
            f"CI95={nash['delta_ci95'][agent_id]} "
            f"jointCI={nash['delta_joint_ci'][agent_id]}"
        )
    print("*" * 111)


def _print_satisfaction_report(
    *,
    algo_name: str,
    satisfaction: Mapping[str, Any],
    agent_ids: tuple[str, ...] | list[str],
) -> None:
    print("*" * 111)
    joint_confidence = satisfaction["joint_confidence"]
    print(
        f"{algo_name} final satisfaction "
        f"(n={satisfaction['eval_episodes']}, "
        f"marginal confidence={satisfaction['confidence']:.3f}, "
        f"joint confidence={joint_confidence['confidence']:.3f}):"
    )
    for agent_id in agent_ids:
        agent = satisfaction["per_agent"][agent_id]
        print(
            f"{agent_id}: p={agent['p_hat']:.3f} "
            f"CI=[{agent['ci'][0]:.3f}, {agent['ci'][1]:.3f}] "
            f"jointCI=[{agent['joint_ci'][0]:.3f}, {agent['joint_ci'][1]:.3f}] "
            f"successes={agent['successes']}"
        )
    certification = satisfaction.get("certification")
    if isinstance(certification, Mapping):
        print(
            f"jointly certified={certification['jointly_certified']} "
            f"joint margin={certification['joint_margin']:+.3f} "
            f"all satisfied on {satisfaction.get('all_satisfied_count', 'n/a')} rollouts"
        )
    violations = {
        group: count
        for group, count in dict(satisfaction.get("exclusivity_violations", {})).items()
        if count
    }
    if violations:
        print(
            f"WARNING: {algo_name} co-satisfied mutually exclusive objectives on {violations} rollout(s); "
            "the evaluation rollouts or objective labels are inconsistent."
        )
    print("*" * 111)


def _save_checkpoint_bundle(
    path: Path,
    *,
    train_states: dict[str, Any],
    agent_ids: tuple[str, ...],
    run_seed: int,
    algo_config: dict[str, Any],
) -> None:
    def _copy_checkpoint_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _copy_checkpoint_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_copy_checkpoint_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_copy_checkpoint_value(item) for item in value)
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, np.generic):
            return value.item()
        return value

    serialized_train_states: dict[str, Any] = {}
    for agent_id, train_state in train_states.items():
        if isinstance(train_state, Mapping):
            serialized_train_states[agent_id] = _copy_checkpoint_value(
                dict(train_state)
            )
        else:
            serialized_train_states[agent_id] = serialization.to_state_dict(train_state)

    bundle = {
        "agent_ids": list(agent_ids),
        "run_seed": int(run_seed),
        "config": _to_builtin(algo_config),
        "train_states": serialized_train_states,
        # Checkpoints without this key predate canonical numbering and were
        # trained under a hash-seed-dependent flloat numbering.
        "monitor_state_numbering": monitor_state_numbering(),
    }
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, pickle.HIGHEST_PROTOCOL)


def _validate_allowed_keys(
    config: Mapping[str, Any],
    *,
    allowed_keys: tuple[str, ...],
    label: str,
) -> None:
    allowed = set(allowed_keys)
    missing_keys = [key for key in allowed_keys if key not in config]
    unknown_keys = sorted(set(config) - allowed)

    if not missing_keys and not unknown_keys:
        return

    problems: list[str] = []
    if missing_keys:
        problems.append(f"missing keys: {', '.join(missing_keys)}")
    if unknown_keys:
        problems.append(f"unknown keys: {', '.join(unknown_keys)}")

    allowed_text = ", ".join(allowed_keys)
    raise KeyError(
        f"Invalid {label} config; "
        + "; ".join(problems)
        + f". Allowed keys: {allowed_text}"
    )


def _coerce_shared_config(
    shared: Mapping[str, Any],
    *,
    extra_shared_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    config = _apply_shared_config_defaults(shared)
    config["num_agents"] = int(config["num_agents"])
    config["num_env_steps"] = int(config["num_env_steps"])
    config["seed"] = int(config["seed"])
    config["num_runs"] = int(config["num_runs"])
    config["progress_chunks"] = int(config["progress_chunks"])
    config["calculate_conv"] = bool(config["calculate_conv"])
    config["calculate_satisfaction"] = bool(config["calculate_satisfaction"])
    config["calculate_conv_quick"] = bool(config["calculate_conv_quick"])
    config["eval_episodes"] = int(config["eval_episodes"])
    config["br_train_iters"] = int(config["br_train_iters"])
    config["br_batch_steps"] = int(config["br_batch_steps"])
    config["satisfaction_eval_episodes"] = int(config["satisfaction_eval_episodes"])
    config["satisfaction_delta"] = float(config["satisfaction_delta"])
    config["nashconv_delta"] = float(config["nashconv_delta"])
    config["nashconv_auto_resume"] = bool(config["nashconv_auto_resume"])
    config["nashconv_auto_resume_threshold"] = float(
        config["nashconv_auto_resume_threshold"]
    )
    config["nashconv_auto_resume_step_fraction"] = float(
        config["nashconv_auto_resume_step_fraction"]
    )
    config["resume_partial_runs"] = bool(config["resume_partial_runs"])
    config["reward_mode"] = str(config["reward_mode"]).lower()
    if config["satisfaction_eval_episodes"] <= 0:
        raise ValueError("satisfaction_eval_episodes must be positive.")
    if not 0.0 < config["satisfaction_delta"] < 1.0:
        raise ValueError("satisfaction_delta must be in the open interval (0, 1).")
    if not 0.0 < config["nashconv_delta"] < 1.0:
        raise ValueError("nashconv_delta must be in the open interval (0, 1).")
    if config["nashconv_auto_resume_threshold"] < 0.0:
        raise ValueError("nashconv_auto_resume_threshold must be non-negative.")
    if config["nashconv_auto_resume_step_fraction"] < 0.0:
        raise ValueError("nashconv_auto_resume_step_fraction must be non-negative.")
    if config["reward_mode"] not in {"objective", "handcrafted"}:
        raise ValueError("reward_mode must be one of {'objective', 'handcrafted'}.")
    config["save_dir"] = Path(config["save_dir"])
    for key in extra_shared_keys:
        config[key] = shared[key]
    return config


def _resolve_ippo_config(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _apply_shared_config_defaults(config)
    _validate_allowed_keys(
        config,
        allowed_keys=_ippo_config_keys(spec),
        label="temporal-game IPPO experiment",
    )
    config = _coerce_shared_config(
        config, extra_shared_keys=tuple(spec.extra_config_keys)
    )

    config["num_envs"] = int(config["num_envs"])
    config["update_epochs"] = int(config["update_epochs"])
    config["lr"] = float(config["lr"])
    config["anneal_lr"] = bool(config["anneal_lr"])
    config["max_grad_norm"] = float(config["max_grad_norm"])
    config["activation"] = str(config["activation"])
    config["gamma"] = float(config["gamma"])
    config["gae_lambda"] = float(config["gae_lambda"])
    config["clip_eps"] = float(config["clip_eps"])
    config["vf_coef"] = float(config["vf_coef"])
    config["ent_coef"] = float(config["ent_coef"])

    if config["num_steps"] is None:
        config["num_steps"] = _default_num_steps(
            config["num_env_steps"],
            config["progress_chunks"],
            config["num_envs"],
        )
    else:
        config["num_steps"] = int(config["num_steps"])

    batch_size = config["num_envs"] * config["num_steps"]
    num_minibatches = _resolve_num_minibatches(
        batch_size, int(config["num_minibatches"])
    )
    num_updates = max(1, math.ceil(config["num_env_steps"] / batch_size))
    total_timesteps = num_updates * batch_size

    ippo_config = {
        "NUM_ENVS": config["num_envs"],
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_STEPS": config["num_steps"],
        "NUM_MINIBATCHES": num_minibatches,
        "UPDATE_EPOCHS": config["update_epochs"],
        "LR": config["lr"],
        "ANNEAL_LR": config["anneal_lr"],
        "MAX_GRAD_NORM": config["max_grad_norm"],
        "ACTIVATION": config["activation"],
        "GAMMA": config["gamma"],
        "GAE_LAMBDA": config["gae_lambda"],
        "CLIP_EPS": config["clip_eps"],
        "VF_COEF": config["vf_coef"],
        "ENT_COEF": config["ent_coef"],
        "SEED": config["seed"],
    }

    config["num_minibatches"] = num_minibatches
    config["total_timesteps"] = total_timesteps
    config["batch_size"] = batch_size
    config["num_updates"] = num_updates
    return config, ippo_config


def _resolve_iql_config(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _apply_shared_config_defaults(config)
    config.setdefault("replay_sequence_length", None)
    config.setdefault("burn_in_steps", None)
    _validate_allowed_keys(
        config,
        allowed_keys=_iql_config_keys(spec),
        label="temporal-game IQL experiment",
    )
    config = _coerce_shared_config(
        config, extra_shared_keys=tuple(spec.extra_config_keys)
    )

    config["network_kind"] = str(config["network_kind"]).lower()
    config["num_envs"] = int(config["num_envs"])
    config["num_epochs"] = int(config["num_epochs"])
    config["lr"] = float(config["lr"])
    config["max_grad_norm"] = float(config["max_grad_norm"])
    config["gamma"] = float(config["gamma"])
    config["hidden_size"] = int(config["hidden_size"])
    config["activation"] = str(config["activation"])
    config["buffer_size"] = int(config["buffer_size"])
    config["buffer_batch_size"] = int(config["buffer_batch_size"])
    config["learning_starts"] = int(config["learning_starts"])
    config["target_update_interval"] = int(config["target_update_interval"])
    config["tau"] = float(config["tau"])
    config["eps_start"] = float(config["eps_start"])
    config["eps_finish"] = float(config["eps_finish"])
    config["eps_decay"] = float(config["eps_decay"])
    config["test_during_training"] = bool(config["test_during_training"])
    config["test_num_envs"] = int(config["test_num_envs"])
    config["test_num_steps"] = int(config["test_num_steps"])
    config["test_interval"] = float(config["test_interval"])

    if config["num_steps"] is None:
        config["num_steps"] = _default_num_steps(
            config["num_env_steps"],
            config["progress_chunks"],
            config["num_envs"],
        )
    else:
        config["num_steps"] = int(config["num_steps"])

    if config["replay_sequence_length"] is None:
        config["replay_sequence_length"] = config["num_steps"]
    else:
        config["replay_sequence_length"] = int(config["replay_sequence_length"])
    if config["burn_in_steps"] is None:
        config["burn_in_steps"] = (
            min(16, config["replay_sequence_length"])
            if config["network_kind"] == "rnn"
            else 0
        )
    else:
        config["burn_in_steps"] = int(config["burn_in_steps"])

    batch_size = config["num_envs"] * config["num_steps"]
    num_updates = max(1, math.ceil(config["num_env_steps"] / batch_size))
    total_timesteps = num_updates * batch_size

    iql_config = {
        "NETWORK_KIND": config["network_kind"],
        "NUM_ENVS": config["num_envs"],
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_STEPS": config["num_steps"],
        "NUM_EPOCHS": config["num_epochs"],
        "LR": config["lr"],
        "MAX_GRAD_NORM": config["max_grad_norm"],
        "GAMMA": config["gamma"],
        "HIDDEN_SIZE": config["hidden_size"],
        "ACTIVATION": config["activation"],
        "BUFFER_SIZE": config["buffer_size"],
        "BUFFER_BATCH_SIZE": config["buffer_batch_size"],
        "REPLAY_SEQUENCE_LENGTH": config["replay_sequence_length"],
        "BURN_IN_STEPS": config["burn_in_steps"],
        "LEARNING_STARTS": config["learning_starts"],
        "TARGET_UPDATE_INTERVAL": config["target_update_interval"],
        "TAU": config["tau"],
        "EPS_START": config["eps_start"],
        "EPS_FINISH": config["eps_finish"],
        "EPS_DECAY": config["eps_decay"],
        "TEST_DURING_TRAINING": config["test_during_training"],
        "TEST_NUM_ENVS": config["test_num_envs"],
        "TEST_NUM_STEPS": config["test_num_steps"],
        "TEST_INTERVAL": config["test_interval"],
        "SEED": config["seed"],
    }

    config["total_timesteps"] = total_timesteps
    config["batch_size"] = batch_size
    config["num_updates"] = num_updates
    return config, iql_config


def _resolve_nashq_config(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _apply_shared_config_defaults(config)
    config.setdefault("equilibrium_selection", "payoff_dominant")
    config.setdefault("counterfactual_max_joint_variants", 1024)
    _validate_allowed_keys(
        config,
        allowed_keys=_nashq_config_keys(spec),
        label="temporal-game Nash-Q experiment",
    )
    config = _coerce_shared_config(
        config, extra_shared_keys=tuple(spec.extra_config_keys)
    )

    config["num_envs"] = int(config["num_envs"])
    config["gamma"] = float(config["gamma"])
    config["eps_start"] = float(config["eps_start"])
    config["eps_finish"] = float(config["eps_finish"])
    config["eps_decay"] = float(config["eps_decay"])
    config["min_alpha"] = float(config["min_alpha"])
    config["equilibrium_selection"] = str(config["equilibrium_selection"]).lower()
    if config["equilibrium_selection"] not in {"first", "payoff_dominant"}:
        raise ValueError(
            "equilibrium_selection must be one of {'first', 'payoff_dominant'}."
        )
    raw_max_joint_variants = config.get("counterfactual_max_joint_variants", 1024)
    config["counterfactual_max_joint_variants"] = (
        None if raw_max_joint_variants is None else int(raw_max_joint_variants)
    )

    if config["num_steps"] is None:
        config["num_steps"] = _default_num_steps(
            config["num_env_steps"],
            config["progress_chunks"],
            config["num_envs"],
        )
    else:
        config["num_steps"] = int(config["num_steps"])

    batch_size = config["num_envs"] * config["num_steps"]
    num_updates = max(1, math.ceil(config["num_env_steps"] / batch_size))
    total_timesteps = num_updates * batch_size

    nashq_config = {
        "NUM_ENVS": config["num_envs"],
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_STEPS": config["num_steps"],
        "GAMMA": config["gamma"],
        "EPS_START": config["eps_start"],
        "EPS_FINISH": config["eps_finish"],
        "EPS_DECAY": config["eps_decay"],
        "MIN_ALPHA": config["min_alpha"],
        "EQUILIBRIUM_SELECTION": config["equilibrium_selection"],
        "COUNTERFACTUAL_MAX_JOINT_VARIANTS": config[
            "counterfactual_max_joint_variants"
        ],
        "SEED": config["seed"],
    }

    config["total_timesteps"] = total_timesteps
    config["batch_size"] = batch_size
    config["num_updates"] = num_updates
    return config, nashq_config


def _resolve_suite_config(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite = _apply_shared_config_defaults(config)
    suite.setdefault("run_ippo_cer", False)
    suite.setdefault("run_pr2_ippo_cer", False)
    suite.setdefault("run_iql_cer", False)
    suite.setdefault("run_pr2_iql_cer", False)
    suite.setdefault("run_nashq_cer", False)
    suite.setdefault("run_hc_ippo", False)
    suite.setdefault("run_hc_pr2_ippo", False)
    suite.setdefault("run_hc_iql", False)
    suite.setdefault("run_hc_pr2_iql", False)
    suite.setdefault("run_hc_nashq", False)
    _validate_allowed_keys(
        suite,
        allowed_keys=_suite_config_keys(spec),
        label="temporal-game experiment suite",
    )

    shared_keys = _shared_config_keys(spec)
    shared = _coerce_shared_config(
        {key: suite[key] for key in shared_keys},
        extra_shared_keys=tuple(spec.extra_config_keys),
    )
    suite["run_ippo"] = bool(suite["run_ippo"])
    suite["run_ippo_cer"] = bool(suite["run_ippo_cer"])
    suite["run_pr2_ippo"] = bool(suite["run_pr2_ippo"])
    suite["run_pr2_ippo_cer"] = bool(suite["run_pr2_ippo_cer"])
    suite["run_iql"] = bool(suite["run_iql"])
    suite["run_iql_cer"] = bool(suite["run_iql_cer"])
    suite["run_pr2_iql"] = bool(suite["run_pr2_iql"])
    suite["run_pr2_iql_cer"] = bool(suite["run_pr2_iql_cer"])
    suite["run_nashq"] = bool(suite["run_nashq"])
    suite["run_nashq_cer"] = bool(suite["run_nashq_cer"])
    suite["run_hc_ippo"] = bool(suite["run_hc_ippo"])
    suite["run_hc_pr2_ippo"] = bool(suite["run_hc_pr2_ippo"])
    suite["run_hc_iql"] = bool(suite["run_hc_iql"])
    suite["run_hc_pr2_iql"] = bool(suite["run_hc_pr2_iql"])
    suite["run_hc_nashq"] = bool(suite["run_hc_nashq"])

    ippo_block = dict(suite["ippo"])
    iql_block = dict(suite["iql"])
    iql_block.setdefault("replay_sequence_length", None)
    iql_block.setdefault("burn_in_steps", None)
    nashq_block = dict(suite["nashq"])
    nashq_block.setdefault("equilibrium_selection", "payoff_dominant")
    nashq_block.setdefault("counterfactual_max_joint_variants", 1024)
    _validate_allowed_keys(
        ippo_block,
        allowed_keys=TEMPORAL_GAME_IPPO_BLOCK_KEYS,
        label="temporal-game IPPO notebook block",
    )
    _validate_allowed_keys(
        iql_block,
        allowed_keys=TEMPORAL_GAME_IQL_BLOCK_KEYS,
        label="temporal-game IQL notebook block",
    )
    _validate_allowed_keys(
        nashq_block,
        allowed_keys=TEMPORAL_GAME_NASHQ_BLOCK_KEYS,
        label="temporal-game Nash-Q notebook block",
    )

    ippo_flat = {**shared, **ippo_block}
    iql_flat = {**shared, **iql_block}
    nashq_flat = {**shared, **nashq_block}
    resolved_ippo, ippo_config = _resolve_ippo_config(spec, ippo_flat)
    resolved_iql, iql_config = _resolve_iql_config(spec, iql_flat)
    resolved_nashq, nashq_config = _resolve_nashq_config(spec, nashq_flat)

    for key, value in shared.items():
        suite[key] = value
    suite["ippo"] = resolved_ippo
    suite["iql"] = resolved_iql
    suite["nashq"] = resolved_nashq
    return suite, ippo_config, iql_config, nashq_config


def _run_temporal_game_algorithm_experiment(
    *,
    algo_name: str,
    config: dict[str, Any],
    algo_config: dict[str, Any],
    config_key_name: str,
    spec: TemporalGameExperimentSpec,
    make_train_fn: Callable[[dict[str, Any], Callable[[int], Any]], Callable[..., Any]],
    nashconv_fn: Callable[..., dict[str, Any]],
    train_timing_name: str,
    satisfaction_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    num_agents = int(config["num_agents"])
    num_env_steps = int(config["num_env_steps"])
    num_runs = int(config["num_runs"])
    seed = int(config["seed"])
    progress_chunks = int(config["progress_chunks"])
    calculate_conv = bool(config["calculate_conv"])
    calculate_satisfaction = bool(
        config["calculate_satisfaction"]
    ) or _is_handcrafted_algorithm(algo_name)
    config["calculate_satisfaction"] = calculate_satisfaction
    calculate_conv_quick = bool(config["calculate_conv_quick"])
    eval_episodes = int(config["eval_episodes"])
    br_train_iters = int(config["br_train_iters"])
    br_batch_steps = int(config["br_batch_steps"])
    satisfaction_eval_episodes = int(config["satisfaction_eval_episodes"])
    satisfaction_delta = float(config["satisfaction_delta"])
    nashconv_delta = float(config["nashconv_delta"])
    nashconv_auto_resume = bool(config["nashconv_auto_resume"])
    nashconv_auto_resume_threshold = float(config["nashconv_auto_resume_threshold"])
    nashconv_auto_resume_step_fraction = float(
        config["nashconv_auto_resume_step_fraction"]
    )
    resume_partial_runs = bool(config["resume_partial_runs"])
    save_dir = Path(config["save_dir"])

    checkpoint_dir = save_dir / "checkpoints"
    timings_path = save_dir / timings_filename(num_agents)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_run_seeds = _expected_run_seeds(seed=seed, num_runs=num_runs)
    config_fingerprint = _partial_resume_fingerprint(
        algo_name=algo_name,
        config=config,
        algo_config=algo_config,
        spec=spec,
    )

    def make_base_env(seed: int | None = None, num_agents: int = num_agents):
        return _make_base_env(spec, config, seed=seed, num_agents=num_agents)

    def make_objective_env(seed: int | None = None, num_agents: int = num_agents):
        objective_config = dict(config)
        objective_config["reward_mode"] = "objective"
        return _make_base_env(spec, objective_config, seed=seed, num_agents=num_agents)

    if satisfaction_fn is None:
        satisfaction_fn = SATISFACTION_FN_BY_ALGORITHM.get(algo_name)
    if calculate_satisfaction and satisfaction_fn is None:
        raise ValueError(
            f"No final satisfaction verifier registered for algorithm {algo_name!r}."
        )

    resume_state = _load_partial_resume_state(
        enabled=resume_partial_runs,
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        algo_name=algo_name,
        num_agents=num_agents,
        seed=seed,
        num_runs=num_runs,
        calculate_conv=calculate_conv,
        calculate_conv_quick=calculate_conv_quick,
        calculate_satisfaction=calculate_satisfaction,
        config_fingerprint=config_fingerprint,
    )
    start_run_idx = int(resume_state["start_run_idx"])
    runs: list[HistoryByAgent] = list(resume_state["runs"])
    nash_runs: list[dict[str, Any]] = list(resume_state["nash_runs"])
    satisfaction_runs: list[dict[str, Any]] = list(resume_state["satisfaction_runs"])
    auto_resume_events: list[dict[str, Any]] = list(resume_state["auto_resume_events"])
    checkpoint_paths: list[Path] = list(resume_state["checkpoint_paths"])
    completed_run_records: list[dict[str, Any]] = list(
        resume_state["completed_run_records"]
    )
    agent_ids: list[str] | None = resume_state["agent_ids"]
    timings: dict[str, dict[str, Any]] = dict(resume_state["timings"])
    if start_run_idx > 0:
        print(
            f"[{algo_name}] resuming from run {start_run_idx + 1}/{num_runs}; reused {start_run_idx} completed run(s)."
        )

    for run_idx in range(start_run_idx, num_runs):
        run_seed = seed + run_idx
        run_algo_config = dict(algo_config)
        run_algo_config["SEED"] = run_seed
        batch_size = int(run_algo_config["NUM_ENVS"] * run_algo_config["NUM_STEPS"])

        pbar = tqdm(
            total=num_env_steps,
            unit="env-step",
            desc=f"{algo_name} Run {run_idx + 1}/{num_runs} (seed={run_seed})",
        )
        chunked_pbar = _ChunkedProgressBar(
            pbar, total=num_env_steps, chunks=progress_chunks
        )

        def on_update(
            update_idx: int, total_updates: int, metric: dict[str, Any]
        ) -> None:
            del update_idx, total_updates
            mean_ret = float(metric["episode_return_mean"])
            chunked_pbar.advance(
                batch_size,
                postfix={"return_mean": f"{mean_ret:.3f}"},
            )

        train = make_train_fn(
            run_algo_config,
            lambda env_seed: make_base_env(seed=env_seed, num_agents=num_agents),
        )
        train_start = perf_counter()
        out = train(seed=run_seed, on_update=on_update, capture_env_histories=True)
        train_seconds = perf_counter() - train_start
        pbar.close()
        record_timing(timings, train_timing_name, train_seconds, run_idx=run_idx)
        save_experiment_timings(timings, save_dir, num_agents=num_agents)

        mean_ret = float(out["metrics"]["episode_return_mean"][-1])
        lifetime = int(run_algo_config["TOTAL_TIMESTEPS"])
        run_history = _merge_env_histories(list(out.get("env_histories", [])))
        runs.append(run_history)
        agent_ids = list(out["agent_ids"])

        run_label = _run_label(run_idx, run_seed)
        checkpoint_path = checkpoint_dir / f"{run_label}.pkl"
        checkpoint_start = perf_counter()
        _save_checkpoint_bundle(
            checkpoint_path,
            train_states=out["train_states"],
            agent_ids=out["agent_ids"],
            run_seed=run_seed,
            algo_config=run_algo_config,
        )
        checkpoint_seconds = perf_counter() - checkpoint_start
        record_timing(timings, "checkpoint_save", checkpoint_seconds, run_idx=run_idx)
        save_experiment_timings(timings, save_dir, num_agents=num_agents)
        checkpoint_paths.append(checkpoint_path)

        auto_resume_event_start = len(auto_resume_events)
        should_run_nash = _should_evaluate_nash_run(
            calculate_conv=calculate_conv,
            calculate_conv_quick=calculate_conv_quick,
            run_idx=run_idx,
        )
        nash_index: int | None = None
        if should_run_nash:
            nash_start = perf_counter()
            nash = nashconv_fn(
                train_states=out["train_states"],
                config=run_algo_config,
                make_base_env_fn=make_base_env,
                agent_ids=list(out["agent_ids"]),
                eval_episodes=eval_episodes,
                eval_seed0=run_seed * 1_000_000 + 7,
                delta=nashconv_delta,
                br_train_iters=br_train_iters,
                br_batch_steps=br_batch_steps,
                success_threshold=spec.success_threshold,
            )
            nash_seconds = perf_counter() - nash_start
            record_timing(timings, "nashconv", nash_seconds, run_idx=run_idx)
            save_experiment_timings(timings, save_dir, num_agents=num_agents)
            timing_bits = [
                f"train: {train_seconds:.2f}s",
                f"checkpoint: {checkpoint_seconds:.2f}s",
                f"nashconv: {nash_seconds:.2f}s",
            ]
            print(
                f"[{algo_name} Run {run_idx + 1}] mean return (last update): "
                f"{mean_ret:.3f} | lifetime steps: {lifetime:,} | {' | '.join(timing_bits)}"
            )
            _print_nashconv_report(
                algo_name=algo_name, nash=nash, agent_ids=out["agent_ids"]
            )

            initial_nashconv = float(nash["nashconv"])
            should_resume = (
                nashconv_auto_resume
                and initial_nashconv > nashconv_auto_resume_threshold
            )
            if should_resume:
                extra_updates, extra_steps = _auto_resume_budget(
                    run_algo_config,
                    nashconv_auto_resume_step_fraction,
                )
                auto_resume_event = {
                    "algorithm": algo_name,
                    "run_idx": run_idx,
                    "run_seed": run_seed,
                    "initial_nashconv": initial_nashconv,
                    "final_nashconv": initial_nashconv,
                    "threshold": nashconv_auto_resume_threshold,
                    "step_fraction": nashconv_auto_resume_step_fraction,
                    "original_steps": int(run_algo_config["TOTAL_TIMESTEPS"]),
                    "extra_updates": int(extra_updates),
                    "extra_steps": int(extra_steps),
                    "converged": False,
                    "status": "skipped_no_budget",
                }

                if extra_updates <= 0:
                    print(
                        f"[{algo_name} Run {run_idx + 1}] NashConv {initial_nashconv:.6f} "
                        f"exceeds threshold {nashconv_auto_resume_threshold:.6f}, but "
                        "auto-resume skipped because the configured extra budget is less than one update."
                    )
                    auto_resume_events.append(auto_resume_event)
                else:
                    resume_algo_config = _make_resume_algo_config(
                        run_algo_config,
                        extra_updates=extra_updates,
                        extra_steps=extra_steps,
                    )
                    resume_seed = run_seed + int(run_algo_config["TOTAL_TIMESTEPS"]) + 1
                    resume_algo_config["SEED"] = resume_seed
                    auto_resume_event["resume_seed"] = resume_seed
                    auto_resume_event["status"] = "resumed"

                    print(
                        f"[{algo_name} Run {run_idx + 1}] NashConv {initial_nashconv:.6f} "
                        f"exceeds threshold {nashconv_auto_resume_threshold:.6f}; "
                        f"auto-resuming for {extra_steps:,} extra steps "
                        f"({extra_updates:,} updates, {nashconv_auto_resume_step_fraction:.0%} cap)."
                    )

                    resume_pbar = tqdm(
                        total=extra_steps,
                        unit="env-step",
                        desc=(
                            f"{algo_name} Run {run_idx + 1}/{num_runs} auto-resume (seed={resume_seed})"
                        ),
                    )
                    resume_batch_size = int(
                        resume_algo_config["NUM_ENVS"] * resume_algo_config["NUM_STEPS"]
                    )
                    chunked_resume_pbar = _ChunkedProgressBar(
                        resume_pbar,
                        total=extra_steps,
                        chunks=progress_chunks,
                    )

                    def on_resume_update(
                        update_idx: int,
                        total_updates: int,
                        metric: dict[str, Any],
                    ) -> None:
                        del update_idx, total_updates
                        resume_mean_ret = float(metric["episode_return_mean"])
                        chunked_resume_pbar.advance(
                            resume_batch_size,
                            postfix={"return_mean": f"{resume_mean_ret:.3f}"},
                        )

                    resume_train = make_train_fn(
                        resume_algo_config,
                        lambda env_seed: make_base_env(
                            seed=env_seed, num_agents=num_agents
                        ),
                    )
                    resume_start = perf_counter()
                    resume_out = resume_train(
                        seed=resume_seed,
                        initial_train_states=out["train_states"],
                        on_update=on_resume_update,
                        capture_env_histories=True,
                    )
                    resume_seconds = perf_counter() - resume_start
                    resume_pbar.close()
                    record_timing(
                        timings,
                        f"{train_timing_name}_auto_resume",
                        resume_seconds,
                        run_idx=run_idx,
                    )
                    save_experiment_timings(timings, save_dir, num_agents=num_agents)

                    resume_history = _merge_env_histories(
                        list(resume_out.get("env_histories", []))
                    )
                    history_offset = int(run_algo_config["TOTAL_TIMESTEPS"]) // max(
                        1, int(run_algo_config["NUM_ENVS"])
                    )
                    runs[run_idx] = _merge_env_histories(
                        [
                            run_history,
                            _offset_history_t_end(resume_history, history_offset),
                        ]
                    )
                    out = resume_out
                    lifetime += extra_steps

                    final_checkpoint_config = _make_final_checkpoint_config(
                        run_algo_config,
                        extra_steps=extra_steps,
                    )
                    final_checkpoint_config["SEED"] = run_seed
                    checkpoint_start = perf_counter()
                    _save_checkpoint_bundle(
                        checkpoint_path,
                        train_states=out["train_states"],
                        agent_ids=out["agent_ids"],
                        run_seed=run_seed,
                        algo_config=final_checkpoint_config,
                    )
                    checkpoint_seconds = perf_counter() - checkpoint_start
                    record_timing(
                        timings,
                        "checkpoint_save_auto_resume",
                        checkpoint_seconds,
                        run_idx=run_idx,
                    )
                    save_experiment_timings(timings, save_dir, num_agents=num_agents)

                    final_nash_start = perf_counter()
                    nash = nashconv_fn(
                        train_states=out["train_states"],
                        config=run_algo_config,
                        make_base_env_fn=make_base_env,
                        agent_ids=list(out["agent_ids"]),
                        eval_episodes=eval_episodes,
                        eval_seed0=run_seed * 1_000_000 + 7,
                        delta=nashconv_delta,
                        br_train_iters=br_train_iters,
                        br_batch_steps=br_batch_steps,
                        success_threshold=spec.success_threshold,
                    )
                    final_nash_seconds = perf_counter() - final_nash_start
                    record_timing(
                        timings,
                        "nashconv_auto_resume",
                        final_nash_seconds,
                        run_idx=run_idx,
                    )
                    save_experiment_timings(timings, save_dir, num_agents=num_agents)

                    final_nashconv = float(nash["nashconv"])
                    auto_resume_event["final_nashconv"] = final_nashconv
                    auto_resume_event["converged"] = (
                        final_nashconv <= nashconv_auto_resume_threshold
                    )
                    auto_resume_events.append(auto_resume_event)

                    resume_mean_ret = float(out["metrics"]["episode_return_mean"][-1])
                    print(
                        f"[{algo_name} Run {run_idx + 1}] auto-resume finished | "
                        f"mean return (last resume update): {resume_mean_ret:.3f} | "
                        f"lifetime steps: {lifetime:,} | "
                        f"resume train: {resume_seconds:.2f}s | "
                        f"checkpoint: {checkpoint_seconds:.2f}s | "
                        f"nashconv: {final_nash_seconds:.2f}s"
                    )
                    print(
                        f"[{algo_name} Run {run_idx + 1}] NashConv after auto-resume: "
                        f"{final_nashconv:.6f} "
                        f"({'converged' if auto_resume_event['converged'] else 'not converged'} "
                        f"at threshold {nashconv_auto_resume_threshold:.6f})"
                    )
                    _print_nashconv_report(
                        algo_name=algo_name,
                        nash=nash,
                        agent_ids=out["agent_ids"],
                    )

            nash_runs.append(nash)
            nash_index = len(nash_runs) - 1
        else:
            timing_bits = [
                f"train: {train_seconds:.2f}s",
                f"checkpoint: {checkpoint_seconds:.2f}s",
            ]
            print(
                f"[{algo_name} Run {run_idx + 1}] mean return (last update): "
                f"{mean_ret:.3f} | lifetime steps: {lifetime:,} | {' | '.join(timing_bits)}"
            )

        satisfaction_index: int | None = None
        if calculate_satisfaction:
            satisfaction_start = perf_counter()
            satisfaction = satisfaction_fn(
                train_states=out["train_states"],
                config=run_algo_config,
                make_base_env_fn=make_objective_env,
                agent_ids=list(out["agent_ids"]),
                eval_episodes=satisfaction_eval_episodes,
                eval_seed0=run_seed * 1_000_000 + 503,
                delta=satisfaction_delta,
                success_threshold=spec.success_threshold,
                probability_thresholds=spec.probability_thresholds,
                exclusive_objective_groups=spec.exclusive_objective_groups,
            )
            satisfaction_seconds = perf_counter() - satisfaction_start
            record_timing(
                timings, "satisfaction", satisfaction_seconds, run_idx=run_idx
            )
            save_experiment_timings(timings, save_dir, num_agents=num_agents)
            satisfaction_runs.append(satisfaction)
            satisfaction_index = len(satisfaction_runs) - 1
            print(
                f"[{algo_name} Run {run_idx + 1}] final satisfaction finished in {satisfaction_seconds:.2f}s"
            )
            _print_satisfaction_report(
                algo_name=algo_name,
                satisfaction=satisfaction,
                agent_ids=out["agent_ids"],
            )

        completed_run_records.append(
            {
                "run_idx": int(run_idx),
                "run_seed": int(run_seed),
                "label": run_label,
                "checkpoint_path": str(checkpoint_path),
                "nash_evaluated": bool(should_run_nash),
                "nash_index": nash_index,
                "satisfaction_evaluated": bool(calculate_satisfaction),
                "satisfaction_index": satisfaction_index,
                "agent_ids": list(out["agent_ids"]),
                "auto_resume_events": auto_resume_events[auto_resume_event_start:],
            }
        )
        artifact_paths = _save_algorithm_artifacts(
            algo_name=algo_name,
            num_agents=num_agents,
            save_dir=save_dir,
            runs=runs,
            nash_runs=nash_runs,
            satisfaction_runs=satisfaction_runs,
        )
        if resume_partial_runs:
            _save_partial_resume_manifest(
                save_dir=save_dir,
                algo_name=algo_name,
                num_agents=num_agents,
                expected_run_seeds=expected_run_seeds,
                config_fingerprint=config_fingerprint,
                completed_run_records=completed_run_records,
            )

    nash_summary = summarize_nashconv_runs(nash_runs)
    satisfaction_summary = summarize_satisfaction_runs(satisfaction_runs)
    timings_summary = summarize_timings(timings)
    artifact_paths = _save_algorithm_artifacts(
        algo_name=algo_name,
        num_agents=num_agents,
        save_dir=save_dir,
        runs=runs,
        nash_runs=nash_runs,
        satisfaction_runs=satisfaction_runs,
    )
    print(f"[{algo_name}] saved result artifacts to {save_dir}")

    return {
        "algorithm": algo_name,
        "config": config,
        config_key_name: algo_config,
        "runs": runs,
        "nash_runs": nash_runs,
        "satisfaction_runs": satisfaction_runs,
        "auto_resume_events": auto_resume_events,
        "nash_summary": nash_summary,
        "satisfaction_summary": satisfaction_summary,
        "num_agents": num_agents,
        "save_dir": save_dir,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_paths": checkpoint_paths,
        "artifact_paths": artifact_paths,
        "timings": timings,
        "timings_path": timings_path,
        "timings_summary": timings_summary,
    }


def run_temporal_game_ippo_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="IPPO",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="ippo_config",
        spec=spec,
        make_train_fn=make_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_ippo,
        train_timing_name="ippo_train",
    )


def run_temporal_game_ippo_cer_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    ippo_config = dict(ippo_config)
    ippo_config["COUNTERFACTUAL_REPLAY"] = True
    return _run_temporal_game_algorithm_experiment(
        algo_name="IPPO_CER",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="ippo_cer_config",
        spec=spec,
        make_train_fn=make_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_ippo,
        train_timing_name="ippo_cer_train",
    )


def run_temporal_game_pr2_ippo_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="PR2_IPPO",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="pr2_ippo_config",
        spec=spec,
        make_train_fn=make_pr2_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_ippo,
        train_timing_name="pr2_ippo_train",
    )


def run_temporal_game_pr2_ippo_cer_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    ippo_config = dict(ippo_config)
    ippo_config["COUNTERFACTUAL_REPLAY"] = True
    return _run_temporal_game_algorithm_experiment(
        algo_name="PR2_IPPO_CER",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="pr2_ippo_cer_config",
        spec=spec,
        make_train_fn=make_pr2_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_ippo,
        train_timing_name="pr2_ippo_cer_train",
    )


def run_temporal_game_iql_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="IQL",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="iql_config",
        spec=spec,
        make_train_fn=make_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_iql,
        train_timing_name="iql_train",
    )


def run_temporal_game_iql_cer_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    iql_config = dict(iql_config)
    iql_config["COUNTERFACTUAL_REPLAY"] = True
    return _run_temporal_game_algorithm_experiment(
        algo_name="IQL_CER",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="iql_cer_config",
        spec=spec,
        make_train_fn=make_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_iql,
        train_timing_name="iql_cer_train",
    )


def run_temporal_game_pr2_iql_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="PR2_IQL",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="pr2_iql_config",
        spec=spec,
        make_train_fn=make_pr2_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_iql,
        train_timing_name="pr2_iql_train",
    )


def run_temporal_game_pr2_iql_cer_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    iql_config = dict(iql_config)
    iql_config["COUNTERFACTUAL_REPLAY"] = True
    return _run_temporal_game_algorithm_experiment(
        algo_name="PR2_IQL_CER",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="pr2_iql_cer_config",
        spec=spec,
        make_train_fn=make_pr2_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_iql,
        train_timing_name="pr2_iql_cer_train",
    )


def run_temporal_game_nashq_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, nashq_config = _resolve_nashq_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="NASHQ",
        config=resolved_config,
        algo_config=nashq_config,
        config_key_name="nashq_config",
        spec=spec,
        make_train_fn=make_nashq_train,
        nashconv_fn=approximate_nashconv_bernoulli_nashq,
        train_timing_name="nashq_train",
    )


def run_temporal_game_nashq_cer_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_config, nashq_config = _resolve_nashq_config(spec, config)
    nashq_config = dict(nashq_config)
    nashq_config["COUNTERFACTUAL_REPLAY"] = True
    return _run_temporal_game_algorithm_experiment(
        algo_name="NASHQ_CER",
        config=resolved_config,
        algo_config=nashq_config,
        config_key_name="nashq_cer_config",
        spec=spec,
        make_train_fn=make_nashq_train,
        nashconv_fn=approximate_nashconv_bernoulli_nashq,
        train_timing_name="nashq_cer_train",
    )


def run_temporal_game_hc_ippo_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    config["reward_mode"] = "handcrafted"
    config["calculate_satisfaction"] = True
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="hc_IPPO",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="hc_ippo_config",
        spec=spec,
        make_train_fn=make_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_ippo,
        train_timing_name="hc_ippo_train",
    )


def run_temporal_game_hc_pr2_ippo_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    config["reward_mode"] = "handcrafted"
    config["calculate_satisfaction"] = True
    resolved_config, ippo_config = _resolve_ippo_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="hc_PR2_IPPO",
        config=resolved_config,
        algo_config=ippo_config,
        config_key_name="hc_pr2_ippo_config",
        spec=spec,
        make_train_fn=make_pr2_ippo_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_ippo,
        train_timing_name="hc_pr2_ippo_train",
    )


def run_temporal_game_hc_iql_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    config["reward_mode"] = "handcrafted"
    config["calculate_satisfaction"] = True
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="hc_IQL",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="hc_iql_config",
        spec=spec,
        make_train_fn=make_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_iql,
        train_timing_name="hc_iql_train",
    )


def run_temporal_game_hc_pr2_iql_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    config["reward_mode"] = "handcrafted"
    config["calculate_satisfaction"] = True
    resolved_config, iql_config = _resolve_iql_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="hc_PR2_IQL",
        config=resolved_config,
        algo_config=iql_config,
        config_key_name="hc_pr2_iql_config",
        spec=spec,
        make_train_fn=make_pr2_iql_train,
        nashconv_fn=approximate_nashconv_bernoulli_pr2_iql,
        train_timing_name="hc_pr2_iql_train",
    )


def run_temporal_game_hc_nashq_experiment(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    config["reward_mode"] = "handcrafted"
    config["calculate_satisfaction"] = True
    resolved_config, nashq_config = _resolve_nashq_config(spec, config)
    return _run_temporal_game_algorithm_experiment(
        algo_name="hc_NASHQ",
        config=resolved_config,
        algo_config=nashq_config,
        config_key_name="hc_nashq_config",
        spec=spec,
        make_train_fn=make_nashq_train,
        nashconv_fn=approximate_nashconv_bernoulli_nashq,
        train_timing_name="hc_nashq_train",
    )


def run_temporal_game_experiments(
    spec: TemporalGameExperimentSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the selected algorithms for a temporal-game experiment suite."""

    resolved_suite, _, _, _ = _resolve_suite_config(spec, config)

    experiments: dict[str, dict[str, Any]] = {}

    if resolved_suite["run_ippo"]:
        ippo_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        ippo_config["save_dir"] = resolved_suite["save_dir"] / "ippo"
        experiments["IPPO"] = run_temporal_game_ippo_experiment(spec, ippo_config)

    if resolved_suite["run_ippo_cer"]:
        ippo_cer_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        ippo_cer_config["save_dir"] = resolved_suite["save_dir"] / "ippo_cer"
        experiments["IPPO_CER"] = run_temporal_game_ippo_cer_experiment(
            spec,
            ippo_cer_config,
        )

    if resolved_suite["run_pr2_ippo"]:
        pr2_ippo_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        pr2_ippo_config["save_dir"] = resolved_suite["save_dir"] / "pr2_ippo"
        experiments["PR2_IPPO"] = run_temporal_game_pr2_ippo_experiment(
            spec, pr2_ippo_config
        )

    if resolved_suite["run_pr2_ippo_cer"]:
        pr2_ippo_cer_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        pr2_ippo_cer_config["save_dir"] = resolved_suite["save_dir"] / "pr2_ippo_cer"
        experiments["PR2_IPPO_CER"] = run_temporal_game_pr2_ippo_cer_experiment(
            spec,
            pr2_ippo_cer_config,
        )

    if resolved_suite["run_iql"]:
        iql_config = {key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)}
        iql_config["save_dir"] = resolved_suite["save_dir"] / "iql"
        experiments["IQL"] = run_temporal_game_iql_experiment(spec, iql_config)

    if resolved_suite["run_iql_cer"]:
        iql_cer_config = {
            key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)
        }
        iql_cer_config["save_dir"] = resolved_suite["save_dir"] / "iql_cer"
        experiments["IQL_CER"] = run_temporal_game_iql_cer_experiment(
            spec, iql_cer_config
        )

    if resolved_suite["run_pr2_iql"]:
        pr2_iql_config = {
            key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)
        }
        pr2_iql_config["save_dir"] = resolved_suite["save_dir"] / "pr2_iql"
        experiments["PR2_IQL"] = run_temporal_game_pr2_iql_experiment(
            spec, pr2_iql_config
        )

    if resolved_suite["run_pr2_iql_cer"]:
        pr2_iql_cer_config = {
            key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)
        }
        pr2_iql_cer_config["save_dir"] = resolved_suite["save_dir"] / "pr2_iql_cer"
        experiments["PR2_IQL_CER"] = run_temporal_game_pr2_iql_cer_experiment(
            spec,
            pr2_iql_cer_config,
        )

    if resolved_suite["run_nashq"]:
        nashq_config = {
            key: resolved_suite["nashq"][key] for key in _nashq_config_keys(spec)
        }
        nashq_config["save_dir"] = resolved_suite["save_dir"] / "nashq"
        experiments["NASHQ"] = run_temporal_game_nashq_experiment(spec, nashq_config)

    if resolved_suite["run_nashq_cer"]:
        nashq_cer_config = {
            key: resolved_suite["nashq"][key] for key in _nashq_config_keys(spec)
        }
        nashq_cer_config["save_dir"] = resolved_suite["save_dir"] / "nashq_cer"
        experiments["NASHQ_CER"] = run_temporal_game_nashq_cer_experiment(
            spec,
            nashq_cer_config,
        )

    if resolved_suite["run_hc_ippo"]:
        hc_ippo_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        hc_ippo_config["save_dir"] = resolved_suite["save_dir"] / "hc_ippo"
        experiments["hc_IPPO"] = run_temporal_game_hc_ippo_experiment(
            spec, hc_ippo_config
        )

    if resolved_suite["run_hc_pr2_ippo"]:
        hc_pr2_ippo_config = {
            key: resolved_suite["ippo"][key] for key in _ippo_config_keys(spec)
        }
        hc_pr2_ippo_config["save_dir"] = resolved_suite["save_dir"] / "hc_pr2_ippo"
        experiments["hc_PR2_IPPO"] = run_temporal_game_hc_pr2_ippo_experiment(
            spec,
            hc_pr2_ippo_config,
        )

    if resolved_suite["run_hc_iql"]:
        hc_iql_config = {
            key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)
        }
        hc_iql_config["save_dir"] = resolved_suite["save_dir"] / "hc_iql"
        experiments["hc_IQL"] = run_temporal_game_hc_iql_experiment(spec, hc_iql_config)

    if resolved_suite["run_hc_pr2_iql"]:
        hc_pr2_iql_config = {
            key: resolved_suite["iql"][key] for key in _iql_config_keys(spec)
        }
        hc_pr2_iql_config["save_dir"] = resolved_suite["save_dir"] / "hc_pr2_iql"
        experiments["hc_PR2_IQL"] = run_temporal_game_hc_pr2_iql_experiment(
            spec,
            hc_pr2_iql_config,
        )

    if resolved_suite["run_hc_nashq"]:
        hc_nashq_config = {
            key: resolved_suite["nashq"][key] for key in _nashq_config_keys(spec)
        }
        hc_nashq_config["save_dir"] = resolved_suite["save_dir"] / "hc_nashq"
        experiments["hc_NASHQ"] = run_temporal_game_hc_nashq_experiment(
            spec, hc_nashq_config
        )

    selected_algorithms = tuple(
        algorithm for algorithm in ALGORITHM_ORDER if algorithm in experiments
    )
    handcrafted_table_path = _save_handcrafted_baseline_table(
        experiments, resolved_suite["save_dir"]
    )

    return {
        "config": resolved_suite,
        "selected_algorithms": selected_algorithms,
        "learning_curve_algorithms": tuple(
            algorithm
            for algorithm in selected_algorithms
            if not _is_handcrafted_algorithm(algorithm)
        ),
        "handcrafted_algorithms": tuple(
            algorithm
            for algorithm in selected_algorithms
            if _is_handcrafted_algorithm(algorithm)
        ),
        "experiments": experiments,
        "handcrafted_baseline_table_path": handcrafted_table_path,
        "save_dir": resolved_suite["save_dir"],
        "num_agents": resolved_suite["num_agents"],
    }


__all__ = [
    "ALGORITHM_ORDER",
    "HistoryByAgent",
    "TEMPORAL_GAME_SHARED_CONFIG_KEYS",
    "TEMPORAL_GAME_SHARED_CONFIG_DEFAULTS",
    "TEMPORAL_GAME_IPPO_BLOCK_KEYS",
    "TEMPORAL_GAME_IQL_BLOCK_KEYS",
    "TEMPORAL_GAME_NASHQ_BLOCK_KEYS",
    "TemporalGameExperimentSpec",
    "run_temporal_game_experiments",
    "run_temporal_game_ippo_experiment",
    "run_temporal_game_ippo_cer_experiment",
    "run_temporal_game_pr2_ippo_experiment",
    "run_temporal_game_pr2_ippo_cer_experiment",
    "run_temporal_game_iql_experiment",
    "run_temporal_game_iql_cer_experiment",
    "run_temporal_game_pr2_iql_experiment",
    "run_temporal_game_pr2_iql_cer_experiment",
    "run_temporal_game_nashq_experiment",
    "run_temporal_game_nashq_cer_experiment",
    "run_temporal_game_hc_ippo_experiment",
    "run_temporal_game_hc_pr2_ippo_experiment",
    "run_temporal_game_hc_iql_experiment",
    "run_temporal_game_hc_pr2_iql_experiment",
    "run_temporal_game_hc_nashq_experiment",
    "_ippo_config_keys",
    "_iql_config_keys",
    "_nashq_config_keys",
    "_resolve_ippo_config",
    "_resolve_iql_config",
    "_resolve_nashq_config",
    "_resolve_suite_config",
    "_run_temporal_game_algorithm_experiment",
]
