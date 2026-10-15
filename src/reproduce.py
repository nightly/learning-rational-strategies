"""Run the benchmark configurations without notebook or scheduler dependencies."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "classic_2",
    "capture_2",
    "classic_3",
    "capture_3",
    "coins",
    "stag_hunt",
    "harvest",
    "territory",
)
ALGORITHMS = (
    "ippo",
    "ippo_cer",
    "pr2_ippo",
    "pr2_ippo_cer",
    "iql",
    "iql_cer",
    "pr2_iql",
    "pr2_iql_cer",
    "nashq",
    "nashq_cer",
    "hc_ippo",
    "hc_pr2_ippo",
    "hc_iql",
    "hc_pr2_iql",
    "hc_nashq",
)


def load_config(task: str) -> dict[str, Any]:
    """Read a fresh copy of a committed benchmark configuration."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return json.loads((ROOT / "configs" / f"{task}.json").read_text())


def prepare_config(
    task: str,
    algorithms: list[str],
    *,
    output: Path,
    seed: int | None = None,
    num_runs: int | None = None,
    steps: int | None = None,
    smoke: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not algorithms or set(algorithms) - set(ALGORITHMS):
        raise ValueError("Select at least one supported algorithm.")
    if task != "stag_hunt" and any("nashq" in name for name in algorithms):
        raise ValueError("The benchmark Nash-Q baseline is restricted to Stag Hunt.")
    payload = load_config(task)
    config = copy.deepcopy(payload["config"])
    if overrides:
        for key, value in overrides.items():
            if key not in config:
                raise ValueError(f"Unknown configuration key: {key}")
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Invalid overrides for {key}")
                config[key].update(value)
            else:
                config[key] = value
    if seed is not None:
        config["seed"] = seed
    if num_runs is not None:
        config["num_runs"] = num_runs
    if steps is not None:
        config["num_env_steps"] = steps
    if smoke:
        # Exercise training, checkpoint persistence, and both evaluators.
        # These deliberately small budgets have no scientific significance.
        config.update(
            num_runs=1,
            num_env_steps=16,
            progress_chunks=1,
            eval_episodes=2,
            br_train_iters=1,
            br_batch_steps=4,
            satisfaction_eval_episodes=2,
            nashconv_auto_resume=False,
            resume_partial_runs=False,
        )
        config["ippo"].update(
            num_envs=1, num_steps=4, num_minibatches=1, update_epochs=1
        )
        config["iql"].update(
            num_envs=1,
            num_steps=4,
            num_epochs=1,
            buffer_size=64,
            buffer_batch_size=2,
            learning_starts=4,
        )
        config["nashq"].update(num_envs=1, num_steps=4)
    if config["num_runs"] <= 0 or config["num_env_steps"] <= 0:
        raise ValueError("num_runs and num_env_steps must be positive.")
    config.update({f"run_{name}": name in algorithms for name in ALGORITHMS})
    config["save_dir"] = (output / payload["task"]).resolve()
    return config


def build_spec(task: str):
    """Construct the same objective wrappers and thresholds as the benchmarks."""
    from src.experiments.objective_thresholds import curated_probability_thresholds

    thresholds = curated_probability_thresholds(load_config(task)["task"])
    if task.startswith(("classic_", "capture_")):
        from src.environments.gridworlds.pursuit.wrappers.constraints import (
            PECaptureVsStealth,
            PEClassicConstraint,
        )
        from src.experiments.pursuit_experiment import (
            PursuitExperimentSpec,
            _as_temporal_spec,
        )

        wrapper = (
            PEClassicConstraint if task.startswith("classic_") else PECaptureVsStealth
        )
        return _as_temporal_spec(
            PursuitExperimentSpec(
                title=task,
                wrapper_factory=wrapper,
                probability_thresholds=thresholds,
            )
        )
    if task == "coins":
        from src.environments.gridworlds.coins import CoinsOwnCoinFirstConstraint
        from src.experiments.coins_experiment import (
            CoinsExperimentSpec,
            _as_temporal_spec,
        )

        return _as_temporal_spec(
            CoinsExperimentSpec(
                title=task,
                wrapper_factory=CoinsOwnCoinFirstConstraint,
                probability_thresholds=thresholds,
            )
        )
    if task == "harvest":
        from src.environments.gridworlds.harvest import CHSafeHarvestConstraint
        from src.experiments.harvest_experiment import (
            HarvestExperimentSpec,
            _as_temporal_spec,
        )

        return _as_temporal_spec(
            HarvestExperimentSpec(
                title=task,
                wrapper_factory=CHSafeHarvestConstraint,
                probability_thresholds=thresholds,
            )
        )
    if task == "territory":
        from src.environments.gridworlds.territory import TerritoryRoleClaimConstraint
        from src.experiments.territory_experiment import (
            TerritoryExperimentSpec,
            _as_temporal_spec,
        )

        return _as_temporal_spec(
            TerritoryExperimentSpec(
                title=task,
                wrapper_factory=TerritoryRoleClaimConstraint,
                probability_thresholds=thresholds,
                exclusive_objective_groups=(("player_0", "player_1"),),
            )
        )
    if task == "stag_hunt":
        from src.environments.matrix.stag_hunt import StagHuntMatrix
        from src.environments.matrix.stag_hunt.wrappers import SHSustainedStagConstraint
        from src.experiments.temporal_game_experiment import TemporalGameExperimentSpec

        def make_env(*, seed=None, num_agents=2, config=None):
            return SHSustainedStagConstraint(
                StagHuntMatrix(
                    num_agents=num_agents,
                    max_moves=int(config["max_moves"]),
                    render_mode=None,
                    seed=seed,
                ),
                max_steps_per_episode=int(config["max_moves"]),
            )

        return TemporalGameExperimentSpec(
            title=task,
            env_factory=make_env,
            extra_config_keys=("max_moves",),
            probability_thresholds=thresholds,
        )
    raise ValueError(f"Unknown task: {task}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=["ippo"])
    parser.add_argument("--output", type=Path, default=ROOT / "exports")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--overrides", type=Path, help="JSON object with configuration overrides"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny integration run, unsuitable for reported results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the configuration without importing JAX or training",
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)
    try:
        overrides = json.loads(args.overrides.read_text()) if args.overrides else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("The overrides file must contain a JSON object.")
        config = prepare_config(
            args.task,
            args.algorithms,
            output=args.output,
            seed=args.seed,
            num_runs=args.num_runs,
            steps=args.steps,
            smoke=args.smoke,
            overrides=overrides,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(config, indent=2, default=str), flush=True)
    if args.dry_run:
        return 0
    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ["JAX_PLATFORM_NAME"] = args.platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from src.experiments.temporal_game_experiment import run_temporal_game_experiments

    # Refuse pre-existing outputs: a new invocation must not silently replace a
    # profile or reset incompatible partial-resume artifacts.
    for name in args.algorithms:
        destination = config["save_dir"] / name
        if destination.exists() and any(destination.iterdir()):
            parser.error(
                f"Output already exists: {destination}. Choose a fresh --output directory."
            )
    spec = build_spec(args.task)
    config["save_dir"].mkdir(parents=True, exist_ok=True)
    manifest = config["save_dir"] / (
        "invocation_" + "_".join(args.algorithms) + ".json"
    )
    manifest.write_text(
        json.dumps(
            {
                "task": args.task,
                "algorithms": args.algorithms,
                "smoke": args.smoke,
                "platform": args.platform,
                "config": config,
                "probability_thresholds": spec.probability_thresholds,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    run_temporal_game_experiments(spec, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
