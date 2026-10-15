"""Summarize per-seed joint certification across exported PLTLf experiments.

Reads every ``<task>/<algorithm>/satisfaction_report.json`` under an exports
root and reports, per profile: the number of seed profiles whose simultaneous
Bonferroni-Hoeffding lower bounds clear their thresholds, the joint margin
``min_i (L_i - q_i)`` (positive exactly when a seed is jointly certified), the
worst-agent joint lower bound (the largest uniform threshold each seed is
jointly certified at), the same count under a family-wise correction over every
seed profile tested on the task or in the whole campaign, and the
confirmatory-batch count when a ``confirmatory_satisfaction_report.json`` exists.

Certification uses the curated per-agent thresholds of
``src/experiments/objective_thresholds.py`` by default, recomputed from each
report's success counts, so reports written under older thresholds are
re-certified consistently. ``--reported-thresholds`` uses the thresholds each
report recorded instead.

Usage:
    uv run python scripts/certification_summary.py [--exports exports] [--json out.json]
        [--reported-thresholds]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.objective_thresholds import curated_probability_thresholds  # noqa: E402
from src.verify.certification import certify_satisfaction_run  # noqa: E402
from src.verify.satisfaction import hoeffding_ci, summarize_satisfaction_runs  # noqa: E402

CURATED_TASKS = {
    "gridworlds/pursuit/classic_2": "Pursuit Classic 2",
    "gridworlds/pursuit/capture_2": "Pursuit Capture 2",
    "gridworlds/pursuit/classic_3": "Pursuit Classic 3",
    "gridworlds/pursuit/capture_3": "Pursuit Capture 3",
    "gridworlds/coins/own_coin_first/2": "Coins Own Coin First",
    "matrix/stag_hunt/sustained_stag/2": "Matrix Sustained Stag",
    "gridworlds/harvest/safe_harvest/4": "Commons Safe Harvest",
    "gridworlds/territory/role_claim/2": "Territory Role Claim",
}
ALGORITHM_LABELS = {
    "ippo_cer": "IPPO-CER",
    "pr2_ippo_cer": "PR2-IPPO-CER",
    "iql_cer": "IQL-CER",
    "pr2_iql_cer": "PR2-IQL-CER",
    "nashq_cer": "Nash-Q-CER",
    "ippo": "IPPO",
    "pr2_ippo": "PR2-IPPO",
    "iql": "IQL",
    "pr2_iql": "PR2-IQL",
    "nashq": "Nash-Q",
    "hc_ippo": "hc-IPPO",
    "hc_pr2_ippo": "hc-PR2-IPPO",
    "hc_iql": "hc-IQL",
    "hc_pr2_iql": "hc-PR2-IQL",
    "hc_nashq": "hc-Nash-Q",
}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _family_certified(run: dict[str, Any], family_size: int) -> bool:
    """Joint certificate with each agent's budget split over ``family_size`` seed profiles."""
    per_agent = run["per_agent"]
    delta = float(run["delta"]) / (len(per_agent) * max(1, family_size))
    n = int(run["eval_episodes"])
    adjusted = {
        **run,
        "per_agent": {
            agent_id: {
                **estimate,
                "joint_ci": hoeffding_ci(float(estimate["p_hat"]), n, delta),
            }
            for agent_id, estimate in per_agent.items()
        },
    }
    return bool(
        certify_satisfaction_run(adjusted, run.get("probability_thresholds"))[
            "jointly_certified"
        ]
    )


def _profile_summary(
    algorithm_dir: Path, thresholds: Any = None
) -> dict[str, Any] | None:
    """Summarize one algorithm directory; ``thresholds=None`` keeps each report's own."""
    report = _read(algorithm_dir / "satisfaction_report.json")
    if report is None or not report.get("runs"):
        return None
    summary = summarize_satisfaction_runs(
        report["runs"], probability_thresholds=thresholds
    )
    runs = summary["runs"]
    certification = summary["certification"]
    nash = _read(algorithm_dir / "nash_report.json") or {}
    confirmatory = _read(algorithm_dir / "confirmatory_satisfaction_report.json")
    confirmatory_summary = (
        summarize_satisfaction_runs(
            confirmatory["runs"], probability_thresholds=thresholds
        )
        if confirmatory
        else None
    )
    return {
        "algorithm": algorithm_dir.name,
        "label": ALGORITHM_LABELS.get(algorithm_dir.name, algorithm_dir.name),
        "handcrafted": algorithm_dir.name.startswith("hc_"),
        "num_agents": len(runs[0]["per_agent"]),
        "num_runs": len(runs),
        "eval_episodes": int(runs[0]["eval_episodes"]),
        "probability_thresholds": runs[0]["probability_thresholds"],
        "num_jointly_certified_runs": int(certification["num_jointly_certified_runs"]),
        "jointly_certified_run_indices": certification["jointly_certified_run_indices"],
        "max_uniform_lower_threshold_by_run": [
            run["certification"]["max_uniform_lower_threshold"] for run in runs
        ],
        "joint_margin_by_run": [run["certification"]["joint_margin"] for run in runs],
        "p_hat_by_run": [
            {
                agent_id: float(estimate["p_hat"])
                for agent_id, estimate in run["per_agent"].items()
            }
            for run in runs
        ],
        "per_agent": summary["per_agent"],
        "exclusivity_violations": summary["metrics"].get("exclusivity_violations", {}),
        "mean_nashconv": nash.get("metrics", {}).get("mean_nashconv"),
        "confirmatory_num_jointly_certified_runs": (
            int(confirmatory_summary["certification"]["num_jointly_certified_runs"])
            if confirmatory_summary
            else None
        ),
        "confirmatory_max_uniform_lower_threshold_by_run": (
            [
                run["certification"]["max_uniform_lower_threshold"]
                for run in confirmatory_summary["runs"]
            ]
            if confirmatory_summary
            else None
        ),
        "confirmatory_joint_margin_by_run": (
            [
                run["certification"]["joint_margin"]
                for run in confirmatory_summary["runs"]
            ]
            if confirmatory_summary
            else None
        ),
        "_runs": runs,
    }


def summarize_exports(
    exports_root: Path, *, reported_thresholds: bool = False
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for rel_path, title in CURATED_TASKS.items():
        task_dir = exports_root / rel_path
        if not task_dir.is_dir():
            continue
        thresholds = (
            None if reported_thresholds else curated_probability_thresholds(rel_path)
        )
        profiles = [
            summary
            for algorithm_dir in sorted(
                path for path in task_dir.iterdir() if path.is_dir()
            )
            if (summary := _profile_summary(algorithm_dir, thresholds)) is not None
        ]
        if profiles:
            tasks[rel_path] = {"title": title, "profiles": profiles}

    task_family_sizes = {
        rel_path: sum(profile["num_runs"] for profile in task["profiles"])
        for rel_path, task in tasks.items()
    }
    campaign_family_size = sum(task_family_sizes.values())
    for rel_path, task in tasks.items():
        for profile in task["profiles"]:
            runs = profile.pop("_runs")
            profile["task_family_size"] = task_family_sizes[rel_path]
            profile["num_task_familywise_certified_runs"] = sum(
                _family_certified(run, task_family_sizes[rel_path]) for run in runs
            )
            profile["num_campaign_familywise_certified_runs"] = sum(
                _family_certified(run, campaign_family_size) for run in runs
            )
    return {
        "campaign_family_size": campaign_family_size,
        "threshold_source": "reported" if reported_thresholds else "curated",
        "tasks": tasks,
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"Campaign family size (seed profiles): {summary['campaign_family_size']}",
        "",
        f"Thresholds: {summary.get('threshold_source', 'reported')}",
        "",
        "| Task | Profile | q_i | Joint cert. | Task FWER | Campaign FWER | Confirm. | "
        "max margin | mean margin | max min_i L_i | ApxNashConv | Excl. viol. |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in summary["tasks"].values():
        for profile in task["profiles"]:
            uniform = [
                value
                for value in profile["max_uniform_lower_threshold_by_run"]
                if value is not None
            ]
            margins = profile["joint_margin_by_run"]
            thresholds = "/".join(
                f"{threshold['comparator']}{threshold['value']:g}"
                for threshold in profile["probability_thresholds"].values()
            )
            confirm = profile["confirmatory_num_jointly_certified_runs"]
            confirm_text = (
                "-" if confirm is None else f"{confirm}/{profile['num_runs']}"
            )
            violations = sum(profile["exclusivity_violations"].values())
            lines.append(
                f"| {task['title']} | {profile['label']} | {thresholds} | "
                f"{profile['num_jointly_certified_runs']}/{profile['num_runs']} | "
                f"{profile['num_task_familywise_certified_runs']}/{profile['num_runs']} | "
                f"{profile['num_campaign_familywise_certified_runs']}/{profile['num_runs']} | "
                f"{confirm_text} | "
                f"{_fmt(max(margins))} | {_fmt(sum(margins) / len(margins))} | "
                f"{_fmt(max(uniform) if uniform else None)} | "
                f"{_fmt(profile['mean_nashconv'])} | {violations} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--exports",
        type=Path,
        default=PROJECT_ROOT / "exports",
        help="exports root directory",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path for the full JSON summary",
    )
    parser.add_argument(
        "--reported-thresholds",
        action="store_true",
        help="certify against the thresholds recorded in each report instead of the curated ones",
    )
    args = parser.parse_args(argv)

    summary = summarize_exports(
        args.exports, reported_thresholds=args.reported_thresholds
    )
    if not summary["tasks"]:
        print(
            f"No curated satisfaction reports found under {args.exports}.",
            file=sys.stderr,
        )
        return 1
    print(render_markdown(summary))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
