from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from src.environments.gridworlds.territory import make_territory_env
from src.experiments.temporal_game_experiment import TemporalGameExperimentSpec
from src.verify.certification import DEFAULT_PROBABILITY_THRESHOLD


WrapperFactory = Callable[..., Any]


@dataclass(frozen=True)
class TerritoryExperimentSpec:
    """Territory wrapper and objective metadata for suite runners."""

    title: str
    wrapper_factory: WrapperFactory
    success_threshold: float = 0.5
    # Per-agent PLTLf thresholds; see src/experiments/objective_thresholds.py.
    probability_thresholds: Any = DEFAULT_PROBABILITY_THRESHOLD
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = ()


def _as_temporal_spec(spec: TerritoryExperimentSpec) -> TemporalGameExperimentSpec:
    def env_factory(*, seed: int | None = None, num_agents: int = 4, config=None):
        if config is None:
            raise ValueError(
                "Territory env factory expects the resolved experiment config."
            )

        base = make_territory_env(
            seed=seed,
            num_agents=num_agents,
            substrate_name=str(config["substrate_name"]),
            max_cycles=int(config["max_cycles"]),
            render_mode=None,
            flatten_observations=True,
        )
        return spec.wrapper_factory(
            base, max_steps_per_episode=int(config["max_cycles"])
        )

    return TemporalGameExperimentSpec(
        title=spec.title,
        env_factory=env_factory,
        success_threshold=spec.success_threshold,
        probability_thresholds=spec.probability_thresholds,
        extra_config_keys=("substrate_name", "max_cycles"),
        exclusive_objective_groups=tuple(
            tuple(group) for group in spec.exclusive_objective_groups
        ),
    )
