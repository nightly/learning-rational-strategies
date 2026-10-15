from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from src.environments.gridworlds.coins import make_coins_env
from src.experiments.temporal_game_experiment import TemporalGameExperimentSpec
from src.verify.certification import DEFAULT_PROBABILITY_THRESHOLD


WrapperFactory = Callable[..., Any]


@dataclass(frozen=True)
class CoinsExperimentSpec:
    """Coins wrapper and objective metadata for suite runners."""

    title: str
    wrapper_factory: WrapperFactory
    success_threshold: float = 0.5
    # Per-agent PLTLf thresholds; see src/experiments/objective_thresholds.py.
    probability_thresholds: Any = DEFAULT_PROBABILITY_THRESHOLD


def _as_temporal_spec(spec: CoinsExperimentSpec) -> TemporalGameExperimentSpec:
    def env_factory(*, seed: int | None = None, num_agents: int = 2, config=None):
        if config is None:
            raise ValueError(
                "Coins env factory expects the resolved experiment config."
            )

        base = make_coins_env(
            seed=seed,
            num_agents=num_agents,
            substrate_name=str(config["substrate_name"]),
            max_cycles=int(config["max_cycles"]),
            regrow_rate=float(config["regrow_rate"]),
            render_mode=None,
            fully_observable=bool(config.get("fully_observable", True)),
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
        extra_config_keys=(
            "substrate_name",
            "max_cycles",
            "regrow_rate",
            "fully_observable",
        ),
    )
