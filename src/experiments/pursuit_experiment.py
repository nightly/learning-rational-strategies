from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from src.environments.gridworlds.pursuit import PursuitEvasion
from src.experiments.temporal_game_experiment import TemporalGameExperimentSpec
from src.verify.certification import DEFAULT_PROBABILITY_THRESHOLD


WrapperFactory = Callable[[Any], Any]


PURSUIT_EXCLUSIVE_OBJECTIVE_GROUPS = (
    ("player_0", "player_1"),
    ("player_0", "player_2"),
)


@dataclass(frozen=True)
class PursuitExperimentSpec:
    """Pursuit wrapper and objective metadata for suite runners."""

    title: str
    wrapper_factory: WrapperFactory
    success_threshold: float = 0.5
    # Per-agent PLTLf thresholds; see src/experiments/objective_thresholds.py.
    probability_thresholds: Any = DEFAULT_PROBABILITY_THRESHOLD
    exclusive_objective_groups: tuple[tuple[str, ...], ...] = (
        PURSUIT_EXCLUSIVE_OBJECTIVE_GROUPS
    )


def _as_temporal_spec(spec: PursuitExperimentSpec) -> TemporalGameExperimentSpec:
    def env_factory(*, seed: int | None = None, num_agents: int = 2):
        base = PursuitEvasion(num_agents=num_agents, render_mode=None, seed=seed)
        env = spec.wrapper_factory(base)
        # The objective wrapper owns the episode horizon; align the native one so
        # the base env also ends the episode (and empties ``agents``) there.
        # Observations do not depend on ``max_steps``, so checkpoints are unaffected.
        horizon = int(getattr(env, "max_steps_per_episode", 0) or 0)
        if horizon > 0:
            base.max_steps = min(int(base.max_steps), horizon)
        return env

    return TemporalGameExperimentSpec(
        title=spec.title,
        env_factory=env_factory,
        success_threshold=spec.success_threshold,
        probability_thresholds=spec.probability_thresholds,
        exclusive_objective_groups=tuple(
            tuple(group) for group in spec.exclusive_objective_groups
        ),
    )
