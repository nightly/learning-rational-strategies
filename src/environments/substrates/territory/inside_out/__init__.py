"""Native Territory: Inside Out substrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.environments.substrates.territory._core import (
    ACTION_TO_COMPONENTS,
    RandomChoiceSpec,
    RenderMode,
    TerritoryConfig,
    TerritoryParallelEnv,
    make_aec_env,
    make_parallel_env,
)
from src.environments.substrates.territory._variants import INSIDE_OUT_ASCII_MAP, INSIDE_OUT_RANDOM_CHOICES

AGENTS: tuple[str, ...] = tuple(f"player_{index}" for index in range(5))
NUM_PLAYERS = 5
ASCII_MAP = INSIDE_OUT_ASCII_MAP
RANDOM_CHOICES: tuple[RandomChoiceSpec, ...] = INSIDE_OUT_RANDOM_CHOICES


@dataclass(frozen=True)
class TerritoryInsideOutConfig(TerritoryConfig):
    """Configuration for Territory: Inside Out."""

    ascii_map: str = ASCII_MAP
    name: str = "inside_out"
    num_players: int = NUM_PLAYERS
    topology: str = "BOUNDED"
    random_choices: tuple[RandomChoiceSpec, ...] = RANDOM_CHOICES


class TerritoryInsideOutParallelEnv(TerritoryParallelEnv):
    """PettingZoo ParallelEnv for Territory: Inside Out."""

    metadata = {
        "name": "territory_inside_out_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }


def parallel_env(
    render_mode: RenderMode = None,
    config: TerritoryInsideOutConfig | None = None,
    **config_overrides: Any,
) -> TerritoryInsideOutParallelEnv:
    """Create the native Territory: Inside Out ParallelEnv."""
    return make_parallel_env(
        config_cls=TerritoryInsideOutConfig,
        env_cls=TerritoryInsideOutParallelEnv,
        render_mode=render_mode,
        config=config,
        **config_overrides,
    )


def env(render_mode: RenderMode = None, **kwargs: Any):
    """Create the native Territory: Inside Out AEC env."""
    return make_aec_env(parallel_factory=parallel_env, render_mode=render_mode, **kwargs)


__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "NUM_PLAYERS",
    "RANDOM_CHOICES",
    "TerritoryInsideOutConfig",
    "TerritoryInsideOutParallelEnv",
    "env",
    "parallel_env",
]
