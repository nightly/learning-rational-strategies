"""Native Territory: Open substrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.environments.substrates.territory._core import (
    ACTION_TO_COMPONENTS,
    RenderMode,
    TerritoryConfig,
    TerritoryParallelEnv,
    make_aec_env,
    make_parallel_env,
)
from src.environments.substrates.territory._variants import OPEN_ASCII_MAP

AGENTS: tuple[str, ...] = tuple(f"player_{index}" for index in range(9))
NUM_PLAYERS = 9
ASCII_MAP = OPEN_ASCII_MAP


@dataclass(frozen=True)
class TerritoryOpenConfig(TerritoryConfig):
    """Configuration for Territory: Open."""

    ascii_map: str = ASCII_MAP
    name: str = "open"
    num_players: int = NUM_PLAYERS
    topology: str = "BOUNDED"


class TerritoryOpenParallelEnv(TerritoryParallelEnv):
    """PettingZoo ParallelEnv for Territory: Open."""

    metadata = {
        "name": "territory_open_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }


def parallel_env(
    render_mode: RenderMode = None,
    config: TerritoryOpenConfig | None = None,
    **config_overrides: Any,
) -> TerritoryOpenParallelEnv:
    """Create the native Territory: Open ParallelEnv."""
    return make_parallel_env(
        config_cls=TerritoryOpenConfig,
        env_cls=TerritoryOpenParallelEnv,
        render_mode=render_mode,
        config=config,
        **config_overrides,
    )


def env(render_mode: RenderMode = None, **kwargs: Any):
    """Create the native Territory: Open AEC env."""
    return make_aec_env(parallel_factory=parallel_env, render_mode=render_mode, **kwargs)


__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "NUM_PLAYERS",
    "TerritoryOpenConfig",
    "TerritoryOpenParallelEnv",
    "env",
    "parallel_env",
]
