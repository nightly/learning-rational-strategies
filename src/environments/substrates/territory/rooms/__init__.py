"""Native Territory: Rooms substrate."""

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
from src.environments.substrates.territory._variants import ROOMS_ASCII_MAP

AGENTS: tuple[str, ...] = tuple(f"player_{index}" for index in range(9))
NUM_PLAYERS = 9
ASCII_MAP = ROOMS_ASCII_MAP


@dataclass(frozen=True)
class TerritoryRoomsConfig(TerritoryConfig):
    """Configuration for Territory: Rooms."""

    ascii_map: str = ASCII_MAP
    name: str = "rooms"
    num_players: int = NUM_PLAYERS
    topology: str = "TORUS"


class TerritoryRoomsParallelEnv(TerritoryParallelEnv):
    """PettingZoo ParallelEnv for Territory: Rooms."""

    metadata = {
        "name": "territory_rooms_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }


def parallel_env(
    render_mode: RenderMode = None,
    config: TerritoryRoomsConfig | None = None,
    **config_overrides: Any,
) -> TerritoryRoomsParallelEnv:
    """Create the native Territory: Rooms ParallelEnv."""
    return make_parallel_env(
        config_cls=TerritoryRoomsConfig,
        env_cls=TerritoryRoomsParallelEnv,
        render_mode=render_mode,
        config=config,
        **config_overrides,
    )


def env(render_mode: RenderMode = None, **kwargs: Any):
    """Create the native Territory: Rooms AEC env."""
    return make_aec_env(parallel_factory=parallel_env, render_mode=render_mode, **kwargs)


__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "NUM_PLAYERS",
    "TerritoryRoomsConfig",
    "TerritoryRoomsParallelEnv",
    "env",
    "parallel_env",
]
