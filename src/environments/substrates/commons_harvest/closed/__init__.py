"""Native Commons Harvest: Closed substrate."""

from __future__ import annotations

from src.environments.substrates.commons_harvest._core import ACTION_TO_COMPONENTS, AGENTS, Apple, Avatar
from src.environments.substrates.commons_harvest._variant_api import make_variant_api
from src.environments.substrates.commons_harvest._variants import CLOSED_ASCII_MAP

ASCII_MAP = CLOSED_ASCII_MAP
NUM_PLAYERS = 7

(
    CommonsHarvestClosedConfig,
    CommonsHarvestClosedParallelEnv,
    parallel_env,
    env,
) = make_variant_api(
    variant_name="closed",
    config_class_name="CommonsHarvestClosedConfig",
    env_class_name="CommonsHarvestClosedParallelEnv",
)

__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "Apple",
    "Avatar",
    "CommonsHarvestClosedConfig",
    "CommonsHarvestClosedParallelEnv",
    "NUM_PLAYERS",
    "env",
    "parallel_env",
]
