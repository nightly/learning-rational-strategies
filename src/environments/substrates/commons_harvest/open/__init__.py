"""Native Commons Harvest: Open substrate."""

from __future__ import annotations

from src.environments.substrates.commons_harvest._core import ACTION_TO_COMPONENTS, AGENTS, Apple, Avatar
from src.environments.substrates.commons_harvest._variant_api import make_variant_api
from src.environments.substrates.commons_harvest._variants import OPEN_ASCII_MAP

ASCII_MAP = OPEN_ASCII_MAP
NUM_PLAYERS = 7

(
    CommonsHarvestOpenConfig,
    CommonsHarvestOpenParallelEnv,
    parallel_env,
    env,
) = make_variant_api(
    variant_name="open",
    config_class_name="CommonsHarvestOpenConfig",
    env_class_name="CommonsHarvestOpenParallelEnv",
)

__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "Apple",
    "Avatar",
    "CommonsHarvestOpenConfig",
    "CommonsHarvestOpenParallelEnv",
    "NUM_PLAYERS",
    "env",
    "parallel_env",
]
