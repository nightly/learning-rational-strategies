"""Native Commons Harvest: Partnership substrate."""

from __future__ import annotations

from src.environments.substrates.commons_harvest._core import ACTION_TO_COMPONENTS, AGENTS, Apple, Avatar
from src.environments.substrates.commons_harvest._variant_api import make_variant_api
from src.environments.substrates.commons_harvest._variants import PARTNERSHIP_ASCII_MAP

ASCII_MAP = PARTNERSHIP_ASCII_MAP
NUM_PLAYERS = 7

(
    CommonsHarvestPartnershipConfig,
    CommonsHarvestPartnershipParallelEnv,
    parallel_env,
    env,
) = make_variant_api(
    variant_name="partnership",
    config_class_name="CommonsHarvestPartnershipConfig",
    env_class_name="CommonsHarvestPartnershipParallelEnv",
)

__all__ = [
    "ACTION_TO_COMPONENTS",
    "AGENTS",
    "ASCII_MAP",
    "Apple",
    "Avatar",
    "CommonsHarvestPartnershipConfig",
    "CommonsHarvestPartnershipParallelEnv",
    "NUM_PLAYERS",
    "env",
    "parallel_env",
]
