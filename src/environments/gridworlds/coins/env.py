from __future__ import annotations

from src.environments.gridworlds.native import (
    NativeObjectObservationWrapper,
    NativeRGBObservationWrapper,
    RGBPreprocessing,
    info_key,
    native_observation_mode,
)
from src.environments.substrates.coins import coins

COINS_COLLECTIVE_REWARD_INFO_KEY = info_key("coins", "COLLECTIVE_REWARD")
COINS_PARTNER_MISMATCH_INFO_KEY = info_key(
    "coins",
    "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
)


class CoinsRGBObservationWrapper(NativeRGBObservationWrapper):
    """Coins-specific normalized RGB policy adapter."""


def make_coins_env(
    *,
    seed: int | None = None,
    num_agents: int = 2,
    substrate_name: str = "coins",
    max_cycles: int = 256,
    regrow_rate: float = 0.0005,
    min_map_size: int = 10,
    max_map_size: int = 15,
    rgb_resize: tuple[int, int] | None = None,
    render_mode: str | None = None,
    fully_observable: bool = True,
    egocentric: bool = False,
    rgb_preprocessing: RGBPreprocessing = "tile_scale",
    flatten_observations: bool = False,
):
    """Build the native two-player Coins substrate."""
    if str(substrate_name) != "coins":
        raise ValueError(f"Unknown native Coins substrate: {substrate_name!r}")
    if int(num_agents) != len(coins.AGENTS):
        raise ValueError(f"Coins requires {len(coins.AGENTS)} agents")

    env = coins.parallel_env(
        render_mode=render_mode,
        min_width=int(min_map_size),
        max_width=int(max_map_size),
        min_height=int(min_map_size),
        max_height=int(max_map_size),
        regrow_rate=float(regrow_rate),
        max_episode_steps=int(max_cycles),
        observation_mode=native_observation_mode(fully_observable, egocentric),
    )
    if seed is not None:
        env._rng.seed(seed)
    if render_mode is None:
        if flatten_observations:
            return NativeObjectObservationWrapper(env, info_key_prefix="coins")
        return env
    if flatten_observations:
        raise ValueError("flatten_observations requires render_mode=None.")

    if fully_observable and egocentric:
        # The focal RGB window is the agent's own observation, not the world frame.
        return CoinsRGBObservationWrapper(
            env,
            fully_observable=False,
            rgb_preprocessing=rgb_preprocessing,
            output_hw=rgb_resize,
            info_key_prefix="coins",
        )
    output_hw = rgb_resize if fully_observable else (rgb_resize or (22, 22))
    return CoinsRGBObservationWrapper(
        env,
        fully_observable=fully_observable,
        rgb_preprocessing=rgb_preprocessing,
        output_hw=output_hw,
        info_key_prefix="coins",
    )


__all__ = [
    "COINS_COLLECTIVE_REWARD_INFO_KEY",
    "COINS_PARTNER_MISMATCH_INFO_KEY",
    "CoinsRGBObservationWrapper",
    "make_coins_env",
]
