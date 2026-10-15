from __future__ import annotations

from src.environments.gridworlds.native import (
    NativeObjectObservationWrapper,
    NativeRGBObservationWrapper,
    RGBPreprocessing,
    info_key,
    native_observation_mode,
)
from src.environments.substrates.commons_harvest import closed, open, partnership

HARVEST_COLLECTIVE_REWARD_INFO_KEY = info_key(
    "harvest",
    "COLLECTIVE_REWARD",
)
HARVEST_READY_TO_SHOOT_INFO_KEY = info_key("harvest", "READY_TO_SHOOT")


class HarvestRGBObservationWrapper(NativeRGBObservationWrapper):
    """Harvest-specific normalized RGB policy adapter."""

    pass


def make_harvest_env(
    *,
    seed: int | None = None,
    num_agents: int = 4,
    substrate_name: str = "commons_harvest__partnership",
    max_cycles: int = 1000,
    rgb_resize: tuple[int, int] | None = None,
    render_mode: str | None = None,
    fully_observable: bool = True,
    egocentric: bool = False,
    rgb_preprocessing: RGBPreprocessing = "tile_scale",
    flatten_observations: bool = False,
):
    """Build a native PettingZoo Commons Harvest substrate."""
    variants = {
        "commons_harvest__open": open,
        "commons_harvest__closed": closed,
        "commons_harvest__partnership": partnership,
    }
    try:
        variant = variants[str(substrate_name)]
    except KeyError as exc:
        raise ValueError(f"Unknown native Commons Harvest substrate: {substrate_name!r}") from exc
    env = variant.parallel_env(
        render_mode=render_mode,
        num_players=int(num_agents),
        max_episode_steps=int(max_cycles),
        observation_mode=native_observation_mode(fully_observable, egocentric),
    )
    if seed is not None:
        env._rng.seed(seed)
    if render_mode is None:
        if flatten_observations:
            return NativeObjectObservationWrapper(env, info_key_prefix="harvest")
        return env
    if flatten_observations:
        raise ValueError("flatten_observations requires render_mode=None.")
    if fully_observable and egocentric:
        # The focal RGB window is the agent's own observation, not the world frame.
        return HarvestRGBObservationWrapper(
            env,
            fully_observable=False,
            rgb_preprocessing=rgb_preprocessing,
            output_hw=rgb_resize,
            info_key_prefix="harvest",
        )
    output_hw = rgb_resize if fully_observable else (rgb_resize or (22, 22))
    return HarvestRGBObservationWrapper(
        env,
        fully_observable=fully_observable,
        rgb_preprocessing=rgb_preprocessing,
        output_hw=output_hw,
        info_key_prefix="harvest",
    )


__all__ = [
    "HARVEST_COLLECTIVE_REWARD_INFO_KEY",
    "HARVEST_READY_TO_SHOOT_INFO_KEY",
    "HarvestRGBObservationWrapper",
    "make_harvest_env",
]
