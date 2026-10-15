from __future__ import annotations

from src.environments.gridworlds.native import (
    NativeObjectObservationWrapper,
    NativeRGBObservationWrapper,
    RGBPreprocessing,
    info_key,
    native_observation_mode,
)
from src.environments.substrates.territory import inside_out, open, rooms

TERRITORY_READY_TO_SHOOT_INFO_KEY = info_key("territory", "READY_TO_SHOOT")
TERRITORY_COLLECTIVE_REWARD_INFO_KEY = info_key(
    "territory",
    "COLLECTIVE_REWARD",
)


class TerritoryRGBObservationWrapper(NativeRGBObservationWrapper):
    """Territory-specific RGB adapter preserving side-channel infos for labels."""

    pass


def make_territory_env(
    *,
    seed: int | None = None,
    num_agents: int = 4,
    substrate_name: str = "territory__rooms",
    max_cycles: int = 1000,
    rgb_resize: tuple[int, int] | None = None,
    render_mode: str | None = None,
    fully_observable: bool = True,
    egocentric: bool = False,
    rgb_preprocessing: RGBPreprocessing = "tile_scale",
    flatten_observations: bool = False,
):
    """Build a native PettingZoo Territory substrate."""
    variants = {
        "territory__rooms": rooms,
        "territory__open": open,
        "territory__inside_out": inside_out,
    }
    try:
        variant = variants[str(substrate_name)]
    except KeyError as exc:
        raise ValueError(f"Unknown native Territory substrate: {substrate_name!r}") from exc
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
            return NativeObjectObservationWrapper(env, info_key_prefix="territory")
        return env
    if flatten_observations:
        raise ValueError("flatten_observations requires render_mode=None.")
    if fully_observable and egocentric:
        # The focal RGB window is the agent's own observation, not the world frame.
        return TerritoryRGBObservationWrapper(
            env,
            fully_observable=False,
            rgb_preprocessing=rgb_preprocessing,
            output_hw=rgb_resize,
            info_key_prefix="territory",
        )
    output_hw = rgb_resize if fully_observable else (rgb_resize or (22, 22))
    return TerritoryRGBObservationWrapper(
        env,
        fully_observable=fully_observable,
        rgb_preprocessing=rgb_preprocessing,
        output_hw=output_hw,
        info_key_prefix="territory",
    )


__all__ = [
    "TERRITORY_COLLECTIVE_REWARD_INFO_KEY",
    "TERRITORY_READY_TO_SHOOT_INFO_KEY",
    "TerritoryRGBObservationWrapper",
    "make_territory_env",
]
