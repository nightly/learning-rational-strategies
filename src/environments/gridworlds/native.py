from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

RGBPreprocessing = Literal["tile_scale", "native"]
_TILE_SCALE_FACTOR = 4


def native_observation_mode(fully_observable: bool, egocentric: bool = False) -> str:
    """Native substrate observation mode.

    Fully observable defaults to the world-frame ``"global"`` map. ``egocentric``
    selects ``"focal"`` instead: an agent-centred window covering the whole map
    from any position, so nearby objects stay on fixed inputs. Off by default so
    existing results stay reproducible; it has no effect on local views.
    """
    if not fully_observable:
        return "local"
    return "focal" if egocentric else "global"


def info_key(prefix: str, raw_key: str) -> str:
    return f"{prefix}.{str(raw_key).lower()}"


def _coerce_output_hw(output_hw: tuple[int, int] | None) -> tuple[int, int] | None:
    if output_hw is None:
        return None
    result = tuple(int(value) for value in output_hw)
    if len(result) != 2 or any(value <= 0 for value in result):
        raise ValueError(f"Expected a positive (height, width), got {output_hw!r}.")
    return result


def _process_rgb(rgb: np.ndarray, *, output_hw: tuple[int, int]) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape (H, W, 3), got {frame.shape}.")
    if frame.shape[:2] != output_hw:
        in_h, in_w, channels = frame.shape
        out_h, out_w = output_hw
        if in_h % out_h != 0 or in_w % out_w != 0:
            raise ValueError(f"Cannot average-pool RGB shape {frame.shape} to {output_hw}.")
        frame = frame.reshape(out_h, in_h // out_h, out_w, in_w // out_w, channels).mean(axis=(1, 3))
    return (frame / 255.0).astype(np.float32, copy=False)


def _coerce_side_channel(value: Any) -> float | np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        return float(array.item())
    return np.asarray(value, dtype=np.float32).copy()


def _augment_infos(
    observations: Mapping[str, Mapping[str, Any]],
    infos: Mapping[str, Mapping[str, Any]],
    *,
    info_key_prefix: str | None,
    scalar_and_vector_only: bool,
):
    augmented = {agent: dict(info) for agent, info in infos.items()}
    for agent, observation in observations.items():
        target = augmented.setdefault(agent, {})
        for key, value in observation.items():
            if key == "RGB" or isinstance(value, Mapping):
                continue
            if scalar_and_vector_only and np.asarray(value).ndim > 1:
                continue
            namespaced = info_key(info_key_prefix, key) if info_key_prefix else key.lower()
            target[namespaced] = _coerce_side_channel(value)
        if info_key_prefix:
            target.setdefault(info_key(info_key_prefix, "COLLECTIVE_REWARD"), 0.0)
    return augmented


def _set_collective_reward(infos, rewards, info_key_prefix: str | None) -> None:
    if not info_key_prefix:
        return
    collective_key = info_key(info_key_prefix, "COLLECTIVE_REWARD")
    collective_reward = float(sum(float(value) for value in rewards.values()))
    for info in infos.values():
        info[collective_key] = collective_reward


class NativeObjectObservationWrapper(BaseParallelWrapper):
    """Flatten compact native object fields into a normalized policy vector.

    Native substrates construct categorical grids directly from simulation
    state when rendering is disabled. This adapter preserves that no-render
    path while exposing the flat ``Box`` observations expected by the temporal
    monitor and neural learners.
    """

    def __init__(self, env, *, info_key_prefix: str | None = None):
        super().__init__(env)
        if getattr(env, "render_mode", None) is not None:
            raise ValueError("NativeObjectObservationWrapper requires render_mode=None.")
        self.info_key_prefix = info_key_prefix
        self._field_specs: dict[str, tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...]] = {}
        self._obs_spaces = {agent: self._build_obs_space(agent) for agent in env.possible_agents}

    def _build_obs_space(self, agent: str) -> gym.spaces.Box:
        source_space = self.env.observation_space(agent)
        if not isinstance(source_space, gym.spaces.Dict):
            raise TypeError("Object observation mode requires a Dict observation space.")

        fields = []
        output_lows = []
        output_highs = []
        for key, field_space in source_space.spaces.items():
            if not isinstance(field_space, gym.spaces.Box):
                raise TypeError(f"Object observation field {key!r} must be a Box space.")
            low = np.broadcast_to(field_space.low, field_space.shape).astype(np.float32, copy=True).reshape(-1)
            high = np.broadcast_to(field_space.high, field_space.shape).astype(np.float32, copy=True).reshape(-1)
            normalizable = np.isfinite(low) & np.isfinite(high) & (high > low)
            fields.append((key, low, high, normalizable))
            output_lows.append(np.where(normalizable, 0.0, low))
            output_highs.append(np.where(normalizable, 1.0, high))

        self._field_specs[agent] = tuple(fields)
        return gym.spaces.Box(
            low=np.concatenate(output_lows).astype(np.float32, copy=False),
            high=np.concatenate(output_highs).astype(np.float32, copy=False),
            dtype=np.float32,
        )

    def _transform_one(self, agent: str, observation: Mapping[str, Any]) -> np.ndarray:
        fields = []
        for key, low, high, normalizable in self._field_specs[agent]:
            value = np.asarray(observation[key], dtype=np.float32).reshape(-1)
            if value.shape != low.shape:
                raise ValueError(
                    f"Object observation field {key!r} for {agent!r} has shape {value.shape}; expected {low.shape}."
                )
            transformed = value.copy()
            transformed[normalizable] = (value[normalizable] - low[normalizable]) / (
                high[normalizable] - low[normalizable]
            )
            fields.append(transformed)
        return np.concatenate(fields).astype(np.float32, copy=False)

    def _transform(self, observations: Mapping[str, Mapping[str, Any]]):
        return {agent: self._transform_one(agent, observation) for agent, observation in observations.items()}

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        augmented_infos = _augment_infos(
            observations,
            infos,
            info_key_prefix=self.info_key_prefix,
            scalar_and_vector_only=True,
        )
        return self._transform(observations), augmented_infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        augmented_infos = _augment_infos(
            observations,
            infos,
            info_key_prefix=self.info_key_prefix,
            scalar_and_vector_only=True,
        )
        _set_collective_reward(augmented_infos, rewards, self.info_key_prefix)
        return (
            self._transform(observations),
            rewards,
            terminations,
            truncations,
            augmented_infos,
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_spaces[agent]


class NativeRGBObservationWrapper(BaseParallelWrapper):
    """Expose a native substrate's RGB field as a normalized policy tensor."""

    def __init__(
        self,
        env,
        *,
        fully_observable: bool = True,
        rgb_preprocessing: RGBPreprocessing = "tile_scale",
        output_hw: tuple[int, int] | None = None,
        info_key_prefix: str | None = None,
    ):
        super().__init__(env)
        self.fully_observable = bool(fully_observable)
        preprocessing = str(rgb_preprocessing).lower()
        if preprocessing not in {"tile_scale", "native"}:
            raise ValueError("rgb_preprocessing must be 'tile_scale' or 'native'.")
        self.rgb_preprocessing: RGBPreprocessing = preprocessing  # type: ignore[assignment]
        self.output_hw = _coerce_output_hw(output_hw)
        self.info_key_prefix = info_key_prefix
        self._obs_spaces = {agent: self._build_obs_space(agent) for agent in env.possible_agents}

    def _input_space(self, agent: str) -> gym.spaces.Box:
        if self.fully_observable:
            shape = tuple(int(value) for value in self.env.config.world_rgb_shape)
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        space = self.env.observation_space(agent)
        if not isinstance(space, gym.spaces.Dict) or "RGB" not in space.spaces:
            raise TypeError("Visual mode requires an RGB field in the native observation.")
        rgb_space = space["RGB"]
        if not isinstance(rgb_space, gym.spaces.Box):
            raise TypeError("The native RGB observation must be a Box space.")
        return rgb_space

    def _build_obs_space(self, agent: str) -> gym.spaces.Box:
        rgb_space = self._input_space(agent)
        in_h, in_w = (int(value) for value in rgb_space.shape[:2])
        output_hw = self.output_hw
        if output_hw is None:
            output_hw = (
                (in_h, in_w)
                if self.rgb_preprocessing == "native"
                else (in_h // _TILE_SCALE_FACTOR, in_w // _TILE_SCALE_FACTOR)
            )
        return gym.spaces.Box(0.0, 1.0, shape=(*output_hw, 3), dtype=np.float32)

    def _transform(self, observations: Mapping[str, Mapping[str, Any]]):
        if not observations:
            return {}
        world_rgb = self.env.state() if self.fully_observable else None
        transformed = {}
        for agent, observation in observations.items():
            rgb = world_rgb if world_rgb is not None else observation["RGB"]
            transformed[agent] = _process_rgb(np.asarray(rgb), output_hw=self._obs_spaces[agent].shape[:2]).copy()
        return transformed

    def _augment_infos(self, observations, infos):
        return _augment_infos(
            observations,
            infos,
            info_key_prefix=self.info_key_prefix,
            scalar_and_vector_only=False,
        )

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        return self._transform(observations), self._augment_infos(observations, infos)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        augmented_infos = self._augment_infos(observations, infos)
        _set_collective_reward(augmented_infos, rewards, self.info_key_prefix)
        return (
            self._transform(observations),
            rewards,
            terminations,
            truncations,
            augmented_infos,
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_spaces[agent]


def unwrap_native_env(env):
    """Walk PettingZoo wrappers to the local substrate implementation."""
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "config") and hasattr(current, "_step_count"):
            return current
        inner = getattr(current, "env", None)
        if inner is None or inner is current:
            break
        current = inner
    return None


def native_lifecycle(env) -> tuple[bool, bool]:
    """Return ``(timeout, terminal)`` from a wrapped native substrate."""
    native = unwrap_native_env(env)
    if native is None:
        return False, False
    max_steps = int(getattr(native.config, "max_episode_steps", 0))
    timeout = max_steps > 0 and int(native._step_count) >= max_steps
    return timeout, len(getattr(native, "agents", ())) == 0


def events_from_infos(infos: Any) -> list[tuple[str, Any]]:
    """Collect each native event once from per-agent info dictionaries."""
    events: list[tuple[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(infos, Mapping):
        return events
    for info in infos.values():
        if not isinstance(info, Mapping):
            continue
        for event in info.get("events", ()):
            if not isinstance(event, Mapping):
                continue
            marker = repr(sorted(event.items(), key=lambda item: str(item[0])))
            if marker in seen:
                continue
            seen.add(marker)
            events.append((str(event.get("event", "")), dict(event)))
    return events


__all__ = [
    "NativeObjectObservationWrapper",
    "NativeRGBObservationWrapper",
    "RGBPreprocessing",
    "info_key",
    "native_observation_mode",
    "events_from_infos",
    "native_lifecycle",
    "unwrap_native_env",
]
