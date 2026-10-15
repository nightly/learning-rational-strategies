"""Shared runtime and configuration contracts for PettingZoo RL trainers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import jax
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
from pettingzoo.utils.env import ParallelEnv

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


EnvFactory = Callable[[int], ParallelEnv]
UpdateCallback = Callable[[int, int, dict[str, Any]], None]


def coerce_seed(seed_or_key: Any) -> int:
    """Return a deterministic unsigned 32-bit seed from an int or JAX key."""

    if seed_or_key is None:
        return 0
    if isinstance(seed_or_key, (int, np.integer)):
        return int(seed_or_key) & 0xFFFFFFFF

    try:
        key_data = jax.random.key_data(seed_or_key)
    except (TypeError, ValueError):
        key_data = seed_or_key

    arr = np.asarray(key_data, dtype=np.uint32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected a non-empty seed or PRNG key.")
    if arr.size == 2 and int(arr[0]) == 0:
        return int(arr[1])
    if arr.size == 1:
        return int(arr[0])

    seed = 0
    for word in arr:
        seed = (seed * 1_664_525 + int(word) + 1_013_904_223) & 0xFFFFFFFF
    return int(seed)


def reset_env(env: ParallelEnv, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset a modern PettingZoo environment and normalize mapping outputs."""

    result = env.reset(seed=int(seed))
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("ParallelEnv.reset(seed=...) must return (observations, infos).")
    observations, infos = result
    if not isinstance(observations, Mapping) or not isinstance(infos, Mapping):
        raise TypeError("ParallelEnv.reset outputs must both be mappings.")
    return dict(observations), dict(infos)


def flatten_observation(observation: Any) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def canonicalize_observations(
    observations: Mapping[str, Any],
    agent_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    missing = [agent_id for agent_id in agent_ids if agent_id not in observations]
    if missing:
        raise KeyError(f"PettingZoo env did not return observations for agents: {missing}")
    return {agent_id: flatten_observation(observations[agent_id]) for agent_id in agent_ids}


def space_dimensions(
    env: ParallelEnv,
    agent_ids: Sequence[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate supported spaces and return per-agent flat dimensions."""

    obs_dims: dict[str, int] = {}
    action_dims: dict[str, int] = {}
    for agent_id in agent_ids:
        obs_space = env.observation_space(agent_id)
        action_space = env.action_space(agent_id)
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"RL trainers require Box observations; agent {agent_id!r} has {type(obs_space).__name__}.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(
                f"RL trainers require Discrete actions; agent {agent_id!r} has {type(action_space).__name__}."
            )
        obs_dims[agent_id] = int(gym.spaces.flatdim(obs_space))
        action_dims[agent_id] = int(action_space.n)
    return obs_dims, action_dims


def possible_agent_ids(env: ParallelEnv) -> tuple[str, ...]:
    agent_ids = tuple(str(agent_id) for agent_id in env.possible_agents)
    if not agent_ids:
        raise ValueError("ParallelEnv must expose at least one possible agent.")
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("ParallelEnv possible_agents must contain unique identifiers.")
    return agent_ids


def coerce_trainable_agent_ids(
    agent_ids: Sequence[str],
    trainable_agent_ids: set[str] | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if trainable_agent_ids is None:
        return tuple(agent_ids)
    trainable = tuple(str(agent_id) for agent_id in trainable_agent_ids)
    if len(set(trainable)) != len(trainable):
        raise ValueError("trainable_agent_ids must not contain duplicates.")
    unknown = sorted(set(trainable) - set(agent_ids))
    if unknown:
        raise KeyError(f"Unknown trainable_agent_ids: {unknown}")
    if not trainable:
        raise ValueError("Expected at least one trainable agent.")
    return trainable


def restore_train_state(
    template_state: TrainState,
    initial_state: TrainState | Mapping[str, Any],
) -> TrainState:
    state_dict = (
        serialization.to_state_dict(initial_state) if isinstance(initial_state, TrainState) else dict(initial_state)
    )
    return serialization.from_state_dict(template_state, state_dict)


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def close_env(env: ParallelEnv) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _find_history_owner(env: ParallelEnv) -> Any | None:
    current: Any = env
    visited: set[int] = set()
    for _ in range(64):
        if current is None or id(current) in visited:
            return None
        visited.add(id(current))
        if hasattr(current, "to_dicts") or hasattr(current, "history_by_agent"):
            return current
        next_env = getattr(current, "par_env", None)
        current = next_env if next_env is not None else getattr(current, "env", None)
    return None


def _serialize_history_entries(entries: Sequence[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            serialized.append(dict(entry))
        else:
            serialized.append(dict(getattr(entry, "__dict__", {})))
    return serialized


def extract_env_history(env: ParallelEnv) -> dict[str, list[dict[str, Any]]]:
    owner = _find_history_owner(env)
    if owner is None:
        return {}
    to_dicts = getattr(owner, "to_dicts", None)
    history = to_dicts() if callable(to_dicts) else getattr(owner, "history_by_agent", {})
    return {str(agent_id): _serialize_history_entries(list(entries)) for agent_id, entries in dict(history).items()}


def maybe_log_to_wandb(metric: Mapping[str, Any]) -> None:
    if wandb is None or wandb.run is None:
        return
    per_agent = dict(metric.get("per_agent_episode_return_mean", {}))
    payload = {key: value for key, value in metric.items() if key != "per_agent_episode_return_mean"}
    payload.update({f"per_agent_episode_return_mean/{agent_id}": value for agent_id, value in per_agent.items()})
    wandb.log(payload)


def require_config_keys(
    config: Mapping[str, Any],
    keys: Sequence[str],
    *,
    algorithm: str,
) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{algorithm} config is missing required keys: {missing}")


def integer_value(config: Mapping[str, Any], key: str, *, algorithm: str) -> int:
    raw_value = config[key]
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{algorithm} requires {key} to be an integer, got a boolean.")
    try:
        value = int(raw_value)
        numeric_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{algorithm} requires {key} to be an integer, got {raw_value!r}.") from exc
    if not np.isfinite(numeric_value) or numeric_value != value:
        raise ValueError(f"{algorithm} requires {key} to be an integer, got {raw_value!r}.")
    return value


def positive_int(config: Mapping[str, Any], key: str, *, algorithm: str) -> int:
    value = integer_value(config, key, algorithm=algorithm)
    if value <= 0:
        raise ValueError(f"{algorithm} requires {key} to be positive, got {value}.")
    return value


def nonnegative_int(config: Mapping[str, Any], key: str, *, algorithm: str) -> int:
    value = integer_value(config, key, algorithm=algorithm)
    if value < 0:
        raise ValueError(f"{algorithm} requires {key} to be nonnegative, got {value}.")
    return value


def finite_float(config: Mapping[str, Any], key: str, *, algorithm: str) -> float:
    value = float(config[key])
    if not np.isfinite(value):
        raise ValueError(f"{algorithm} requires finite {key}, got {value}.")
    return value


def positive_float(config: Mapping[str, Any], key: str, *, algorithm: str) -> float:
    value = finite_float(config, key, algorithm=algorithm)
    if value <= 0.0:
        raise ValueError(f"{algorithm} requires {key} to be positive, got {value}.")
    return value


def nonnegative_float(config: Mapping[str, Any], key: str, *, algorithm: str) -> float:
    value = finite_float(config, key, algorithm=algorithm)
    if value < 0.0:
        raise ValueError(f"{algorithm} requires {key} to be nonnegative, got {value}.")
    return value


def unit_interval_float(
    config: Mapping[str, Any],
    key: str,
    *,
    algorithm: str,
    include_zero: bool = True,
) -> float:
    value = finite_float(config, key, algorithm=algorithm)
    lower_ok = value >= 0.0 if include_zero else value > 0.0
    if not lower_ok or value > 1.0:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{algorithm} requires {key} in {interval}, got {value}.")
    return value


def validate_rollout_budget(config: Mapping[str, Any], *, algorithm: str) -> int:
    total = positive_int(config, "TOTAL_TIMESTEPS", algorithm=algorithm)
    num_steps = positive_int(config, "NUM_STEPS", algorithm=algorithm)
    num_envs = positive_int(config, "NUM_ENVS", algorithm=algorithm)
    batch_size = num_steps * num_envs
    if total < batch_size:
        raise ValueError(f"{algorithm} requires TOTAL_TIMESTEPS >= NUM_STEPS * NUM_ENVS ({batch_size}), got {total}.")
    if total % batch_size != 0:
        raise ValueError(
            f"{algorithm} requires TOTAL_TIMESTEPS to be divisible by NUM_STEPS * NUM_ENVS ({batch_size}), got {total}."
        )
    return total // batch_size


__all__ = [
    "EnvFactory",
    "UpdateCallback",
    "canonicalize_observations",
    "close_env",
    "coerce_seed",
    "coerce_trainable_agent_ids",
    "extract_env_history",
    "finite_float",
    "flatten_observation",
    "integer_value",
    "maybe_log_to_wandb",
    "mean_or_nan",
    "nonnegative_float",
    "nonnegative_int",
    "positive_float",
    "positive_int",
    "possible_agent_ids",
    "require_config_keys",
    "reset_env",
    "restore_train_state",
    "space_dimensions",
    "unit_interval_float",
    "validate_rollout_budget",
]
