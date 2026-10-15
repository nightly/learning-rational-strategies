from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pettingzoo.utils.env import ParallelEnv

from src.monitor.reward_monitor import BoolMonitor

COUNTERFACTUAL_TRUE_PROPS_KEY = "ltlf_true_props"


@dataclass(frozen=True)
class CounterfactualMonitorModel:
    """Agent-specific DFA adapter used by counterfactual replay."""

    agent_id: str
    monitor: BoolMonitor

    @property
    def state_ids(self) -> tuple[int, ...]:
        return self.monitor.state_ids

    @property
    def initial_state_id(self) -> int:
        return self.monitor.initial_state_id()

    def successor_state_id(self, state_id: int, true_props: Iterable[str]) -> int:
        return self.monitor.successor_state_id(state_id, true_props)

    def is_accepting_state_id(self, state_id: int) -> bool:
        return self.monitor.is_accepting_state_id(state_id)


@dataclass(frozen=True)
class CounterfactualTrajectory:
    """Relabeled monitor-state trajectory for one factual rollout window."""

    factual: bool
    start_state_id: int | None
    obs_state_ids: np.ndarray
    next_obs_state_ids: np.ndarray
    rewards: np.ndarray


def split_monitor_state(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split augmented observations into base observations and monitor ids."""

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] <= 0:
        raise ValueError("Expected augmented observations with a monitor-state component.")

    base_obs = arr[..., :-1].astype(np.float32, copy=True)
    state_ids = np.rint(arr[..., -1]).astype(np.int32)
    return base_obs, state_ids


def append_monitor_state(
    base_obs: np.ndarray,
    state_ids: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Append monitor ids as the final feature of a base observation batch."""

    base_arr = np.asarray(base_obs, dtype=np.float32)
    if base_arr.ndim == 0:
        raise ValueError("Expected observation tensors with a feature dimension.")

    state_arr = np.asarray(state_ids, dtype=np.float32)
    try:
        state_arr = np.broadcast_to(state_arr, base_arr.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "Expected monitor-state ids to align with the observation batch shape."
        ) from exc

    return np.concatenate([base_arr, state_arr[..., None]], axis=-1).astype(
        np.float32,
        copy=False,
    )


def next_state_ids_from_final_state(
    obs_state_ids: np.ndarray | Sequence[int],
    final_next_state_id: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Build next-observation monitor ids from current ids and a final id."""

    state_ids = np.asarray(obs_state_ids, dtype=np.int32)
    if state_ids.ndim != 1:
        raise ValueError("Expected a 1D monitor-state sequence.")

    final_state_id = int(np.asarray(final_next_state_id, dtype=np.int32))
    return np.concatenate(
        [
            state_ids[1:],
            np.asarray([final_state_id], dtype=np.int32),
        ],
        axis=0,
    )


def _find_counterfactual_owner(env: ParallelEnv) -> Any | None:
    current = env
    visited: set[int] = set()

    for _ in range(64):
        if current is None:
            return None

        current_id = id(current)
        if current_id in visited:
            return None
        visited.add(current_id)

        if (
            hasattr(current, "supports_counterfactual_replay")
            and hasattr(current, "counterfactual_monitor")
        ):
            return current

        next_env = getattr(current, "par_env", None)
        if next_env is None:
            next_env = getattr(current, "env", None)
        current = next_env

    return None


def extract_counterfactual_models(
    env: ParallelEnv,
    agent_ids: Sequence[str],
) -> dict[str, CounterfactualMonitorModel]:
    """Find the reward wrapper and return replay-ready monitor models."""

    owner = _find_counterfactual_owner(env)
    if owner is None:
        raise ValueError(
            "COUNTERFACTUAL_REPLAY requires a BoolRewardWrapper with "
            "counterfactual monitor support."
        )

    if not bool(owner.supports_counterfactual_replay()):
        raise ValueError(
            "COUNTERFACTUAL_REPLAY requires temporally-extended monitor-state "
            "observations."
        )

    models: dict[str, CounterfactualMonitorModel] = {}
    for agent_id in agent_ids:
        models[str(agent_id)] = CounterfactualMonitorModel(
            agent_id=str(agent_id),
            monitor=owner.counterfactual_monitor(agent_id),
        )
    return models


def extract_counterfactual_true_props(
    infos: Mapping[str, Mapping[str, Any]] | None,
    agent_ids: Sequence[str],
) -> frozenset[str]:
    """Read the true LTLf propositions written into step infos."""

    if not infos:
        raise ValueError(
            "COUNTERFACTUAL_REPLAY expected step infos with monitor label metadata."
        )

    for agent_id in agent_ids:
        info = infos.get(agent_id)
        if info is None:
            continue
        raw = info.get(COUNTERFACTUAL_TRUE_PROPS_KEY)
        if raw is not None:
            return frozenset(str(item) for item in raw)

    raise ValueError(
        "COUNTERFACTUAL_REPLAY expected "
        f"'{COUNTERFACTUAL_TRUE_PROPS_KEY}' in step infos."
    )


def monitor_state_ids_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return only the monitor-state component of augmented observations."""

    _, state_ids = split_monitor_state(obs)
    return state_ids


def replace_monitor_state_ids(obs: np.ndarray, state_ids: np.ndarray | Sequence[int] | int) -> np.ndarray:
    """Return augmented observations with replacement monitor-state ids."""

    base_obs, _ = split_monitor_state(obs)
    return append_monitor_state(base_obs, state_ids)


def _trajectory_key(
    obs_state_ids: Sequence[int],
    next_obs_state_ids: Sequence[int],
    rewards: Sequence[float],
) -> tuple[int, ...]:
    reward_bits = tuple(int(float(reward) > 0.0) for reward in rewards)
    return tuple(int(state_id) for state_id in obs_state_ids) + (-1,) + tuple(
        int(state_id) for state_id in next_obs_state_ids
    ) + (-1,) + reward_bits


def enumerate_counterfactual_trajectories(
    *,
    model: CounterfactualMonitorModel,
    factual_obs_state_ids: Sequence[int],
    factual_next_obs_state_ids: Sequence[int],
    factual_rewards: Sequence[float],
    labels_sequence: Sequence[Iterable[str]],
    done: Sequence[float],
    reset: Sequence[float] | None = None,
) -> list[CounterfactualTrajectory]:
    """Enumerate unique LTLf monitor relabelings for a factual trajectory."""

    labels = [frozenset(str(item) for item in step_labels) for step_labels in labels_sequence]
    done_flags = np.asarray(done, dtype=np.float32).astype(bool)
    reset_flags = (
        np.asarray(reset, dtype=np.float32).astype(bool)
        if reset is not None
        else None
    )
    factual_obs = np.asarray(factual_obs_state_ids, dtype=np.int32)
    factual_next_obs = np.asarray(factual_next_obs_state_ids, dtype=np.int32)
    factual_reward_arr = np.asarray(factual_rewards, dtype=np.float32)

    if factual_obs.shape != factual_next_obs.shape:
        raise ValueError("Expected factual obs and next-obs state sequences to align.")
    if factual_obs.shape[0] != len(labels) or factual_obs.shape[0] != done_flags.shape[0]:
        raise ValueError("Expected label, done, and state-id sequences to have equal length.")
    if reset_flags is not None and reset_flags.shape[0] != factual_obs.shape[0]:
        raise ValueError("Expected reset masks to align with state-id sequences.")

    variants: list[CounterfactualTrajectory] = []
    seen: set[tuple[int, ...]] = set()

    factual_key = _trajectory_key(factual_obs, factual_next_obs, factual_reward_arr)
    seen.add(factual_key)
    variants.append(
        CounterfactualTrajectory(
            factual=True,
            start_state_id=None,
            obs_state_ids=factual_obs.copy(),
            next_obs_state_ids=factual_next_obs.copy(),
            rewards=factual_reward_arr.copy(),
        )
    )

    horizon = factual_obs.shape[0]
    for start_state_id in model.state_ids:
        obs_state_ids = np.empty((horizon,), dtype=np.int32)
        next_obs_state_ids = np.empty((horizon,), dtype=np.int32)
        rewards = np.zeros((horizon,), dtype=np.float32)

        current_state_id = int(start_state_id)
        for t in range(horizon):
            if t > 0 and reset_flags is not None and bool(reset_flags[t]):
                current_state_id = int(start_state_id)

            obs_state_ids[t] = current_state_id
            post_state_id = model.successor_state_id(current_state_id, labels[t])
            if done_flags[t]:
                rewards[t] = 1.0 if model.is_accepting_state_id(post_state_id) else 0.0
                current_state_id = int(start_state_id)
            else:
                current_state_id = int(post_state_id)
            next_obs_state_ids[t] = current_state_id

        variant_key = _trajectory_key(obs_state_ids, next_obs_state_ids, rewards)
        if variant_key in seen:
            continue
        seen.add(variant_key)
        variants.append(
            CounterfactualTrajectory(
                factual=False,
                start_state_id=int(start_state_id),
                obs_state_ids=obs_state_ids,
                next_obs_state_ids=next_obs_state_ids,
                rewards=rewards,
            )
        )

    return variants
