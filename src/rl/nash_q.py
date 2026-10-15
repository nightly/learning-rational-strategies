"""
PettingZoo-compatible tabular Nash-Q trainer.

This module mirrors the broad trainer contract used by ``src/rl/ippo.py`` and
``src/rl/iql.py`` while keeping one tabular joint-action value function per
agent. Stage-game equilibria are solved from those joint payoff tables with
``pygambit``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import pygambit
from pettingzoo.utils.env import ParallelEnv

from src.monitor.counterfactual import (
    CounterfactualMonitorModel,
    append_monitor_state,
    enumerate_counterfactual_trajectories,
    extract_counterfactual_models,
    extract_counterfactual_true_props,
    split_monitor_state,
)
from src.rl.common import (
    EnvFactory,
    UpdateCallback,
    close_env,
    coerce_seed,
    coerce_trainable_agent_ids,
    extract_env_history,
    maybe_log_to_wandb,
    mean_or_nan,
    reset_env,
)

StateKey = tuple[bytes, ...]
PolicyProfile = tuple[np.ndarray, ...]
QTableByState = dict[StateKey, np.ndarray]
VisitCountByState = dict[StateKey, np.ndarray]
Snapshot = dict[str, Any]


class SolverOutcome(NamedTuple):
    policies: PolicyProfile
    solver: str


class CounterfactualJointStep(NamedTuple):
    state_key: StateKey
    next_state_key: StateKey | None
    rewards: tuple[float, ...]


def _coerce_equilibrium_selection(selection: str | None) -> str:
    value = "payoff_dominant" if selection is None else str(selection).lower()
    if value not in {"first", "payoff_dominant"}:
        raise ValueError("EQUILIBRIUM_SELECTION must be one of {'first', 'payoff_dominant'}.")
    return value


def _action_specs(
    env: ParallelEnv,
    agent_ids: list[str],
) -> dict[str, int]:
    action_dims: dict[str, int] = {}

    for agent_id in agent_ids:
        action_space = env.action_space(agent_id)
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(
                f"Nash-Q only supports Discrete actions in this pass, got "
                f"{type(action_space).__name__} for agent '{agent_id}'."
            )
        action_dims[agent_id] = int(action_space.n)

    return action_dims


def _obs_bytes(obs: Any) -> bytes:
    arr = np.asarray(obs, dtype=np.float32)
    shape = np.asarray(arr.shape, dtype=np.int32)
    return shape.tobytes() + arr.reshape(-1).tobytes()


def joint_observation_key(
    obs: Mapping[str, Any],
    agent_ids: Sequence[str],
) -> StateKey:
    missing = [agent_id for agent_id in agent_ids if agent_id not in obs]
    if missing:
        raise KeyError(f"PettingZoo env did not return observations for agents: {missing}")
    return tuple(_obs_bytes(obs[agent_id]) for agent_id in agent_ids)


def _normalize_policy(
    probs: Sequence[float] | np.ndarray,
    *,
    action_dim: int,
) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if arr.size != int(action_dim):
        raise ValueError(f"Expected {action_dim} action probabilities, got {arr.size}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Expected a policy with positive probability mass.")
    return arr / total


def _copy_q_values(q_values: Mapping[StateKey, Any]) -> QTableByState:
    return {tuple(state_key): np.asarray(value, dtype=np.float32).copy() for state_key, value in q_values.items()}


def _copy_visit_counts(visit_counts: Mapping[StateKey, Any]) -> VisitCountByState:
    return {tuple(state_key): np.asarray(value, dtype=np.int64).copy() for state_key, value in visit_counts.items()}


def _restore_snapshot(initial_state: Mapping[str, Any]) -> Snapshot:
    q_values = _copy_q_values(initial_state.get("q_values", {}))
    visit_counts = _copy_visit_counts(initial_state.get("visit_counts", {}))
    return {
        "q_values": q_values,
        "visit_counts": visit_counts,
        "timesteps": int(initial_state.get("timesteps", 0)),
        "n_updates": int(initial_state.get("n_updates", 0)),
    }


def _ensure_state_tables(
    q_tables: Mapping[str, QTableByState],
    visit_counts: VisitCountByState,
    *,
    state_key: StateKey,
    agent_ids: Sequence[str],
    action_shape: Sequence[int],
) -> None:
    if state_key not in visit_counts:
        visit_counts[state_key] = np.zeros(tuple(action_shape), dtype=np.int64)

    for agent_id in agent_ids:
        agent_table = q_tables[agent_id]
        if state_key not in agent_table:
            agent_table[state_key] = np.zeros(tuple(action_shape), dtype=np.float32)


def _extract_profile_policies(
    game: Any,
    profile: Any,
    *,
    action_shape: Sequence[int],
) -> PolicyProfile:
    players = list(game.players)
    if len(players) != len(action_shape):
        raise ValueError(f"Expected {len(action_shape)} players, got {len(players)} in profile.")

    def _mixed_strategy_probs(mixed_strategy: Any) -> np.ndarray:
        values = []
        for entry in mixed_strategy:
            if isinstance(entry, tuple) and len(entry) == 2:
                _, prob = entry
            else:
                prob = entry
            values.append(float(prob))
        return np.asarray(values, dtype=np.float64)

    by_label: dict[str, np.ndarray] = {}
    if hasattr(profile, "mixed_strategies"):
        for player, mixed_strategy in profile.mixed_strategies():
            label = str(getattr(player, "label", ""))
            by_label[label] = _mixed_strategy_probs(mixed_strategy)

    policies: list[np.ndarray] = []
    for player_index, player in enumerate(players):
        label = str(getattr(player, "label", ""))
        probs = by_label.get(label)

        if probs is None:
            strategies = list(getattr(player, "strategies", []))
            if strategies:
                probs = np.asarray(
                    [float(profile[strategy]) for strategy in strategies],
                    dtype=np.float64,
                )

        if probs is None:
            raise ValueError(f"Unable to extract mixed strategy for player index {player_index}.")

        policies.append(_normalize_policy(probs, action_dim=int(action_shape[player_index])))

    return tuple(policies)


def _expected_value_under_policies(
    q_tensor: np.ndarray,
    policies: PolicyProfile,
) -> float:
    reduced = np.asarray(q_tensor, dtype=np.float64)
    for axis in range(reduced.ndim - 1, -1, -1):
        reduced = np.tensordot(reduced, policies[axis], axes=([axis], [0]))
    return float(reduced)


def _select_stage_game_outcome(
    game: Any,
    equilibria: Sequence[Any],
    *,
    action_shape: Sequence[int],
    q_tensors: Sequence[np.ndarray],
    equilibrium_selection: str,
    solver: str,
) -> SolverOutcome | None:
    if not equilibria:
        return None

    if equilibrium_selection == "first":
        return SolverOutcome(
            _extract_profile_policies(
                game,
                equilibria[0],
                action_shape=action_shape,
            ),
            solver,
        )

    best: tuple[tuple[float, float, int], PolicyProfile] | None = None
    for equilibrium_index, equilibrium in enumerate(equilibria):
        policies = _extract_profile_policies(
            game,
            equilibrium,
            action_shape=action_shape,
        )
        payoff_values = tuple(_expected_value_under_policies(q_tensor, policies) for q_tensor in q_tensors)
        score = (
            float(sum(payoff_values)),
            float(min(payoff_values)),
            -int(equilibrium_index),
        )
        if best is None or score > best[0]:
            best = (score, policies)

    if best is None:
        return None
    return SolverOutcome(best[1], solver)


def _solve_stage_game(
    q_tensors: Sequence[np.ndarray],
    *,
    equilibrium_selection: str = "payoff_dominant",
) -> SolverOutcome:
    if not q_tensors:
        raise ValueError("Expected at least one payoff tensor.")

    equilibrium_selection = _coerce_equilibrium_selection(equilibrium_selection)

    try:
        action_shape = tuple(int(size) for size in q_tensors[0].shape)
        game = pygambit.Game.from_arrays(
            *(np.asarray(q_tensor, dtype=np.float64) for q_tensor in q_tensors),
            title="Nash-Q stage game",
        )
    except Exception as exc:
        raise RuntimeError("Failed to build the Nash-Q stage game for pygambit.") from exc

    pure_error: Exception | None = None
    try:
        pure_result = pygambit.nash.enumpure_solve(game)
        pure_equilibria = list(getattr(pure_result, "equilibria", []))
        pure_outcome = _select_stage_game_outcome(
            game,
            pure_equilibria,
            action_shape=action_shape,
            q_tensors=q_tensors,
            equilibrium_selection=equilibrium_selection,
            solver="enumpure",
        )
        if pure_outcome is not None:
            return pure_outcome
    except Exception as exc:
        pure_error = exc

    try:
        poly_result = pygambit.nash.enumpoly_solve(
            game,
            use_strategic=True,
            stop_after=1,
        )
        poly_equilibria = list(getattr(poly_result, "equilibria", []))
        poly_outcome = _select_stage_game_outcome(
            game,
            poly_equilibria,
            action_shape=action_shape,
            q_tensors=q_tensors,
            equilibrium_selection=equilibrium_selection,
            solver="enumpoly",
        )
        if poly_outcome is not None:
            return poly_outcome
    except Exception as exc:
        message = "pygambit failed to solve the Nash-Q stage game."
        if pure_error is not None:
            message += f" enumpure_solve also failed with {type(pure_error).__name__}: {pure_error}"
        raise RuntimeError(message) from exc

    raise RuntimeError("pygambit did not return an equilibrium for the Nash-Q stage game.")


def _expected_action_values(
    q_tensor: np.ndarray,
    *,
    agent_index: int,
    policies: PolicyProfile,
) -> np.ndarray:
    moved = np.moveaxis(np.asarray(q_tensor, dtype=np.float64), agent_index, 0)
    opponent_policies = [
        np.asarray(policy, dtype=np.float64) for idx, policy in enumerate(policies) if idx != agent_index
    ]
    result = moved
    for axis in range(result.ndim - 1, 0, -1):
        result = np.tensordot(result, opponent_policies[axis - 1], axes=([axis], [0]))
    return np.asarray(result, dtype=np.float64)


def _best_response_policy(
    q_tensor: np.ndarray,
    *,
    agent_index: int,
    policies: PolicyProfile,
) -> np.ndarray:
    action_values = _expected_action_values(
        q_tensor,
        agent_index=agent_index,
        policies=policies,
    )
    best_value = float(np.max(action_values))
    best_actions = np.flatnonzero(np.isclose(action_values, best_value))
    probs = np.zeros_like(action_values, dtype=np.float64)
    if best_actions.size == 0:
        probs.fill(1.0 / max(1, probs.size))
        return probs
    probs[best_actions] = 1.0 / float(best_actions.size)
    return probs


def _best_response_value(
    q_tensor: np.ndarray,
    *,
    agent_index: int,
    policies: PolicyProfile,
) -> float:
    action_values = _expected_action_values(
        q_tensor,
        agent_index=agent_index,
        policies=policies,
    )
    return float(np.max(action_values))


def _sample_action(
    policy: np.ndarray,
    *,
    rng: np.random.Generator,
) -> int:
    probs = _normalize_policy(policy, action_dim=int(policy.shape[0]))
    return int(rng.choice(np.arange(probs.shape[0]), p=probs))


def _epsilon_schedule(config: Mapping[str, Any], update_index: int) -> float:
    start = float(config["EPS_START"])
    finish = float(config["EPS_FINISH"])
    schedule_update_index = max(0, int(config.get("SCHEDULE_UPDATE_OFFSET", 0))) + int(update_index)
    num_updates = max(1, int(config.get("SCHEDULE_TOTAL_UPDATES", config["NUM_UPDATES"])))
    transition_steps = max(
        1,
        int(float(config["EPS_DECAY"]) * num_updates),
    )
    progress = min(1.0, float(schedule_update_index) / float(transition_steps))
    return float(start + (finish - start) * progress)


def _policy_for_state(
    state_key: StateKey,
    *,
    q_tables: Mapping[str, QTableByState],
    visit_counts: VisitCountByState,
    agent_ids: Sequence[str],
    action_shape: Sequence[int],
    cache: dict[StateKey, SolverOutcome],
    equilibrium_selection: str = "payoff_dominant",
) -> SolverOutcome:
    cached = cache.get(state_key)
    if cached is not None:
        return cached

    _ensure_state_tables(
        q_tables,
        visit_counts,
        state_key=state_key,
        agent_ids=agent_ids,
        action_shape=action_shape,
    )
    q_tensors = [q_tables[agent_id][state_key] for agent_id in agent_ids]
    outcome = _solve_stage_game(q_tensors, equilibrium_selection=equilibrium_selection)
    cache[state_key] = outcome
    return outcome


def _enumerate_joint_counterfactual_steps(
    *,
    agent_ids: Sequence[str],
    models: Mapping[str, CounterfactualMonitorModel],
    current_obs: Mapping[str, Any],
    next_obs: Mapping[str, Any],
    rewards: Mapping[str, float],
    labels: frozenset[str],
    done: bool,
    max_joint_variants: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[CounterfactualJointStep]:
    per_agent_variants: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
    for agent_id in agent_ids:
        model = models[str(agent_id)]
        current_agent_obs = np.asarray(current_obs[agent_id], dtype=np.float32)
        next_agent_obs = np.asarray(next_obs[agent_id], dtype=np.float32)
        current_agent_base_obs, current_agent_state_id = split_monitor_state(current_agent_obs)
        next_agent_base_obs, next_agent_state_id = split_monitor_state(next_agent_obs)
        factual_reward = float(rewards.get(agent_id, 0.0))
        trajectories = enumerate_counterfactual_trajectories(
            model=model,
            factual_obs_state_ids=np.asarray(
                [int(current_agent_state_id)],
                dtype=np.int32,
            ),
            factual_next_obs_state_ids=np.asarray(
                [int(next_agent_state_id)],
                dtype=np.int32,
            ),
            factual_rewards=np.asarray([factual_reward], dtype=np.float32),
            labels_sequence=[labels],
            done=np.asarray([1.0 if done else 0.0], dtype=np.float32),
            reset=np.asarray([0.0], dtype=np.float32),
        )

        agent_variants: list[tuple[np.ndarray, np.ndarray, float]] = []
        for trajectory in trajectories:
            if trajectory.factual:
                variant_obs = current_agent_obs.astype(np.float32, copy=True)
                variant_next_obs = next_agent_obs.astype(np.float32, copy=True)
                variant_reward = factual_reward
            else:
                variant_obs = append_monitor_state(
                    current_agent_base_obs,
                    int(trajectory.obs_state_ids[0]),
                )
                variant_next_obs = append_monitor_state(
                    next_agent_base_obs,
                    int(trajectory.next_obs_state_ids[0]),
                )
                variant_reward = float(trajectory.rewards[0])
            agent_variants.append((variant_obs, variant_next_obs, variant_reward))
        per_agent_variants.append(agent_variants)

    variants: list[CounterfactualJointStep] = []
    seen: set[tuple[StateKey, StateKey | None, tuple[float, ...]]] = set()

    def _append_combo(combo: Sequence[tuple[np.ndarray, np.ndarray, float]]) -> None:
        variant_obs = {agent_id: combo[agent_index][0] for agent_index, agent_id in enumerate(agent_ids)}
        variant_rewards = tuple(float(combo[agent_index][2]) for agent_index in range(len(agent_ids)))
        state_key = joint_observation_key(variant_obs, agent_ids)
        if done:
            next_state_key = None
        else:
            variant_next_obs = {agent_id: combo[agent_index][1] for agent_index, agent_id in enumerate(agent_ids)}
            next_state_key = joint_observation_key(variant_next_obs, agent_ids)

        variant_key = (state_key, next_state_key, variant_rewards)
        if variant_key in seen:
            return
        seen.add(variant_key)
        variants.append(
            CounterfactualJointStep(
                state_key=state_key,
                next_state_key=next_state_key,
                rewards=variant_rewards,
            )
        )

    cap = None if max_joint_variants is None or int(max_joint_variants) <= 0 else int(max_joint_variants)
    total_joint_variants = 1
    for agent_variants in per_agent_variants:
        total_joint_variants *= len(agent_variants)

    if cap is None or total_joint_variants <= cap:
        for combo in product(*per_agent_variants):
            _append_combo(combo)
        return variants

    factual_combo = tuple(agent_variants[0] for agent_variants in per_agent_variants)
    _append_combo(factual_combo)

    if cap <= 1:
        return variants

    sample_rng = rng if rng is not None else np.random.default_rng(0)
    max_attempts = max(cap * 50, cap + 1)
    attempts = 0
    while len(variants) < cap and attempts < max_attempts:
        attempts += 1
        combo = tuple(
            agent_variants[int(sample_rng.integers(0, len(agent_variants)))] for agent_variants in per_agent_variants
        )
        _append_combo(combo)

    return variants


def make_train(config: dict[str, Any], env_factory: EnvFactory):
    config = dict(config)
    config["COUNTERFACTUAL_REPLAY"] = bool(config.get("COUNTERFACTUAL_REPLAY", False))
    config["COUNTERFACTUAL_MAX_JOINT_VARIANTS"] = config.get(
        "COUNTERFACTUAL_MAX_JOINT_VARIANTS",
        1024,
    )
    config["EQUILIBRIUM_SELECTION"] = _coerce_equilibrium_selection(
        config.get("EQUILIBRIUM_SELECTION", "payoff_dominant")
    )

    probe_seed = coerce_seed(config.get("SEED", 0))
    probe_env = env_factory(probe_seed)
    try:
        agent_ids = list(getattr(probe_env, "possible_agents", []))
        if not agent_ids:
            raise ValueError("PettingZoo env must expose a non-empty possible_agents list.")
        action_dims = _action_specs(probe_env, agent_ids)
        counterfactual_models = (
            extract_counterfactual_models(probe_env, agent_ids) if config["COUNTERFACTUAL_REPLAY"] else {}
        )
    finally:
        close_env(probe_env)

    action_shape = tuple(action_dims[agent_id] for agent_id in agent_ids)
    config["NUM_AGENTS"] = len(agent_ids)
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    if int(config["NUM_UPDATES"]) <= 0:
        raise ValueError("TOTAL_TIMESTEPS is too small for even one Nash-Q update.")
    config["SCHEDULE_UPDATE_OFFSET"] = max(
        0,
        int(config.get("SCHEDULE_UPDATE_OFFSET", 0)),
    )
    config["SCHEDULE_TOTAL_UPDATES"] = max(
        1,
        int(config.get("SCHEDULE_TOTAL_UPDATES", config["NUM_UPDATES"])),
    )

    def train(
        seed: Any | None = None,
        *,
        initial_train_states: Mapping[str, Mapping[str, Any]] | None = None,
        trainable_agent_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        on_update: UpdateCallback | None = None,
        capture_env_histories: bool = False,
    ):
        base_seed = coerce_seed(seed if seed is not None else config.get("SEED", 0))
        rng = np.random.default_rng(base_seed)
        active_trainable_agent_ids = coerce_trainable_agent_ids(
            agent_ids,
            trainable_agent_ids,
        )
        active_trainable_set = set(active_trainable_agent_ids)

        q_tables: dict[str, QTableByState] = {}
        visit_counts: VisitCountByState
        timesteps_by_agent: dict[str, int] = {}
        n_updates_by_agent: dict[str, int] = {}

        if initial_train_states is None:
            q_tables = {agent_id: {} for agent_id in agent_ids}
            visit_counts = {}
            timesteps_by_agent = {agent_id: 0 for agent_id in agent_ids}
            n_updates_by_agent = {agent_id: 0 for agent_id in agent_ids}
        else:
            missing = [agent_id for agent_id in agent_ids if agent_id not in initial_train_states]
            if missing:
                raise KeyError(f"Missing initial_train_states entry for agent(s): {missing}")
            restored = {agent_id: _restore_snapshot(initial_train_states[agent_id]) for agent_id in agent_ids}
            q_tables = {agent_id: restored[agent_id]["q_values"] for agent_id in agent_ids}
            visit_counts = _copy_visit_counts(restored[agent_ids[0]]["visit_counts"])
            timesteps_by_agent = {agent_id: int(restored[agent_id]["timesteps"]) for agent_id in agent_ids}
            n_updates_by_agent = {agent_id: int(restored[agent_id]["n_updates"]) for agent_id in agent_ids}

        best_response_mode = len(active_trainable_agent_ids) == 1
        br_agent_id = active_trainable_agent_ids[0] if best_response_mode else None
        br_agent_index = agent_ids.index(br_agent_id) if br_agent_id is not None else None

        frozen_q_tables = {agent_id: _copy_q_values(q_tables[agent_id]) for agent_id in agent_ids}
        frozen_visit_counts = _copy_visit_counts(visit_counts)

        policy_cache: dict[StateKey, SolverOutcome] = {}
        frozen_policy_cache: dict[StateKey, SolverOutcome] = {}

        envs = [env_factory(base_seed + idx) for idx in range(int(config["NUM_ENVS"]))]
        episode_returns = [{agent_id: 0.0 for agent_id in agent_ids} for _ in envs]
        episode_lengths = [0 for _ in envs]
        last_obs_by_env: list[dict[str, Any]] = []
        next_reset_seed = base_seed + int(config["NUM_ENVS"])

        try:
            for env_idx, env in enumerate(envs):
                obs, _ = reset_env(env, base_seed + env_idx)
                last_obs_by_env.append(dict(obs))

            metric_history: dict[str, list[float]] = defaultdict(list)
            per_agent_return_history = {agent_id: [] for agent_id in agent_ids}

            for update_index in range(int(config["NUM_UPDATES"])):
                epsilon = _epsilon_schedule(config, update_index)

                completed_episode_returns: list[float] = []
                completed_episode_lengths: list[float] = []
                completed_returns_by_agent = {agent_id: [] for agent_id in agent_ids}

                for _ in range(int(config["NUM_STEPS"])):
                    step_records: list[tuple[StateKey, PolicyProfile, tuple[int, ...]]] = []

                    for env_obs in last_obs_by_env:
                        state_key = joint_observation_key(env_obs, agent_ids)
                        if best_response_mode and br_agent_id is not None and br_agent_index is not None:
                            _ensure_state_tables(
                                q_tables,
                                visit_counts,
                                state_key=state_key,
                                agent_ids=[br_agent_id],
                                action_shape=action_shape,
                            )
                            _ensure_state_tables(
                                frozen_q_tables,
                                frozen_visit_counts,
                                state_key=state_key,
                                agent_ids=agent_ids,
                                action_shape=action_shape,
                            )
                            frozen_outcome = _policy_for_state(
                                state_key,
                                q_tables=frozen_q_tables,
                                visit_counts=frozen_visit_counts,
                                agent_ids=agent_ids,
                                action_shape=action_shape,
                                cache=frozen_policy_cache,
                                equilibrium_selection=config["EQUILIBRIUM_SELECTION"],
                            )
                            policies = list(frozen_outcome.policies)
                            policies[br_agent_index] = _best_response_policy(
                                q_tables[br_agent_id][state_key],
                                agent_index=br_agent_index,
                                policies=frozen_outcome.policies,
                            )
                            active_policies = tuple(policies)
                        else:
                            outcome = _policy_for_state(
                                state_key,
                                q_tables=q_tables,
                                visit_counts=visit_counts,
                                agent_ids=agent_ids,
                                action_shape=action_shape,
                                cache=policy_cache,
                                equilibrium_selection=config["EQUILIBRIUM_SELECTION"],
                            )
                            active_policies = outcome.policies

                        joint_action: list[int] = []
                        for agent_index, agent_id in enumerate(agent_ids):
                            if agent_id in active_trainable_set and rng.random() < epsilon:
                                joint_action.append(int(rng.integers(0, action_shape[agent_index])))
                            else:
                                joint_action.append(
                                    _sample_action(
                                        active_policies[agent_index],
                                        rng=rng,
                                    )
                                )
                        step_records.append((state_key, active_policies, tuple(joint_action)))

                    next_obs_by_env: list[dict[str, Any]] = []

                    for env_idx, env in enumerate(envs):
                        state_key, active_policies, joint_action = step_records[env_idx]
                        action_dict = {
                            agent_id: int(joint_action[agent_index]) for agent_index, agent_id in enumerate(agent_ids)
                        }
                        obs, rewards, terminations, truncations, infos = env.step(action_dict)
                        env_done = any(
                            bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in agent_ids
                        )

                        episode_lengths[env_idx] += 1
                        for agent_id in agent_ids:
                            episode_returns[env_idx][agent_id] += float(rewards.get(agent_id, 0.0))

                        if env_done:
                            returns_for_env = [episode_returns[env_idx][agent_id] for agent_id in agent_ids]
                            completed_episode_returns.append(
                                float(np.mean(np.asarray(returns_for_env, dtype=np.float32)))
                            )
                            completed_episode_lengths.append(float(episode_lengths[env_idx]))
                            for agent_id in agent_ids:
                                completed_returns_by_agent[agent_id].append(episode_returns[env_idx][agent_id])

                            reset_obs, _ = reset_env(env, next_reset_seed)
                            next_reset_seed += 1
                            next_obs_by_env.append(dict(reset_obs))
                            episode_returns[env_idx] = {agent_id: 0.0 for agent_id in agent_ids}
                            episode_lengths[env_idx] = 0
                        else:
                            next_obs_by_env.append(dict(obs))

                        if config["COUNTERFACTUAL_REPLAY"]:
                            step_variants = _enumerate_joint_counterfactual_steps(
                                agent_ids=agent_ids,
                                models=counterfactual_models,
                                current_obs=last_obs_by_env[env_idx],
                                next_obs=next_obs_by_env[-1],
                                rewards=rewards,
                                labels=extract_counterfactual_true_props(
                                    infos,
                                    agent_ids,
                                ),
                                done=env_done,
                                max_joint_variants=config["COUNTERFACTUAL_MAX_JOINT_VARIANTS"],
                                rng=rng,
                            )
                        else:
                            next_state_key = None if env_done else joint_observation_key(obs, agent_ids)
                            step_variants = [
                                CounterfactualJointStep(
                                    state_key=state_key,
                                    next_state_key=next_state_key,
                                    rewards=tuple(float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids),
                                )
                            ]

                        for variant in step_variants:
                            variant_state_key = variant.state_key
                            variant_next_state_key = variant.next_state_key

                            if best_response_mode and br_agent_id is not None and br_agent_index is not None:
                                _ensure_state_tables(
                                    q_tables,
                                    visit_counts,
                                    state_key=variant_state_key,
                                    agent_ids=[br_agent_id],
                                    action_shape=action_shape,
                                )
                            else:
                                _ensure_state_tables(
                                    q_tables,
                                    visit_counts,
                                    state_key=variant_state_key,
                                    agent_ids=agent_ids,
                                    action_shape=action_shape,
                                )

                            visit_counts[variant_state_key][joint_action] += 1
                            visit_count = int(visit_counts[variant_state_key][joint_action])
                            alpha = max(
                                1.0 / max(1, visit_count),
                                float(config["MIN_ALPHA"]),
                            )

                            if variant_next_state_key is not None:
                                if best_response_mode and br_agent_id is not None and br_agent_index is not None:
                                    _ensure_state_tables(
                                        q_tables,
                                        visit_counts,
                                        state_key=variant_next_state_key,
                                        agent_ids=[br_agent_id],
                                        action_shape=action_shape,
                                    )
                                    _ensure_state_tables(
                                        frozen_q_tables,
                                        frozen_visit_counts,
                                        state_key=variant_next_state_key,
                                        agent_ids=agent_ids,
                                        action_shape=action_shape,
                                    )
                                    frozen_next = _policy_for_state(
                                        variant_next_state_key,
                                        q_tables=frozen_q_tables,
                                        visit_counts=frozen_visit_counts,
                                        agent_ids=agent_ids,
                                        action_shape=action_shape,
                                        cache=frozen_policy_cache,
                                        equilibrium_selection=config["EQUILIBRIUM_SELECTION"],
                                    )
                                    next_outcome = None
                                else:
                                    _ensure_state_tables(
                                        q_tables,
                                        visit_counts,
                                        state_key=variant_next_state_key,
                                        agent_ids=agent_ids,
                                        action_shape=action_shape,
                                    )
                                    next_outcome = _policy_for_state(
                                        variant_next_state_key,
                                        q_tables=q_tables,
                                        visit_counts=visit_counts,
                                        agent_ids=agent_ids,
                                        action_shape=action_shape,
                                        cache=policy_cache,
                                        equilibrium_selection=config["EQUILIBRIUM_SELECTION"],
                                    )
                                    frozen_next = None
                            else:
                                next_outcome = None
                                frozen_next = None

                            for agent_index, agent_id in enumerate(agent_ids):
                                if agent_id not in active_trainable_set:
                                    continue

                                if variant_next_state_key is None:
                                    continuation_value = 0.0
                                elif (
                                    best_response_mode
                                    and br_agent_id == agent_id
                                    and br_agent_index is not None
                                    and frozen_next is not None
                                ):
                                    continuation_value = _best_response_value(
                                        q_tables[agent_id][variant_next_state_key],
                                        agent_index=br_agent_index,
                                        policies=frozen_next.policies,
                                    )
                                elif next_outcome is not None:
                                    continuation_value = _expected_value_under_policies(
                                        q_tables[agent_id][variant_next_state_key],
                                        next_outcome.policies,
                                    )
                                else:
                                    continuation_value = 0.0

                                current_value = float(q_tables[agent_id][variant_state_key][joint_action])
                                td_target = variant.rewards[agent_index] + (float(config["GAMMA"]) * continuation_value)
                                q_tables[agent_id][variant_state_key][joint_action] = current_value + alpha * (
                                    td_target - current_value
                                )

                            policy_cache.pop(variant_state_key, None)

                    last_obs_by_env = next_obs_by_env

                current_timesteps = timesteps_by_agent[agent_ids[0]] + int(config["NUM_STEPS"]) * int(
                    config["NUM_ENVS"]
                )
                for agent_id in agent_ids:
                    timesteps_by_agent[agent_id] = current_timesteps
                    n_updates_by_agent[agent_id] += 1

                metric = {
                    "episode_return_mean": mean_or_nan(completed_episode_returns),
                    "episode_length_mean": mean_or_nan(completed_episode_lengths),
                    "completed_episodes": float(len(completed_episode_returns)),
                    "epsilon": float(epsilon),
                    "per_agent_episode_return_mean": {
                        agent_id: mean_or_nan(completed_returns_by_agent[agent_id]) for agent_id in agent_ids
                    },
                }

                for key, value in metric.items():
                    if key == "per_agent_episode_return_mean":
                        for agent_id, agent_value in value.items():
                            per_agent_return_history[agent_id].append(agent_value)
                    else:
                        metric_history[key].append(value)

                maybe_log_to_wandb(metric)
                if on_update is not None:
                    on_update(update_index + 1, int(config["NUM_UPDATES"]), metric)

            metrics = {key: np.asarray(values, dtype=np.float32) for key, values in metric_history.items()}
            metrics["per_agent_episode_return_mean"] = {
                agent_id: np.asarray(values, dtype=np.float32) for agent_id, values in per_agent_return_history.items()
            }

            train_states = {
                agent_id: {
                    "q_values": _copy_q_values(q_tables[agent_id]),
                    "visit_counts": _copy_visit_counts(visit_counts),
                    "timesteps": int(timesteps_by_agent[agent_id]),
                    "n_updates": int(n_updates_by_agent[agent_id]),
                }
                for agent_id in agent_ids
            }

            out = {
                "train_states": train_states,
                "metrics": metrics,
                "agent_ids": tuple(agent_ids),
            }
            if capture_env_histories:
                out["env_histories"] = [extract_env_history(env) for env in envs]
            return out
        finally:
            for env in envs:
                close_env(env)

    return train


__all__ = [
    "PolicyProfile",
    "SolverOutcome",
    "joint_observation_key",
    "make_train",
]
