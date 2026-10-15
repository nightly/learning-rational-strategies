from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from src.rl.nash_q import (
    _best_response_policy,
    _copy_q_values,
    _copy_visit_counts,
    _ensure_state_tables,
    _policy_for_state,
    joint_observation_key,
)
from src.rl.nash_q import (
    make_train as make_nashq_train,
)

from .common import (
    MakeBaseEnv,
    NashConvProgress,
    bernoulli_ci95,
    call_make_base_env,
    close_env,
    nashconv_joint_confidence_payload,
    paired_diff_ci95,
    rollout_episode_done,
    safe_reset,
)


def _probe_nashq_action_shape(
    make_base_env_fn: MakeBaseEnv,
    agent_ids: list[str],
) -> tuple[int, ...]:
    env = call_make_base_env(make_base_env_fn, seed=0, num_agents=len(agent_ids))
    try:
        action_shape: list[int] = []
        for agent_id in agent_ids:
            action_space = env.action_space(agent_id)
            if not isinstance(action_space, gym.spaces.Discrete):
                raise TypeError(
                    f"Nash-Q verification expects Discrete actions, got "
                    f"{type(action_space).__name__} for agent '{agent_id}'."
                )
            action_shape.append(int(action_space.n))
        return tuple(action_shape)
    finally:
        close_env(env)


def _restore_nashq_tables(
    train_states: Mapping[str, Mapping[str, Any]],
    agent_ids: list[str],
) -> tuple[dict[str, dict[Any, np.ndarray]], dict[Any, np.ndarray]]:
    missing = [agent_id for agent_id in agent_ids if agent_id not in train_states]
    if missing:
        raise KeyError(f"Missing train state(s) for agents: {missing}")

    q_tables = {agent_id: _copy_q_values(train_states[agent_id].get("q_values", {})) for agent_id in agent_ids}
    visit_counts = _copy_visit_counts(train_states[agent_ids[0]].get("visit_counts", {}))
    return q_tables, visit_counts


def rollout_successes_parallel_nashq(
    train_states: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    make_base_env_fn: MakeBaseEnv,
    agent_ids: list[str],
    episode_seeds: list[int],
    *,
    success_threshold: float = 0.5,
    frozen_train_states: Mapping[str, Mapping[str, Any]] | None = None,
    best_response_target_agent: str | None = None,
    progress: NashConvProgress | None = None,
):
    q_tables, visit_counts = _restore_nashq_tables(train_states, agent_ids)
    action_shape = _probe_nashq_action_shape(make_base_env_fn, agent_ids)
    policy_cache: dict[Any, Any] = {}
    equilibrium_selection = str(config.get("EQUILIBRIUM_SELECTION", "payoff_dominant"))
    rng = np.random.default_rng(int(config.get("SEED", 0)) + 1234)

    frozen_q_tables = None
    frozen_visit_counts = None
    frozen_policy_cache: dict[Any, Any] | None = None
    target_index = None
    if frozen_train_states is not None:
        frozen_q_tables, frozen_visit_counts = _restore_nashq_tables(
            frozen_train_states,
            agent_ids,
        )
        frozen_policy_cache = {}
    if best_response_target_agent is not None:
        target_index = agent_ids.index(best_response_target_agent)

    n_episodes = len(episode_seeds)
    success_bits = {aid: np.zeros(n_episodes, dtype=np.int8) for aid in agent_ids}
    returns = {aid: np.zeros(n_episodes, dtype=np.float32) for aid in agent_ids}

    initial_env_seed = int(episode_seeds[0]) if episode_seeds else int(config.get("SEED", 0))
    env = call_make_base_env(
        make_base_env_fn,
        seed=initial_env_seed,
        num_agents=len(agent_ids),
    )
    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            obs, _ = safe_reset(env, seed=int(episode_seed))
            obs = dict(obs)
            episode_returns = {aid: 0.0 for aid in agent_ids}

            while env.agents:
                state_key = joint_observation_key(obs, agent_ids)
                _ensure_state_tables(
                    q_tables,
                    visit_counts,
                    state_key=state_key,
                    agent_ids=agent_ids,
                    action_shape=action_shape,
                )

                if best_response_target_agent is not None:
                    if (
                        frozen_q_tables is None
                        or frozen_visit_counts is None
                        or frozen_policy_cache is None
                        or target_index is None
                    ):
                        raise ValueError("Expected frozen_train_states when evaluating a Nash-Q best response.")

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
                        equilibrium_selection=equilibrium_selection,
                    )
                    active_policies = list(frozen_outcome.policies)
                    active_policies[target_index] = _best_response_policy(
                        q_tables[best_response_target_agent][state_key],
                        agent_index=target_index,
                        policies=frozen_outcome.policies,
                    )
                    policies = tuple(active_policies)
                else:
                    outcome = _policy_for_state(
                        state_key,
                        q_tables=q_tables,
                        visit_counts=visit_counts,
                        agent_ids=agent_ids,
                        action_shape=action_shape,
                        cache=policy_cache,
                        equilibrium_selection=equilibrium_selection,
                    )
                    policies = outcome.policies

                actions = {
                    agent_id: int(rng.choice(np.arange(action_shape[agent_index]), p=policies[agent_index]))
                    for agent_index, agent_id in enumerate(agent_ids)
                }

                obs, rewards, terminations, truncations, _ = env.step(actions)
                obs = dict(obs)
                for agent_id, reward in rewards.items():
                    episode_returns[agent_id] += float(reward)
                if rollout_episode_done(env, terminations, truncations):
                    break

            for agent_id in agent_ids:
                returns[agent_id][episode_index] = episode_returns[agent_id]
                success_bits[agent_id][episode_index] = 1 if episode_returns[agent_id] >= success_threshold else 0
            if progress is not None:
                progress.advance()
    finally:
        close_env(env)

    return success_bits, returns


def approximate_nashconv_bernoulli_nashq(
    train_states: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    make_base_env_fn: MakeBaseEnv,
    agent_ids: list[str],
    *,
    eval_episodes: int = 400,
    eval_seed0: int = 2026001,
    delta: float = 0.05,
    br_train_iters: int = 15,
    br_batch_steps: int = 4096,
    success_threshold: float = 0.5,
    show_progress: bool = True,
    progress_refresh_percent: float = 5.0,
):
    config = dict(config)
    eval_episodes = int(eval_episodes)
    br_batch_steps = int(br_batch_steps)
    br_train_iters = int(br_train_iters)
    should_train_br = br_train_iters > 0 and br_batch_steps > 0
    episode_seeds = [int(eval_seed0) + i for i in range(eval_episodes)]
    progress_total = eval_episodes * (1 + len(agent_ids))
    if should_train_br:
        progress_total += len(agent_ids)

    with NashConvProgress(
        total=progress_total,
        description="Nash-Q NashConv",
        enabled=show_progress,
        refresh_percent=progress_refresh_percent,
    ) as progress:
        base_success, _ = rollout_successes_parallel_nashq(
            train_states=train_states,
            config=config,
            make_base_env_fn=make_base_env_fn,
            agent_ids=agent_ids,
            episode_seeds=episode_seeds,
            success_threshold=success_threshold,
            progress=progress,
        )

        baseline_p, baseline_ci = {}, {}
        for aid in agent_ids:
            p_hat = float(base_success[aid].mean())
            baseline_p[aid] = p_hat
            baseline_ci[aid] = bernoulli_ci95(p_hat, eval_episodes)

        br_p, br_ci = {}, {}
        delta_p, delta_ci = {}, {}

        for target_index, target_agent in enumerate(agent_ids):
            candidate_states = train_states

            if should_train_br:
                br_config = dict(config)
                br_config["NUM_ENVS"] = 1
                br_config["NUM_STEPS"] = br_batch_steps
                br_config["TOTAL_TIMESTEPS"] = br_train_iters * br_batch_steps
                br_seed = int(config.get("SEED", 0)) + (target_index + 1) * 10_000
                br_config["SEED"] = br_seed

                br_train = make_nashq_train(
                    br_config,
                    lambda seed: call_make_base_env(
                        make_base_env_fn,
                        seed=seed,
                        num_agents=len(agent_ids),
                    ),
                )
                br_out = br_train(
                    seed=br_seed,
                    initial_train_states=train_states,
                    trainable_agent_ids={target_agent},
                )
                candidate_states = br_out["train_states"]
                progress.advance()

            br_success, _ = rollout_successes_parallel_nashq(
                train_states=candidate_states,
                config=config,
                make_base_env_fn=make_base_env_fn,
                agent_ids=agent_ids,
                episode_seeds=episode_seeds,
                success_threshold=success_threshold,
                frozen_train_states=train_states,
                best_response_target_agent=target_agent,
                progress=progress,
            )

            p_hat = float(br_success[target_agent].mean())
            br_p[target_agent] = p_hat
            br_ci[target_agent] = bernoulli_ci95(p_hat, eval_episodes)

            diff = br_success[target_agent].astype(np.float32) - base_success[target_agent].astype(np.float32)
            delta_p[target_agent] = float(diff.mean())
            delta_ci[target_agent] = paired_diff_ci95(diff)

    nashconv = float(sum(max(0.0, delta_p[aid]) for aid in agent_ids))
    joint_confidence = nashconv_joint_confidence_payload(
        delta_p,
        eval_episodes=eval_episodes,
        delta=delta,
    )

    return {
        "nashconv": nashconv,
        **joint_confidence,
        "baseline_p": baseline_p,
        "baseline_ci95": baseline_ci,
        "br_p": br_p,
        "br_ci95": br_ci,
        "delta_p": delta_p,
        "delta_ci95": delta_ci,
        "eval_episodes": eval_episodes,
        "br_train_iters": br_train_iters,
        "br_batch_steps": br_batch_steps,
        "eval_seed0": eval_seed0,
    }


__all__ = [
    "approximate_nashconv_bernoulli_nashq",
    "make_nashq_train",
    "rollout_successes_parallel_nashq",
]
