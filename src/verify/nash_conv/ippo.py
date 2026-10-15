from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from src.rl.ippo import ActorCritic
from src.rl.ippo import make_train as make_ippo_train

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
    train_state_params,
)


def _probe_ippo_networks(
    make_base_env_fn: MakeBaseEnv,
    agent_ids: list[str],
    activation: str,
) -> dict[str, ActorCritic]:
    env = call_make_base_env(make_base_env_fn, seed=0, num_agents=len(agent_ids))
    try:
        networks: dict[str, ActorCritic] = {}
        for agent_id in agent_ids:
            obs_space = env.observation_space(agent_id)
            action_space = env.action_space(agent_id)
            if not isinstance(obs_space, gym.spaces.Box):
                raise TypeError(
                    f"IPPO verification expects Box observations, got "
                    f"{type(obs_space).__name__} for agent '{agent_id}'."
                )
            if not isinstance(action_space, gym.spaces.Discrete):
                raise TypeError(
                    f"IPPO verification expects Discrete actions, got "
                    f"{type(action_space).__name__} for agent '{agent_id}'."
                )
            networks[agent_id] = ActorCritic(
                action_dim=int(action_space.n),
                activation=activation,
            )
        return networks
    finally:
        close_env(env)


def _ippo_action(
    network: ActorCritic,
    train_state: TrainState,
    obs: Any,
    *,
    deterministic: bool,
    rng: jax.Array | None = None,
) -> int:
    obs_batch = jnp.asarray(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    logits, _ = network.apply(train_state_params(train_state), obs_batch)
    logits = logits.squeeze(axis=0)

    if deterministic:
        action = jnp.argmax(logits, axis=-1)
    else:
        if rng is None:
            raise ValueError("Expected rng when deterministic=False.")
        action = jax.random.categorical(rng, logits, axis=-1)
    return int(np.asarray(action))


def _make_ippo_action_selection_fn(network: ActorCritic):
    @jax.jit
    def _select_actions(
        params: Any,
        obs_batch: jnp.ndarray,
        rng: jax.Array,
        deterministic: jnp.ndarray,
    ) -> jnp.ndarray:
        logits, _ = network.apply(params, obs_batch)

        def _greedy_action(_: None) -> jnp.ndarray:
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        def _sample_action(_: None) -> jnp.ndarray:
            return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)

        return jax.lax.cond(
            jnp.asarray(deterministic, dtype=jnp.bool_),
            _greedy_action,
            _sample_action,
            operand=None,
        )

    return _select_actions


def rollout_successes_parallel_ippo(
    train_states: Mapping[str, TrainState],
    config: Mapping[str, Any],
    make_base_env_fn: MakeBaseEnv,
    agent_ids: list[str],
    episode_seeds: list[int],
    *,
    deterministic: bool = False,
    explore: bool | None = None,
    success_threshold: float = 0.5,
    progress: NashConvProgress | None = None,
):
    if explore is not None:
        deterministic = not bool(explore)

    missing = [agent_id for agent_id in agent_ids if agent_id not in train_states]
    if missing:
        raise KeyError(f"Missing train state(s) for agents: {missing}")

    networks = _probe_ippo_networks(
        make_base_env_fn,
        agent_ids,
        activation=str(config.get("ACTIVATION", "tanh")),
    )
    action_selection_fns = {agent_id: _make_ippo_action_selection_fn(networks[agent_id]) for agent_id in agent_ids}
    rng = jax.random.PRNGKey(int(config.get("SEED", 0)))

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
            episode_returns = {aid: 0.0 for aid in agent_ids}

            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    rng, action_rng = jax.random.split(rng)
                    obs_batch = np.asarray(obs[agent_id], dtype=np.float32).reshape(1, -1)
                    chosen_actions = action_selection_fns[agent_id](
                        train_state_params(train_states[agent_id]),
                        jnp.asarray(obs_batch, dtype=jnp.float32),
                        action_rng,
                        jnp.asarray(deterministic, dtype=jnp.bool_),
                    )
                    actions[agent_id] = int(np.asarray(chosen_actions[0]))

                obs, rewards, terminations, truncations, _ = env.step(actions)
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


def approximate_nashconv_bernoulli_ippo(
    train_states: Mapping[str, TrainState],
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
        description="IPPO NashConv",
        enabled=show_progress,
        refresh_percent=progress_refresh_percent,
    ) as progress:
        base_success, _ = rollout_successes_parallel_ippo(
            train_states=train_states,
            config=config,
            make_base_env_fn=make_base_env_fn,
            agent_ids=agent_ids,
            episode_seeds=episode_seeds,
            deterministic=False,
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
                br_config["NUM_MINIBATCHES"] = 1
                br_config["TOTAL_TIMESTEPS"] = br_train_iters * br_batch_steps
                br_seed = int(config.get("SEED", 0)) + (target_index + 1) * 10_000
                br_config["SEED"] = br_seed

                br_train = make_ippo_train(
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

            br_success, _ = rollout_successes_parallel_ippo(
                train_states=candidate_states,
                config=config,
                make_base_env_fn=make_base_env_fn,
                agent_ids=agent_ids,
                episode_seeds=episode_seeds,
                deterministic=False,
                success_threshold=success_threshold,
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
    "approximate_nashconv_bernoulli_ippo",
    "make_ippo_train",
    "rollout_successes_parallel_ippo",
]
