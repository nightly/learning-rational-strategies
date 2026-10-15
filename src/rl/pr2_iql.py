"""
PettingZoo-compatible PR2-style independent Q-learning trainer.

This variant keeps the broad trainer contract used by ``src/rl/iql.py`` but
learns a joint-action critic per agent with output shape
``[own_action_dim, opponent_joint_action_dim]``. Action selection and
bootstrapping use PR2 response-aware action values derived from the critic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal

from src.monitor.counterfactual import (
    CounterfactualMonitorModel,
    append_monitor_state,
    extract_counterfactual_models,
    extract_counterfactual_true_props,
    split_monitor_state,
)
from src.rl.common import (
    EnvFactory,
    UpdateCallback,
    canonicalize_observations,
    close_env,
    coerce_seed,
    coerce_trainable_agent_ids,
    extract_env_history,
    maybe_log_to_wandb,
    mean_or_nan,
    possible_agent_ids,
    reset_env,
    restore_train_state,
    space_dimensions,
)
from src.rl.iql import (
    IQL_REPLAY_FIELDS,
    IQLTrainState,
    ScannedRNN,
    counterfactual_sequences_for_agent,
    validate_iql_config,
)
from src.rl.pr2_common import (
    OpponentJointActionIndexer,
    build_opponent_indexers,
    encode_opponent_joint_action,
    reshape_joint_q,
    response_aware_action_values,
    select_joint_q_value,
)
from src.rl.trajectory import SequenceReplayBuffer, masked_mean, stack_replay_samples

PR2_REPLAY_FIELDS = IQL_REPLAY_FIELDS + ("opponent_index",)


def _activation_fn(name: str):
    if str(name).lower() == "relu":
        return nn.relu
    return nn.tanh


class PR2RNNQNetwork(nn.Module):
    action_dim: int
    opponent_joint_size: int
    hidden_dim: int
    activation: str = "tanh"
    init_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        activation = _activation_fn(self.activation)
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, resets))
        q_vals = nn.Dense(
            self.action_dim * self.opponent_joint_size,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)
        return hidden, q_vals


class PR2MLPQNetwork(nn.Module):
    action_dim: int
    opponent_joint_size: int
    hidden_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del resets
        activation = _activation_fn(self.activation)
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)
        q_vals = nn.Dense(
            self.action_dim * self.opponent_joint_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(embedding)
        return hidden, q_vals


class PR2ChunkBatch(NamedTuple):
    obs: jnp.ndarray
    reset: jnp.ndarray
    action: jnp.ndarray
    opponent_index: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    final_next_obs: jnp.ndarray
    loss_mask: jnp.ndarray


def _network_cls(network_kind: str):
    if str(network_kind).lower() == "mlp":
        return PR2MLPQNetwork
    if str(network_kind).lower() == "rnn":
        return PR2RNNQNetwork
    raise ValueError("NETWORK_KIND must be one of {'rnn', 'mlp'}.")


def make_pr2_action_selection_fn(
    network: nn.Module,
    *,
    action_dim: int,
    opponent_joint_size: int,
):
    @jax.jit
    def _select_actions(
        params: Any,
        hidden: jnp.ndarray,
        obs_batch: jnp.ndarray,
        reset_batch: jnp.ndarray,
        rng: jax.Array,
        epsilon: jnp.ndarray,
        explore: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        new_hidden, q_vals = network.apply(
            params,
            hidden,
            obs_batch[None, ...],
            reset_batch[None, ...],
        )
        joint_q = reshape_joint_q(
            q_vals.squeeze(axis=0),
            own_action_dim=action_dim,
            opponent_joint_size=opponent_joint_size,
        )
        greedy_actions = jnp.argmax(response_aware_action_values(joint_q), axis=-1).astype(jnp.int32)
        action_rng, mask_rng = jax.random.split(rng)
        random_actions = jax.random.randint(
            action_rng,
            shape=greedy_actions.shape,
            minval=0,
            maxval=int(action_dim),
        )
        explore_mask = jnp.logical_and(
            jnp.asarray(explore, dtype=jnp.bool_),
            jax.random.uniform(mask_rng, shape=greedy_actions.shape) < jnp.asarray(epsilon, dtype=jnp.float32),
        )
        chosen = jnp.where(explore_mask, random_actions, greedy_actions)
        return new_hidden, chosen

    return _select_actions


def make_update_fn(
    network: nn.Module,
    config: Mapping[str, Any],
    *,
    own_action_dim: int,
    opponent_joint_size: int,
):
    gamma = float(config["GAMMA"])
    hidden_size = int(config["HIDDEN_SIZE"])
    burn_in_steps = int(config["BURN_IN_STEPS"])

    @jax.jit
    def _update_agent(
        train_state: IQLTrainState,
        batch: PR2ChunkBatch,
    ) -> tuple[IQLTrainState, dict[str, jnp.ndarray]]:
        def _update_epoch(
            epoch_train_state: IQLTrainState,
            epoch_batch: PR2ChunkBatch,
        ) -> tuple[IQLTrainState, dict[str, jnp.ndarray]]:
            batch_size = epoch_batch.obs.shape[1]
            init_hidden = jnp.zeros(
                (batch_size, hidden_size),
                dtype=jnp.float32,
            )

            def _loss_fn(params: Any):
                online_hidden = init_hidden
                target_hidden = init_hidden
                if burn_in_steps:
                    online_hidden, _ = network.apply(
                        params,
                        online_hidden,
                        epoch_batch.obs[:burn_in_steps],
                        epoch_batch.reset[:burn_in_steps],
                    )
                    target_hidden, _ = network.apply(
                        epoch_train_state.target_network_params,
                        target_hidden,
                        epoch_batch.obs[:burn_in_steps],
                        epoch_batch.reset[:burn_in_steps],
                    )
                    online_hidden = jax.lax.stop_gradient(online_hidden)
                    target_hidden = jax.lax.stop_gradient(target_hidden)

                learn_obs = epoch_batch.obs[burn_in_steps:]
                learn_reset = epoch_batch.reset[burn_in_steps:]
                learn_action = epoch_batch.action[burn_in_steps:]
                learn_opponent_index = epoch_batch.opponent_index[burn_in_steps:]
                learn_reward = epoch_batch.reward[burn_in_steps:]
                learn_done = epoch_batch.done[burn_in_steps:]
                learn_mask = epoch_batch.loss_mask[burn_in_steps:]
                ext_obs = jnp.concatenate(
                    [learn_obs, epoch_batch.final_next_obs[None, ...]],
                    axis=0,
                )
                ext_reset = jnp.concatenate(
                    [learn_reset, learn_done[-1:, ...]],
                    axis=0,
                )

                _, q_online_ext = network.apply(
                    params,
                    online_hidden,
                    ext_obs,
                    ext_reset,
                )
                _, q_target_ext = network.apply(
                    epoch_train_state.target_network_params,
                    target_hidden,
                    ext_obs,
                    ext_reset,
                )
                q_online_ext = reshape_joint_q(
                    q_online_ext,
                    own_action_dim=own_action_dim,
                    opponent_joint_size=opponent_joint_size,
                )
                q_target_ext = reshape_joint_q(
                    q_target_ext,
                    own_action_dim=own_action_dim,
                    opponent_joint_size=opponent_joint_size,
                )

                q_current = q_online_ext[:-1, ...]
                q_next_target = q_target_ext[1:, ...]

                chosen_q = select_joint_q_value(
                    q_current,
                    learn_action,
                    learn_opponent_index,
                )
                next_q_pr2 = response_aware_action_values(q_next_target)
                next_best = jnp.max(next_q_pr2, axis=-1)

                targets = learn_reward + (1.0 - learn_done) * gamma * next_best
                td_error = chosen_q - jax.lax.stop_gradient(targets)
                loss = masked_mean(jnp.square(td_error), learn_mask)
                return loss, {
                    "loss": loss,
                    "qvals": masked_mean(chosen_q, learn_mask),
                }

            (_, loss_info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(epoch_train_state.params)
            epoch_train_state = epoch_train_state.apply_gradients(grads=grads)
            epoch_train_state = epoch_train_state.replace(grad_steps=epoch_train_state.grad_steps + 1)
            return epoch_train_state, loss_info

        train_state, loss_info = jax.lax.scan(_update_epoch, train_state, batch)
        return train_state, jax.tree_util.tree_map(jnp.mean, loss_info)

    return _update_agent


def counterfactual_pr2_sequences_for_agent(
    *,
    model: CounterfactualMonitorModel,
    obs_base_chunk: np.ndarray,
    obs_state_ids: np.ndarray,
    reset_chunk: np.ndarray,
    action_chunk: np.ndarray,
    opponent_index_chunk: np.ndarray,
    reward_chunk: np.ndarray,
    done_chunk: np.ndarray,
    final_next_base_obs: np.ndarray,
    final_next_state_id: int,
    labels_sequence: Sequence[frozenset[str]],
) -> list[dict[str, np.ndarray]]:
    sequences = counterfactual_sequences_for_agent(
        model=model,
        obs_base_chunk=obs_base_chunk,
        obs_state_ids=obs_state_ids,
        reset_chunk=reset_chunk,
        action_chunk=action_chunk,
        reward_chunk=reward_chunk,
        done_chunk=done_chunk,
        final_next_base_obs=final_next_base_obs,
        final_next_state_id=final_next_state_id,
        labels_sequence=labels_sequence,
    )
    for sequence in sequences:
        sequence["opponent_index"] = np.asarray(
            opponent_index_chunk,
            dtype=np.int32,
        )
    return sequences


def make_train(config: dict[str, Any], env_factory: EnvFactory):
    config = validate_iql_config(config, algorithm="PR2-IQL")

    probe_seed = coerce_seed(config.get("SEED", 0))
    probe_env = env_factory(probe_seed)
    try:
        agent_ids = list(possible_agent_ids(probe_env))
        obs_dims, action_dims = space_dimensions(probe_env, agent_ids)
        counterfactual_models = (
            extract_counterfactual_models(probe_env, agent_ids) if config["COUNTERFACTUAL_REPLAY"] else {}
        )
    finally:
        close_env(probe_env)

    indexers = build_opponent_indexers(agent_ids, action_dims)
    config["NUM_AGENTS"] = len(agent_ids)
    eps_schedule = optax.linear_schedule(
        init_value=float(config["EPS_START"]),
        end_value=float(config["EPS_FINISH"]),
        transition_steps=max(
            1,
            int(float(config["EPS_DECAY"]) * int(config["SCHEDULE_TOTAL_UPDATES"])),
        ),
    )
    network_cls = _network_cls(config["NETWORK_KIND"])

    def build_optimizer():
        return optax.chain(
            optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
            optax.adam(learning_rate=float(config["LR"])),
        )

    def train(
        seed: Any | None = None,
        *,
        initial_train_states: Mapping[str, IQLTrainState | Mapping[str, Any]] | None = None,
        trainable_agent_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        on_update: UpdateCallback | None = None,
        capture_env_histories: bool = False,
    ):
        base_seed = coerce_seed(seed if seed is not None else config.get("SEED", 0))
        action_rng = jax.random.PRNGKey(base_seed)
        buffer_rng = np.random.default_rng(base_seed + 10_000)
        active_trainable_agent_ids = coerce_trainable_agent_ids(
            agent_ids,
            trainable_agent_ids,
        )

        networks = {
            agent_id: network_cls(
                action_dim=action_dims[agent_id],
                opponent_joint_size=indexers[agent_id].opponent_joint_size,
                hidden_dim=int(config["HIDDEN_SIZE"]),
                activation=config["ACTIVATION"],
            )
            for agent_id in agent_ids
        }
        update_fns = {
            agent_id: make_update_fn(
                networks[agent_id],
                config,
                own_action_dim=action_dims[agent_id],
                opponent_joint_size=indexers[agent_id].opponent_joint_size,
            )
            for agent_id in agent_ids
        }
        action_selection_fns = {
            agent_id: make_pr2_action_selection_fn(
                networks[agent_id],
                action_dim=action_dims[agent_id],
                opponent_joint_size=indexers[agent_id].opponent_joint_size,
            )
            for agent_id in agent_ids
        }

        train_states: dict[str, IQLTrainState] = {}
        for agent_id in agent_ids:
            action_rng, init_rng = jax.random.split(action_rng)
            init_hidden = jnp.zeros((1, int(config["HIDDEN_SIZE"])), dtype=jnp.float32)
            init_obs = jnp.zeros((1, 1, obs_dims[agent_id]), dtype=jnp.float32)
            init_reset = jnp.ones((1, 1), dtype=jnp.float32)
            params = networks[agent_id].init(
                init_rng,
                init_hidden,
                init_obs,
                init_reset,
            )
            template_state = IQLTrainState.create(
                apply_fn=networks[agent_id].apply,
                params=params,
                target_network_params=params,
                tx=build_optimizer(),
            )
            if initial_train_states is not None:
                if agent_id not in initial_train_states:
                    raise KeyError(f"Missing initial_train_states entry for agent '{agent_id}'.")
                train_states[agent_id] = restore_train_state(
                    template_state,
                    initial_train_states[agent_id],
                )
            else:
                train_states[agent_id] = template_state

        buffers = {
            agent_id: SequenceReplayBuffer(
                int(config["BUFFER_SIZE"]),
                required_fields=PR2_REPLAY_FIELDS,
            )
            for agent_id in agent_ids
        }
        next_synthetic_stream_id = {agent_id: int(config["NUM_ENVS"]) for agent_id in agent_ids}
        hidden_states = {
            agent_id: np.zeros(
                (int(config["NUM_ENVS"]), int(config["HIDDEN_SIZE"])),
                dtype=np.float32,
            )
            for agent_id in agent_ids
        }

        envs = [env_factory(base_seed + idx) for idx in range(int(config["NUM_ENVS"]))]
        episode_returns = [{agent_id: 0.0 for agent_id in agent_ids} for _ in envs]
        episode_lengths = [0 for _ in envs]
        last_obs_by_env: list[dict[str, np.ndarray]] = []
        last_done_by_env = [{agent_id: 1.0 for agent_id in agent_ids} for _ in envs]
        next_reset_seed = base_seed + int(config["NUM_ENVS"])

        def _evaluate_greedy(eval_seed: int) -> dict[str, float]:
            if not bool(config.get("TEST_DURING_TRAINING", True)):
                return {}

            num_eval_envs = int(config.get("TEST_NUM_ENVS", 1))
            num_eval_steps = int(config.get("TEST_NUM_STEPS", config["NUM_STEPS"]))
            eval_envs = [env_factory(eval_seed + idx) for idx in range(num_eval_envs)]
            eval_hidden = {
                agent_id: np.zeros(
                    (num_eval_envs, int(config["HIDDEN_SIZE"])),
                    dtype=np.float32,
                )
                for agent_id in agent_ids
            }
            eval_obs_by_env: list[dict[str, np.ndarray]] = []
            eval_done_by_env = [{agent_id: 0.0 for agent_id in agent_ids} for _ in eval_envs]
            eval_episode_returns = [{agent_id: 0.0 for agent_id in agent_ids} for _ in eval_envs]
            eval_episode_lengths = [0 for _ in eval_envs]
            eval_next_seed = eval_seed + num_eval_envs

            try:
                for env_idx, env in enumerate(eval_envs):
                    obs, _ = reset_env(env, eval_seed + env_idx)
                    eval_obs_by_env.append(canonicalize_observations(obs, agent_ids))

                completed_episode_returns: list[float] = []
                completed_episode_lengths: list[float] = []
                completed_returns_by_agent = {agent_id: [] for agent_id in agent_ids}

                for _ in range(num_eval_steps):
                    obs_batches = {
                        agent_id: np.stack(
                            [env_obs[agent_id] for env_obs in eval_obs_by_env],
                            axis=0,
                        ).astype(np.float32)
                        for agent_id in agent_ids
                    }
                    reset_batches = {
                        agent_id: np.asarray(
                            [done[agent_id] for done in eval_done_by_env],
                            dtype=np.float32,
                        )
                        for agent_id in agent_ids
                    }
                    step_actions: dict[str, np.ndarray] = {}

                    for agent_id in agent_ids:
                        new_hidden, chosen_actions = action_selection_fns[agent_id](
                            train_states[agent_id].params,
                            jnp.asarray(eval_hidden[agent_id], dtype=jnp.float32),
                            jnp.asarray(obs_batches[agent_id], dtype=jnp.float32),
                            jnp.asarray(reset_batches[agent_id], dtype=jnp.float32),
                            jax.random.PRNGKey(0),
                            jnp.asarray(0.0, dtype=jnp.float32),
                            jnp.asarray(False),
                        )
                        eval_hidden[agent_id] = np.asarray(new_hidden, dtype=np.float32)
                        step_actions[agent_id] = np.asarray(chosen_actions, dtype=np.int32)

                    next_obs_by_env: list[dict[str, np.ndarray]] = []
                    next_done_by_env: list[dict[str, float]] = []

                    for env_idx, env in enumerate(eval_envs):
                        env_actions = {agent_id: int(step_actions[agent_id][env_idx]) for agent_id in agent_ids}
                        obs, rewards, terminations, truncations, _ = env.step(env_actions)
                        env_done = any(
                            bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in agent_ids
                        )

                        eval_episode_lengths[env_idx] += 1
                        for agent_id in agent_ids:
                            eval_episode_returns[env_idx][agent_id] += float(rewards.get(agent_id, 0.0))

                        if env_done:
                            returns_for_env = [eval_episode_returns[env_idx][agent_id] for agent_id in agent_ids]
                            completed_episode_returns.append(
                                float(np.mean(np.asarray(returns_for_env, dtype=np.float32)))
                            )
                            completed_episode_lengths.append(float(eval_episode_lengths[env_idx]))
                            for agent_id in agent_ids:
                                completed_returns_by_agent[agent_id].append(eval_episode_returns[env_idx][agent_id])

                            reset_obs, _ = reset_env(env, eval_next_seed)
                            eval_next_seed += 1
                            next_obs_by_env.append(canonicalize_observations(reset_obs, agent_ids))
                            next_done_by_env.append({agent_id: 1.0 for agent_id in agent_ids})
                            eval_episode_returns[env_idx] = {agent_id: 0.0 for agent_id in agent_ids}
                            eval_episode_lengths[env_idx] = 0
                        else:
                            next_obs_by_env.append(canonicalize_observations(obs, agent_ids))
                            next_done_by_env.append({agent_id: 0.0 for agent_id in agent_ids})

                    eval_obs_by_env = next_obs_by_env
                    eval_done_by_env = next_done_by_env

                metrics = {
                    "test_episode_return_mean": mean_or_nan(completed_episode_returns),
                    "test_episode_length_mean": mean_or_nan(completed_episode_lengths),
                    "test_completed_episodes": float(len(completed_episode_returns)),
                }
                for agent_id in agent_ids:
                    metrics[f"test_per_agent_episode_return_mean/{agent_id}"] = mean_or_nan(
                        completed_returns_by_agent[agent_id]
                    )
                return metrics
            finally:
                for env in eval_envs:
                    close_env(env)

        try:
            for env_idx, env in enumerate(envs):
                obs, _ = reset_env(env, base_seed + env_idx)
                last_obs_by_env.append(canonicalize_observations(obs, agent_ids))

            metric_history: dict[str, list[float]] = defaultdict(list)
            per_agent_return_history = {agent_id: [] for agent_id in agent_ids}

            for update_index in range(int(config["NUM_UPDATES"])):
                schedule_update_index = int(config["SCHEDULE_UPDATE_OFFSET"]) + update_index
                epsilon = float(np.asarray(eps_schedule(schedule_update_index)))
                rollout_buffers = {
                    agent_id: {
                        "obs": [],
                        "reset": [],
                        "action": [],
                        "opponent_index": [],
                        "reward": [],
                        "done": [],
                    }
                    for agent_id in agent_ids
                }
                label_batches: list[list[frozenset[str]]] = []

                completed_episode_returns: list[float] = []
                completed_episode_lengths: list[float] = []
                completed_returns_by_agent = {agent_id: [] for agent_id in agent_ids}

                for _ in range(int(config["NUM_STEPS"])):
                    obs_batches = {
                        agent_id: np.stack(
                            [env_obs[agent_id] for env_obs in last_obs_by_env],
                            axis=0,
                        ).astype(np.float32)
                        for agent_id in agent_ids
                    }
                    reset_batches = {
                        agent_id: np.asarray(
                            [done[agent_id] for done in last_done_by_env],
                            dtype=np.float32,
                        )
                        for agent_id in agent_ids
                    }

                    step_actions: dict[str, np.ndarray] = {}
                    for agent_id in agent_ids:
                        action_rng, step_rng = jax.random.split(action_rng)
                        new_hidden, chosen_actions = action_selection_fns[agent_id](
                            train_states[agent_id].params,
                            jnp.asarray(hidden_states[agent_id], dtype=jnp.float32),
                            jnp.asarray(obs_batches[agent_id], dtype=jnp.float32),
                            jnp.asarray(reset_batches[agent_id], dtype=jnp.float32),
                            step_rng,
                            jnp.asarray(epsilon, dtype=jnp.float32),
                            jnp.asarray(agent_id in active_trainable_agent_ids),
                        )
                        hidden_states[agent_id] = np.asarray(
                            new_hidden,
                            dtype=np.float32,
                        )
                        step_actions[agent_id] = np.asarray(chosen_actions, dtype=np.int32)

                    opponent_index_batches = {
                        agent_id: np.zeros((int(config["NUM_ENVS"]),), dtype=np.int32) for agent_id in agent_ids
                    }
                    reward_batches = {
                        agent_id: np.zeros((int(config["NUM_ENVS"]),), dtype=np.float32) for agent_id in agent_ids
                    }
                    done_batches = {
                        agent_id: np.zeros((int(config["NUM_ENVS"]),), dtype=np.float32) for agent_id in agent_ids
                    }
                    next_obs_by_env: list[dict[str, np.ndarray]] = []
                    next_done_by_env: list[dict[str, float]] = []
                    step_true_props: list[frozenset[str]] = []

                    for env_idx, env in enumerate(envs):
                        env_actions = {agent_id: int(step_actions[agent_id][env_idx]) for agent_id in agent_ids}
                        for agent_id in agent_ids:
                            opponent_index_batches[agent_id][env_idx] = encode_opponent_joint_action(
                                indexers=indexers,
                                agent_id=agent_id,
                                joint_actions=env_actions,
                            )

                        obs, rewards, terminations, truncations, infos = env.step(env_actions)
                        if config["COUNTERFACTUAL_REPLAY"]:
                            step_true_props.append(extract_counterfactual_true_props(infos, agent_ids))
                        env_done = any(
                            bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in agent_ids
                        )

                        episode_lengths[env_idx] += 1
                        for agent_id in agent_ids:
                            reward = float(rewards.get(agent_id, 0.0))
                            episode_returns[env_idx][agent_id] += reward
                            reward_batches[agent_id][env_idx] = reward

                        if env_done:
                            returns_for_env = [episode_returns[env_idx][agent_id] for agent_id in agent_ids]
                            completed_episode_returns.append(
                                float(np.mean(np.asarray(returns_for_env, dtype=np.float32)))
                            )
                            completed_episode_lengths.append(float(episode_lengths[env_idx]))
                            for agent_id in agent_ids:
                                done_batches[agent_id][env_idx] = 1.0
                                completed_returns_by_agent[agent_id].append(episode_returns[env_idx][agent_id])

                            reset_obs, _ = reset_env(env, next_reset_seed)
                            next_reset_seed += 1
                            next_obs_by_env.append(canonicalize_observations(reset_obs, agent_ids))
                            next_done_by_env.append({agent_id: 1.0 for agent_id in agent_ids})
                            episode_returns[env_idx] = {agent_id: 0.0 for agent_id in agent_ids}
                            episode_lengths[env_idx] = 0
                        else:
                            next_obs_by_env.append(canonicalize_observations(obs, agent_ids))
                            next_done_by_env.append({agent_id: 0.0 for agent_id in agent_ids})

                    for agent_id in agent_ids:
                        rollout_buffers[agent_id]["obs"].append(obs_batches[agent_id])
                        rollout_buffers[agent_id]["reset"].append(reset_batches[agent_id])
                        rollout_buffers[agent_id]["action"].append(step_actions[agent_id])
                        rollout_buffers[agent_id]["opponent_index"].append(opponent_index_batches[agent_id])
                        rollout_buffers[agent_id]["reward"].append(reward_batches[agent_id])
                        rollout_buffers[agent_id]["done"].append(done_batches[agent_id])
                    if config["COUNTERFACTUAL_REPLAY"]:
                        label_batches.append(step_true_props)

                    last_obs_by_env = next_obs_by_env
                    last_done_by_env = next_done_by_env

                for agent_id in agent_ids:
                    obs_array = np.stack(rollout_buffers[agent_id]["obs"], axis=0)
                    reset_array = np.stack(rollout_buffers[agent_id]["reset"], axis=0)
                    action_array = np.stack(rollout_buffers[agent_id]["action"], axis=0)
                    opponent_index_array = np.stack(
                        rollout_buffers[agent_id]["opponent_index"],
                        axis=0,
                    )
                    reward_array = np.stack(rollout_buffers[agent_id]["reward"], axis=0)
                    done_array = np.stack(rollout_buffers[agent_id]["done"], axis=0)
                    final_next_obs = np.stack(
                        [env_obs[agent_id] for env_obs in last_obs_by_env],
                        axis=0,
                    ).astype(np.float32)
                    next_obs_array = np.concatenate(
                        [obs_array[1:], final_next_obs[None, ...]],
                        axis=0,
                    )

                    for env_idx in range(int(config["NUM_ENVS"])):
                        factual_sequence = {
                            "obs": obs_array[:, env_idx, ...].astype(np.float32),
                            "reset": reset_array[:, env_idx].astype(np.float32),
                            "action": action_array[:, env_idx].astype(np.int32),
                            "opponent_index": opponent_index_array[:, env_idx].astype(np.int32),
                            "reward": reward_array[:, env_idx].astype(np.float32),
                            "done": done_array[:, env_idx].astype(np.float32),
                            "next_obs": next_obs_array[:, env_idx, ...].astype(np.float32),
                        }
                        if config["COUNTERFACTUAL_REPLAY"]:
                            obs_base_chunk, obs_state_ids = split_monitor_state(
                                obs_array[:, env_idx, ...].astype(np.float32)
                            )
                            final_next_base_obs, final_next_state_id = split_monitor_state(
                                final_next_obs[env_idx].astype(np.float32)
                            )
                            labels_sequence = [
                                label_batches[step_idx][env_idx] for step_idx in range(int(config["NUM_STEPS"]))
                            ]
                            sequences = counterfactual_pr2_sequences_for_agent(
                                model=counterfactual_models[agent_id],
                                obs_base_chunk=obs_base_chunk,
                                obs_state_ids=np.asarray(obs_state_ids, dtype=np.int32),
                                reset_chunk=factual_sequence["reset"],
                                action_chunk=factual_sequence["action"],
                                opponent_index_chunk=factual_sequence["opponent_index"],
                                reward_chunk=factual_sequence["reward"],
                                done_chunk=factual_sequence["done"],
                                final_next_base_obs=final_next_base_obs,
                                final_next_state_id=int(final_next_state_id),
                                labels_sequence=labels_sequence,
                            )
                            for sequence_index, sequence in enumerate(sequences):
                                materialized = {
                                    "obs": append_monitor_state(
                                        sequence["obs_base"],
                                        sequence["obs_state_ids"],
                                    ).astype(np.float32),
                                    "reset": sequence["reset"],
                                    "action": sequence["action"],
                                    "reward": sequence["reward"],
                                    "done": sequence["done"],
                                    "next_obs": append_monitor_state(
                                        sequence["next_obs_base"],
                                        sequence["next_obs_state_ids"],
                                    ).astype(np.float32),
                                    "opponent_index": sequence["opponent_index"],
                                }
                                if sequence_index == 0:
                                    buffers[agent_id].add(
                                        materialized,
                                        stream_id=env_idx,
                                    )
                                else:
                                    synthetic_stream_id = next_synthetic_stream_id[agent_id]
                                    buffers[agent_id].add(
                                        materialized,
                                        stream_id=synthetic_stream_id,
                                        force_episode_start=True,
                                    )
                                    next_synthetic_stream_id[agent_id] += 1
                        else:
                            buffers[agent_id].add(
                                factual_sequence,
                                stream_id=env_idx,
                            )

                current_timesteps = train_states[agent_ids[0]].timesteps + int(config["NUM_STEPS"]) * int(
                    config["NUM_ENVS"]
                )

                agent_loss_info: dict[str, list[dict[str, float]]] = {
                    agent_id: [] for agent_id in active_trainable_agent_ids
                }
                can_learn = current_timesteps >= int(config["LEARNING_STARTS"]) and all(
                    buffers[agent_id].can_sample(
                        sequence_length=int(config["REPLAY_SEQUENCE_LENGTH"]),
                        burn_in_steps=int(config["BURN_IN_STEPS"]),
                    )
                    for agent_id in active_trainable_agent_ids
                )

                if can_learn:
                    for agent_id in active_trainable_agent_ids:
                        sample = stack_replay_samples(
                            [
                                buffers[agent_id].sample(
                                    batch_size=int(config["BUFFER_BATCH_SIZE"]),
                                    sequence_length=int(config["REPLAY_SEQUENCE_LENGTH"]),
                                    burn_in_steps=int(config["BURN_IN_STEPS"]),
                                    rng=buffer_rng,
                                )
                                for _ in range(int(config["NUM_EPOCHS"]))
                            ]
                        )
                        batch = PR2ChunkBatch(
                            obs=jnp.asarray(
                                np.swapaxes(sample["obs"], 1, 2),
                                dtype=jnp.float32,
                            ),
                            reset=jnp.asarray(
                                np.swapaxes(sample["reset"], 1, 2),
                                dtype=jnp.float32,
                            ),
                            action=jnp.asarray(
                                np.swapaxes(sample["action"], 1, 2),
                                dtype=jnp.int32,
                            ),
                            opponent_index=jnp.asarray(
                                np.swapaxes(sample["opponent_index"], 1, 2),
                                dtype=jnp.int32,
                            ),
                            reward=jnp.asarray(
                                np.swapaxes(sample["reward"], 1, 2),
                                dtype=jnp.float32,
                            ),
                            done=jnp.asarray(
                                np.swapaxes(sample["done"], 1, 2),
                                dtype=jnp.float32,
                            ),
                            final_next_obs=jnp.asarray(
                                sample["next_obs"][:, :, -1, ...],
                                dtype=jnp.float32,
                            ),
                            loss_mask=jnp.asarray(
                                np.swapaxes(sample["loss_mask"], 1, 2),
                                dtype=jnp.float32,
                            ),
                        )
                        train_states[agent_id], loss_info = update_fns[agent_id](
                            train_states[agent_id],
                            batch,
                        )
                        agent_loss_info[agent_id].append({key: float(value) for key, value in loss_info.items()})

                        if ((update_index + 1) % int(config["TARGET_UPDATE_INTERVAL"])) == 0:
                            train_states[agent_id] = train_states[agent_id].replace(
                                target_network_params=optax.incremental_update(
                                    train_states[agent_id].params,
                                    train_states[agent_id].target_network_params,
                                    float(config["TAU"]),
                                )
                            )

                for agent_id in agent_ids:
                    train_states[agent_id] = train_states[agent_id].replace(
                        timesteps=current_timesteps,
                        n_updates=train_states[agent_id].n_updates + 1,
                    )

                trainable_list = list(active_trainable_agent_ids)
                if can_learn and trainable_list:
                    loss_values = [info["loss"] for agent_id in trainable_list for info in agent_loss_info[agent_id]]
                    qval_values = [info["qvals"] for agent_id in trainable_list for info in agent_loss_info[agent_id]]
                    loss_mean = float(np.mean(np.asarray(loss_values, dtype=np.float32)))
                    qvals_mean = float(np.mean(np.asarray(qval_values, dtype=np.float32)))
                else:
                    loss_mean = 0.0
                    qvals_mean = 0.0

                metric = {
                    "episode_return_mean": mean_or_nan(completed_episode_returns),
                    "episode_length_mean": mean_or_nan(completed_episode_lengths),
                    "completed_episodes": float(len(completed_episode_returns)),
                    "loss": loss_mean,
                    "qvals": qvals_mean,
                    "epsilon": epsilon,
                    "per_agent_episode_return_mean": {
                        agent_id: mean_or_nan(completed_returns_by_agent[agent_id]) for agent_id in agent_ids
                    },
                }

                if bool(config.get("TEST_DURING_TRAINING", True)):
                    test_interval = max(
                        1,
                        int(int(config["NUM_UPDATES"]) * float(config.get("TEST_INTERVAL", 1.0))),
                    )
                    if (
                        update_index == 0
                        or (update_index + 1) % test_interval == 0
                        or (update_index + 1) == int(config["NUM_UPDATES"])
                    ):
                        metric.update(_evaluate_greedy(base_seed + 100_000 + update_index * 100))

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
    "IQLTrainState",
    "OpponentJointActionIndexer",
    "PR2ChunkBatch",
    "PR2MLPQNetwork",
    "PR2_REPLAY_FIELDS",
    "PR2RNNQNetwork",
    "counterfactual_pr2_sequences_for_agent",
    "make_pr2_action_selection_fn",
    "make_train",
    "make_update_fn",
]
