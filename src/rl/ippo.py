"""
PettingZoo-compatible independent PPO trainer.

The original version of this module was written around JAXMARL's pure-JAX
environment API. This version keeps the JAX/Flax PPO updates, but interacts
with PettingZoo ParallelEnv instances from Python.
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
from flax.training.train_state import TrainState

from src.monitor.counterfactual import (
    CounterfactualMonitorModel,
    append_monitor_state,
    enumerate_counterfactual_trajectories,
    extract_counterfactual_models,
    extract_counterfactual_true_props,
    next_state_ids_from_final_state,
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
    nonnegative_float,
    positive_float,
    positive_int,
    possible_agent_ids,
    require_config_keys,
    reset_env,
    restore_train_state,
    space_dimensions,
    unit_interval_float,
    validate_rollout_budget,
)
from src.rl.trajectory import (
    categorical_entropy,
    categorical_log_prob,
    clipped_value_loss,
    compute_gae,
    masked_mean,
    normalize_masked,
)


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return actor_mean, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray


def _counterfactual_ippo_trajectories_for_agent(
    *,
    model: CounterfactualMonitorModel,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    reward_array: np.ndarray,
    done_array: np.ndarray,
    final_next_obs: np.ndarray,
    label_batches: Sequence[Sequence[frozenset[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_steps, num_envs = done_array.shape
    trajectories: list[dict[str, np.ndarray]] = []

    for env_idx in range(num_envs):
        obs_chunk = obs_array[:, env_idx, ...].astype(np.float32)
        action_chunk = action_array[:, env_idx].astype(np.int32)
        reward_chunk = reward_array[:, env_idx].astype(np.float32)
        done_chunk = done_array[:, env_idx].astype(np.float32)
        final_next_obs_chunk = final_next_obs[env_idx].astype(np.float32)
        reset_chunk = np.zeros((num_steps,), dtype=np.float32)
        if num_steps > 1:
            reset_chunk[1:] = done_chunk[:-1]

        obs_base_chunk, factual_obs_state_ids = split_monitor_state(obs_chunk)
        final_next_base_obs, factual_final_next_state_id = split_monitor_state(final_next_obs_chunk)
        factual_next_obs_state_ids = next_state_ids_from_final_state(
            factual_obs_state_ids,
            factual_final_next_state_id,
        )
        labels_sequence = [label_batches[step_idx][env_idx] for step_idx in range(num_steps)]
        relabeled = enumerate_counterfactual_trajectories(
            model=model,
            factual_obs_state_ids=factual_obs_state_ids,
            factual_next_obs_state_ids=factual_next_obs_state_ids,
            factual_rewards=reward_chunk,
            labels_sequence=labels_sequence,
            done=done_chunk,
            reset=reset_chunk,
        )

        for trajectory in relabeled:
            trajectories.append(
                {
                    "obs_base": np.asarray(obs_base_chunk, dtype=np.float32),
                    "obs_state_ids": np.asarray(trajectory.obs_state_ids, dtype=np.int32),
                    "action": np.asarray(action_chunk, dtype=np.int32),
                    "reward": np.asarray(
                        reward_chunk if trajectory.factual else trajectory.rewards,
                        dtype=np.float32,
                    ),
                    "done": np.asarray(done_chunk, dtype=np.float32),
                    "last_obs_base": np.asarray(final_next_base_obs, dtype=np.float32),
                    "last_obs_state_id": np.asarray(
                        (factual_final_next_state_id if trajectory.factual else int(trajectory.next_obs_state_ids[-1])),
                        dtype=np.int32,
                    ),
                }
            )

    return (
        np.stack([trajectory["obs_base"] for trajectory in trajectories], axis=1),
        np.stack([trajectory["obs_state_ids"] for trajectory in trajectories], axis=1),
        np.stack([trajectory["action"] for trajectory in trajectories], axis=1),
        np.stack([trajectory["reward"] for trajectory in trajectories], axis=1),
        np.stack([trajectory["done"] for trajectory in trajectories], axis=1),
        np.stack([trajectory["last_obs_base"] for trajectory in trajectories], axis=0),
        np.stack([trajectory["last_obs_state_id"] for trajectory in trajectories], axis=0),
    )


def _recompute_ippo_rollout_statistics(
    *,
    network: ActorCritic,
    params: Any,
    obs: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_obs = jnp.asarray(obs.reshape((-1, obs.shape[-1])), dtype=jnp.float32)
    flat_action = jnp.asarray(action.reshape((-1,)), dtype=jnp.int32)
    logits, value = network.apply(params, flat_obs)
    log_prob = categorical_log_prob(logits, flat_action)
    return (
        np.asarray(value, dtype=np.float32).reshape(action.shape),
        np.asarray(log_prob, dtype=np.float32).reshape(action.shape),
    )


def _pad_transition_batch(
    traj_batch: Transition,
    last_obs: jnp.ndarray,
    *,
    num_minibatches: int,
    min_trajectory_count: int | None = None,
) -> tuple[Transition, jnp.ndarray]:
    num_steps = int(traj_batch.obs.shape[0])
    traj_count = int(traj_batch.obs.shape[1])
    target_traj_count = max(traj_count, int(min_trajectory_count or traj_count))
    while (num_steps * target_traj_count) % int(num_minibatches) != 0:
        target_traj_count += 1

    if target_traj_count == traj_count:
        return traj_batch, last_obs

    pad_count = target_traj_count - traj_count
    pad_indices = jnp.asarray(np.arange(pad_count) % traj_count, dtype=jnp.int32)
    padded_values = {}
    for field_name, value in zip(traj_batch._fields, traj_batch, strict=True):
        pad_values = jnp.take(value, pad_indices, axis=1)
        if field_name == "mask":
            pad_values = jnp.zeros_like(pad_values)
        padded_values[field_name] = jnp.concatenate([value, pad_values], axis=1)
    padded_batch = Transition(**padded_values)
    padded_last_obs = jnp.concatenate(
        [last_obs, jnp.take(last_obs, pad_indices, axis=0)],
        axis=0,
    )
    return padded_batch, padded_last_obs


def _make_update_fn(network: ActorCritic, config: dict[str, Any]):
    def _loss_fn(
        params: Any,
        traj_batch: Transition,
        advantages: jnp.ndarray,
        targets: jnp.ndarray,
    ):
        logits, value = network.apply(params, traj_batch.obs)
        log_prob = categorical_log_prob(logits, traj_batch.action)
        mask = traj_batch.mask
        value_loss = clipped_value_loss(
            value=value,
            old_value=traj_batch.value,
            target=targets,
            mask=mask,
            clip_eps=config["CLIP_EPS"],
        )
        normalized_adv = normalize_masked(advantages, mask)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        loss_actor1 = ratio * normalized_adv
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * normalized_adv
        )
        actor_loss = -masked_mean(jnp.minimum(loss_actor1, loss_actor2), mask)
        entropy = masked_mean(categorical_entropy(logits), mask)

        total_loss = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        aux = {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": value_loss,
            "entropy": entropy,
            "ratio": masked_mean(ratio, mask),
        }
        return total_loss, aux

    @jax.jit
    def _update_agent(
        train_state: TrainState,
        traj_batch: Transition,
        last_obs: jnp.ndarray,
        rng: jax.Array,
    ):
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, targets = compute_gae(
            reward=traj_batch.reward,
            value=traj_batch.value,
            done=traj_batch.done,
            last_value=last_val,
            gamma=config["GAMMA"],
            gae_lambda=config["GAE_LAMBDA"],
        )

        def _update_epoch(
            update_state: tuple[TrainState, Transition, jnp.ndarray, jnp.ndarray, jax.Array],
            _,
        ):
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, perm_rng = jax.random.split(rng)

            batch_size = traj_batch.obs.shape[0] * traj_batch.obs.shape[1]
            minibatch_size = batch_size // config["NUM_MINIBATCHES"]
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                batch,
            )

            permutation = jax.random.permutation(perm_rng, batch_size)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                batch,
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x,
                    (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:],
                ),
                shuffled_batch,
            )

            def _update_minibatch(train_state: TrainState, batch_info):
                traj_mb, adv_mb, targets_mb = batch_info
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (_, loss_info), grads = grad_fn(train_state.params, traj_mb, adv_mb, targets_mb)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss_info

            train_state, loss_info = jax.lax.scan(
                _update_minibatch,
                train_state,
                minibatches,
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch,
            update_state,
            None,
            length=config["UPDATE_EPOCHS"],
        )
        loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
        return update_state[0], loss_info

    return _update_agent


def validate_ippo_config(
    config: Mapping[str, Any],
    *,
    algorithm: str = "IPPO",
) -> dict[str, Any]:
    """Validate and normalize the complete IPPO trainer configuration."""

    normalized = dict(config)
    require_config_keys(
        normalized,
        (
            "NUM_ENVS",
            "TOTAL_TIMESTEPS",
            "NUM_STEPS",
            "NUM_MINIBATCHES",
            "UPDATE_EPOCHS",
            "LR",
            "MAX_GRAD_NORM",
            "GAMMA",
            "GAE_LAMBDA",
            "CLIP_EPS",
            "VF_COEF",
            "ENT_COEF",
        ),
        algorithm=algorithm,
    )
    normalized["ACTIVATION"] = str(normalized.get("ACTIVATION", "tanh")).lower()
    normalized.setdefault("ANNEAL_LR", False)
    normalized.setdefault("COUNTERFACTUAL_REPLAY", False)
    if normalized["ACTIVATION"] not in {"tanh", "relu"}:
        raise ValueError(f"{algorithm} ACTIVATION must be one of {{'tanh', 'relu'}}.")
    for key in ("ANNEAL_LR", "COUNTERFACTUAL_REPLAY"):
        if not isinstance(normalized[key], (bool, np.bool_)):
            raise ValueError(f"{algorithm} {key} must be a boolean.")

    normalized["NUM_UPDATES"] = validate_rollout_budget(
        normalized,
        algorithm=algorithm,
    )
    normalized["NUM_ENVS"] = positive_int(
        normalized,
        "NUM_ENVS",
        algorithm=algorithm,
    )
    normalized["NUM_STEPS"] = positive_int(
        normalized,
        "NUM_STEPS",
        algorithm=algorithm,
    )
    normalized["BATCH_SIZE"] = normalized["NUM_ENVS"] * normalized["NUM_STEPS"]
    normalized["NUM_MINIBATCHES"] = positive_int(
        normalized,
        "NUM_MINIBATCHES",
        algorithm=algorithm,
    )
    normalized["UPDATE_EPOCHS"] = positive_int(
        normalized,
        "UPDATE_EPOCHS",
        algorithm=algorithm,
    )
    if not normalized["COUNTERFACTUAL_REPLAY"] and normalized["BATCH_SIZE"] % normalized["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            f"{algorithm} requires NUM_ENVS * NUM_STEPS to be divisible by "
            "NUM_MINIBATCHES when counterfactual replay is disabled."
        )
    normalized["LR"] = positive_float(normalized, "LR", algorithm=algorithm)
    normalized["MAX_GRAD_NORM"] = positive_float(
        normalized,
        "MAX_GRAD_NORM",
        algorithm=algorithm,
    )
    normalized["GAMMA"] = unit_interval_float(
        normalized,
        "GAMMA",
        algorithm=algorithm,
    )
    normalized["GAE_LAMBDA"] = unit_interval_float(
        normalized,
        "GAE_LAMBDA",
        algorithm=algorithm,
    )
    normalized["CLIP_EPS"] = positive_float(
        normalized,
        "CLIP_EPS",
        algorithm=algorithm,
    )
    normalized["VF_COEF"] = nonnegative_float(
        normalized,
        "VF_COEF",
        algorithm=algorithm,
    )
    normalized["ENT_COEF"] = nonnegative_float(
        normalized,
        "ENT_COEF",
        algorithm=algorithm,
    )
    normalized["MINIBATCH_SIZE"] = max(
        1,
        normalized["BATCH_SIZE"] // normalized["NUM_MINIBATCHES"],
    )
    normalized["SCHEDULE_TOTAL_UPDATES"] = positive_int(
        {
            "SCHEDULE_TOTAL_UPDATES": normalized.get(
                "SCHEDULE_TOTAL_UPDATES",
                normalized["NUM_UPDATES"],
            )
        },
        "SCHEDULE_TOTAL_UPDATES",
        algorithm=algorithm,
    )
    return normalized


def make_train(config: dict[str, Any], env_factory: EnvFactory):
    config = validate_ippo_config(config)

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

    config["NUM_AGENTS"] = len(agent_ids)
    activation = config["ACTIVATION"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["SCHEDULE_TOTAL_UPDATES"]
        return jnp.asarray(config["LR"], dtype=jnp.float32) * jnp.maximum(0.0, frac)

    def build_optimizer():
        learning_rate = linear_schedule if config.get("ANNEAL_LR", False) else config["LR"]
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )

    def train(
        seed: Any | None = None,
        *,
        initial_train_states: Mapping[str, TrainState | Mapping[str, Any]] | None = None,
        trainable_agent_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        on_update: UpdateCallback | None = None,
        capture_env_histories: bool = False,
    ):
        base_seed = coerce_seed(seed if seed is not None else config.get("SEED", 0))
        rng = jax.random.PRNGKey(base_seed)
        active_trainable_agent_ids = coerce_trainable_agent_ids(
            agent_ids,
            trainable_agent_ids,
        )

        networks = {agent_id: ActorCritic(action_dims[agent_id], activation=activation) for agent_id in agent_ids}
        update_fns = {agent_id: _make_update_fn(networks[agent_id], config) for agent_id in agent_ids}

        train_states: dict[str, TrainState] = {}
        for agent_id in agent_ids:
            rng, init_rng = jax.random.split(rng)
            init_obs = jnp.zeros((obs_dims[agent_id],), dtype=jnp.float32)
            params = networks[agent_id].init(init_rng, init_obs)
            template_state = TrainState.create(
                apply_fn=networks[agent_id].apply,
                params=params,
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

        envs = [env_factory(base_seed + idx) for idx in range(config["NUM_ENVS"])]
        episode_returns = [{agent_id: 0.0 for agent_id in agent_ids} for _ in envs]
        episode_lengths = [0 for _ in envs]
        last_obs_by_env: list[dict[str, np.ndarray]] = []
        next_reset_seed = base_seed + config["NUM_ENVS"]

        try:
            for env_idx, env in enumerate(envs):
                obs, _ = reset_env(env, base_seed + env_idx)
                last_obs_by_env.append(canonicalize_observations(obs, agent_ids))

            metric_history: dict[str, list[float]] = defaultdict(list)
            per_agent_return_history = {agent_id: [] for agent_id in agent_ids}

            for update_index in range(config["NUM_UPDATES"]):
                rollout_buffers = {
                    agent_id: {
                        "done": [],
                        "action": [],
                        "value": [],
                        "reward": [],
                        "log_prob": [],
                        "obs": [],
                    }
                    for agent_id in agent_ids
                }
                label_batches: list[list[frozenset[str]]] = []

                completed_episode_returns: list[float] = []
                completed_episode_lengths: list[float] = []
                completed_returns_by_agent = {agent_id: [] for agent_id in agent_ids}

                for _ in range(config["NUM_STEPS"]):
                    obs_batches = {
                        agent_id: np.stack(
                            [env_obs[agent_id] for env_obs in last_obs_by_env],
                            axis=0,
                        ).astype(np.float32)
                        for agent_id in agent_ids
                    }

                    step_actions: dict[str, np.ndarray] = {}
                    step_values: dict[str, np.ndarray] = {}
                    step_log_probs: dict[str, np.ndarray] = {}

                    for agent_id in agent_ids:
                        rng, action_rng = jax.random.split(rng)
                        logits, value = networks[agent_id].apply(
                            train_states[agent_id].params,
                            jnp.asarray(obs_batches[agent_id]),
                        )
                        action = jax.random.categorical(action_rng, logits, axis=-1)
                        log_prob = jax.nn.log_softmax(logits)
                        log_prob = jnp.take_along_axis(
                            log_prob,
                            action[..., None],
                            axis=-1,
                        ).squeeze(axis=-1)

                        step_actions[agent_id] = np.asarray(action, dtype=np.int32)
                        step_values[agent_id] = np.asarray(value, dtype=np.float32)
                        step_log_probs[agent_id] = np.asarray(log_prob, dtype=np.float32)

                    reward_batches = {
                        agent_id: np.zeros((config["NUM_ENVS"],), dtype=np.float32) for agent_id in agent_ids
                    }
                    done_batches = {
                        agent_id: np.zeros((config["NUM_ENVS"],), dtype=np.float32) for agent_id in agent_ids
                    }

                    next_obs_by_env: list[dict[str, np.ndarray]] = []
                    step_true_props: list[frozenset[str]] = []

                    for env_idx, env in enumerate(envs):
                        env_actions = {agent_id: int(step_actions[agent_id][env_idx]) for agent_id in agent_ids}
                        obs, rewards, terminations, truncations, infos = env.step(env_actions)
                        if config["COUNTERFACTUAL_REPLAY"]:
                            step_true_props.append(extract_counterfactual_true_props(infos, agent_ids))

                        episode_lengths[env_idx] += 1
                        for agent_id in agent_ids:
                            reward = float(rewards.get(agent_id, 0.0))
                            episode_returns[env_idx][agent_id] += reward
                            reward_batches[agent_id][env_idx] = reward

                        env_done = any(
                            bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                            for agent_id in agent_ids
                        )

                        if env_done:
                            episode_return_values = [episode_returns[env_idx][agent_id] for agent_id in agent_ids]
                            completed_episode_returns.append(
                                float(np.mean(np.asarray(episode_return_values, dtype=np.float32)))
                            )
                            completed_episode_lengths.append(float(episode_lengths[env_idx]))
                            for agent_id in agent_ids:
                                done_batches[agent_id][env_idx] = 1.0
                                completed_returns_by_agent[agent_id].append(episode_returns[env_idx][agent_id])

                            reset_obs, _ = reset_env(env, next_reset_seed)
                            next_reset_seed += 1
                            next_obs_by_env.append(canonicalize_observations(reset_obs, agent_ids))
                            episode_returns[env_idx] = {agent_id: 0.0 for agent_id in agent_ids}
                            episode_lengths[env_idx] = 0
                        else:
                            next_obs_by_env.append(canonicalize_observations(obs, agent_ids))

                    for agent_id in agent_ids:
                        rollout_buffers[agent_id]["done"].append(done_batches[agent_id])
                        rollout_buffers[agent_id]["action"].append(step_actions[agent_id])
                        rollout_buffers[agent_id]["value"].append(step_values[agent_id])
                        rollout_buffers[agent_id]["reward"].append(reward_batches[agent_id])
                        rollout_buffers[agent_id]["log_prob"].append(step_log_probs[agent_id])
                        rollout_buffers[agent_id]["obs"].append(obs_batches[agent_id])
                    if config["COUNTERFACTUAL_REPLAY"]:
                        label_batches.append(step_true_props)

                    last_obs_by_env = next_obs_by_env

                agent_loss_info: dict[str, dict[str, float]] = {}
                for agent_id in active_trainable_agent_ids:
                    obs_array = np.stack(rollout_buffers[agent_id]["obs"], axis=0)
                    action_array = np.stack(rollout_buffers[agent_id]["action"], axis=0)
                    reward_array = np.stack(rollout_buffers[agent_id]["reward"], axis=0)
                    done_array = np.stack(rollout_buffers[agent_id]["done"], axis=0)
                    final_next_obs = np.stack(
                        [env_obs[agent_id] for env_obs in last_obs_by_env],
                        axis=0,
                    ).astype(np.float32)

                    if config["COUNTERFACTUAL_REPLAY"]:
                        (
                            batch_obs_base,
                            batch_obs_state_ids,
                            batch_action,
                            batch_reward,
                            batch_done,
                            batch_last_obs_base,
                            batch_last_obs_state_ids,
                        ) = _counterfactual_ippo_trajectories_for_agent(
                            model=counterfactual_models[agent_id],
                            obs_array=obs_array,
                            action_array=action_array,
                            reward_array=reward_array,
                            done_array=done_array,
                            final_next_obs=final_next_obs,
                            label_batches=label_batches,
                        )
                        batch_obs = append_monitor_state(
                            batch_obs_base,
                            batch_obs_state_ids,
                        )
                        batch_last_obs = append_monitor_state(
                            batch_last_obs_base,
                            batch_last_obs_state_ids,
                        )
                        batch_value, batch_log_prob = _recompute_ippo_rollout_statistics(
                            network=networks[agent_id],
                            params=train_states[agent_id].params,
                            obs=batch_obs,
                            action=batch_action,
                        )
                    else:
                        batch_obs = obs_array.astype(np.float32)
                        batch_action = action_array.astype(np.int32)
                        batch_reward = reward_array.astype(np.float32)
                        batch_done = done_array.astype(np.float32)
                        batch_last_obs = final_next_obs
                        batch_value = np.stack(rollout_buffers[agent_id]["value"], axis=0)
                        batch_log_prob = np.stack(
                            rollout_buffers[agent_id]["log_prob"],
                            axis=0,
                        )

                    traj_batch = Transition(
                        done=jnp.asarray(batch_done, dtype=jnp.float32),
                        action=jnp.asarray(batch_action, dtype=jnp.int32),
                        value=jnp.asarray(batch_value, dtype=jnp.float32),
                        reward=jnp.asarray(batch_reward, dtype=jnp.float32),
                        log_prob=jnp.asarray(batch_log_prob, dtype=jnp.float32),
                        obs=jnp.asarray(batch_obs, dtype=jnp.float32),
                        mask=jnp.ones_like(jnp.asarray(batch_done, dtype=jnp.float32)),
                    )
                    last_obs = jnp.asarray(batch_last_obs, dtype=jnp.float32)
                    traj_batch, last_obs = _pad_transition_batch(
                        traj_batch,
                        last_obs,
                        num_minibatches=int(config["NUM_MINIBATCHES"]),
                        min_trajectory_count=(
                            int(config["NUM_ENVS"]) * (len(counterfactual_models[agent_id].state_ids) + 1)
                            if config["COUNTERFACTUAL_REPLAY"]
                            else None
                        ),
                    )
                    rng, update_rng = jax.random.split(rng)
                    train_states[agent_id], loss_info = update_fns[agent_id](
                        train_states[agent_id],
                        traj_batch,
                        last_obs,
                        update_rng,
                    )
                    agent_loss_info[agent_id] = {key: float(value) for key, value in loss_info.items()}

                loss_agents = list(active_trainable_agent_ids)
                metric = {
                    "episode_return_mean": mean_or_nan(completed_episode_returns),
                    "episode_length_mean": mean_or_nan(completed_episode_lengths),
                    "completed_episodes": float(len(completed_episode_returns)),
                    "total_loss": float(np.mean([agent_loss_info[agent_id]["total_loss"] for agent_id in loss_agents])),
                    "actor_loss": float(np.mean([agent_loss_info[agent_id]["actor_loss"] for agent_id in loss_agents])),
                    "critic_loss": float(
                        np.mean([agent_loss_info[agent_id]["critic_loss"] for agent_id in loss_agents])
                    ),
                    "entropy": float(np.mean([agent_loss_info[agent_id]["entropy"] for agent_id in loss_agents])),
                    "ratio": float(np.mean([agent_loss_info[agent_id]["ratio"] for agent_id in loss_agents])),
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
                    on_update(update_index + 1, config["NUM_UPDATES"], metric)

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


__all__ = ["ActorCritic", "Transition", "make_train", "validate_ippo_config"]
