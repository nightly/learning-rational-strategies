from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class OpponentJointActionIndexer:
    agent_id: str
    agent_ids: tuple[str, ...]
    action_dims: tuple[int, ...]
    opponent_agent_ids: tuple[str, ...]
    opponent_action_dims: tuple[int, ...]
    opponent_joint_size: int
    _strides: tuple[int, ...]

    def encode_opponent_actions(self, joint_actions: dict[str, int] | tuple[int, ...]) -> int:
        if isinstance(joint_actions, dict):
            actions = tuple(int(joint_actions[agent_id]) for agent_id in self.opponent_agent_ids)
        else:
            actions = tuple(int(value) for value in joint_actions)

        if len(actions) != len(self.opponent_agent_ids):
            raise ValueError(
                f"Expected {len(self.opponent_agent_ids)} opponent actions for "
                f"agent '{self.agent_id}', got {len(actions)}."
            )

        index = 0
        for action, action_dim, stride in zip(
            actions,
            self.opponent_action_dims,
            self._strides,
            strict=True,
        ):
            if action < 0 or action >= int(action_dim):
                raise ValueError(
                    f"Opponent action {action} is out of bounds for dim {action_dim}."
                )
            index += int(action) * int(stride)
        return int(index)

    def decode_opponent_index(self, index: int) -> tuple[int, ...]:
        if index < 0 or index >= self.opponent_joint_size:
            raise ValueError(
                f"Opponent joint-action index {index} is out of bounds for size "
                f"{self.opponent_joint_size}."
            )

        remainder = int(index)
        actions: list[int] = []
        for action_dim, stride in zip(
            self.opponent_action_dims,
            self._strides,
            strict=True,
        ):
            actions.append((remainder // int(stride)) % int(action_dim))
        return tuple(actions)

    def enumerate_opponent_actions(self) -> list[tuple[int, ...]]:
        return [
            self.decode_opponent_index(index)
            for index in range(self.opponent_joint_size)
        ]


def build_opponent_indexers(
    agent_ids: list[str] | tuple[str, ...],
    action_dims: dict[str, int],
) -> dict[str, OpponentJointActionIndexer]:
    ordered_agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
    action_shape = tuple(int(action_dims[agent_id]) for agent_id in ordered_agent_ids)
    indexers: dict[str, OpponentJointActionIndexer] = {}

    for agent_id in ordered_agent_ids:
        opponent_agent_ids = tuple(other for other in ordered_agent_ids if other != agent_id)
        opponent_action_dims = tuple(int(action_dims[other]) for other in opponent_agent_ids)

        strides: list[int] = []
        running = 1
        for action_dim in opponent_action_dims:
            strides.append(running)
            running *= int(action_dim)

        indexers[agent_id] = OpponentJointActionIndexer(
            agent_id=agent_id,
            agent_ids=ordered_agent_ids,
            action_dims=action_shape,
            opponent_agent_ids=opponent_agent_ids,
            opponent_action_dims=opponent_action_dims,
            opponent_joint_size=int(max(1, running)),
            _strides=tuple(int(stride) for stride in strides),
        )

    return indexers


def reshape_joint_q(
    q_values: jnp.ndarray,
    *,
    own_action_dim: int,
    opponent_joint_size: int,
) -> jnp.ndarray:
    return jnp.reshape(q_values, q_values.shape[:-1] + (int(own_action_dim), int(opponent_joint_size)))


def response_probs_from_joint_q(joint_q: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(joint_q, axis=-1)


def response_aware_action_values(joint_q: jnp.ndarray) -> jnp.ndarray:
    rho = response_probs_from_joint_q(joint_q)
    return jnp.sum(rho * joint_q, axis=-1)


def response_aware_state_values(
    actor_logits: jnp.ndarray,
    joint_q: jnp.ndarray,
) -> jnp.ndarray:
    action_probs = jax.nn.softmax(actor_logits, axis=-1)
    q_pr2 = response_aware_action_values(joint_q)
    return jnp.sum(action_probs * q_pr2, axis=-1)


def select_joint_q_value(
    joint_q: jnp.ndarray,
    own_actions: jnp.ndarray,
    opponent_indices: jnp.ndarray,
) -> jnp.ndarray:
    own_actions = jnp.asarray(own_actions, dtype=jnp.int32)
    opponent_indices = jnp.asarray(opponent_indices, dtype=jnp.int32)
    gathered = jnp.take_along_axis(
        joint_q,
        own_actions[..., None, None],
        axis=-2,
    ).squeeze(axis=-2)
    return jnp.take_along_axis(
        gathered,
        opponent_indices[..., None],
        axis=-1,
    ).squeeze(axis=-1)


def greedy_pr2_actions(joint_q: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(response_aware_action_values(joint_q), axis=-1)


def expected_joint_q_np(joint_q: np.ndarray) -> np.ndarray:
    joint_q_arr = jnp.asarray(np.asarray(joint_q, dtype=np.float32))
    return np.asarray(response_aware_action_values(joint_q_arr), dtype=np.float32)


def encode_opponent_joint_action(
    *,
    indexers: dict[str, OpponentJointActionIndexer],
    agent_id: str,
    joint_actions: dict[str, int],
) -> int:
    return indexers[agent_id].encode_opponent_actions(joint_actions)


__all__ = [
    "OpponentJointActionIndexer",
    "build_opponent_indexers",
    "encode_opponent_joint_action",
    "expected_joint_q_np",
    "greedy_pr2_actions",
    "reshape_joint_q",
    "response_aware_action_values",
    "response_aware_state_values",
    "response_probs_from_joint_q",
    "select_joint_q_value",
]
