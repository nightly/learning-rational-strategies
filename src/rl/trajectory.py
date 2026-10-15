"""Trajectory math and contiguous sequence replay shared by RL trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import index

import jax
import jax.numpy as jnp
import numpy as np


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got a boolean.")
    try:
        return int(index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def categorical_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(logits)
    return -(probs * jax.nn.log_softmax(logits)).sum(axis=-1)


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    denominator = jnp.maximum(jnp.asarray(mask, dtype=jnp.float32).sum(), 1.0)
    return jnp.sum(values * mask) / denominator


def normalize_masked(values: jax.Array, mask: jax.Array) -> jax.Array:
    mean = masked_mean(values, mask)
    variance = masked_mean(jnp.square(values - mean), mask)
    return (values - mean) / jnp.sqrt(variance + 1e-8)


def compute_gae(
    *,
    reward: jax.Array,
    value: jax.Array,
    done: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute finite-episode GAE for time-major arrays."""

    def scan_step(
        carry: tuple[jax.Array, jax.Array],
        transition: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        gae, next_value = carry
        step_reward, step_value, step_done = transition
        continuation = 1.0 - step_done
        delta = step_reward + gamma * next_value * continuation - step_value
        gae = delta + gamma * gae_lambda * continuation * gae
        return (gae, step_value), gae

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_value), last_value),
        (reward, value, done),
        reverse=True,
    )
    return advantages, advantages + value


def clipped_value_loss(
    *,
    value: jax.Array,
    old_value: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    clip_eps: float,
) -> jax.Array:
    clipped = old_value + jnp.clip(value - old_value, -clip_eps, clip_eps)
    losses = jnp.square(value - target)
    clipped_losses = jnp.square(clipped - target)
    return 0.5 * masked_mean(jnp.maximum(losses, clipped_losses), mask)


@dataclass(frozen=True)
class _SequenceCandidate:
    context_indices: np.ndarray
    learn_indices: np.ndarray


class SequenceReplayBuffer:
    """Fixed-capacity transition replay with contiguous per-stream sampling.

    Each insertion belongs to a logical environment stream. Sampling never
    crosses ring-buffer gaps or switches streams. A learning window is preceded
    by up to ``burn_in_steps`` real transitions. Missing left context is only
    allowed when the first available transition is an explicit recurrent reset.
    """

    def __init__(self, capacity: int, *, required_fields: tuple[str, ...]):
        parsed_capacity = _strict_int(capacity, name="Replay capacity")
        if parsed_capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        if not required_fields or len(set(required_fields)) != len(required_fields):
            raise ValueError("required_fields must be non-empty and unique.")
        if "reset" not in required_fields:
            raise ValueError("Sequence replay requires a 'reset' field.")
        self.capacity = parsed_capacity
        self.required_fields = tuple(required_fields)
        self._storage: dict[str, np.ndarray] | None = None
        self._stream_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._step_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._next_step_by_stream: dict[int, int] = {}
        self._size = 0
        self._cursor = 0
        self._candidate_cache: dict[tuple[int, int], tuple[_SequenceCandidate, ...]] = {}

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        sequence: Mapping[str, np.ndarray],
        *,
        stream_id: int,
        force_episode_start: bool = False,
    ) -> None:
        fields = tuple(sequence)
        if set(fields) != set(self.required_fields):
            missing = sorted(set(self.required_fields) - set(fields))
            extra = sorted(set(fields) - set(self.required_fields))
            raise ValueError(f"Replay schema mismatch; missing={missing}, extra={extra}.")

        arrays = {name: np.asarray(sequence[name]).copy() for name in self.required_fields}
        scalar_fields = [name for name, value in arrays.items() if value.ndim == 0]
        if scalar_fields:
            raise ValueError(f"Replay fields require a leading time dimension: {scalar_fields}.")
        lengths = {name: int(value.shape[0]) for name, value in arrays.items()}
        if not lengths or len(set(lengths.values())) != 1:
            raise ValueError(f"Replay fields must share a leading length, got {lengths}.")
        item_count = next(iter(lengths.values()))
        if item_count <= 0:
            raise ValueError("Cannot add an empty replay sequence.")

        reset = arrays["reset"]
        if reset.ndim != 1:
            raise ValueError(f"Replay reset must have shape [time], got {reset.shape}.")
        if force_episode_start:
            reset[0] = 1

        if self._storage is None:
            self._storage = {
                name: np.empty((self.capacity,) + value.shape[1:], dtype=value.dtype) for name, value in arrays.items()
            }
        else:
            for name, value in arrays.items():
                expected = self._storage[name].shape[1:]
                if value.shape[1:] != expected:
                    raise ValueError(
                        f"Replay field {name!r} has trailing shape {value.shape[1:]}, expected {expected}."
                    )
                if value.dtype != self._storage[name].dtype:
                    raise ValueError(
                        f"Replay field {name!r} has dtype {value.dtype}, expected {self._storage[name].dtype}."
                    )

        stream = _strict_int(stream_id, name="Replay stream_id")
        next_step = self._next_step_by_stream.get(stream, 0)
        step_ids = np.arange(next_step, next_step + item_count, dtype=np.int64)
        self._next_step_by_stream[stream] = next_step + item_count

        if item_count >= self.capacity:
            source = slice(item_count - self.capacity, item_count)
            for name, value in arrays.items():
                self._storage[name][:] = value[source]
            self._stream_ids[:] = stream
            self._step_ids[:] = step_ids[source]
            self._size = self.capacity
            self._cursor = 0
            self._candidate_cache.clear()
            return

        remaining = item_count
        offset = 0
        while remaining:
            count = min(remaining, self.capacity - self._cursor)
            target = slice(self._cursor, self._cursor + count)
            source = slice(offset, offset + count)
            for name, value in arrays.items():
                self._storage[name][target] = value[source]
            self._stream_ids[target] = stream
            self._step_ids[target] = step_ids[source]
            self._cursor = (self._cursor + count) % self.capacity
            self._size = min(self.capacity, self._size + count)
            remaining -= count
            offset += count
        self._candidate_cache.clear()

    def _valid_indices(self) -> np.ndarray:
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)
        return np.arange(self.capacity, dtype=np.int64)

    def _candidates(
        self,
        sequence_length: int,
        burn_in_steps: int,
    ) -> tuple[_SequenceCandidate, ...]:
        key = (
            _strict_int(sequence_length, name="Replay sequence_length"),
            _strict_int(burn_in_steps, name="Replay burn_in_steps"),
        )
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        if self._storage is None:
            return ()

        learn_length, burn = key
        if learn_length <= 0 or burn < 0:
            return ()
        reset_store = np.asarray(self._storage["reset"], dtype=bool)
        candidates: list[_SequenceCandidate] = []
        valid = self._valid_indices()
        for stream in np.unique(self._stream_ids[valid]):
            indices = valid[self._stream_ids[valid] == stream]
            indices = indices[np.argsort(self._step_ids[indices])]
            if indices.size == 0:
                continue
            gaps = np.flatnonzero(np.diff(self._step_ids[indices]) != 1) + 1
            for segment in np.split(indices, gaps):
                if segment.size < learn_length:
                    continue
                for learn_start in range(segment.size - learn_length + 1):
                    context_start = max(0, learn_start - burn)
                    if learn_start - context_start < burn:
                        reset_positions = np.flatnonzero(reset_store[segment[: learn_start + 1]])
                        if reset_positions.size == 0:
                            continue
                        context_start = max(context_start, int(reset_positions[-1]))
                    candidates.append(
                        _SequenceCandidate(
                            context_indices=segment[context_start:learn_start],
                            learn_indices=segment[learn_start : learn_start + learn_length],
                        )
                    )
        result = tuple(candidates)
        self._candidate_cache[key] = result
        return result

    def can_sample(self, *, sequence_length: int, burn_in_steps: int) -> bool:
        return bool(self._candidates(sequence_length, burn_in_steps))

    def sample(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        burn_in_steps: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        batch = _strict_int(batch_size, name="Replay batch_size")
        learn = _strict_int(sequence_length, name="Replay sequence_length")
        burn = _strict_int(burn_in_steps, name="Replay burn_in_steps")
        if batch <= 0:
            raise ValueError("Replay batch_size must be positive.")
        if learn <= 0:
            raise ValueError("Replay sequence_length must be positive.")
        if burn < 0:
            raise ValueError("Replay burn_in_steps must be nonnegative.")
        candidates = self._candidates(learn, burn)
        if not candidates or self._storage is None:
            raise ValueError("Replay does not contain a complete contiguous sequence.")

        total = burn + learn
        sampled = {
            name: np.zeros((batch, total) + values.shape[1:], dtype=values.dtype)
            for name, values in self._storage.items()
        }
        sampled["reset"].fill(1)
        sampled["loss_mask"] = np.zeros((batch, total), dtype=np.float32)
        sampled["sequence_valid"] = np.zeros((batch, total), dtype=np.float32)

        choices = rng.integers(0, len(candidates), size=batch)
        for batch_index, choice in enumerate(choices):
            candidate = candidates[int(choice)]
            context = candidate.context_indices[-burn:] if burn else np.empty(0, dtype=np.int64)
            context_count = int(context.size)
            target_start = burn - context_count
            source = np.concatenate([context, candidate.learn_indices])
            target = slice(target_start, total)
            for name, values in self._storage.items():
                sampled[name][batch_index, target] = values[source]
            sampled["sequence_valid"][batch_index, target] = 1.0
            sampled["loss_mask"][batch_index, burn:] = 1.0
        return sampled


def stack_replay_samples(
    samples: list[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Expected at least one replay sample.")
    keys = tuple(samples[0])
    if any(tuple(sample) != keys for sample in samples[1:]):
        raise ValueError("Replay samples must share an ordered schema.")
    return {key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0) for key in keys}


__all__ = [
    "SequenceReplayBuffer",
    "categorical_entropy",
    "categorical_log_prob",
    "clipped_value_loss",
    "compute_gae",
    "masked_mean",
    "normalize_masked",
    "stack_replay_samples",
]
