from typing import Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training.replay_buffers import (
    ReplayBuffer,
    ReplayBufferState,
)
from brax.training.types import PRNGKey, Transition


@flax.struct.dataclass
class PytreeReplayBufferState:
    """Contains data related to a replay buffer."""

    data: Transition
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: PRNGKey


class PytreeUniformSamplingQueue(ReplayBuffer[PytreeReplayBufferState, Transition]):
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Transition,
        sample_batch_size: int,
    ):
        # Create per-field data arrays with shapes [max_replay_size, ...]
        self._data_template = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(
                shape=(max_replay_size,) + x.shape,
                dtype=x.dtype,
            ),
            dummy_data_sample,
        )
        self._max_replay_size = max_replay_size
        self._sample_batch_size = sample_batch_size
        self._size = 0

    def init(self, key: jax.random.PRNGKey) -> PytreeReplayBufferState:
        data = jax.tree_util.tree_map(
            lambda template: jnp.zeros(template.shape, template.dtype),
            self._data_template,
        )
        return PytreeReplayBufferState(  # type: ignore
            data=data,
            insert_position=jnp.zeros((), jnp.int32),
            sample_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert_internal(
        self, buffer_state: PytreeReplayBufferState, samples: Transition
    ) -> PytreeReplayBufferState:
        """Insert samples using circular writes instead of rolling the buffer."""

        capacity = self._max_replay_size
        batch_size = samples.reward.shape[0]
        write_position = buffer_state.insert_position % capacity
        idx = (jnp.arange(batch_size, dtype=jnp.int32) + write_position) % capacity

        def insert_one_field(field_data, field_sample):
            return field_data.at[idx].set(field_sample)

        new_data = jax.tree_util.tree_map(insert_one_field, buffer_state.data, samples)
        insert_position = buffer_state.insert_position + batch_size
        sample_position = jnp.maximum(
            buffer_state.sample_position,
            insert_position - capacity,
        )

        # Keep counters bounded while preserving their difference and wrap order.
        offset = (sample_position // capacity) * capacity
        insert_position = insert_position - offset
        sample_position = sample_position - offset
        return buffer_state.replace(  # type: ignore
            data=new_data,
            insert_position=insert_position,
            sample_position=sample_position,
        )

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )  # pytype: disable=bad-return-type  # jax-ndarray

    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Transition]:
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )

        # Sample each field from the buffer using gathered indices
        def sample_field(field_data):
            return jnp.take(field_data, idx, axis=0, mode="wrap")

        batch = jax.tree_util.tree_map(sample_field, buffer_state.data)
        return buffer_state.replace(key=key), batch
