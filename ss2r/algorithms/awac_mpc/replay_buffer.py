"""Host-backed replay buffer utilities for AWAC-MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from brax.training.types import PRNGKey
from jax import tree_util

Data = Any


def get_size(data: Data) -> int:
    sizes = tree_util.tree_map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))


def _to_numpy_leaf(leaf: Any) -> Any:
    try:
        return np.asarray(jax.device_get(leaf))
    except TypeError:
        return leaf


def _to_numpy_tree(data: Data) -> Data:
    return tree_util.tree_map(_to_numpy_leaf, data)


def _seed_from_key(key: PRNGKey) -> int:
    key_array = np.asarray(jax.device_get(key), dtype=np.uint32).reshape(-1)
    seed = 0
    for value in key_array:
        seed = ((seed << 32) ^ int(value)) & 0xFFFFFFFFFFFFFFFF
    return seed


class Dataset:
    """Stores nested arrays and supports subset sampling on host."""

    def __init__(self, data: Data):
        self.data = _to_numpy_tree(data)
        self.size = get_size(self.data)

    def sample(self, batch_size: int, indx: np.ndarray | None = None) -> Data:
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx: np.ndarray) -> Data:
        return tree_util.tree_map(lambda arr: arr[indx], self.data)


class ReplayBuffer(Dataset):
    """Ring buffer that stores nested transition trees in NumPy arrays."""

    @classmethod
    def create(cls, transition: Data, size: int) -> "ReplayBuffer":
        def create_buffer(example):
            example = np.asarray(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        return cls(tree_util.tree_map(create_buffer, _to_numpy_tree(transition)))

    @classmethod
    def create_from_initial_dataset(
        cls,
        init_dataset: Data,
        size: int,
    ) -> "ReplayBuffer":
        def create_buffer(init_buffer):
            init_buffer = np.asarray(init_buffer)
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        dataset = cls(tree_util.tree_map(create_buffer, _to_numpy_tree(init_dataset)))
        dataset.size = min(get_size(init_dataset), size)
        dataset.pointer = dataset.size % dataset.max_size
        return dataset

    def __init__(self, data: Data):
        super().__init__(data)
        self.max_size = get_size(self.data)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition: Data) -> None:
        self.add_transitions(
            tree_util.tree_map(
                lambda arr: np.expand_dims(np.asarray(arr), axis=0), transition
            )
        )

    def add_transitions(self, transitions: Data) -> None:
        transitions = jax.device_get(transitions)
        # transitions = _to_numpy_tree(transitions)
        insert_size = get_size(transitions)
        if insert_size > self.max_size:
            raise ValueError(
                "Trying to insert more samples than max_size. "
                f"insert_size={insert_size}, max_size={self.max_size}"
            )

        start = self.pointer
        end = start + insert_size

        def set_batch(buffer, new_elements):
            if end <= self.max_size:
                buffer[start:end] = new_elements
            else:
                first_chunk = self.max_size - start
                buffer[start:] = new_elements[:first_chunk]
                buffer[: end % self.max_size] = new_elements[first_chunk:]
            return buffer

        self.data = tree_util.tree_map(set_batch, self.data, transitions)
        self.pointer = end % self.max_size
        self.size = min(self.max_size, self.size + insert_size)


@dataclass
class CpuReplayBufferState:
    data: ReplayBuffer
    rng: np.random.Generator


class CpuUniformSamplingQueue:
    """Minimal Brax-like replay buffer wrapper backed by NumPy on CPU."""

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Data,
        sample_batch_size: int,
    ):
        self._max_replay_size = int(max_replay_size)
        self._dummy_data_sample = dummy_data_sample
        self._sample_batch_size = int(sample_batch_size)

    def init(self, key: PRNGKey) -> CpuReplayBufferState:
        return CpuReplayBufferState(
            data=ReplayBuffer.create(self._dummy_data_sample, self._max_replay_size),
            rng=np.random.default_rng(_seed_from_key(key)),
        )

    def insert(
        self, buffer_state: CpuReplayBufferState, samples: Data
    ) -> CpuReplayBufferState:
        buffer_state.data.add_transitions(samples)
        return buffer_state

    def sample(
        self, buffer_state: CpuReplayBufferState
    ) -> tuple[CpuReplayBufferState, Data]:
        if buffer_state.data.size < self._sample_batch_size:
            raise ValueError(
                f"Trying to sample {self._sample_batch_size} elements, but only "
                f"{buffer_state.data.size} available."
            )
        indices = buffer_state.rng.integers(
            low=0,
            high=buffer_state.data.size,
            size=self._sample_batch_size,
        )
        return buffer_state, buffer_state.data.get_subset(indices)

    def size(self, buffer_state: CpuReplayBufferState) -> int:
        return buffer_state.data.size
