import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Policy, PRNGKey

from ss2r.algorithms.sac import CollectDataFn, ReplayBufferState, UnrollFn, float16


def get_collection_fn(cfg):
    if cfg.agent.data_collection.name == "step":
        return collect_single_step
    elif cfg.agent.data_collection.name == "episodic":
        return make_collection_fn(
            functools.partial(acting.generate_unroll, cfg.training.episode_length)
        )
    else:
        raise ValueError(f"Unknown data collection {cfg.agent.data_collection.name}")


def make_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        policy: Policy,
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env, env_state, policy, key, extra_fields=extra_fields
        )
        if transitions.observation.ndim == 3:
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
        normalizer_params = running_statistics.update(
            normalizer_params, transitions.observation
        )
        buffer_state = replay_buffer.insert(buffer_state, float16(transitions))
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(acting.actor_step)
