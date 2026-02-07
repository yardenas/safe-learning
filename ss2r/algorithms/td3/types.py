import functools
from typing import Any, Callable, Tuple, TypeAlias

import flax
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey

from ss2r.rl.types import MakePolicyFn

Metrics: TypeAlias = types.Metrics
Transition: TypeAlias = types.Transition
InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState: TypeAlias = Any

make_float = lambda x, t: jax.tree.map(lambda y: y.astype(t), x)
float16 = functools.partial(make_float, t=jnp.float16)
float32 = functools.partial(make_float, t=jnp.float32)


CollectDataFn = Callable[
    [
        envs.Env,
        MakePolicyFn,
        Params,
        running_statistics.RunningStatisticsState,
        ReplayBuffer,
        envs.State,
        ReplayBufferState,
        PRNGKey,
        Tuple[str, ...],
    ],
    Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ],
]


@flax.struct.dataclass
class TrainingState:
    # Contains training state for the learner.
    policy_optimizer_state: optax.OptState
    policy_params: Params
    target_policy_params: Params
    qr_optimizer_state: optax.OptState
    qr_params: Params
    target_qr_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState
