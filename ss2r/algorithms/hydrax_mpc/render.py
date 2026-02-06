import jax
import numpy as np
from mujoco_playground._src import mjx_env

from ss2r.common.pytree import pytrees_unstack


def _rollout_state(
    env: mjx_env.MjxEnv,
    policy,
    steps: int,
    rng: jax.Array,
    state: mjx_env.State,
):
    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        action, _ = policy(state, current_key)
        nstate = env.step(state, action)
        return (nstate, next_key), nstate

    (final_state, _), data = jax.lax.scan(f, (state, rng), (), length=steps)
    return final_state, data


def render(
    env: mjx_env.MjxEnv,
    policy,
    steps: int,
    rng: jax.Array,
    num_envs: int = 5,
):
    state = env.reset(rng)
    batch_size = 1
    if hasattr(state.data, "qpos") and getattr(state.data.qpos, "ndim", 0) >= 2:
        batch_size = state.data.qpos.shape[0]
    if num_envs is None:
        num_envs = batch_size
    else:
        num_envs = min(num_envs, batch_size)
    if batch_size > 1 and num_envs != batch_size:
        state = jax.tree.map(lambda x: x[:num_envs], state)

    rollout_key = rng[0] if getattr(rng, "ndim", 0) >= 2 else rng
    _, trajectory = _rollout_state(env, policy, steps, rollout_key, state)
    videos = []
    for i in range(num_envs):
        ep_trajectory = jax.tree.map(lambda x: x[:, i], trajectory)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        video = env.render(ep_trajectory)
        videos.append(video)
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)
