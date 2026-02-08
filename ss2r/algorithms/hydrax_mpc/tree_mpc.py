from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import struct
from hydrax.alg_base import Trajectory
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .task import MujocoPlaygroundTask


@struct.dataclass
class TreeMPCParams:
    actions: jax.Array
    rng: jax.Array


def _squash_to_bounds(u: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    z = jnp.tanh(u)
    return low + 0.5 * (z + 1.0) * (high - low)


def _softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    x = x - jnp.max(x, axis=axis, keepdims=True)
    ex = jnp.exp(x)
    return ex / jnp.sum(ex, axis=axis, keepdims=True)


def _topk_indices(scores: jax.Array, k: int) -> jax.Array:
    _, idx = jax.lax.top_k(scores, k)
    return idx


def _repeat_tree(tree: Any, repeats: int) -> Any:
    return jax.tree.map(lambda x: jnp.repeat(x, repeats, axis=0), tree)


def _broadcast_tree(tree: Any, batch: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), tree)


def _gather_tree(tree: Any, idx: jax.Array) -> Any:
    return jax.tree.map(lambda x: x[idx], tree)


@struct.dataclass
class _TreeRollout:
    traj_data: mjx.Data
    traj_actions: jax.Array
    traj_rewards: jax.Array
    returns: jax.Array


class TreeMPC:
    """Tree expansion planner with MPPI-style resampling."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        *,
        width: int,
        branch: int,
        horizon: Optional[int] = None,
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_noise_std: float = 0.3,
        mode: str = "resample",
        iterations: int = 1,
    ) -> None:
        self.task = task
        self.dt = float(self.task.dt)

        self.width = int(width)
        self.branch = int(branch)
        if horizon is None:
            raise ValueError("TreeMPC requires horizon to be set explicitly.")
        self.horizon = int(horizon)
        self.plan_horizon = float(self.horizon) * self.dt
        self.ctrl_steps = int(self.horizon)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.action_noise_std = action_noise_std
        self.mode = mode
        self.iterations = int(iterations)
        self.uses_env_step = True

    def _tree_expand(
        self, key: jax.Array, state: mjx_env.State, params: TreeMPCParams
    ) -> _TreeRollout:
        width = self.width
        branch = self.branch
        horizon = self.horizon
        act_dim = self.task.u_min.shape[-1]
        # Initialize B identical roots
        states = _broadcast_tree(state, width)
        returns = jnp.zeros((width,), dtype=jnp.float32)

        traj_actions = jnp.zeros((width, horizon, act_dim), dtype=jnp.float32)
        traj_rewards = jnp.zeros((width, horizon), dtype=jnp.float32)
        traj_data = jax.tree.map(
            lambda x: jnp.zeros((width, horizon + 1) + x.shape, dtype=x.dtype),
            state.data,
        )
        traj_data = jax.tree.map(lambda t, d: t.at[:, 0].set(d), traj_data, states.data)

        def _step_fn(carry, t):
            key, states, returns, traj_data, traj_actions, traj_rewards = carry
            key, k_noise, k_sel = jax.random.split(key, 3)

            mu = jnp.broadcast_to(params.actions[t], (width, act_dim))
            eps = jax.random.normal(k_noise, (width, branch, act_dim))
            std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
            u = mu[:, None, :] + eps * std
            actions = _squash_to_bounds(u, self.task.u_min, self.task.u_max)

            flat_states = _repeat_tree(states, branch)
            flat_actions = actions.reshape((width * branch, act_dim))

            def _env_step(s, a):
                return self.task.env.step(s, a)

            flat_next_states = jax.vmap(_env_step)(flat_states, flat_actions)
            flat_rewards = flat_next_states.reward

            exp_traj_data = _repeat_tree(traj_data, branch)
            exp_traj_actions = jnp.repeat(traj_actions, branch, axis=0)
            exp_traj_rewards = jnp.repeat(traj_rewards, branch, axis=0)

            exp_traj_actions = exp_traj_actions.at[:, t, :].set(flat_actions)
            exp_traj_rewards = exp_traj_rewards.at[:, t].set(flat_rewards)
            exp_traj_data = jax.tree.map(
                lambda td, d: td.at[:, t + 1].set(d),
                exp_traj_data,
                flat_next_states.data,
            )

            scores = jnp.sum(exp_traj_rewards, axis=1)
            if self.mode == "beam":
                idx = _topk_indices(scores, width)
            elif self.mode == "resample":
                w = _softmax(scores / self.temperature, axis=0).squeeze()
                idx = jax.random.choice(
                    k_sel, scores.shape[0], shape=(width,), p=w, replace=True
                )
            else:
                raise ValueError(f"Unknown mode={self.mode}. Use 'beam' or 'resample'.")

            states = _gather_tree(flat_next_states, idx)
            traj_data = _gather_tree(exp_traj_data, idx)
            traj_actions = exp_traj_actions[idx]
            traj_rewards = exp_traj_rewards[idx]
            returns = jnp.sum(traj_rewards, axis=1)

            return (key, states, returns, traj_data, traj_actions, traj_rewards), None

        carry0 = (key, states, returns, traj_data, traj_actions, traj_rewards)
        carryN, _ = jax.lax.scan(_step_fn, carry0, jnp.arange(horizon))
        _, states, returns, traj_data, traj_actions, traj_rewards = carryN

        return _TreeRollout(  # type: ignore
            traj_data=traj_data,
            traj_actions=traj_actions,
            traj_rewards=traj_rewards,
            returns=returns,
        )

    def optimize(self, state: mjx_env.State, params: TreeMPCParams):
        def _iter_step(carry, _):
            params, rng = carry
            rng, tree_rng = jax.random.split(rng)
            rollouts = self._tree_expand(tree_rng, state, params)

            best_idx = jnp.argmax(rollouts.returns)
            best_actions = rollouts.traj_actions[best_idx]
            params = params.replace(actions=best_actions, rng=rng)

            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _iter_step, (params, params.rng), jnp.arange(self.iterations)
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)

        costs = -rollouts.traj_rewards

        def _trace_sites(x):
            return self.task.get_trace_sites(x)

        trace_sites = jax.vmap(jax.vmap(_trace_sites))(rollouts.traj_data)
        traj_knots = rollouts.traj_actions

        return params, Trajectory(
            controls=rollouts.traj_actions,
            knots=traj_knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.horizon, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
