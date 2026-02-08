from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydrax.alg_base import Trajectory
from hydrax.utils.spline import get_interp_func
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .task import MujocoPlaygroundTask


@struct.dataclass
class TreeMPCParams:
    actions: jax.Array
    rng: jax.Array


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
        num_knots: int = 4,
        spline_type: str = "zero",
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
        self.num_knots = int(num_knots)
        if self.num_knots < 1:
            raise ValueError("num_knots must be >= 1.")
        if self.num_knots > self.horizon:
            raise ValueError("num_knots cannot exceed horizon.")
        self.spline_type = spline_type
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.action_noise_std = action_noise_std
        self.mode = mode
        self.iterations = int(iterations)
        self.uses_env_step = True
        self.interp_func = get_interp_func(self.spline_type)

        knot_indices = np.round(
            np.linspace(0, self.horizon - 1, self.num_knots)
        ).astype(int)
        knot_indices[0] = 0
        knot_indices[-1] = self.horizon - 1
        if np.unique(knot_indices).size != self.num_knots:
            raise ValueError(
                "num_knots results in duplicate knot indices; "
                "reduce num_knots or increase horizon."
            )
        self._knot_indices = jnp.asarray(knot_indices, dtype=jnp.int32)
        knot_id = np.full((self.horizon,), -1, dtype=np.int32)
        for idx, t in enumerate(knot_indices):
            knot_id[t] = idx
        self._knot_id = jnp.asarray(knot_id, dtype=jnp.int32)
        self._tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        self._tq = jnp.linspace(self._tk[0], self._tk[-1], self.ctrl_steps)

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
        base_knots = params.actions[self._knot_indices]
        knot_actions = jnp.broadcast_to(
            base_knots, (width,) + base_knots.shape
        )  # (width, num_knots, act_dim)

        def _step_fn(carry, t):
            (
                key,
                states,
                returns,
                knot_actions,
                traj_data,
                traj_actions,
                traj_rewards,
            ) = carry
            knot_id = self._knot_id[t]

            def _expand_at_knot():
                key_out, k_noise, k_sel = jax.random.split(key, 3)

                mu = knot_actions[:, knot_id, :]
                eps = jax.random.normal(k_noise, (width, branch, act_dim))
                std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
                new_knot = mu[:, None, :] + eps * std

                exp_states = _repeat_tree(states, branch)
                exp_returns = jnp.repeat(returns, branch, axis=0)
                exp_knot_actions = jnp.repeat(knot_actions, branch, axis=0)
                exp_traj_data = _repeat_tree(traj_data, branch)
                exp_traj_actions = jnp.repeat(traj_actions, branch, axis=0)
                exp_traj_rewards = jnp.repeat(traj_rewards, branch, axis=0)

                exp_knot_actions = exp_knot_actions.at[:, knot_id, :].set(
                    new_knot.reshape(width * branch, act_dim)
                )

                tq = jax.lax.dynamic_slice(self._tq, (t,), (1,))
                action_t = self.interp_func(tq, self._tk, exp_knot_actions)[:, 0]
                next_states = jax.vmap(self.task.env.step)(exp_states, action_t)
                rewards = next_states.reward

                exp_returns = exp_returns + rewards
                exp_traj_actions = exp_traj_actions.at[:, t, :].set(action_t)
                exp_traj_rewards = exp_traj_rewards.at[:, t].set(rewards)
                exp_traj_data = jax.tree.map(
                    lambda td, d: td.at[:, t + 1].set(d),
                    exp_traj_data,
                    next_states.data,
                )

                scores = exp_returns
                if self.mode == "beam":
                    idx = _topk_indices(scores, width)
                elif self.mode == "resample":
                    w = _softmax(scores / self.temperature, axis=0).squeeze()
                    idx = jax.random.choice(
                        k_sel, scores.shape[0], shape=(width,), p=w, replace=True
                    )
                else:
                    raise ValueError(
                        f"Unknown mode={self.mode}. Use 'beam' or 'resample'."
                    )

                states_out = _gather_tree(next_states, idx)
                returns_out = exp_returns[idx]
                knot_actions_out = exp_knot_actions[idx]
                traj_data_out = _gather_tree(exp_traj_data, idx)
                traj_actions_out = exp_traj_actions[idx]
                traj_rewards_out = exp_traj_rewards[idx]

                return (
                    key_out,
                    states_out,
                    returns_out,
                    knot_actions_out,
                    traj_data_out,
                    traj_actions_out,
                    traj_rewards_out,
                )

            def _propagate():
                tq = jax.lax.dynamic_slice(self._tq, (t,), (1,))
                action_t = self.interp_func(tq, self._tk, knot_actions)[:, 0]
                next_states = jax.vmap(self.task.env.step)(states, action_t)
                rewards = next_states.reward

                returns_out = returns + rewards
                traj_actions_out = traj_actions.at[:, t, :].set(action_t)
                traj_rewards_out = traj_rewards.at[:, t].set(rewards)
                traj_data_out = jax.tree.map(
                    lambda td, d: td.at[:, t + 1].set(d),
                    traj_data,
                    next_states.data,
                )
                return (
                    key,
                    next_states,
                    returns_out,
                    knot_actions,
                    traj_data_out,
                    traj_actions_out,
                    traj_rewards_out,
                )

            out = jax.lax.cond(knot_id >= 0, _expand_at_knot, _propagate)
            return out, None

        carry0 = (
            key,
            states,
            returns,
            knot_actions,
            traj_data,
            traj_actions,
            traj_rewards,
        )
        carryN, _ = jax.lax.scan(_step_fn, carry0, jnp.arange(horizon))
        _, states, returns, _, traj_data, traj_actions, traj_rewards = carryN

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
