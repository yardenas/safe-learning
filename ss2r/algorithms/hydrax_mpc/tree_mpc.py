from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import struct
from hydrax.alg_base import Trajectory
from hydrax.utils.spline import get_interp_func
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .task import MujocoPlaygroundTask


@struct.dataclass
class TreeMPCParams:
    tk: jax.Array
    mean: jax.Array
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
    traj_knots: jax.Array
    returns: jax.Array


def _segment_lengths(ctrl_steps: int, num_knots: int) -> tuple[int, ...]:
    if num_knots < 2:
        raise ValueError("TreeMPC requires num_knots >= 2.")
    if num_knots > ctrl_steps:
        raise ValueError("num_knots cannot exceed horizon/ctrl_steps in TreeMPC.")
    num_segments = num_knots - 1
    base = ctrl_steps // num_segments
    remainder = ctrl_steps % num_segments
    return tuple(base + 1 if i < remainder else base for i in range(num_segments))


class TreeMPC:
    """Tree expansion planner with MPPI-style resampling."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        *,
        width: int,
        branch: int,
        horizon: Optional[int] = None,
        spline_type: str = "zero",
        num_knots: Optional[int] = None,
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
        self.spline_type = spline_type
        self.num_knots = self.horizon if num_knots is None else int(num_knots)
        self.interp_func = get_interp_func(spline_type)
        if self.spline_type not in {"zero", "linear"}:
            raise ValueError(
                "TreeMPC only supports spline_type 'zero' or 'linear' when using "
                "knot-level branching. Set spline_type accordingly for tree runs."
            )
        self._segment_lengths = _segment_lengths(self.ctrl_steps, self.num_knots)
        seg_starts: list[int] = []
        seg_ends: list[int] = []
        offset = 0
        for seg_len in self._segment_lengths:
            seg_starts.append(offset)
            offset += seg_len
            seg_ends.append(offset)
        self._segment_starts = tuple(seg_starts)
        self._segment_ends = tuple(seg_ends)
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
        tq = jnp.linspace(params.tk[0], params.tk[-1], self.ctrl_steps)
        # Initialize B identical roots
        states = _broadcast_tree(state, width)
        returns = jnp.zeros((width,), dtype=jnp.float32)
        current_knots = jnp.broadcast_to(params.mean[0], (width, act_dim))

        traj_actions = jnp.zeros((width, horizon, act_dim), dtype=jnp.float32)
        traj_rewards = jnp.zeros((width, horizon), dtype=jnp.float32)
        traj_knots = jnp.zeros((width, self.num_knots, act_dim), dtype=jnp.float32)
        traj_knots = traj_knots.at[:, 0, :].set(current_knots)
        traj_data = jax.tree.map(
            lambda x: jnp.zeros((width, horizon + 1) + x.shape, dtype=x.dtype),
            state.data,
        )
        traj_data = jax.tree.map(lambda t, d: t.at[:, 0].set(d), traj_data, states.data)

        for k, (seg_start, seg_end) in enumerate(
            zip(self._segment_starts, self._segment_ends)
        ):
            key, k_noise, k_sel = jax.random.split(key, 3)

            mu = jnp.broadcast_to(params.mean[k + 1], (width, act_dim))
            eps = jax.random.normal(k_noise, (width, branch, act_dim))
            std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
            next_knots = jnp.clip(
                mu[:, None, :] + eps * std, self.task.u_min, self.task.u_max
            )

            tq_seg = tq[seg_start:seg_end]
            tk_seg = jnp.array([params.tk[k], params.tk[k + 1]])
            k0 = jnp.repeat(current_knots, branch, axis=0)
            k1 = next_knots.reshape(width * branch, act_dim)
            seg_knots = jnp.stack([k0, k1], axis=1)
            seg_actions = self.interp_func(tq_seg, tk_seg, seg_knots)
            actions = seg_actions.reshape(width, branch, tq_seg.shape[0], act_dim)

            flat_states = _repeat_tree(states, branch)
            flat_actions = actions.reshape((width * branch, tq_seg.shape[0], act_dim))
            flat_actions_t = jnp.swapaxes(flat_actions, 0, 1)

            def _env_step(s, a):
                s = jax.vmap(lambda ss, aa: self.task.env.step(ss, aa))(s, a)
                return s, (s.data, s.reward)

            flat_next_states, (seg_data, seg_rewards) = jax.lax.scan(
                _env_step, flat_states, flat_actions_t
            )
            seg_data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), seg_data)
            seg_rewards = jnp.swapaxes(seg_rewards, 0, 1)

            exp_traj_data = _repeat_tree(traj_data, branch)
            exp_traj_actions = jnp.repeat(traj_actions, branch, axis=0)
            exp_traj_rewards = jnp.repeat(traj_rewards, branch, axis=0)
            exp_traj_knots = jnp.repeat(traj_knots, branch, axis=0)

            exp_traj_actions = exp_traj_actions.at[:, seg_start:seg_end, :].set(
                flat_actions
            )
            exp_traj_rewards = exp_traj_rewards.at[:, seg_start:seg_end].set(
                seg_rewards
            )
            exp_traj_data = jax.tree.map(
                lambda td, d: td.at[:, seg_start + 1 : seg_end + 1].set(d),
                exp_traj_data,
                seg_data,
            )
            exp_traj_knots = exp_traj_knots.at[:, k + 1, :].set(
                next_knots.reshape(width * branch, act_dim)
            )

            exp_returns = jnp.repeat(returns, branch, axis=0) + jnp.sum(
                seg_rewards, axis=1
            )

            if self.mode == "beam":
                idx = _topk_indices(exp_returns, width)
            elif self.mode == "resample":
                w = _softmax(exp_returns / self.temperature, axis=0).squeeze()
                idx = jax.random.choice(
                    k_sel, exp_returns.shape[0], shape=(width,), p=w, replace=True
                )
            else:
                raise ValueError(f"Unknown mode={self.mode}. Use 'beam' or 'resample'.")

            states = _gather_tree(flat_next_states, idx)
            traj_data = _gather_tree(exp_traj_data, idx)
            traj_actions = exp_traj_actions[idx]
            traj_rewards = exp_traj_rewards[idx]
            traj_knots = exp_traj_knots[idx]
            returns = exp_returns[idx]
            current_knots = next_knots.reshape(width * branch, act_dim)[idx]

        return _TreeRollout(  # type: ignore
            traj_data=traj_data,
            traj_actions=traj_actions,
            traj_rewards=traj_rewards,
            traj_knots=traj_knots,
            returns=returns,
        )

    def optimize(self, state: mjx_env.State, params: TreeMPCParams):
        data = state.data
        if not hasattr(data, "time"):
            raise ValueError("state.data has no time attribute.")
        tk = params.tk
        new_tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) + data.time
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)  # type: ignore

        def _iter_step(carry, _):
            params, rng = carry
            rng, tree_rng = jax.random.split(rng)
            rollouts = self._tree_expand(tree_rng, state, params)

            best_idx = jnp.argmax(rollouts.returns)
            best_knots = rollouts.traj_knots[best_idx]
            params = params.replace(mean=best_knots, rng=rng)

            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _iter_step, (params, params.rng), jnp.arange(self.iterations)
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)

        costs = -rollouts.traj_rewards

        def _trace_sites(x):
            return self.task.get_trace_sites(x)

        trace_sites = jax.vmap(jax.vmap(_trace_sites))(rollouts.traj_data)
        traj_knots = rollouts.traj_knots

        return params, Trajectory(
            controls=rollouts.traj_actions,
            knots=traj_knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.num_knots, self.task.u_min.shape[-1]), dtype=jnp.float32)
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        return TreeMPCParams(tk=tk, mean=mean, rng=rng)  # type: ignore
