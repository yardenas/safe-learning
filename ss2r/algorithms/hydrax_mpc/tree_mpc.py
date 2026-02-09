from typing import Any, Optional, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint as sac_checkpoint
from flax import struct
from hydrax.alg_base import Trajectory
from mujoco import mjx
from mujoco_playground._src import mjx_env

from ss2r.algorithms.sac import networks as sac_networks

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


def _reshape_tree(tree: Any, width: int, branch: int) -> Any:
    return jax.tree.map(
        lambda x: x.reshape((width, branch) + x.shape[1:]),
        tree,
    )


def _take_per_parent(tree: Any, idx: jax.Array) -> Any:
    def _take(x: jax.Array) -> jax.Array:
        take_idx = idx.reshape((idx.shape[0],) + (1,) * (x.ndim - 1))
        return jnp.take_along_axis(x, take_idx, axis=1).squeeze(axis=1)

    return jax.tree.map(_take, tree)


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
        policy_checkpoint_path: str | None = None,
        policy_noise_std: float = 0.05,
        td_lambda: float = 0.0,
        policy_action_only: bool = False,
        normalize_observations: bool = True,
        policy_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
        value_hidden_layer_sizes: Sequence[int] = (512, 512),
        activation: str = "swish",
        n_critics: int = 2,
        n_heads: int = 1,
        use_bro: bool = True,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
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
        self.policy_noise_std = float(policy_noise_std)
        self.td_lambda = float(td_lambda)
        self.policy_action_only = bool(policy_action_only)

        self._policy_params = None
        self._qr_params = None
        self._normalizer_params = None
        self._sac_network = None
        self._n_critics = int(n_critics)
        self._n_heads = int(n_heads)
        self._normalize_observations = bool(normalize_observations)

        if policy_checkpoint_path is not None:
            self._load_policy_and_critic(
                policy_checkpoint_path,
                policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                value_hidden_layer_sizes=value_hidden_layer_sizes,
                activation=activation,
                use_bro=use_bro,
                policy_obs_key=policy_obs_key,
                value_obs_key=value_obs_key,
            )

    def _load_policy_and_critic(
        self,
        checkpoint_path: str,
        *,
        policy_hidden_layer_sizes: Sequence[int],
        value_hidden_layer_sizes: Sequence[int],
        activation: str,
        use_bro: bool,
        policy_obs_key: str,
        value_obs_key: str,
    ) -> None:
        params = sac_checkpoint.load(checkpoint_path)
        self._normalizer_params = params[0]
        self._policy_params = params[1]
        self._qr_params = params[3]

        obs_size = self.task.env.observation_size
        action_size = (
            self.task.env.action_size
            if hasattr(self.task.env, "action_size")
            else self.task.u_min.shape[-1]
        )
        preprocess_fn = (
            running_statistics.normalize
            if self._normalize_observations
            else (lambda x, y: x)
        )
        activation_fn = getattr(jnn, activation)
        self._sac_network = sac_networks.make_sac_networks(  # type: ignore
            observation_size=obs_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_fn,  # ty:ignore[invalid-argument-type]
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation_fn,
            use_bro=use_bro,
            n_critics=self._n_critics,
            n_heads=self._n_heads,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key,
        )

    def _has_policy(self) -> bool:
        return self._sac_network is not None

    def _sample_policy_actions(self, obs: Any, key: jax.Array) -> jax.Array:
        assert self._sac_network is not None
        assert self._policy_params is not None
        dist_params = self._sac_network.policy_network.apply(
            self._normalizer_params,
            self._policy_params,
            obs,
        )
        dist = self._sac_network.parametric_action_distribution
        return dist.sample(dist_params, key)

    def _critic_value(self, obs: Any, action: jax.Array) -> jax.Array:
        assert self._sac_network is not None
        assert self._qr_params is not None
        q_values = self._sac_network.qr_network.apply(
            self._normalizer_params,
            self._qr_params,
            obs,
            action,
        )
        if q_values.ndim == 1:
            q_values = q_values[:, None]
        q_values = q_values.reshape((q_values.shape[0], self._n_critics, self._n_heads))
        q_values = jnp.min(q_values, axis=1)
        return jnp.mean(q_values, axis=-1)

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
            if self._has_policy():
                key, k_policy, k_value, k_sel = jax.random.split(key, 4)
                obs = states.obs
                obs_flat = _repeat_tree(obs, branch)
                flat_actions = self._sample_policy_actions(obs_flat, k_policy)
                flat_actions = jnp.clip(flat_actions, self.task.u_min, self.task.u_max)
                actions = flat_actions.reshape((width, branch, act_dim))
            else:
                key, k_noise, k_sel = jax.random.split(key, 3)
                mu = jnp.broadcast_to(params.actions[t], (width, act_dim))
                eps = jax.random.normal(k_noise, (width, branch, act_dim))
                std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
                u = mu[:, None, :] + eps * std
                actions = _squash_to_bounds(u, self.task.u_min, self.task.u_max)
                flat_actions = actions.reshape((width * branch, act_dim))

            flat_states = _repeat_tree(states, branch)
            if self._has_policy():
                flat_actions = flat_actions.reshape((width * branch, act_dim))

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

            exp_returns = jnp.repeat(returns, branch, axis=0) + flat_rewards
            if self._has_policy() and self.td_lambda < 1.0:
                value_actions = self._sample_policy_actions(
                    flat_next_states.obs, k_value
                )
                value_actions = jnp.clip(
                    value_actions, self.task.u_min, self.task.u_max
                )
                values = self._critic_value(flat_next_states.obs, value_actions)
                exp_scores = (
                    exp_returns
                    + (self.gamma ** (t + 1)) * (1.0 - self.td_lambda) * values
                )
            else:
                exp_scores = exp_returns

            if self.mode == "beam":
                idx = _topk_indices(exp_scores, width)
                states = _gather_tree(flat_next_states, idx)
                traj_data = _gather_tree(exp_traj_data, idx)
                traj_actions = exp_traj_actions[idx]
                traj_rewards = exp_traj_rewards[idx]
                returns = exp_scores[idx]
            elif self.mode == "resample":
                w = _softmax(exp_scores / self.temperature, axis=0).squeeze()
                idx = jax.random.choice(
                    k_sel, exp_scores.shape[0], shape=(width,), p=w, replace=True
                )
                states = _gather_tree(flat_next_states, idx)
                traj_data = _gather_tree(exp_traj_data, idx)
                traj_actions = exp_traj_actions[idx]
                traj_rewards = exp_traj_rewards[idx]
                returns = exp_scores[idx]
            elif self.mode in {"myopic", "per_parent"}:
                per_parent_scores = exp_scores.reshape((width, branch))
                per_parent_idx = jnp.argmax(per_parent_scores, axis=1)

                states = _take_per_parent(
                    _reshape_tree(flat_next_states, width, branch), per_parent_idx
                )
                traj_data = _take_per_parent(
                    _reshape_tree(exp_traj_data, width, branch), per_parent_idx
                )
                traj_actions = _take_per_parent(
                    exp_traj_actions.reshape(
                        (width, branch) + exp_traj_actions.shape[1:]
                    ),
                    per_parent_idx,
                )
                traj_rewards = _take_per_parent(
                    exp_traj_rewards.reshape(
                        (width, branch) + exp_traj_rewards.shape[1:]
                    ),
                    per_parent_idx,
                )
                returns = per_parent_scores[jnp.arange(width), per_parent_idx]
            else:
                raise ValueError(
                    f"Unknown mode={self.mode}. Use 'beam', 'resample', or 'myopic'."
                )

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
        if self.policy_action_only:
            if not self._has_policy():
                raise ValueError(
                    "policy_action_only=True requires a loaded policy checkpoint."
                )
            rng, action_key = jax.random.split(params.rng)
            action = self._sample_policy_actions(state.obs, action_key)
            action = jnp.clip(action, self.task.u_min, self.task.u_max)
            if action.ndim > 1:
                action = action[0]
            action_seq = jnp.broadcast_to(
                action, (self.horizon, self.task.u_min.shape[-1])
            )
            params = params.replace(actions=action_seq, rng=rng)  # type: ignore

            sites = self.task.get_trace_sites(state.data)
            trace_sites = jnp.broadcast_to(sites, (self.horizon + 1,) + sites.shape)[
                None, ...
            ]
            controls = action_seq[None, ...]
            costs = jnp.zeros((1, self.horizon), dtype=jnp.float32)
            return params, Trajectory(
                controls=controls,
                knots=controls,
                costs=costs,
                trace_sites=trace_sites,
            )

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
