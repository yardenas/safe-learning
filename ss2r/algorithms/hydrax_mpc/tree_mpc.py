from typing import Any, Optional

import jax
import jax.nn as jnn
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


@struct.dataclass
class TreeMPCModelParams:
    normalizer_params: Any
    policy_params: Any
    qr_params: Any


def _squash_to_bounds(u: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    z = jnp.tanh(u)
    return low + 0.5 * (z + 1.0) * (high - low)


def _broadcast_tree(tree: Any, batch: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), tree)


def _gather_tree(tree: Any, idx: jax.Array) -> Any:
    return jax.tree.map(lambda x: x[idx], tree)


@struct.dataclass
class _TreeRollout:
    traj_data: mjx.Data
    traj_actions: jax.Array
    traj_rewards: jax.Array
    all_traj_obs: Any
    all_traj_next_obs: Any
    all_traj_actions: jax.Array
    all_traj_rewards: jax.Array
    returns: jax.Array


class TreeMPC:
    """SMC planner using SIR-style resampling."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        *,
        num_samples: int,
        horizon: Optional[int] = None,
        gae_lambda: float = 0.0,
        use_policy: bool = False,
        use_critic: bool = False,
        n_critics: int = 2,
        n_heads: int = 1,
        zoh_steps: int = 1,
        planner: str = "sir",
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_noise_std: float = 0.3,
        mode: str = "resample",
        iterations: int = 1,
    ) -> None:
        self.task = task
        self.dt = float(self.task.dt)

        self.num_samples = int(num_samples)
        if horizon is None:
            raise ValueError("TreeMPC requires horizon to be set explicitly.")
        self.horizon = int(horizon)
        self.plan_horizon = float(self.horizon) * self.dt
        self.zoh_steps = max(int(zoh_steps), 1)
        if self.horizon % self.zoh_steps != 0:
            raise ValueError("TreeMPC requires horizon to be divisible by zoh_steps.")
        self.ctrl_steps = self.horizon // self.zoh_steps
        self.gamma = float(gamma)
        self.gamma_decision = self.gamma**self.zoh_steps
        self.temperature = float(temperature)
        self.action_noise_std = action_noise_std
        self.planner = planner
        self.mode = mode
        self.iterations = int(iterations)
        self.uses_env_step = True
        self.gae_lambda = float(gae_lambda)
        self.use_policy = bool(use_policy)
        self.use_critic = bool(use_critic)
        self._n_critics = int(n_critics)
        self._n_heads = int(n_heads)
        if self._n_critics < 1 or self._n_heads < 1:
            raise ValueError("TreeMPC requires n_critics >= 1 and n_heads >= 1.")

        self._sac_network = None

    def bind_sac_network(self, sac_network: Any) -> None:
        self._sac_network = sac_network

    def _has_policy(self, model_params: TreeMPCModelParams | None = None) -> bool:
        return (
            self._sac_network is not None
            and model_params is not None
            and model_params.policy_params is not None
        )

    def _has_critic(self, model_params: TreeMPCModelParams | None = None) -> bool:
        return (
            self._sac_network is not None
            and model_params is not None
            and model_params.qr_params is not None
        )

    def _validate_model_params(self, model_params: TreeMPCModelParams | None) -> None:
        if self.use_policy and not self._has_policy(model_params):
            raise ValueError(
                "TreeMPC with use_policy=True requires a bound SAC network and "
                "model_params with policy_params."
            )
        if self.use_critic and not self._has_critic(model_params):
            raise ValueError(
                "TreeMPC with use_critic=True requires a bound SAC network and "
                "model_params with qr_params."
            )

    def _sample_policy_actions(
        self,
        obs: Any,
        key: jax.Array,
        model_params: TreeMPCModelParams | None = None,
    ) -> jax.Array:
        if self._sac_network is None or model_params is None:
            raise ValueError(
                "_sample_policy_actions requires a bound SAC network and model_params."
            )
        policy_params = model_params.policy_params
        if policy_params is None:
            raise ValueError(
                "_sample_policy_actions requires model_params.policy_params."
            )
        dist_params = self._sac_network.policy_network.apply(
            model_params.normalizer_params,
            policy_params,
            obs,
        )
        dist = self._sac_network.parametric_action_distribution
        return dist.sample(dist_params, key)

    def _critic_value(
        self,
        obs: Any,
        action: jax.Array,
        model_params: TreeMPCModelParams | None = None,
    ) -> jax.Array:
        if self._sac_network is None or model_params is None:
            raise ValueError(
                "_critic_value requires a bound SAC network and model_params."
            )
        qr_params = model_params.qr_params
        if qr_params is None:
            raise ValueError("_critic_value requires model_params.qr_params.")
        q_values = self._sac_network.qr_network.apply(
            model_params.normalizer_params,
            qr_params,
            obs,
            action,
        )
        if q_values.ndim == 1:
            q_values = q_values[:, None]
        expected_width = self._n_critics * self._n_heads
        if q_values.shape[-1] != expected_width:
            raise ValueError(
                "Unexpected critic output width for TreeMPC. "
                f"got={q_values.shape[-1]}, expected={expected_width} "
                "(n_critics * n_heads)."
            )
        q_values = q_values.reshape((q_values.shape[0], self._n_critics, self._n_heads))
        q_values = jnp.min(q_values, axis=1)
        return jnp.mean(q_values, axis=-1)

    def _step_env_repeat(
        self, state: mjx_env.State, action: jax.Array
    ) -> tuple[mjx_env.State, jax.Array]:
        if self.zoh_steps == 1:
            next_state = self.task.env.step(state, action)
            return next_state, next_state.reward

        def _repeat_fn(carry, _):
            s = carry
            s = self.task.env.step(s, action)
            return s, s.reward

        final_state, rewards = jax.lax.scan(
            _repeat_fn, state, jnp.arange(self.zoh_steps)
        )
        return final_state, jnp.sum(rewards)

    def _tree_expand(
        self,
        key: jax.Array,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ) -> _TreeRollout:
        num_particles = self.num_samples
        decision_steps = self.ctrl_steps
        act_dim = self.task.u_min.shape[-1]
        states = _broadcast_tree(state, num_particles)
        returns = jnp.zeros((num_particles,), dtype=jnp.float32)
        gae_trace = jnp.zeros((num_particles,), dtype=jnp.float32)
        value_start = jnp.zeros((num_particles,), dtype=jnp.float32)

        traj_actions = jnp.zeros(
            (num_particles, decision_steps, act_dim), dtype=jnp.float32
        )
        traj_rewards = jnp.zeros((num_particles, decision_steps), dtype=jnp.float32)
        all_traj_actions = jnp.zeros(
            (num_particles, decision_steps, act_dim), dtype=jnp.float32
        )
        all_traj_rewards = jnp.zeros((num_particles, decision_steps), dtype=jnp.float32)
        traj_data = jax.tree.map(
            lambda x: jnp.zeros(
                (num_particles, decision_steps + 1) + x.shape, dtype=x.dtype
            ),
            state.data,
        )
        traj_data = jax.tree.map(lambda t, d: t.at[:, 0].set(d), traj_data, states.data)
        all_traj_obs = jax.tree.map(
            lambda x: jnp.zeros(
                (num_particles, decision_steps) + x.shape, dtype=x.dtype
            ),
            state.obs,
        )
        all_traj_next_obs = jax.tree.map(
            lambda x: jnp.zeros(
                (num_particles, decision_steps) + x.shape, dtype=x.dtype
            ),
            state.obs,
        )

        def _step_fn(carry, t):
            (
                key,
                states,
                traj_data,
                traj_actions,
                traj_rewards,
                all_traj_obs,
                all_traj_next_obs,
                all_traj_actions,
                all_traj_rewards,
                returns,
                gae_trace,
                value_start,
            ) = carry
            key, k_sel = jax.random.split(key)
            if self.use_policy and self._has_policy(model_params):
                key, k_policy, k_value = jax.random.split(key, 3)
                actions = self._sample_policy_actions(
                    states.obs, k_policy, model_params
                )
                actions = jnp.clip(actions, self.task.u_min, self.task.u_max)
            else:
                key, k_noise = jax.random.split(key, 2)
                mu = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
                eps = jax.random.normal(k_noise, (num_particles, act_dim))
                std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
                u = mu + eps * std
                actions = _squash_to_bounds(u, self.task.u_min, self.task.u_max)

            def _env_step(s, a):
                return self._step_env_repeat(s, a)

            next_states, rewards = jax.vmap(_env_step)(states, actions)

            traj_actions = traj_actions.at[:, t, :].set(actions)
            traj_rewards = traj_rewards.at[:, t].set(rewards)
            all_traj_actions = all_traj_actions.at[:, t, :].set(actions)
            all_traj_rewards = all_traj_rewards.at[:, t].set(rewards)
            all_traj_obs = jax.tree.map(
                lambda to, o: to.at[:, t].set(o),
                all_traj_obs,
                states.obs,
            )
            all_traj_next_obs = jax.tree.map(
                lambda to, o: to.at[:, t].set(o),
                all_traj_next_obs,
                next_states.obs,
            )
            traj_data = jax.tree.map(
                lambda td, d: td.at[:, t + 1].set(d),
                traj_data,
                next_states.data,
            )
            returns = returns + rewards

            if self.use_critic and self._has_critic(model_params):
                value_s = self._critic_value(states.obs, actions, model_params)
                if self.use_policy and self._has_policy(model_params):
                    value_actions = self._sample_policy_actions(
                        next_states.obs, k_value, model_params
                    )
                    value_actions = jnp.clip(
                        value_actions, self.task.u_min, self.task.u_max
                    )
                else:
                    value_actions = actions
                value_next = self._critic_value(
                    next_states.obs, value_actions, model_params
                )
                delta = rewards + self.gamma_decision * value_next - value_s
                gae_trace = delta + self.gamma_decision * self.gae_lambda * gae_trace
                value_start = jnp.where(t == 0, value_s, value_start)
                advantage = gae_trace
            else:
                advantage = returns

            weights = jnn.softmax(advantage / self.temperature, axis=0)
            idx = jax.random.choice(
                k_sel, num_particles, shape=(num_particles,), p=weights, replace=True
            )
            states = _gather_tree(next_states, idx)
            traj_data = _gather_tree(traj_data, idx)
            traj_actions = traj_actions[idx]
            traj_rewards = traj_rewards[idx]
            returns = returns[idx]
            gae_trace = gae_trace[idx]
            value_start = value_start[idx]

            return (
                key,
                states,
                traj_data,
                traj_actions,
                traj_rewards,
                all_traj_obs,
                all_traj_next_obs,
                all_traj_actions,
                all_traj_rewards,
                returns,
                gae_trace,
                value_start,
            ), None

        carry0 = (
            key,
            states,
            traj_data,
            traj_actions,
            traj_rewards,
            all_traj_obs,
            all_traj_next_obs,
            all_traj_actions,
            all_traj_rewards,
            returns,
            gae_trace,
            value_start,
        )
        carryN, _ = jax.lax.scan(_step_fn, carry0, jnp.arange(decision_steps))
        (
            key,
            states,
            traj_data,
            traj_actions,
            traj_rewards,
            all_traj_obs,
            all_traj_next_obs,
            all_traj_actions,
            all_traj_rewards,
            returns,
            gae_trace,
            value_start,
        ) = carryN

        if self.use_critic and self._has_critic(model_params):
            returns = value_start + gae_trace

        return _TreeRollout(  # type: ignore
            traj_data=traj_data,
            traj_actions=traj_actions,
            traj_rewards=traj_rewards,
            all_traj_obs=all_traj_obs,
            all_traj_next_obs=all_traj_next_obs,
            all_traj_actions=all_traj_actions,
            all_traj_rewards=all_traj_rewards,
            returns=returns,
        )

    def _mppi_expand(
        self,
        key: jax.Array,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ) -> tuple[_TreeRollout, jax.Array]:
        num_particles = self.num_samples
        decision_steps = self.ctrl_steps
        act_dim = self.task.u_min.shape[-1]

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, t):
            states, k = carry
            k, k_policy, k_noise = jax.random.split(k, 3)
            if self.use_policy and self._has_policy(model_params):
                policy_keys = jax.random.split(k_policy, num_particles)
                actions = jax.vmap(
                    lambda o, pk: self._sample_policy_actions(o, pk, model_params)
                )(states.obs, policy_keys)
            else:
                mu = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
                actions = mu
                noise = (
                    jax.random.normal(k_noise, actions.shape) * self.action_noise_std
                )
                actions = jnp.clip(actions + noise, self.task.u_min, self.task.u_max)
            next_states, rewards = jax.vmap(lambda s, a: self._step_env_repeat(s, a))(
                states, actions
            )
            return (next_states, k), (
                actions,
                rewards,
                next_states.data,
                next_states.obs,
            )

        (
            (final_states, post_rollout_key),
            (actions, rewards, traj_data_steps, traj_obs_steps),
        ) = jax.lax.scan(_scan_fn, (states0, key), jnp.arange(decision_steps))
        del final_states

        actions = jnp.swapaxes(actions, 0, 1)
        traj_rewards = jnp.swapaxes(rewards, 0, 1)
        traj_data_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_data_steps)
        traj_obs_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_obs_steps)

        traj_data = jax.tree.map(
            lambda step_data, s0: jnp.concatenate(
                [s0[:, None, ...], step_data], axis=1
            ),
            traj_data_steps,
            states0.data,
        )
        returns = jnp.sum(traj_rewards, axis=1)
        if self.use_critic and self._has_critic(model_params):
            if self.use_policy and self._has_policy(model_params):
                value_keys = jax.random.split(
                    post_rollout_key, num_particles * decision_steps
                )
                value_keys = value_keys.reshape(
                    (num_particles, decision_steps) + value_keys.shape[1:]
                )
                value_actions = jax.vmap(
                    lambda obs_p, keys_p: jax.vmap(
                        lambda o, k: self._sample_policy_actions(o, k, model_params)
                    )(obs_p, keys_p)
                )(traj_obs_steps, value_keys)
                value_actions = jnp.clip(
                    value_actions, self.task.u_min, self.task.u_max
                )
            else:
                value_actions = actions

            flat_next_obs = jax.tree.map(
                lambda x: x.reshape((num_particles * decision_steps,) + x.shape[2:]),
                traj_obs_steps,
            )
            flat_value_actions = value_actions.reshape(
                (num_particles * decision_steps, act_dim)
            )
            flat_values_next = self._critic_value(
                flat_next_obs,
                flat_value_actions,
                model_params,
            )
            values_next = flat_values_next.reshape((num_particles, decision_steps))

            # Direct TD(lambda) recursion at planner decision timescale:
            # G_t = r_t + gamma_decision * ((1-lambda) * V_{t+1} + lambda * G_{t+1})
            rev_ts = jnp.arange(decision_steps - 1, -1, -1)

            def _td_lambda_step(g_next, t):
                r_t = traj_rewards[:, t]
                v_tp1 = values_next[:, t]
                g_t = r_t + self.gamma_decision * (
                    (1.0 - self.gae_lambda) * v_tp1 + self.gae_lambda * g_next
                )
                return g_t, g_t

            bootstrap = values_next[:, -1]
            td_lambda_return, _ = jax.lax.scan(_td_lambda_step, bootstrap, rev_ts)
            returns = td_lambda_return

        weights = jnn.softmax(returns / self.temperature, axis=0)
        mean_actions = jnp.sum(weights[:, None, None] * actions, axis=0)
        all_traj_obs = jax.tree.map(
            lambda step_obs, o0: jnp.concatenate(
                [o0[:, None, ...], step_obs[:, :-1, ...]], axis=1
            ),
            traj_obs_steps,
            states0.obs,
        )
        all_traj_next_obs = traj_obs_steps

        return (
            _TreeRollout(  # type: ignore
                traj_data=traj_data,
                traj_actions=actions,
                traj_rewards=traj_rewards,
                all_traj_obs=all_traj_obs,
                all_traj_next_obs=all_traj_next_obs,
                all_traj_actions=actions,
                all_traj_rewards=traj_rewards,
                returns=returns,
            ),
            mean_actions,
        )

    def optimize(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ):
        self._validate_model_params(model_params)
        if self.planner == "mppi":

            def _mppi_iter_step(carry, _):
                params, rng = carry
                rng, mppi_rng = jax.random.split(rng)
                rollouts, mean_actions = self._mppi_expand(
                    mppi_rng, state, params, model_params
                )
                params = params.replace(actions=mean_actions, rng=rng)
                return (params, rng), rollouts

            (params, _), rollouts = jax.lax.scan(
                _mppi_iter_step, (params, params.rng), jnp.arange(self.iterations)
            )
            rollouts = jax.tree.map(lambda x: x[-1], rollouts)

            costs = -rollouts.traj_rewards

            def _trace_sites_mppi(x):
                return self.task.get_trace_sites(x)

            trace_sites = jax.vmap(jax.vmap(_trace_sites_mppi))(rollouts.traj_data)
            traj_knots = rollouts.traj_actions

            return params, Trajectory(
                controls=rollouts.traj_actions,
                knots=traj_knots,
                costs=costs,
                trace_sites=trace_sites,
            )

        def _sir_iter_step(carry, _):
            params, rng = carry
            rng, tree_rng = jax.random.split(rng)
            rollouts = self._tree_expand(tree_rng, state, params, model_params)

            rng, select_rng = jax.random.split(rng)
            idx = jax.random.randint(select_rng, (), 0, rollouts.traj_actions.shape[0])
            sampled_actions = rollouts.traj_actions[idx]
            params = params.replace(actions=sampled_actions, rng=rng)

            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _sir_iter_step, (params, params.rng), jnp.arange(self.iterations)
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)

        costs = jnp.zeros(
            (rollouts.returns.shape[0], self.ctrl_steps), dtype=jnp.float32
        )
        costs = costs.at[:, -1].set(-rollouts.returns)

        def _trace_sites_sir(x):
            return self.task.get_trace_sites(x)

        trace_sites = jax.vmap(jax.vmap(_trace_sites_sir))(rollouts.traj_data)
        traj_knots = rollouts.traj_actions

        return params, Trajectory(
            controls=rollouts.traj_actions,
            knots=traj_knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def optimize_with_rollouts(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ) -> tuple[TreeMPCParams, _TreeRollout]:
        self._validate_model_params(model_params)
        if self.planner == "mppi":

            def _mppi_iter_step(carry, _):
                params, rng = carry
                rng, mppi_rng = jax.random.split(rng)
                rollouts, mean_actions = self._mppi_expand(
                    mppi_rng, state, params, model_params
                )
                params = params.replace(actions=mean_actions, rng=rng)
                return (params, rng), rollouts

            (params, _), rollouts = jax.lax.scan(
                _mppi_iter_step, (params, params.rng), jnp.arange(self.iterations)
            )
            rollouts = jax.tree.map(lambda x: x[-1], rollouts)
            return params, rollouts

        def _sir_iter_step(carry, _):
            params, rng = carry
            rng, tree_rng = jax.random.split(rng)
            rollouts = self._tree_expand(tree_rng, state, params, model_params)

            rng, select_rng = jax.random.split(rng)
            idx = jax.random.randint(select_rng, (), 0, rollouts.traj_actions.shape[0])
            sampled_actions = rollouts.traj_actions[idx]
            params = params.replace(actions=sampled_actions, rng=rng)

            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _sir_iter_step, (params, params.rng), jnp.arange(self.iterations)
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.ctrl_steps, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
