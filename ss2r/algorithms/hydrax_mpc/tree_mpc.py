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


@struct.dataclass
class _TreeRollout:
    traj_data: mjx.Data
    traj_actions: jax.Array
    traj_rewards: jax.Array
    all_traj_obs: Any
    all_traj_next_obs: Any
    all_traj_actions: jax.Array
    all_traj_rewards: jax.Array
    all_traj_discount: jax.Array
    all_traj_truncation: jax.Array
    all_traj_advantages: jax.Array
    returns: jax.Array


class TreeMPC:
    """Tree-structured MPPI planner with optional policy prior."""

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
        use_bro: bool = True,
        zoh_steps: int = 1,
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_noise_std: float = 0.3,
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
        self.temperature = float(temperature)
        self.action_noise_std = action_noise_std
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("TreeMPC requires iterations >= 1.")
        self.uses_env_step = True
        self.gae_lambda = float(gae_lambda)
        self.use_policy = bool(use_policy)
        self.use_critic = bool(use_critic)
        self._n_critics = int(n_critics)
        self._n_heads = int(n_heads)
        self.use_bro = bool(use_bro)
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
        q_values = q_values.reshape((q_values.shape[0], expected_width))
        if self.use_bro:
            return jnp.mean(q_values, axis=-1)
        return jnp.min(q_values, axis=-1)

    def _rollout_zoh(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> tuple[mjx_env.State, Any, Any, jax.Array, jax.Array, jax.Array]:
        def _repeat_fn(carry, _):
            current_state = carry
            next_state = self.task.env.step(current_state, action)
            reward = jnp.asarray(next_state.reward, dtype=jnp.float32)
            discount = jnp.asarray(1.0 - next_state.done, dtype=jnp.float32)
            truncation = (
                jnp.asarray(next_state.info["truncation"], dtype=jnp.float32)
                if "truncation" in next_state.info
                else jnp.zeros_like(reward)
            )
            return next_state, (
                current_state.obs,
                next_state.obs,
                reward,
                discount,
                truncation,
            )

        (
            final_state,
            (
                obs_steps,
                next_obs_steps,
                rewards_steps,
                discount_steps,
                truncation_steps,
            ),
        ) = jax.lax.scan(_repeat_fn, state, jnp.arange(self.zoh_steps))

        return (
            final_state,
            obs_steps,
            next_obs_steps,
            rewards_steps,
            discount_steps,
            truncation_steps,
        )

    def _mppi_expand(
        self,
        key: jax.Array,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
        *,
        sample_from_policy: bool,
    ) -> tuple[_TreeRollout, jax.Array]:
        num_particles = self.num_samples
        decision_steps = self.ctrl_steps
        horizon_steps = self.horizon
        act_dim = self.task.u_min.shape[-1]

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, t):
            states, k = carry
            if sample_from_policy:
                k, k_policy = jax.random.split(k)
                policy_keys = jax.random.split(k_policy, num_particles)
                decision_actions = jax.vmap(
                    lambda o, pk: self._sample_policy_actions(o, pk, model_params)
                )(states.obs, policy_keys)
                decision_actions = jnp.clip(
                    decision_actions, self.task.u_min, self.task.u_max
                )
            else:
                k, k_noise = jax.random.split(k)
                mu = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
                noise = (
                    jax.random.normal(k_noise, (num_particles, act_dim))
                    * self.action_noise_std
                )
                decision_actions = _squash_to_bounds(
                    mu + noise, self.task.u_min, self.task.u_max
                )

            (
                next_states,
                obs_steps,
                next_obs_steps,
                rewards_steps,
                discount_steps,
                truncation_steps,
            ) = jax.vmap(self._rollout_zoh)(states, decision_actions)

            repeated_actions = jnp.broadcast_to(
                decision_actions[:, None, :],
                (num_particles, self.zoh_steps, act_dim),
            )
            decision_rewards = jnp.sum(rewards_steps, axis=1)

            return (next_states, k), (
                decision_actions,
                decision_rewards,
                next_states.data,
                obs_steps,
                next_obs_steps,
                repeated_actions,
                rewards_steps,
                discount_steps,
                truncation_steps,
            )

        (
            (final_states, _),
            (
                decision_actions,
                decision_rewards,
                traj_data_steps,
                obs_steps,
                next_obs_steps,
                repeated_actions,
                rewards_steps,
                discount_steps,
                truncation_steps,
            ),
        ) = jax.lax.scan(_scan_fn, (states0, key), jnp.arange(decision_steps))
        del final_states

        decision_actions = jnp.swapaxes(decision_actions, 0, 1)
        traj_rewards = jnp.swapaxes(decision_rewards, 0, 1)
        traj_data_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_data_steps)

        def _flatten_time(x: jax.Array) -> jax.Array:
            x = jnp.swapaxes(x, 0, 1)
            return x.reshape((num_particles, horizon_steps) + x.shape[3:])

        all_traj_obs = jax.tree.map(_flatten_time, obs_steps)
        all_traj_next_obs = jax.tree.map(_flatten_time, next_obs_steps)
        all_traj_actions = _flatten_time(repeated_actions)
        all_traj_rewards = _flatten_time(rewards_steps)
        all_traj_discount = _flatten_time(discount_steps)
        all_traj_truncation = _flatten_time(truncation_steps)

        traj_data = jax.tree.map(
            lambda step_data, s0: jnp.concatenate(
                [s0[:, None, ...], step_data], axis=1
            ),
            traj_data_steps,
            states0.data,
        )

        bootstrap_discount = (
            self.gamma * all_traj_discount * (1.0 - all_traj_truncation)
        )

        def _return_step(carry, x):
            reward_t, discount_t = x
            ret_t = reward_t + discount_t * carry
            return ret_t, ret_t

        _, returns_rev = jax.lax.scan(
            _return_step,
            jnp.zeros_like(all_traj_rewards[:, -1]),
            (
                jnp.flip(all_traj_rewards, axis=1).T,
                jnp.flip(bootstrap_discount, axis=1).T,
            ),
        )
        return_to_go = jnp.flip(returns_rev.T, axis=1)
        all_traj_advantages = return_to_go
        returns = return_to_go[:, 0]

        if self.use_critic and self._has_critic(model_params):
            terminal_obs = jax.tree.map(lambda x: x[:, -1, ...], all_traj_next_obs)
            terminal_action = all_traj_actions[:, -1, :]
            terminal_value = self._critic_value(
                terminal_obs, terminal_action, model_params
            )
            terminal_discount = jnp.prod(bootstrap_discount, axis=1)
            returns = returns + terminal_discount * terminal_value

        weights = jnn.softmax(returns / self.temperature, axis=0)
        mean_actions = jnp.sum(weights[:, None, None] * decision_actions, axis=0)
        winner_idx = jnp.argmax(returns)
        winner_rollout = _TreeRollout(  # type: ignore
            traj_data=jax.tree.map(lambda x: x[winner_idx], traj_data),
            traj_actions=decision_actions[winner_idx],
            traj_rewards=traj_rewards[winner_idx],
            all_traj_obs=jax.tree.map(lambda x: x[winner_idx], all_traj_obs),
            all_traj_next_obs=jax.tree.map(lambda x: x[winner_idx], all_traj_next_obs),
            all_traj_actions=all_traj_actions[winner_idx],
            all_traj_rewards=all_traj_rewards[winner_idx],
            all_traj_discount=all_traj_discount[winner_idx],
            all_traj_truncation=all_traj_truncation[winner_idx],
            all_traj_advantages=all_traj_advantages[winner_idx],
            returns=returns[winner_idx],
        )

        return (
            winner_rollout,
            mean_actions,
        )

    def _mppi_iterate(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None,
    ) -> tuple[TreeMPCParams, _TreeRollout]:
        rng = params.rng
        policy_prior_rollouts = None

        if self.use_policy and self._has_policy(model_params):
            rng, prior_rng = jax.random.split(rng)
            policy_prior_rollouts, policy_prior_actions = self._mppi_expand(
                prior_rng,
                state,
                params,
                model_params,
                sample_from_policy=True,
            )
            params = params.replace(actions=policy_prior_actions, rng=rng)  # type: ignore

            remaining_iterations = self.iterations - 1
            if remaining_iterations <= 0:
                assert policy_prior_rollouts is not None
                return params, policy_prior_rollouts
        else:
            remaining_iterations = self.iterations

        def _mppi_iter_step(carry, _):
            params, rng = carry
            rng, mppi_rng = jax.random.split(rng)
            rollouts, mean_actions = self._mppi_expand(
                mppi_rng,
                state,
                params,
                model_params,
                sample_from_policy=False,
            )
            params = params.replace(actions=mean_actions, rng=rng)
            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _mppi_iter_step,
            (params, params.rng),
            jnp.arange(remaining_iterations),
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts

    def optimize(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ):
        self._validate_model_params(model_params)
        params, rollouts = self._mppi_iterate(state, params, model_params)

        costs = -rollouts.all_traj_rewards[None, :]

        def _trace_sites_mppi(x):
            return self.task.get_trace_sites(x)

        trace_sites = jax.vmap(_trace_sites_mppi)(rollouts.traj_data)
        trace_sites = jax.tree.map(lambda x: x[None, ...], trace_sites)

        return params, Trajectory(
            controls=rollouts.all_traj_actions[None, ...],
            knots=rollouts.traj_actions[None, ...],
            costs=costs,
            trace_sites=trace_sites,
        )

    def optimize_with_winner(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ) -> tuple[TreeMPCParams, _TreeRollout]:
        self._validate_model_params(model_params)
        return self._mppi_iterate(state, params, model_params)

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.ctrl_steps, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
