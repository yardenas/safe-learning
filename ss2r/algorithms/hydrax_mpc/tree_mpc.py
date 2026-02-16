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
    returns: jax.Array


class TreeMPC:
    """MPPI planner with optional policy-seeded first iteration."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        *,
        num_samples: int,
        horizon: Optional[int] = None,
        use_policy: bool = False,
        use_critic: bool = False,
        n_critics: int = 2,
        n_heads: int = 1,
        action_repeat: int = 1,
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_noise_std: float = 0.3,
        iterations: int = 1,
    ) -> None:
        self.task = task
        self.dt = float(self.task.dt)

        self.num_samples = int(num_samples)
        if self.num_samples < 1:
            raise ValueError("TreeMPC requires num_samples >= 1.")

        if horizon is None:
            raise ValueError("TreeMPC requires horizon to be set explicitly.")
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("TreeMPC requires horizon >= 1.")

        self.plan_horizon = float(self.horizon) * self.dt
        self.action_repeat = max(int(action_repeat), 1)
        if self.horizon % self.action_repeat != 0:
            raise ValueError(
                "TreeMPC requires horizon to be divisible by action_repeat."
            )
        self.ctrl_steps = self.horizon // self.action_repeat

        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.action_noise_std = float(action_noise_std)
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("TreeMPC requires iterations >= 1.")

        self.uses_env_step = True
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
        if self.action_repeat == 1:
            next_state = self.task.env.step(state, action)
            return next_state, next_state.reward

        def _repeat_fn(carry, _):
            s = carry
            s = self.task.env.step(s, action)
            return s, s.reward

        final_state, rewards = jax.lax.scan(
            _repeat_fn, state, jnp.arange(self.action_repeat)
        )
        return final_state, jnp.sum(rewards)

    def _mppi_expand(
        self,
        key: jax.Array,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
        *,
        use_policy_proposal: bool,
    ) -> tuple[_TreeRollout, jax.Array]:
        num_particles = self.num_samples
        decision_steps = self.ctrl_steps
        act_dim = self.task.u_min.shape[-1]

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, t):
            states, k = carry
            k, k_policy, k_noise = jax.random.split(k, 3)

            mean_action = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
            noise = (
                jax.random.normal(k_noise, mean_action.shape) * self.action_noise_std
            )
            noisy_actions = jnp.clip(
                mean_action + noise,
                self.task.u_min,
                self.task.u_max,
            )

            if (
                use_policy_proposal
                and self.use_policy
                and self._has_policy(model_params)
            ):
                policy_keys = jax.random.split(k_policy, num_particles)
                actions = jax.vmap(
                    lambda o, pk: self._sample_policy_actions(o, pk, model_params)
                )(states.obs, policy_keys)
                actions = jnp.clip(actions, self.task.u_min, self.task.u_max)
            else:
                actions = noisy_actions

            next_states, rewards = jax.vmap(lambda s, a: self._step_env_repeat(s, a))(
                states, actions
            )
            return (next_states, k), (
                states.obs,
                actions,
                rewards,
                next_states.obs,
                next_states.data,
            )

        (
            (final_states, post_rollout_key),
            (
                obs_steps,
                actions,
                rewards,
                next_obs_steps,
                traj_data_steps,
            ),
        ) = jax.lax.scan(_scan_fn, (states0, key), jnp.arange(decision_steps))

        actions = jnp.swapaxes(actions, 0, 1)
        traj_rewards = jnp.swapaxes(rewards, 0, 1)
        obs_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs_steps)
        next_obs_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), next_obs_steps)
        traj_data_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_data_steps)

        traj_data = jax.tree.map(
            lambda step_data, s0: jnp.concatenate(
                [s0[:, None, ...], step_data], axis=1
            ),
            traj_data_steps,
            states0.data,
        )

        discounts = jnp.power(self.gamma, jnp.arange(decision_steps, dtype=jnp.float32))
        returns = jnp.sum(traj_rewards * discounts[None, :], axis=1)

        # Optional terminal bootstrap only; no in-horizon TD(lambda)/GAE recursion.
        if self.use_critic and self._has_critic(model_params):
            if self.use_policy and self._has_policy(model_params):
                terminal_keys = jax.random.split(post_rollout_key, num_particles)
                terminal_actions = jax.vmap(
                    lambda o, tk: self._sample_policy_actions(o, tk, model_params)
                )(final_states.obs, terminal_keys)
                terminal_actions = jnp.clip(
                    terminal_actions,
                    self.task.u_min,
                    self.task.u_max,
                )
            else:
                terminal_actions = actions[:, -1, :]

            terminal_value = self._critic_value(
                final_states.obs,
                terminal_actions,
                model_params,
            )
            terminal_discount = jnp.power(
                jnp.asarray(self.gamma, dtype=jnp.float32),
                jnp.asarray(decision_steps, dtype=jnp.float32),
            )
            returns = returns + terminal_discount * terminal_value

        weights = jnn.softmax(returns / self.temperature, axis=0)
        mean_actions = jnp.sum(weights[:, None, None] * actions, axis=0)

        rollout = _TreeRollout(  # type: ignore
            traj_data=traj_data,
            traj_actions=actions,
            traj_rewards=traj_rewards,
            all_traj_obs=obs_steps,
            all_traj_next_obs=next_obs_steps,
            all_traj_actions=actions,
            all_traj_rewards=traj_rewards,
            returns=returns,
        )
        return rollout, mean_actions

    def _run_mppi_iterations(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None,
    ) -> tuple[TreeMPCParams, _TreeRollout]:
        rng = params.rng
        latest_rollout = None

        if self.use_policy and self._has_policy(model_params):
            rng, seed_rng = jax.random.split(rng)
            latest_rollout, seeded_actions = self._mppi_expand(
                seed_rng,
                state,
                params,
                model_params,
                use_policy_proposal=True,
            )
            params = params.replace(actions=seeded_actions, rng=rng)  # type: ignore
            remaining_iterations = self.iterations - 1
        else:
            remaining_iterations = self.iterations

        if remaining_iterations > 0:

            def _iter_step(carry, _):
                current_params, current_rng = carry
                current_rng, mppi_rng = jax.random.split(current_rng)
                rollouts, mean_actions = self._mppi_expand(
                    mppi_rng,
                    state,
                    current_params,
                    model_params,
                    use_policy_proposal=False,
                )
                current_params = current_params.replace(
                    actions=mean_actions,
                    rng=current_rng,
                )
                return (current_params, current_rng), rollouts

            (params, rng), rollout_seq = jax.lax.scan(
                _iter_step,
                (params, rng),
                None,
                length=remaining_iterations,
            )
            latest_rollout = jax.tree.map(lambda x: x[-1], rollout_seq)
            params = params.replace(rng=rng)  # type: ignore

        if latest_rollout is None:
            raise RuntimeError("TreeMPC failed to produce rollout outputs.")
        return params, latest_rollout

    def optimize(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None = None,
    ):
        self._validate_model_params(model_params)
        params, rollouts = self._run_mppi_iterations(state, params, model_params)

        costs = -rollouts.traj_rewards

        def _trace_sites_fn(x):
            return self.task.get_trace_sites(x)

        trace_sites = jax.vmap(jax.vmap(_trace_sites_fn))(rollouts.traj_data)
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
        return self._run_mppi_iterations(state, params, model_params)

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.ctrl_steps, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
