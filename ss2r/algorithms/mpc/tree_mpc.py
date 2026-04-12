from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env


@struct.dataclass
class Trajectory:
    controls: jax.Array
    knots: jax.Array
    costs: jax.Array
    trace_sites: Any


class MujocoPlaygroundTask:
    def __init__(self, env: mjx_env.MjxEnv, dt: float) -> None:
        self.env = env
        self.dt = float(dt)
        ctrl_range = jnp.asarray(env.mj_model.actuator_ctrlrange, dtype=jnp.float32)
        self.u_min = ctrl_range[:, 0]
        self.u_max = ctrl_range[:, 1]

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        del u
        if isinstance(x, dict):
            reward = x.get("reward", None)
            if reward is None:
                raise ValueError(
                    "Task payload missing reward; ensure env.step provides reward."
                )
            return -reward
        if hasattr(x, "reward"):
            return -x.reward
        raise ValueError(
            "Task running_cost expects reward in state payload or mjx.State."
        )

    def terminal_cost(self, x: jax.Array) -> float:
        del x
        return 0.0

    def get_trace_sites(self, x: mjx.Data) -> jax.Array:
        if hasattr(x, "site_xpos"):
            return jnp.asarray(x.site_xpos, dtype=jnp.float32)
        return jnp.zeros((0, 3), dtype=jnp.float32)


def make_task(env: mjx_env.MjxEnv) -> MujocoPlaygroundTask:
    env_dt = getattr(env, "_ctrl_dt", None)
    if env_dt is None:
        raise ValueError("Unable to infer controller dt from environment.")
    return MujocoPlaygroundTask(env, float(env_dt))


@struct.dataclass
class TreeMPCParams:
    actions: jax.Array
    rng: jax.Array


@struct.dataclass
class TreeMPCModelParams:
    normalizer_params: Any
    policy_params: Any
    value_params: Any


def _broadcast_tree(tree: Any, batch: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), tree)


def _compute_discounted_returns(
    rewards: jax.Array,
    discounts: jax.Array,
    truncations: jax.Array,
    *,
    gamma: float,
    reward_scaling: float,
) -> jax.Array:
    bootstrap_discount = gamma * discounts * (1.0 - truncations)

    def _return_step(carry, x):
        reward_t, discount_t = x
        ret_t = reward_scaling * reward_t + discount_t * carry
        return ret_t, ret_t

    _, returns_rev = jax.lax.scan(
        _return_step,
        jnp.zeros_like(rewards[:, -1]),
        (
            jnp.flip(rewards, axis=1).T,
            jnp.flip(bootstrap_discount, axis=1).T,
        ),
    )
    return jnp.flip(returns_rev.T, axis=1)


def _compute_generalized_advantages(
    rewards: jax.Array,
    discounts: jax.Array,
    truncations: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
    reward_scaling: float,
) -> tuple[jax.Array, jax.Array]:
    bootstrap_discount = gamma * discounts * (1.0 - truncations)
    deltas = reward_scaling * rewards + bootstrap_discount * next_values - values
    rev_ts = jnp.arange(rewards.shape[1] - 1, -1, -1)

    def _gae_step(carry: jax.Array, t: jax.Array):
        delta_t = jax.lax.dynamic_index_in_dim(deltas, t, axis=1, keepdims=False)
        discount_t = jax.lax.dynamic_index_in_dim(
            bootstrap_discount, t, axis=1, keepdims=False
        )
        advantage_t = delta_t + discount_t * gae_lambda * carry
        return advantage_t, advantage_t

    _, advantages_rev = jax.lax.scan(
        _gae_step,
        jnp.zeros((rewards.shape[0],), dtype=jnp.float32),
        rev_ts,
    )
    advantages = jnp.flip(advantages_rev, axis=0).T
    returns = values[:, 0] + advantages[:, 0]
    return advantages, returns


@struct.dataclass
class _TreeRollout:
    traj_data: Any
    raw_action_sequences: jax.Array
    traj_actions: jax.Array
    traj_rewards: jax.Array
    all_traj_actions: jax.Array
    all_traj_rewards: jax.Array
    candidate_raw_actions: jax.Array
    candidate_actions: jax.Array
    candidate_advantages: jax.Array
    returns: jax.Array


class TreeMPC:
    """Tree-structured MPC planner that samples candidate actions from a policy."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        awac_network: Any,
        *,
        num_samples: int,
        horizon: int,
        gae_lambda: float = 0.0,
        use_value: bool = False,
        n_critics: int = 2,
        use_bro: bool = True,
        zoh_steps: int = 1,
        gamma: float = 0.99,
        reward_scaling: float = 1.0,
        temperature: float = 1.0,
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
        self.reward_scaling = float(reward_scaling)
        self.temperature = float(temperature)
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("TreeMPC requires iterations >= 1.")
        self.gae_lambda = float(gae_lambda)
        self.use_value = bool(use_value)
        self._n_critics = int(n_critics)
        self.use_bro = bool(use_bro)
        if self._n_critics < 1:
            raise ValueError("TreeMPC requires n_critics >= 1.")
        if awac_network is None:
            raise ValueError("TreeMPC requires a policy/value network at init.")
        self._awac_network = awac_network

    def _validate_model_params(self, model_params: TreeMPCModelParams | None) -> None:
        if model_params is None or model_params.policy_params is None:
            raise ValueError("TreeMPC requires model_params with policy_params.")
        if self.use_value and model_params.value_params is None:
            raise ValueError(
                "TreeMPC with use_value=True requires a bound value network and "
                "model_params with value_params."
            )

    def _raw_to_action(self, raw_action: jax.Array) -> jax.Array:
        action = self._awac_network.parametric_action_distribution.postprocess(
            raw_action
        )
        return jnp.clip(action, self.task.u_min, self.task.u_max)

    def action_sequence(self, raw_actions: jax.Array) -> jax.Array:
        flat_actions = raw_actions.reshape((-1, raw_actions.shape[-1]))
        actions = jax.vmap(self._raw_to_action)(flat_actions)
        return actions.reshape(raw_actions.shape)

    def raw_action_sequence_from_actions(self, actions: jax.Array) -> jax.Array:
        clipped_actions = jnp.clip(actions, self.task.u_min, self.task.u_max)
        normalized = (
            2.0
            * (clipped_actions - self.task.u_min)
            / (self.task.u_max - self.task.u_min)
            - 1.0
        )
        normalized = jnp.clip(normalized, -0.999999, 0.999999)
        return jnp.arctanh(normalized)

    def _sample_policy_raw_actions(
        self,
        obs: Any,
        key: jax.Array,
        model_params: TreeMPCModelParams | None = None,
    ) -> jax.Array:
        if model_params is None:
            raise ValueError("_sample_policy_raw_actions requires model_params.")
        policy_params = model_params.policy_params
        if policy_params is None:
            raise ValueError(
                "_sample_policy_raw_actions requires model_params.policy_params."
            )
        dist_params = self._awac_network.policy_network.apply(
            model_params.normalizer_params,
            policy_params,
            obs,
        )
        dist = self._awac_network.parametric_action_distribution
        return dist.sample_no_postprocessing(dist_params, key)

    def _value(
        self,
        obs: Any,
        model_params: TreeMPCModelParams | None = None,
    ) -> jax.Array:
        if model_params is None:
            raise ValueError("_value requires model_params.")
        value_params = model_params.value_params
        if value_params is None:
            raise ValueError("_value requires model_params.value_params.")
        values = self._awac_network.value_network.apply(
            model_params.normalizer_params,
            value_params,
            obs,
        )
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[-1] != self._n_critics:
            raise ValueError(
                "Unexpected value output width for TreeMPC. "
                f"got={values.shape[-1]}, expected={self._n_critics}."
            )
        if self.use_bro:
            return jnp.mean(values, axis=-1)
        return jnp.min(values, axis=-1)

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
        model_params: TreeMPCModelParams | None = None,
    ) -> _TreeRollout:
        num_particles = self.num_samples
        decision_steps = self.ctrl_steps
        horizon_steps = self.horizon
        act_dim = self.task.u_min.shape[-1]

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, t):
            states, k = carry
            del t
            k, k_policy = jax.random.split(k)
            policy_keys = jax.random.split(k_policy, num_particles)
            raw_decision_actions = jax.vmap(
                lambda o, pk: self._sample_policy_raw_actions(o, pk, model_params)
            )(states.obs, policy_keys)
            decision_actions = self.action_sequence(raw_decision_actions)

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
                raw_decision_actions,
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
            (_, _),
            (
                decision_raw_actions,
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

        decision_raw_actions = jnp.swapaxes(decision_raw_actions, 0, 1)
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

        if self.use_value:
            split_idx = num_particles * horizon_steps
            flat_value_obs = jax.tree.map(
                lambda obs, next_obs: jnp.concatenate(
                    [
                        obs.reshape((split_idx,) + obs.shape[2:]),
                        next_obs[:, -1, ...],
                    ],
                    axis=0,
                ),
                all_traj_obs,
                all_traj_next_obs,
            )
            value_sequence = self._value(flat_value_obs, model_params)
            values = value_sequence[:split_idx].reshape((num_particles, horizon_steps))
            next_values = jnp.concatenate(
                [values[:, 1:], value_sequence[split_idx:, None]],
                axis=1,
            )
            all_traj_advantages, returns = _compute_generalized_advantages(
                all_traj_rewards,
                all_traj_discount,
                all_traj_truncation,
                values,
                next_values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scaling=self.reward_scaling,
            )
        else:
            discounted_returns = _compute_discounted_returns(
                all_traj_rewards,
                all_traj_discount,
                all_traj_truncation,
                gamma=self.gamma,
                reward_scaling=self.reward_scaling,
            )
            all_traj_advantages = discounted_returns
            returns = discounted_returns[:, 0]

        winner_idx = jnp.argmax(returns)
        return _TreeRollout(  # type: ignore
            traj_data=jax.tree.map(lambda x: x[winner_idx], traj_data),
            raw_action_sequences=decision_raw_actions,
            traj_actions=decision_actions[winner_idx],
            traj_rewards=traj_rewards[winner_idx],
            all_traj_actions=all_traj_actions[winner_idx],
            all_traj_rewards=all_traj_rewards[winner_idx],
            candidate_raw_actions=decision_raw_actions[:, 0, :],
            candidate_actions=decision_actions[:, 0, :],
            candidate_advantages=all_traj_advantages[:, 0],
            returns=returns,
        )

    def _mppi_iterate(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams | None,
    ) -> tuple[TreeMPCParams, _TreeRollout]:
        act_dim = self.task.u_min.shape[-1]

        def _mppi_iter_step(rng, _):
            rng, mppi_rng = jax.random.split(rng)
            rollouts = self._mppi_expand(
                mppi_rng,
                state,
                model_params,
            )
            return rng, rollouts

        # FIXME: iterations currently only repeats independent policy-sampled
        # candidate batches; there is no refinement state passed between passes.
        rng, rollouts = jax.lax.scan(
            _mppi_iter_step,
            params.rng,
            jnp.arange(self.iterations),
        )

        flat_returns = rollouts.returns.reshape((self.iterations * self.num_samples,))
        flat_raw_action_sequences = rollouts.raw_action_sequences.reshape(
            (self.iterations * self.num_samples, self.ctrl_steps, act_dim)
        )
        weights = jnn.softmax(flat_returns / self.temperature, axis=0)
        mean_raw_actions = jnp.sum(
            weights[:, None, None] * flat_raw_action_sequences,
            axis=0,
        )

        best_iteration_idx = jnp.argmax(jnp.max(rollouts.returns, axis=1))
        best_rollout = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(
                x, best_iteration_idx, axis=0, keepdims=False
            ),
            rollouts,
        )
        best_rollout = best_rollout.replace(
            raw_action_sequences=flat_raw_action_sequences,
            candidate_raw_actions=rollouts.candidate_raw_actions.reshape(
                (self.iterations * self.num_samples, act_dim)
            ),
            candidate_actions=rollouts.candidate_actions.reshape(
                (self.iterations * self.num_samples, act_dim)
            ),
            candidate_advantages=rollouts.candidate_advantages.reshape(
                (self.iterations * self.num_samples,)
            ),
            returns=flat_returns,
        )
        params = params.replace(actions=mean_raw_actions, rng=rng)  # type: ignore
        return params, best_rollout

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

    def optimize_with_candidates(
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
