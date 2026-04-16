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
    target_policy_params: Any
    qr_params: Any


def _broadcast_tree(tree: Any, batch: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), tree)


def _reduce_q(q_values: jax.Array, use_bro: bool) -> jax.Array:
    if q_values.ndim == 1:
        q_values = q_values[:, None]
    if use_bro:
        return jnp.mean(q_values, axis=-1)
    return jnp.min(q_values, axis=-1)


def _state_truncation(state: mjx_env.State) -> jax.Array:
    if "truncation" in state.info:
        return jnp.asarray(state.info["truncation"], dtype=jnp.float32)
    return jnp.zeros_like(state.done, dtype=jnp.float32)


def _state_is_active(state: mjx_env.State) -> jax.Array:
    truncation = _state_truncation(state)
    done = jnp.asarray(state.done, dtype=jnp.bool_)
    return jnp.logical_not(jnp.logical_or(done, truncation > 0.0))


@struct.dataclass
class _TreeRollout:
    traj_data: Any
    raw_action_sequences: jax.Array
    traj_actions: jax.Array
    traj_rewards: jax.Array
    all_traj_actions: jax.Array
    all_traj_rewards: jax.Array
    candidate_raw_actions: jax.Array
    candidate_q_values: jax.Array
    returns: jax.Array


class TreeMPC:
    """Tree-structured MPC planner that scores candidates with rollout or direct Q."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        sac_network: Any,
        *,
        num_samples: int,
        horizon: int,
        use_bro: bool = True,
        zoh_steps: int = 1,
        gamma: float = 0.99,
        reward_scaling: float = 1.0,
        temperature: float = 1.0,
        iterations: int = 1,
        rollout_actions: bool = True,
        terminal_action_samples: int = 1,
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
        self.rollout_actions = bool(rollout_actions)
        self.terminal_action_samples = int(terminal_action_samples)
        if self.iterations < 1:
            raise ValueError("TreeMPC requires iterations >= 1.")
        if self.terminal_action_samples < 1:
            raise ValueError("TreeMPC requires terminal_action_samples >= 1.")
        self.use_bro = bool(use_bro)
        if sac_network is None:
            raise ValueError("TreeMPC requires a policy/Q network at init.")
        self._sac_network = sac_network

    def _direct_q_expand(
        self,
        key: jax.Array,
        state: mjx_env.State,
        model_params: TreeMPCModelParams,
    ) -> _TreeRollout:
        num_particles = self.num_samples
        act_dim = self.task.u_min.shape[-1]

        policy_keys = jax.random.split(key, num_particles)
        raw_root_actions = jax.vmap(
            lambda policy_key: self._sample_policy_raw_actions(
                state.obs,
                policy_key,
                model_params.target_policy_params,
                model_params.normalizer_params,
            )
        )(policy_keys)
        root_actions = self.action_sequence(raw_root_actions)
        q_values = jax.vmap(lambda action: self._q(state.obs, action, model_params))(
            root_actions
        )

        raw_action_sequences = jnp.broadcast_to(
            raw_root_actions[:, None, :],
            (num_particles, self.ctrl_steps, act_dim),
        )
        decision_actions = jnp.broadcast_to(
            root_actions[:, None, :],
            (num_particles, self.ctrl_steps, act_dim),
        )
        repeated_actions = jnp.broadcast_to(
            root_actions[:, None, :],
            (num_particles, self.horizon, act_dim),
        )
        repeated_state_data = _broadcast_tree(state.data, self.ctrl_steps + 1)
        zero_step_rewards = jnp.zeros((self.ctrl_steps,), dtype=jnp.float32)
        zero_traj_rewards = jnp.zeros((self.horizon,), dtype=jnp.float32)

        winner_idx = jnp.argmax(q_values)
        return _TreeRollout(  # type: ignore
            traj_data=repeated_state_data,
            raw_action_sequences=raw_action_sequences,
            traj_actions=decision_actions[winner_idx],
            traj_rewards=zero_step_rewards,
            all_traj_actions=repeated_actions[winner_idx],
            all_traj_rewards=zero_traj_rewards,
            candidate_raw_actions=raw_root_actions,
            candidate_q_values=q_values,
            returns=q_values,
        )

    def _validate_model_params(self, model_params: TreeMPCModelParams | None) -> None:
        if model_params is None:
            raise ValueError("TreeMPC requires model_params.")
        if model_params.target_policy_params is None:
            raise ValueError("TreeMPC requires model_params.target_policy_params.")
        if model_params.qr_params is None:
            raise ValueError("TreeMPC requires model_params.qr_params.")

    def _raw_to_action(self, raw_action: jax.Array) -> jax.Array:
        action = self._sac_network.parametric_action_distribution.postprocess(
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
        policy_params: Any,
        normalizer_params: Any,
    ) -> jax.Array:
        dist_params = self._sac_network.policy_network.apply(
            normalizer_params,
            policy_params,
            obs,
        )
        dist = self._sac_network.parametric_action_distribution
        return dist.sample_no_postprocessing(dist_params, key)

    def _sample_target_policy_action(
        self,
        obs: Any,
        key: jax.Array,
        model_params: TreeMPCModelParams,
    ) -> jax.Array:
        raw_action = self._sample_policy_raw_actions(
            obs,
            key,
            model_params.target_policy_params,
            model_params.normalizer_params,
        )
        return self._raw_to_action(raw_action)

    def _q(
        self,
        obs: Any,
        action: jax.Array,
        model_params: TreeMPCModelParams,
    ) -> jax.Array:
        q_values = self._sac_network.qr_network.apply(
            model_params.normalizer_params,
            model_params.qr_params,
            obs,
            action,
        )
        return _reduce_q(q_values, self.use_bro)

    def _estimate_target_q(
        self,
        obs: Any,
        key: jax.Array,
        model_params: TreeMPCModelParams,
    ) -> jax.Array:
        batch_size = jax.tree_util.tree_leaves(obs)[0].shape[0]
        sample_shape = (batch_size, self.terminal_action_samples)
        keys = jax.random.split(key, batch_size * self.terminal_action_samples)
        keys = keys.reshape(sample_shape + keys.shape[1:])

        q_samples = jax.vmap(
            lambda obs_i, keys_i: jax.vmap(
                lambda key_i: self._q(
                    obs_i,
                    self._sample_target_policy_action(obs_i, key_i, model_params),
                    model_params,
                )
            )(keys_i)
        )(obs, keys)
        return jnp.mean(q_samples, axis=1)

    def _rollout_zoh(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> tuple[mjx_env.State, Any, Any, jax.Array, jax.Array, jax.Array]:
        def _repeat_fn(carry, _):
            current_state, active = carry

            def _step_active(s: mjx_env.State) -> mjx_env.State:
                return self.task.env.step(s, action)

            next_state = jax.lax.cond(
                active,
                _step_active,
                lambda s: s,
                current_state,
            )
            raw_reward = jnp.asarray(next_state.reward, dtype=jnp.float32)
            raw_discount = jnp.asarray(1.0 - next_state.done, dtype=jnp.float32)
            raw_truncation = _state_truncation(next_state)
            reward = jnp.where(active, raw_reward, jnp.zeros_like(raw_reward))
            discount = jnp.where(active, raw_discount, jnp.zeros_like(raw_discount))
            truncation = jnp.where(
                active,
                raw_truncation,
                jnp.zeros_like(raw_truncation),
            )
            next_active = jnp.logical_and(
                active,
                jnp.logical_and(raw_discount > 0.0, raw_truncation <= 0.0),
            )
            return (next_state, next_active), (
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
        ) = jax.lax.scan(
            _repeat_fn,
            (state, _state_is_active(state)),
            jnp.arange(self.zoh_steps),
        )

        return (
            final_state[0],
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
        model_params: TreeMPCModelParams,
    ) -> _TreeRollout:
        if not self.rollout_actions:
            return self._direct_q_expand(key, state, model_params)

        num_particles = self.num_samples
        horizon_steps = self.horizon
        act_dim = self.task.u_min.shape[-1]

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, _):
            states, rng = carry
            rng, policy_rng = jax.random.split(rng)
            policy_keys = jax.random.split(policy_rng, num_particles)
            raw_decision_actions = jax.vmap(
                lambda obs, pk: self._sample_policy_raw_actions(
                    obs,
                    pk,
                    model_params.target_policy_params,
                    model_params.normalizer_params,
                )
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

            return (next_states, rng), (
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
            (_, terminal_rng),
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
        ) = jax.lax.scan(_scan_fn, (states0, key), jnp.arange(self.ctrl_steps))

        decision_raw_actions = jnp.swapaxes(decision_raw_actions, 0, 1)
        decision_actions = jnp.swapaxes(decision_actions, 0, 1)
        traj_rewards = jnp.swapaxes(decision_rewards, 0, 1)
        traj_data_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_data_steps)

        def _flatten_time(x: jax.Array) -> jax.Array:
            x = jnp.swapaxes(x, 0, 1)
            return x.reshape((num_particles, horizon_steps) + x.shape[3:])

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

        terminal_obs = jax.tree.map(lambda x: x[:, -1, ...], all_traj_next_obs)
        terminal_q = self._estimate_target_q(terminal_obs, terminal_rng, model_params)

        step_discount = self.gamma * all_traj_discount * (1.0 - all_traj_truncation)
        reward_weights = jnp.cumprod(
            jnp.concatenate(
                [
                    jnp.ones((num_particles, 1), dtype=step_discount.dtype),
                    step_discount[:, :-1],
                ],
                axis=1,
            ),
            axis=1,
        )
        terminal_weight = jnp.prod(step_discount, axis=1)
        returns = (
            jnp.sum(
                reward_weights * (self.reward_scaling * all_traj_rewards),
                axis=1,
            )
            + terminal_weight * terminal_q
        )

        winner_idx = jnp.argmax(returns)
        return _TreeRollout(  # type: ignore
            traj_data=jax.tree.map(lambda x: x[winner_idx], traj_data),
            raw_action_sequences=decision_raw_actions,
            traj_actions=decision_actions[winner_idx],
            traj_rewards=traj_rewards[winner_idx],
            all_traj_actions=all_traj_actions[winner_idx],
            all_traj_rewards=all_traj_rewards[winner_idx],
            candidate_raw_actions=decision_raw_actions[:, 0, :],
            candidate_q_values=returns,
            returns=returns,
        )

    def _mppi_iterate(
        self,
        state: mjx_env.State,
        params: TreeMPCParams,
        model_params: TreeMPCModelParams,
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
            candidate_q_values=rollouts.candidate_q_values.reshape(
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
        assert model_params is not None
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
        assert model_params is not None
        return self._mppi_iterate(state, params, model_params)

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.ctrl_steps, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
