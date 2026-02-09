from typing import Any, Mapping, Optional, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint as sac_checkpoint
from flax import struct
from hydrax.alg_base import Trajectory
from mujoco import mjx
from mujoco_playground._src import mjx_env

from ss2r.algorithms.sac import networks as sac_networks
from ss2r.rl.utils import remove_pixels

from .task import MujocoPlaygroundTask


@struct.dataclass
class TreeMPCParams:
    actions: jax.Array
    rng: jax.Array


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
    returns: jax.Array


class TreeMPC:
    """SMC planner using SIR-style resampling."""

    def __init__(
        self,
        task: MujocoPlaygroundTask,
        *,
        n_particles: int,
        horizon: Optional[int] = None,
        policy_checkpoint_path: str | None = None,
        policy_noise_std: float = 0.05,
        gae_lambda: float = 0.0,
        policy_action_only: bool = False,
        use_policy: bool = True,
        normalize_observations: bool = True,
        policy_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
        value_hidden_layer_sizes: Sequence[int] = (512, 512),
        activation: str = "swish",
        n_critics: int = 2,
        n_heads: int = 1,
        use_bro: bool = True,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
        planner: str = "sir",
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_noise_std: float = 0.3,
        mode: str = "resample",
        iterations: int = 1,
    ) -> None:
        self.task = task
        self.dt = float(self.task.dt)

        self.n_particles = int(n_particles)
        if horizon is None:
            raise ValueError("TreeMPC requires horizon to be set explicitly.")
        self.horizon = int(horizon)
        self.plan_horizon = float(self.horizon) * self.dt
        self.ctrl_steps = int(self.horizon)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.action_noise_std = action_noise_std
        self.planner = planner
        self.mode = mode
        self.iterations = int(iterations)
        self.uses_env_step = True
        self.policy_noise_std = float(policy_noise_std)
        self.gae_lambda = float(gae_lambda)
        self.policy_action_only = bool(policy_action_only)
        self.use_policy = bool(use_policy)

        self._policy_params = None
        self._qr_params = None
        self._normalizer_params = None
        self._sac_network = None
        self._n_critics = int(n_critics)
        self._n_heads = int(n_heads)
        self._normalize_observations = bool(normalize_observations)

        if self.use_policy:
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
            else:
                self._init_policy_and_critic(
                    policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                    value_hidden_layer_sizes=value_hidden_layer_sizes,
                    activation=activation,
                    use_bro=use_bro,
                    policy_obs_key=policy_obs_key,
                    value_obs_key=value_obs_key,
                )

    def _init_policy_and_critic(
        self,
        *,
        policy_hidden_layer_sizes: Sequence[int],
        value_hidden_layer_sizes: Sequence[int],
        activation: str,
        use_bro: bool,
        policy_obs_key: str,
        value_obs_key: str,
        seed: int = 0,
    ) -> None:
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
        rng = jax.random.PRNGKey(seed)
        rng, key_policy = jax.random.split(rng)
        rng, key_qr = jax.random.split(rng)
        self._policy_params = self._sac_network.policy_network.init(key_policy)  # type: ignore
        self._qr_params = self._sac_network.qr_network.init(key_qr)  # type: ignore

        if isinstance(obs_size, Mapping):
            obs_shape = {
                k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
            }
        else:
            obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
        self._normalizer_params = running_statistics.init_state(
            remove_pixels(obs_shape)
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
        num_particles = self.n_particles
        horizon = self.horizon
        act_dim = self.task.u_min.shape[-1]
        states = _broadcast_tree(state, num_particles)
        returns = jnp.zeros((num_particles,), dtype=jnp.float32)
        gae_trace = jnp.zeros((num_particles,), dtype=jnp.float32)
        value_start = jnp.zeros((num_particles,), dtype=jnp.float32)

        traj_actions = jnp.zeros((num_particles, horizon, act_dim), dtype=jnp.float32)
        traj_rewards = jnp.zeros((num_particles, horizon), dtype=jnp.float32)
        traj_data = jax.tree.map(
            lambda x: jnp.zeros((num_particles, horizon + 1) + x.shape, dtype=x.dtype),
            state.data,
        )
        traj_data = jax.tree.map(lambda t, d: t.at[:, 0].set(d), traj_data, states.data)

        def _step_fn(carry, t):
            (
                key,
                states,
                traj_data,
                traj_actions,
                traj_rewards,
                returns,
                gae_trace,
                value_start,
            ) = carry
            if self._has_policy():
                key, k_policy, k_value, k_sel = jax.random.split(key, 4)
                actions = self._sample_policy_actions(states.obs, k_policy)
                actions = jnp.clip(actions, self.task.u_min, self.task.u_max)
            else:
                key, k_noise, k_sel = jax.random.split(key, 3)
                mu = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
                eps = jax.random.normal(k_noise, (num_particles, act_dim))
                std = jnp.asarray(self.action_noise_std, dtype=jnp.float32)
                u = mu + eps * std
                actions = _squash_to_bounds(u, self.task.u_min, self.task.u_max)

            def _env_step(s, a):
                return self.task.env.step(s, a)

            next_states = jax.vmap(_env_step)(states, actions)
            rewards = next_states.reward

            traj_actions = traj_actions.at[:, t, :].set(actions)
            traj_rewards = traj_rewards.at[:, t].set(rewards)
            traj_data = jax.tree.map(
                lambda td, d: td.at[:, t + 1].set(d),
                traj_data,
                next_states.data,
            )
            returns = returns + rewards

            if self._has_policy():
                value_s = self._critic_value(states.obs, actions)
                value_actions = self._sample_policy_actions(next_states.obs, k_value)
                value_actions = jnp.clip(
                    value_actions, self.task.u_min, self.task.u_max
                )
                value_next = self._critic_value(next_states.obs, value_actions)
                delta = rewards + self.gamma * value_next - value_s
                gae_trace = delta + self.gamma * self.gae_lambda * gae_trace
                value_start = jnp.where(t == 0, value_s, value_start)
                advantage = gae_trace
            else:
                advantage = rewards

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
            returns,
            gae_trace,
            value_start,
        )
        carryN, _ = jax.lax.scan(_step_fn, carry0, jnp.arange(horizon))
        (
            key,
            states,
            traj_data,
            traj_actions,
            traj_rewards,
            returns,
            gae_trace,
            value_start,
        ) = carryN

        if self._has_policy():
            returns = value_start + gae_trace

        return _TreeRollout(  # type: ignore
            traj_data=traj_data,
            traj_actions=traj_actions,
            traj_rewards=traj_rewards,
            returns=returns,
        )

    def _mppi_expand(
        self, key: jax.Array, state: mjx_env.State, params: TreeMPCParams
    ) -> tuple[_TreeRollout, jax.Array]:
        num_particles = self.n_particles
        horizon = self.horizon
        act_dim = self.task.u_min.shape[-1]

        if self._has_policy():
            noise_std = self.policy_noise_std
        else:
            noise_std = self.action_noise_std

        states0 = _broadcast_tree(state, num_particles)

        def _scan_fn(carry, t):
            states, k = carry
            k, k_policy, k_noise = jax.random.split(k, 3)
            if self._has_policy():
                policy_keys = jax.random.split(k_policy, num_particles)
                actions = jax.vmap(self._sample_policy_actions)(states.obs, policy_keys)
            else:
                mu = jnp.broadcast_to(params.actions[t], (num_particles, act_dim))
                actions = mu
            noise = jax.random.normal(k_noise, actions.shape) * noise_std
            actions = jnp.clip(actions + noise, self.task.u_min, self.task.u_max)
            next_states = jax.vmap(self.task.env.step)(states, actions)
            return (next_states, k), (actions, next_states.reward, next_states.data)

        (final_states, _), (actions, rewards, traj_data_steps) = jax.lax.scan(
            _scan_fn, (states0, key), jnp.arange(horizon)
        )
        del final_states

        actions = jnp.swapaxes(actions, 0, 1)
        traj_rewards = jnp.swapaxes(rewards, 0, 1)
        traj_data_steps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_data_steps)

        traj_data = jax.tree.map(
            lambda step_data, s0: jnp.concatenate(
                [s0[:, None, ...], step_data], axis=1
            ),
            traj_data_steps,
            states0.data,
        )
        returns = jnp.sum(traj_rewards, axis=1)

        weights = jnn.softmax(returns / self.temperature, axis=0)
        mean_actions = jnp.sum(weights[:, None, None] * actions, axis=0)

        return (
            _TreeRollout(  # type: ignore
                traj_data=traj_data,
                traj_actions=actions,
                traj_rewards=traj_rewards,
                returns=returns,
            ),
            mean_actions,
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

        if self.planner == "mppi":

            def _mppi_iter_step(carry, _):
                params, rng = carry
                rng, mppi_rng = jax.random.split(rng)
                rollouts, mean_actions = self._mppi_expand(mppi_rng, state, params)
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
            rollouts = self._tree_expand(tree_rng, state, params)

            rng, select_rng = jax.random.split(rng)
            idx = jax.random.randint(select_rng, (), 0, rollouts.traj_actions.shape[0])
            sampled_actions = rollouts.traj_actions[idx]
            params = params.replace(actions=sampled_actions, rng=rng)

            return (params, rng), rollouts

        (params, _), rollouts = jax.lax.scan(
            _sir_iter_step, (params, params.rng), jnp.arange(self.iterations)
        )
        rollouts = jax.tree.map(lambda x: x[-1], rollouts)

        costs = jnp.zeros((rollouts.returns.shape[0], self.horizon), dtype=jnp.float32)
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

    def init_params(self, seed: int = 0) -> TreeMPCParams:
        rng = jax.random.key(seed)
        actions = jnp.zeros(
            (self.horizon, self.task.u_min.shape[-1]), dtype=jnp.float32
        )
        return TreeMPCParams(actions=actions, rng=rng)  # type: ignore
