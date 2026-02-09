from functools import partial
from typing import Any, Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint as sac_checkpoint
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco_playground._src import mjx_env

from ss2r.algorithms.sac import networks as sac_networks
from ss2r.rl.utils import remove_pixels


class PolicyPrior:
    def __init__(
        self,
        *,
        env: mjx_env.MjxEnv,
        checkpoint_path: str | None = None,
        random_init: bool = False,
        seed: int = 0,
        normalize_observations: bool = True,
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 256, 256),
        value_hidden_layer_sizes: tuple[int, ...] = (512, 512),
        activation: str = "swish",
        use_bro: bool = True,
        n_critics: int = 2,
        n_heads: int = 1,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
    ) -> None:
        self._env = env
        obs_size = env.observation_size
        action_size = env.action_size
        preprocess_fn = (
            running_statistics.normalize if normalize_observations else (lambda x, y: x)
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
            n_critics=n_critics,
            n_heads=n_heads,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key,
        )
        if checkpoint_path is not None:
            params = sac_checkpoint.load(checkpoint_path)
            self._normalizer_params = params[0]
            self._policy_params = params[1]
        else:
            if not random_init:
                raise ValueError(
                    "PolicyPrior requires checkpoint_path or random_init=True."
                )
            rng = jax.random.PRNGKey(seed)
            rng, key_policy = jax.random.split(rng)
            self._policy_params = self._sac_network.policy_network.init(key_policy)
            if isinstance(obs_size, Mapping):
                obs_shape = {
                    k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
                }
            else:
                obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
            self._normalizer_params = running_statistics.init_state(
                remove_pixels(obs_shape)
            )

    def _sample_policy_action(self, obs: Any, key: jax.Array) -> jax.Array:
        dist_params = self._sac_network.policy_network.apply(
            self._normalizer_params,
            self._policy_params,
            obs,
        )
        dist = self._sac_network.parametric_action_distribution
        return dist.sample(dist_params, key)

    def prior_knots(
        self,
        state: mjx_env.State,
        controller: SamplingBasedController,
        key: jax.Array,
    ) -> jax.Array:
        def _scan_fn(carry, _):
            s, k = carry
            k, sub = jax.random.split(k)
            action = self._sample_policy_action(s.obs, sub)
            action = jnp.clip(action, controller.task.u_min, controller.task.u_max)
            next_state = self._env.step(s, action)
            return (next_state, k), action

        (_, _), actions = jax.lax.scan(
            _scan_fn, (state, key), (), length=controller.ctrl_steps
        )
        if controller.num_knots == 1:
            knot_idx = jnp.asarray([0], dtype=jnp.int32)
        else:
            knot_idx = jnp.round(
                jnp.linspace(0, controller.ctrl_steps - 1, controller.num_knots)
            ).astype(jnp.int32)
        return actions[knot_idx]


def wrap_controller_for_env_step(
    controller: SamplingBasedController, env: mjx_env.MjxEnv
) -> SamplingBasedController:
    """Override controller methods to step the environment State."""
    policy_prior_cfg = getattr(controller, "_policy_prior_cfg", None)
    if policy_prior_cfg is not None and not hasattr(controller, "_policy_prior"):
        controller._policy_prior = PolicyPrior(env=env, **policy_prior_cfg)

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def eval_rollouts(model, state_payload, controls, knots):
        del model
        state = state_payload

        def _scan_fn(x_state: mjx_env.State, u: jax.Array):
            x_state = env.step(x_state, u)
            cost = -x_state.reward
            sites = controller.task.get_trace_sites(x_state.data)
            return x_state, (x_state.data, cost, sites)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = controller.task.terminal_cost(final_state.data)
        final_trace_sites = controller.task.get_trace_sites(final_state.data)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def optimize(state_payload: mjx_env.State, params: Any):
        data = state_payload.data
        if not hasattr(data, "time"):
            raise ValueError("state.data has no time attribute.")
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, controller.plan_horizon, controller.num_knots) + data.time
        )
        new_mean = controller.interp_func(new_tk, tk, params.mean[None, ...])[0]
        policy_prior = getattr(controller, "_policy_prior", None)
        if policy_prior is not None:
            rng, prior_rng = jax.random.split(params.rng)
            prior_knots = policy_prior.prior_knots(state_payload, controller, prior_rng)
            params = params.replace(tk=new_tk, mean=prior_knots, rng=rng)
        else:
            params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, _: Any):
            knots, params = controller.sample_knots(params)
            knots = jnp.clip(knots, controller.task.u_min, controller.task.u_max)
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = controller.rollout_with_randomizations(
                state_payload, new_tk, knots, dr_rng
            )
            params = params.replace(rng=rng)
            params = controller.update_params(params, rollouts)
            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(controller.iterations)
        )
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final

    controller.eval_rollouts = eval_rollouts
    controller.optimize = optimize
    return controller
