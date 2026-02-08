from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco_playground._src import mjx_env


def wrap_controller_for_env_step(
    controller: SamplingBasedController, env: mjx_env.MjxEnv
) -> SamplingBasedController:
    """Override controller methods to step the environment State."""

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
