from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from hydrax.alg_base import Trajectory


def wrap_controller_with_env(controller: Any, env: Any, template_state: Any) -> Any:
    """Override controller.eval_rollouts to step the environment."""

    def _state_from_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return template_state.replace(
                data=payload.get("data", template_state.data),
                obs=payload.get("obs", template_state.obs),
                reward=payload.get("reward", template_state.reward),
                done=payload.get("done", template_state.done),
                metrics=payload.get("metrics", template_state.metrics),
                info=payload.get("info", template_state.info),
            )
        return template_state.replace(data=payload)

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def eval_rollouts(model, state_payload, controls, knots):
        del model
        state = _state_from_payload(state_payload)

        def _scan_fn(x_state, u):
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

    controller.eval_rollouts = eval_rollouts
    return controller
