import jax
import jax.numpy as jnp
from brax import envs
from hydrax.alg_base import SamplingBasedController
from mujoco_playground._src import mjx_env
from omegaconf import DictConfig

from ss2r.algorithms.hydrax_mpc.controller import wrap_controller_for_env_step
from ss2r.algorithms.hydrax_mpc.factory import make_controller, make_task
from ss2r.rl.evaluation import Evaluator


def _make_policy(
    controller: SamplingBasedController,
    params: jax.Array,
    action_low: jax.Array | None,
    action_high: jax.Array | None,
    batch_size: int | None,
):
    batched_params = _tile_params(params, batch_size) if batch_size else None

    def _policy(state: mjx_env.State, rng: jax.Array):
        del rng
        if _is_batched(state):
            current_batch = state.data.qpos.shape[0]
            params_for_batch = (
                batched_params
                if batched_params is not None
                and getattr(batched_params, "shape", (0,))[0] == current_batch
                else _tile_params(params, current_batch)
            )
            actions, _, metrics = jax.vmap(
                lambda s, p: _optimize_and_action(controller, s, p)
            )(state, params_for_batch)
            action = _clip_action(actions, action_low, action_high)
        else:
            action, _, metrics = _optimize_and_action(controller, state, params)
            action = _clip_action(action, action_low, action_high)
        metrics = jax.tree.map(jnp.mean, metrics)
        return action, metrics

    return _policy


def _is_batched(state: mjx_env.State) -> bool:
    data = state.data
    return hasattr(data, "qpos") and getattr(data.qpos, "ndim", 0) >= 2


def _tile_params(params: jax.Array, batch_size: int) -> jax.Array:
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape),
        params,
    )


def _optimize_and_action(
    controller: SamplingBasedController, state: mjx_env.State, params: jax.Array
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    params_out, rollouts = controller.optimize(state, params)
    action = _action_from_params(controller, params_out, state)
    total_cost = jnp.sum(rollouts.costs, axis=-1)
    mean_cost = jnp.mean(total_cost)
    min_cost = jnp.min(total_cost)
    max_cost = jnp.max(total_cost)
    metrics = {
        "planner/rollout_cost_mean": mean_cost,
        "planner/rollout_cost_min": min_cost,
        "planner/rollout_cost_max": max_cost,
        "planner/estimated_return": -min_cost,
        "planner/best_vs_mean_cost": min_cost - mean_cost,
    }
    return action, params_out, metrics


# TODO (yarden): check that this function is how the hydrax people also do.
def _action_from_params(
    controller: SamplingBasedController, params: jax.Array, state: mjx_env.State
) -> jax.Array:
    if not hasattr(params, "tk") or not hasattr(params, "mean"):
        raise ValueError("Hydrax params must expose tk and mean for interpolation.")
    if not hasattr(state.data, "time"):
        raise ValueError("state.data has no time attribute.")
    # Be careful: this is wrong, hydrax originally plans over continuous-time systems
    # with knots that coincide with the planning frequency. Below we do something different.
    t_curr = state.data.time
    sim_dt = controller.task.dt
    steps = controller.ctrl_steps
    tq = jnp.arange(0, steps) * sim_dt + t_curr
    tk = params.tk
    knots = params.mean[None, ...]  # ty:ignore[not-subscriptable]
    us = controller.interp_func(tq, tk, knots)
    return _first_action(us)


def _first_action(us: jax.Array) -> jax.Array:
    if us.ndim == 3:
        return us[0, 0]
    return us[0]


def _clip_action(
    action: jax.Array, low: jax.Array | None, high: jax.Array | None
) -> jax.Array:
    if low is None and high is None:
        return action
    low_arr = None if low is None else jnp.asarray(low)
    high_arr = None if high is None else jnp.asarray(high)
    if low_arr is None and high_arr is not None:
        return jnp.minimum(action, high_arr)
    if high_arr is None and low_arr is not None:
        return jnp.maximum(action, low_arr)
    return jnp.clip(action, low_arr, high_arr)


def train(
    environment: envs.Env,
    eval_env: envs.Env,
    progress_fn,
    *,
    episode_length: int,
    seed: int,
    cfg: DictConfig,
):
    task = make_task(cfg, environment)  # ty:ignore[invalid-argument-type]
    controller = make_controller(
        cfg, task, env=environment
    )  # ty:ignore[invalid-argument-type]
    if not getattr(controller, "uses_env_step", False):
        controller = wrap_controller_for_env_step(controller, environment)
    action_low = cfg.agent.get("action_low", None)
    action_high = cfg.agent.get("action_high", None)

    key = jax.random.PRNGKey(seed)
    key, eval_key = jax.random.split(key)
    params = controller.init_params()
    if cfg.agent.get("debug_random_policy", False):
        action_size = environment.action_size
        low = -1.0 if action_low is None else action_low
        high = 1.0 if action_high is None else action_high

        def make_policy(state: mjx_env.State, rng: jax.Array):
            if _is_batched(state):
                batch_size = state.data.qpos.shape[0]
                keys = jax.random.split(rng, batch_size)
                actions = jax.vmap(
                    lambda k: jax.random.uniform(
                        k, (action_size,), minval=low, maxval=high
                    )
                )(keys)
            else:
                actions = jax.random.uniform(
                    rng, (action_size,), minval=low, maxval=high
                )
            return actions, {}
    else:
        make_policy = _make_policy(
            controller,
            params,
            action_low,
            action_high,
            cfg.training.num_eval_envs,
        )

    evaluator = Evaluator(
        eval_env,
        lambda _: make_policy,
        num_eval_envs=cfg.training.num_eval_envs,
        episode_length=episode_length,
        action_repeat=cfg.training.action_repeat,
        key=eval_key,
    )

    metrics = evaluator.run_evaluation(params, training_metrics={})
    progress_fn(0, metrics)
    return make_policy, params, metrics
