from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper
from brax.envs.wrappers import training as brax_training

from ss2r.algorithms.hydrax_mpc.controller import wrap_controller_with_env
from ss2r.algorithms.hydrax_mpc.factory import make_controller, make_task
from ss2r.rl.evaluation import ConstraintsEvaluator


class _MjxDataObsWrapper(Wrapper):
    def reset(self, rng):
        state = self.env.reset(rng)
        if not hasattr(state, "data"):
            raise ValueError("Expected mjx environment state with `data`.")
        return state.replace(
            obs={
                "data": state.data,
                "info": state.info,
                "metrics": state.metrics,
                "done": state.done,
                "reward": state.reward,
                "obs": state.obs,
            }
        )

    def step(self, state, action):
        nstate = self.env.step(state, action)
        if not hasattr(nstate, "data"):
            raise ValueError("Expected mjx environment state with `data`.")
        return nstate.replace(
            obs={
                "data": nstate.data,
                "info": nstate.info,
                "metrics": nstate.metrics,
                "done": nstate.done,
                "reward": nstate.reward,
                "obs": nstate.obs,
            }
        )


def _make_policy(controller, params, action_low, action_high, batch_size: int | None):
    batched_params = _tile_params(params, batch_size) if batch_size else None

    def _policy(obs, rng):
        del rng
        data = obs
        if _is_batched(data):
            actions, _ = jax.vmap(lambda d, p: _optimize_and_action(controller, d, p))(
                data, batched_params
            )
            action = _clip_action(actions, action_low, action_high)
        else:
            action, _ = _optimize_and_action(controller, data, params)
            action = _clip_action(action, action_low, action_high)
        return action, {}

    return _policy


def _is_batched(data: Any) -> bool:
    if isinstance(data, dict):
        data = data.get("data")
    return hasattr(data, "qpos") and getattr(data.qpos, "ndim", 0) >= 2


def _tile_params(params: Any, batch_size: int) -> Any:
    return jax.tree_map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape),
        params,
    )


def _optimize_and_action(controller: Any, data: Any, params: Any) -> tuple[Any, Any]:
    params_out, _ = controller.optimize(data, params)
    action = _action_from_params(controller, params_out, data)
    return action, params_out


def _action_from_params(controller: Any, params: Any, data: Any) -> Any:
    if not hasattr(params, "tk") or not hasattr(params, "mean"):
        raise ValueError("Hydrax params must expose tk and mean for interpolation.")
    if isinstance(data, dict):
        data = data.get("data")
    t_curr = getattr(data, "time", 0.0)
    tq = jnp.atleast_1d(jnp.asarray(t_curr))
    tk = params.tk
    knots = params.mean[None, ...]
    us = controller.interp_func(tq, tk, knots)
    return _first_action(us)


def _first_action(us: Any) -> Any:
    if us.ndim == 3:
        return us[0, 0]
    return us[0]


def _clip_action(action: Any, low: Any, high: Any) -> Any:
    if low is None and high is None:
        return action
    low_arr = None if low is None else jnp.asarray(low)
    high_arr = None if high is None else jnp.asarray(high)
    if low_arr is None:
        return jnp.minimum(action, high_arr)
    if high_arr is None:
        return jnp.maximum(action, low_arr)
    return jnp.clip(action, low_arr, high_arr)


def train(
    environment: Any,
    eval_env: Any,
    progress_fn,
    *,
    episode_length: int,
    seed: int,
    cfg: Any,
) -> tuple[Any, Any, dict[str, float]]:
    if not eval_env:
        eval_env = environment
    eval_env = brax_training.VmapWrapper(eval_env)
    eval_env = _MjxDataObsWrapper(eval_env)

    task = make_task(cfg, environment)
    controller = make_controller(cfg, task, env=environment)
    template_state = environment.reset(jax.random.PRNGKey(seed))
    controller = wrap_controller_with_env(controller, environment, template_state)
    action_low = cfg.agent.get("action_low", None)
    action_high = cfg.agent.get("action_high", None)

    key = jax.random.PRNGKey(seed)
    key, eval_key = jax.random.split(key)
    params = controller.init_params()
    make_policy = _make_policy(
        controller,
        params,
        action_low,
        action_high,
        cfg.training.num_eval_envs,
    )

    evaluator = ConstraintsEvaluator(
        eval_env,
        lambda _: make_policy,
        num_eval_envs=cfg.training.num_eval_envs,
        episode_length=episode_length,
        action_repeat=cfg.training.action_repeat,
        key=eval_key,
        budget=cfg.training.safety_budget,
        num_episodes=cfg.training.num_eval_episodes,
    )

    metrics = evaluator.run_evaluation(
        None,
        training_metrics={},
    )
    progress_fn(0, metrics)
    return make_policy, None, metrics
