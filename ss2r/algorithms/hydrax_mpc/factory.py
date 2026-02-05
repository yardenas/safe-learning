from typing import Any

from hydrax.algs import MPPI

from ss2r.algorithms.hydrax_mpc.task import MujocoPlaygroundTask


def get_mjx_model(env: Any) -> Any:
    if hasattr(env, "mjx_model"):
        return env.mjx_model
    if hasattr(env, "_mjx_model"):
        return env._mjx_model
    raise ValueError(
        "Environment does not expose an MJX model. "
        "Provide a mjx.Model explicitly or use a mujoco_playground/safety_gym env."
    )


def make_task(cfg, env: Any) -> Any:
    task_name = cfg.environment.task_name
    randomization_cfg = cfg.environment.train_params
    running_cost_scale = cfg.agent.get("running_cost_scale", 1.0)
    terminal_cost_scale = cfg.agent.get("terminal_cost_scale", 0.0)
    return MujocoPlaygroundTask(
        env,
        task_name=task_name,
        randomization_cfg=randomization_cfg,
        running_cost_scale=running_cost_scale,
        terminal_cost_scale=terminal_cost_scale,
    )


def make_controller(cfg, task: Any, *, env: Any | None = None) -> Any:
    controller_kwargs = dict(cfg.agent.get("controller_kwargs", {}))
    if "dt" not in controller_kwargs and env is not None:
        env_dt = _get_env_ctrl_dt(env)
        if env_dt is None:
            raise ValueError("Unable to infer controller dt from environment.")
        if hasattr(task, "dt"):
            task.dt = env_dt
    return MPPI(task, **controller_kwargs)


def _get_env_ctrl_dt(env: Any) -> float | None:
    for attr in ("ctrl_dt", "dt"):
        value = getattr(env, attr, None)
        if value is not None:
            return float(value)
    cfg = getattr(env, "_config", None)
    if cfg is not None and getattr(cfg, "ctrl_dt", None) is not None:
        return float(cfg.ctrl_dt)
    return None
