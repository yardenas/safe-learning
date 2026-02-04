from importlib import import_module
from typing import Any

from ss2r.algorithms.hydrax_mpc.task import (
    MujocoPlaygroundDomainRandomizedTask,
    MujocoPlaygroundTask,
)


def _import_from_path(target: str) -> Any:
    module_path, attr = target.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)


def _instantiate(target: str, *args: Any, **kwargs: Any) -> Any:
    cls = _import_from_path(target)
    return cls(*args, **kwargs)


def _resolve_controller_target(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip().lower()
    mapping = {
        "predictivesampling": "hydrax.algs.PredictiveSampling",
        "predictive_sampling": "hydrax.algs.PredictiveSampling",
        "ps": "hydrax.algs.PredictiveSampling",
        "mppi": "hydrax.algs.MPPI",
        "cem": "hydrax.algs.CEM",
    }
    return mapping.get(normalized, name)


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
    use_randomization = bool(cfg.agent.get("domain_randomization", False))
    task_cls = (
        MujocoPlaygroundDomainRandomizedTask
        if use_randomization
        else MujocoPlaygroundTask
    )
    return task_cls(
        env,
        task_name=task_name,
        randomization_cfg=randomization_cfg,
        running_cost_scale=running_cost_scale,
        terminal_cost_scale=terminal_cost_scale,
    )


def make_risk_strategy(cfg) -> Any | None:
    risk_target = cfg.agent.get("risk_strategy_target", None)
    if not risk_target:
        return None
    risk_kwargs = dict(cfg.agent.get("risk_strategy_kwargs", {}))
    return _instantiate(risk_target, **risk_kwargs)


def make_controller(cfg, task: Any) -> Any:
    controller_target = cfg.agent.get("controller_target", None)
    if controller_target is None:
        controller_target = _resolve_controller_target(
            cfg.agent.get("controller_name", None)
        )
    if not controller_target:
        raise ValueError(
            "hydrax_mpc requires cfg.agent.controller_target or controller_name."
        )
    controller_kwargs = dict(cfg.agent.get("controller_kwargs", {}))
    risk_strategy = make_risk_strategy(cfg)
    if risk_strategy is not None and "risk_strategy" not in controller_kwargs:
        controller_kwargs["risk_strategy"] = risk_strategy
    return _instantiate(controller_target, task, **controller_kwargs)
