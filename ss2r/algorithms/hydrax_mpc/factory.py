from hydrax.algs import MPPI
from mujoco_playground._src import mjx_env

from ss2r.algorithms.hydrax_mpc.task import MujocoPlaygroundTask
from ss2r.algorithms.hydrax_mpc.tree_mpc import TreeMPC


def make_task(cfg, env: mjx_env.MjxEnv) -> MujocoPlaygroundTask:
    task_name = cfg.environment.task_name
    randomization_cfg = cfg.environment.train_params
    running_cost_scale = cfg.agent.get("running_cost_scale", 1.0)
    terminal_cost_scale = cfg.agent.get("terminal_cost_scale", 0.0)
    env_dt = env._ctrl_dt
    if env_dt is None:
        raise ValueError("Unable to infer controller dt from environment.")
    return MujocoPlaygroundTask(
        env,
        env_dt,
        task_name=task_name,
        randomization_cfg=randomization_cfg,
        running_cost_scale=running_cost_scale,
        terminal_cost_scale=terminal_cost_scale,
    )


def make_controller(
    cfg, task: MujocoPlaygroundTask, *, env: mjx_env.MjxEnv
) -> MPPI | TreeMPC:
    controller_kwargs = dict(cfg.agent.get("controller_kwargs", {}))
    controller_name = cfg.agent.get("controller_name", "mppi")
    if controller_name == "mppi":
        return MPPI(task, **controller_kwargs)
    if controller_name == "tree":
        return TreeMPC(task, **controller_kwargs)
    raise ValueError(f"Unknown controller_name: {controller_name}")
