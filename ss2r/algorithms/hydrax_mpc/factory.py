from hydrax.algs import MPPI
from mujoco_playground._src import mjx_env

from ss2r.algorithms.hydrax_mpc.task import MujocoPlaygroundTask
from ss2r.algorithms.hydrax_mpc.tree_mpc import TreeMPC


def make_task(env: mjx_env.MjxEnv) -> MujocoPlaygroundTask:
    env_dt = env._ctrl_dt
    if env_dt is None:
        raise ValueError("Unable to infer controller dt from environment.")
    return MujocoPlaygroundTask(
        env,
        env_dt,
    )


def make_controller(
    cfg, task: MujocoPlaygroundTask, *, env: mjx_env.MjxEnv
) -> MPPI | TreeMPC:
    controller_kwargs = dict(cfg.agent.get("controller_kwargs", {}))
    controller_name = cfg.agent.get("controller_name", "mppi")
    if controller_name == "mppi":
        return MPPI(task, **controller_kwargs)
    if controller_name == "tree":
        allowed_keys = {
            "width",
            "branch",
            "horizon",
            "gamma",
            "temperature",
            "action_noise_std",
            "mode",
            "plan_horizon",
            "iterations",
        }
        controller_kwargs = {
            key: value
            for key, value in controller_kwargs.items()
            if key in allowed_keys
        }
        return TreeMPC(task, **controller_kwargs)
    raise ValueError(f"Unknown controller_name: {controller_name}")
