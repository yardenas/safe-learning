from hydrax.algs import MPPI, PredictiveSampling
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
    cfg,
    task: MujocoPlaygroundTask,
    *,
    env: mjx_env.MjxEnv | None = None,
    policy_checkpoint_path: str | None = None,
) -> MPPI | TreeMPC | PredictiveSampling:
    del env
    controller_kwargs = dict(cfg.agent.get("controller_kwargs", {}))
    controller_name = cfg.agent.get("controller_name", "mppi")
    if controller_name in {"mppi", "predictive_sampling", "ps"}:
        if "horizon" in controller_kwargs and "plan_horizon" not in controller_kwargs:
            controller_kwargs["plan_horizon"] = (
                float(controller_kwargs["horizon"]) * task.dt
            )
        controller_kwargs.pop("horizon", None)
    if controller_name == "mppi":
        return MPPI(task, **controller_kwargs)
    if controller_name in {"predictive_sampling", "ps"}:
        allowed_keys = {
            "num_samples",
            "noise_level",
            "num_randomizations",
            "risk_strategy",
            "seed",
            "plan_horizon",
            "spline_type",
            "num_knots",
            "iterations",
        }
        controller_kwargs = {
            key: value
            for key, value in controller_kwargs.items()
            if key in allowed_keys
        }
        return PredictiveSampling(task, **controller_kwargs)
    if controller_name == "tree":
        allowed_keys = {
            "width",
            "branch",
            "horizon",
            "policy_checkpoint_path",
            "policy_noise_std",
            "td_lambda",
            "policy_action_only",
            "normalize_observations",
            "policy_hidden_layer_sizes",
            "value_hidden_layer_sizes",
            "activation",
            "n_critics",
            "n_heads",
            "use_bro",
            "policy_obs_key",
            "value_obs_key",
            "gamma",
            "temperature",
            "action_noise_std",
            "mode",
            "iterations",
        }
        controller_kwargs = {
            key: value
            for key, value in controller_kwargs.items()
            if key in allowed_keys
        }
        if policy_checkpoint_path is not None:
            controller_kwargs["policy_checkpoint_path"] = policy_checkpoint_path
        return TreeMPC(task, **controller_kwargs)
    raise ValueError(f"Unknown controller_name: {controller_name}")
