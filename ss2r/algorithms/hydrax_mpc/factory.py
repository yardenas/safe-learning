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
        policy_prior_cfg = {}
        policy_prior_keys = {
            "policy_prior_random_init",
            "normalize_observations",
            "policy_hidden_layer_sizes",
            "value_hidden_layer_sizes",
            "activation",
            "use_bro",
            "n_critics",
            "n_heads",
            "policy_obs_key",
            "value_obs_key",
        }
        for key in list(controller_kwargs.keys()):
            if key in policy_prior_keys:
                policy_prior_cfg[key] = controller_kwargs.pop(key)
        controller = MPPI(task, **controller_kwargs)
        policy_prior_random_init = bool(
            policy_prior_cfg.get("policy_prior_random_init", False)
        )
        policy_prior_seed = int(cfg.training.seed)
        if policy_checkpoint_path is not None or policy_prior_random_init:
            policy_prior_cfg = {
                "checkpoint_path": policy_checkpoint_path,
                "random_init": policy_prior_random_init,
                "seed": policy_prior_seed,
                "normalize_observations": policy_prior_cfg.get(
                    "normalize_observations", True
                ),
                "policy_hidden_layer_sizes": policy_prior_cfg.get(
                    "policy_hidden_layer_sizes", (256, 256, 256)
                ),
                "value_hidden_layer_sizes": policy_prior_cfg.get(
                    "value_hidden_layer_sizes", (512, 512)
                ),
                "activation": policy_prior_cfg.get("activation", "swish"),
                "use_bro": policy_prior_cfg.get("use_bro", True),
                "n_critics": policy_prior_cfg.get("n_critics", 2),
                "n_heads": policy_prior_cfg.get("n_heads", 1),
                "policy_obs_key": policy_prior_cfg.get("policy_obs_key", "state"),
                "value_obs_key": policy_prior_cfg.get("value_obs_key", "state"),
            }
            controller._policy_prior_cfg = policy_prior_cfg
        return controller
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
            "num_samples",
            "horizon",
            "gae_lambda",
            "use_policy",
            "use_critic",
            "n_critics",
            "n_heads",
            "use_bro",
            "zoh_steps",
            "gamma",
            "temperature",
            "action_noise_std",
            "iterations",
        }
        controller_kwargs = {
            key: value
            for key, value in controller_kwargs.items()
            if key in allowed_keys
        }
        return TreeMPC(task, **controller_kwargs)
    raise ValueError(f"Unknown controller_name: {controller_name}")
