import logging
import os
import time

import hydra
import jax
import jax.nn as jnn
import omegaconf
import wandb
from brax.envs.wrappers.training import EpisodeWrapper
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.types import Metrics, Policy

from ss2r.algorithms.sac.networks import make_inference_fn, make_sac_networks
from ss2r.benchmark_suites.rccar import hardware, rccar
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.benchmark_suites.wrappers import (
    ActionObservationDelayWrapper,
    FrameActionStack,
)
from ss2r.rl.evaluation import ConstraintEvalWrapper

_LOG = logging.getLogger(__name__)


def make_env(controller, cfg):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_car_params")
    task_cfg.pop("eval_car_params")
    dynamics = hardware.HardwareDynamics(
        controller=controller, max_throttle=task_cfg["max_throttle"]
    )
    action_delay, obs_delay = (
        task_cfg.pop("action_delay"),
        task_cfg.pop("observation_delay"),
    )
    sliding_window = task_cfg.pop("sliding_window")
    env = rccar.RCCar(train_car_params["nominal"], **task_cfg, hardware=dynamics)
    if action_delay > 0 or obs_delay > 0:
        env = ActionObservationDelayWrapper(
            env, action_delay=action_delay, obs_delay=obs_delay
        )
    if sliding_window > 0:
        env = FrameActionStack(env, num_stack=sliding_window)
    env = EpisodeWrapper(env, cfg.episode_length, cfg.action_repeat)
    env = ConstraintEvalWrapper(env)
    return env


def collect_trajectory(
    env: rccar.RCCar, controller, policy: Policy, rng: jax.Array
) -> Metrics:
    t = time.time()
    with hardware.start(controller):
        state = env.reset(rng)
        while not state.done:
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            state = env.step(state, action)
    epoch_eval_time = time.time() - t
    eval_metrics = state.info["eval_metrics"]
    metrics = {"eval/walltime": epoch_eval_time, **eval_metrics}
    return metrics


def fetch_policy(run_id):
    api = wandb.Api()
    run = api.run(f"ss2r/{run_id}")
    policy_artifact = api.artifact(f"ss2r/policy:{run_id}")
    policy_dir = policy_artifact.download()
    path = os.path.join(policy_dir, "policy.pkl")
    policy_params = model.load_params(path)
    config = run.config
    activation = getattr(jnn, config["agent"]["activation"])
    if config["agent"]["normalize_observations"]:
        normalize = running_statistics.normalize
    else:
        normalize = lambda x, y: x
    sac_network = make_sac_networks(
        observation_size=7,
        action_size=2,
        preprocess_observations_fn=normalize,
        hidden_layer_sizes=config["agent"]["hidden_layer_sizes"],
        activation=activation,
    )
    make_policy = make_inference_fn(sac_network)
    return make_policy(policy_params, True)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="rccar_hardware")
def main(cfg):
    traj_count = 0
    rng = jax.random.PRNGKey(cfg.seed)
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project="ss2r", resume=True, config=config_dict, **cfg.wandb
    ), hardware.connect(
        car_id=cfg.car_id,
        port_number=cfg.port_number,
        control_frequency=cfg.control_frequency,
    ) as controller, jax.disable_jit():
        policy_fn = fetch_policy(cfg.policy_id)
        env = make_env(controller, cfg)
        while traj_count < cfg.num_trajectories:
            answer = input("Press Y/y when ready to collect trajectory\n")
            if not (answer == "Y" or answer == "y"):
                _LOG.info("Skipping trajectory")
                continue
            _LOG.info(f"Collecting trajectory {traj_count}")
            rng, key = jax.random.split(rng)
            metrics = collect_trajectory(env, controller, policy_fn, key)
            print(metrics)
            _LOG.info(f"Done trajectory: {traj_count}")
            traj_count += 1


if __name__ == "__main__":
    main()
