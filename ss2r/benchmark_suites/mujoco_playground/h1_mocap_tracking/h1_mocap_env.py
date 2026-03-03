"""H1 mocap tracking adapter backed by vendored loco_mujoco MJX env."""

from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.environments  # noqa: F401
from ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.task_factories.imitation_factory import (
    ImitationFactory,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        env_name="MjxUnitreeH1",
        disable_arms=False,
        goal_type="GoalTrajMimic",
        goal_params=config_dict.create(visualize_goal=False),
        control_type="DefaultControl",
        control_params=config_dict.create(),
        reward_type="MimicReward",
        reward_params=config_dict.create(
            qpos_w_exp=10.0,
            qvel_w_exp=2.0,
            rpos_w_exp=100.0,
            rquat_w_exp=10.0,
            rvel_w_exp=0.1,
            qpos_w_sum=0.0,
            qvel_w_sum=0.0,
            rpos_w_sum=0.5,
            rquat_w_sum=0.3,
            rvel_w_sum=0.0,
            action_out_of_bounds_coeff=0.01,
            joint_acc_coeff=0.0,
            joint_torque_coeff=0.0,
            action_rate_coeff=0.0,
            sites_for_mimic=[
                "upper_body_mimic",
                "left_hand_mimic",
                "left_foot_mimic",
                "right_hand_mimic",
                "right_foot_mimic",
            ],
        ),
        loco=config_dict.create(
            dataset_name="walk1_subject5",
            reference_source="hf",
            reference_repo_id="robfiras/loco-mujoco-datasets",
            reference_repo_type="dataset",
            reference_dir="Lafan1/mocap/UnitreeH1",
        ),
    )


def _lafan1_dataset_name(config: config_dict.ConfigDict) -> str:
    raw_name: Any = config.loco.dataset_name
    if isinstance(raw_name, (list, tuple)):
        raw_name = raw_name[0] if raw_name else "walk1_subject5"
    name = str(raw_name).strip()
    if not name:
        name = "walk1_subject5"
    if name.endswith(".npz"):
        name = name[:-4]
    if name.endswith(".csv"):
        name = name[:-4]
    return name or "walk1_subject5"


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


class H1MocapTracking(mjx_env.MjxEnv):
    """Adapter that presents loco_mujoco MjxUnitreeH1 as a playground MjxEnv."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        del task
        super().__init__(config, config_overrides)

        n_substeps = max(
            1, int(round(float(self._config.ctrl_dt / self._config.sim_dt)))
        )
        dataset_name = _lafan1_dataset_name(self._config)
        self._dataset_name = dataset_name

        self._loco_env = ImitationFactory.make(
            self._config.env_name,
            lafan1_dataset_conf={"dataset_name": dataset_name},
            disable_arms=bool(self._config.disable_arms),
            goal_type=str(self._config.goal_type),
            goal_params=_to_plain(self._config.goal_params),
            reward_type=str(self._config.reward_type),
            reward_params=_to_plain(self._config.reward_params),
            control_type=str(self._config.control_type),
            control_params=_to_plain(self._config.control_params),
            timestep=float(self._config.sim_dt),
            n_substeps=n_substeps,
            horizon=int(self._config.episode_length),
        )

        self._xml_path = self._loco_env.__class__.get_default_xml_file_path()
        self._mj_model = self._loco_env.model
        self._mjx_model = self._loco_env.sys
        self._action_size = int(np.prod(self._loco_env.info.action_space.shape))

    def _to_playground_state(
        self,
        loco_state: Any,
        *,
        rng: jax.Array | None = None,
        previous_state: mjx_env.State | None = None,
    ) -> mjx_env.State:
        if previous_state is None:
            info = dict(loco_state.info)
            metrics = {}
        else:
            # Preserve wrapper-added carry structure for JAX scans.
            info = dict(previous_state.info)
            metrics = dict(previous_state.metrics)

        info["_loco_state"] = loco_state
        if rng is not None and "rng" not in info:
            info["rng"] = rng

        if previous_state is None:
            reward = jp.asarray(loco_state.reward)
            done = jp.asarray(loco_state.done, dtype=jp.float32)
        else:
            reward = jp.asarray(loco_state.reward, dtype=previous_state.reward.dtype)
            done = jp.asarray(loco_state.done, dtype=previous_state.done.dtype)

        return mjx_env.State(
            data=loco_state.data,
            obs=loco_state.observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        loco_state = self._loco_env.mjx_reset(rng)
        return self._to_playground_state(loco_state, rng=rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        loco_state = state.info.get("_loco_state")
        if loco_state is None:
            raise ValueError("Missing loco state in state.info['_loco_state']")
        next_loco_state = self._loco_env.mjx_step(loco_state, action)
        return self._to_playground_state(next_loco_state, previous_state=state)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def sample_command(self, rng: jax.Array) -> jax.Array:
        del rng
        return jp.zeros((3,))
