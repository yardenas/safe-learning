"""H1 mocap tracking adapter backed by vendored loco_mujoco MJX env."""

from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from flax import struct
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.environments  # noqa: F401
from ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.core.mujoco_mjx import (
    MjxState,
)
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


@struct.dataclass
class H1DataSeed:
    time: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    act: jax.Array
    qacc_warmstart: jax.Array
    ctrl: jax.Array
    qfrc_applied: jax.Array
    xfrc_applied: jax.Array
    eq_active: jax.Array
    mocap_pos: jax.Array
    mocap_quat: jax.Array
    act_dot: jax.Array
    userdata: jax.Array


@struct.dataclass
class H1PlannerState:
    data: H1DataSeed
    observation: jax.Array
    reward: jax.Array
    absorbing: jax.Array
    done: jax.Array
    additional_carry: Any
    info: Dict[str, Any]
    truncation: jax.Array


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

    def _compress_data_seed(self, data: mjx.Data) -> H1DataSeed:
        return H1DataSeed(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            qacc_warmstart=data.qacc_warmstart,
            ctrl=data.ctrl,
            qfrc_applied=data.qfrc_applied,
            xfrc_applied=data.xfrc_applied,
            eq_active=data.eq_active,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            act_dot=data.act_dot,
            userdata=data.userdata,
        )

    def _restore_data_seed(self, seed: H1DataSeed) -> mjx.Data:
        # Rebuild large derived MJX caches from the compact simulation state and
        # then restore the few stateful fields that `mjx.forward` does not keep.
        data = mjx_env.init(
            self._mjx_model,
            qpos=seed.qpos,
            qvel=seed.qvel,
            ctrl=seed.ctrl,
            act=seed.act,
            mocap_pos=seed.mocap_pos if self._mjx_model.nmocap else None,
            mocap_quat=seed.mocap_quat if self._mjx_model.nmocap else None,
        )
        return data.replace(
            time=seed.time,
            qacc_warmstart=seed.qacc_warmstart,
            qfrc_applied=seed.qfrc_applied,
            xfrc_applied=seed.xfrc_applied,
            eq_active=seed.eq_active,
            act_dot=seed.act_dot,
            userdata=seed.userdata,
        )

    def compress_planner_state(self, state: Any) -> Any:
        loco_state = getattr(state, "info", {}).get("_loco_state")
        if loco_state is None:
            return state
        truncation = state.info.get("truncation", jp.zeros_like(state.done))
        return H1PlannerState(
            data=self._compress_data_seed(loco_state.data),
            observation=loco_state.observation,
            reward=jp.asarray(loco_state.reward),
            absorbing=jp.asarray(loco_state.absorbing, dtype=bool),
            done=jp.asarray(loco_state.done, dtype=bool),
            additional_carry=loco_state.additional_carry,
            info=dict(loco_state.info),
            truncation=jp.asarray(truncation, dtype=jp.float32),
        )

    def restore_planner_state(self, planner_state: Any) -> mjx_env.State:
        if isinstance(planner_state, mjx_env.State):
            return planner_state
        if hasattr(planner_state, "loco_state"):
            # Backward compatibility for planner states stored before compact
            # `mjx.Data` reconstruction was introduced.
            loco_state = planner_state.loco_state
            truncation = getattr(
                planner_state,
                "truncation",
                jp.zeros_like(loco_state.done, dtype=jp.float32),
            )
        elif isinstance(planner_state, H1PlannerState):
            loco_state = MjxState(
                data=self._restore_data_seed(planner_state.data),
                observation=planner_state.observation,
                reward=planner_state.reward,
                absorbing=planner_state.absorbing,
                done=planner_state.done,
                additional_carry=planner_state.additional_carry,
                info=dict(planner_state.info),
            )
            truncation = planner_state.truncation
        else:
            loco_state = planner_state
            truncation = jp.zeros_like(loco_state.done, dtype=jp.float32)

        state = self._to_playground_state(loco_state)
        info = dict(state.info)
        info["truncation"] = jp.asarray(truncation, dtype=jp.float32)
        return state.replace(info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        loco_state = state.info.get("_loco_state")
        if loco_state is None:
            raise ValueError("Missing loco state in state.info['_loco_state']")
        next_loco_state = self._loco_env.mjx_step(loco_state, action)
        nan_terminated = (
            jp.isnan(next_loco_state.data.qpos).any()
            | jp.isnan(next_loco_state.data.qvel).any()
        )
        next_loco_state = next_loco_state.replace(
            done=jp.logical_or(next_loco_state.done, nan_terminated)
        )
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
