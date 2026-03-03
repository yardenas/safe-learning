"""G1 mocap tracking adapter backed by vendored loco_mujoco MJX env."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco.environments  # noqa: F401
from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco.task_factories.imitation_factory import (
    ImitationFactory,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        loco=config_dict.create(
            dataset_name="dance1_subject3",
            reference_source="hf",
            reference_repo_id="robfiras/loco-mujoco-datasets",
            reference_repo_type="dataset",
            reference_dir="Lafan1/mocap/UnitreeG1",
            control_type="DefaultControl",
            control_params=config_dict.create(),
            mimic_reward=config_dict.create(
                qpos_w_exp=10.0,
                qvel_w_exp=2.0,
                rpos_w_exp=100.0,
                rquat_w_exp=10.0,
                rvel_w_exp=0.1,
                qpos_w_sum=0.4,
                qvel_w_sum=0.2,
                rpos_w_sum=0.5,
                rquat_w_sum=0.3,
                rvel_w_sum=0.1,
                action_out_of_bounds_coeff=0.01,
                joint_acc_coeff=0.0,
                joint_torque_coeff=0.0,
                action_rate_coeff=0.0,
            ),
        ),
    )


def _lafan1_dataset_name(config: config_dict.ConfigDict) -> str:
    dataset_candidates = [
        _cfg_path_get(config, ("loco", "dataset_name"), None),
        _cfg_path_get(
            config,
            ("task_factory", "params", "lafan1_dataset_conf", "dataset_name"),
            None,
        ),
        _cfg_path_get(
            config,
            (
                "experiment",
                "task_factory",
                "params",
                "lafan1_dataset_conf",
                "dataset_name",
            ),
            None,
        ),
    ]

    raw_name: Any = "dance1_subject3"
    for candidate in dataset_candidates:
        if candidate is None:
            continue
        raw_name = candidate
        break

    if isinstance(raw_name, (list, tuple)):
        raw_name = raw_name[0] if raw_name else "dance1_subject3"
    name = str(raw_name).strip()
    if not name:
        name = "dance1_subject3"

    name = Path(name).name
    if name.endswith(".npz"):
        name = name[:-4]
    if name.endswith(".csv"):
        name = name[:-4]
    return name or "dance1_subject3"


def _cfg_path_get(config: Any, path: tuple[str, ...], default: Any) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        if hasattr(current, key):
            current = getattr(current, key)
            continue
        return default
    return current


def _episode_length(config: config_dict.ConfigDict) -> int:
    candidates = [
        _cfg_path_get(config, ("episode_length",), None),
        _cfg_path_get(config, ("horizon",), None),
        _cfg_path_get(config, ("env_params", "horizon"), None),
        _cfg_path_get(config, ("experiment", "env_params", "horizon"), None),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 1000


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _control_spec(config: config_dict.ConfigDict) -> tuple[str, dict[str, Any]]:
    control_type_candidates = [
        _cfg_path_get(config, ("loco", "control_type"), None),
        _cfg_path_get(config, ("control_type",), None),
        _cfg_path_get(config, ("env_params", "control_type"), None),
        _cfg_path_get(config, ("experiment", "env_params", "control_type"), None),
    ]
    control_type = next((c for c in control_type_candidates if c is not None), None)
    control_type_str = (
        str(control_type) if control_type is not None else "DefaultControl"
    )

    control_params_candidates = [
        _cfg_path_get(config, ("loco", "control_params"), None),
        _cfg_path_get(config, ("control_params",), None),
        _cfg_path_get(config, ("env_params", "control_params"), None),
        _cfg_path_get(config, ("experiment", "env_params", "control_params"), None),
    ]
    control_params_raw = next(
        (c for c in control_params_candidates if c is not None), {}
    )
    control_params = (
        _to_plain(control_params_raw) if isinstance(control_params_raw, Mapping) else {}
    )
    return control_type_str, control_params


def _mimic_reward_params(config: config_dict.ConfigDict) -> dict[str, Any]:
    defaults = {
        "qpos_w_exp": 10.0,
        "qvel_w_exp": 2.0,
        "rpos_w_exp": 100.0,
        "rquat_w_exp": 10.0,
        "rvel_w_exp": 0.1,
        "qpos_w_sum": 0.4,
        "qvel_w_sum": 0.2,
        "rpos_w_sum": 0.5,
        "rquat_w_sum": 0.3,
        "rvel_w_sum": 0.1,
        "action_out_of_bounds_coeff": 0.01,
        "joint_acc_coeff": 0.0,
        "joint_torque_coeff": 0.0,
        "action_rate_coeff": 0.0,
    }
    candidate_cfgs = [
        _cfg_path_get(config, ("loco", "mimic_reward"), None),
        _cfg_path_get(config, ("reward_params",), None),
        _cfg_path_get(config, ("env_params", "reward_params"), None),
        _cfg_path_get(config, ("experiment", "env_params", "reward_params"), None),
    ]
    reward_cfg = next((c for c in candidate_cfgs if c is not None), None)
    if reward_cfg is None:
        return defaults

    out: dict[str, Any] = {}
    for key, default_value in defaults.items():
        value = _cfg_path_get(reward_cfg, (key,), default_value)
        out[key] = float(value)
    sites_for_mimic = _cfg_path_get(reward_cfg, ("sites_for_mimic",), None)
    if sites_for_mimic is not None:
        out["sites_for_mimic"] = list(sites_for_mimic)
    return out


class G1MocapTracking(mjx_env.MjxEnv):
    """Adapter that presents loco_mujoco MjxUnitreeG1 as a playground MjxEnv."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        del task
        super().__init__(config, config_overrides)

        self._xml_path = str(
            Path(__file__).resolve().parent
            / "loco_mujoco"
            / "models"
            / "unitree_g1"
            / "g1_23dof.xml"
        )

        n_substeps = max(
            1, int(round(float(self._config.ctrl_dt / self._config.sim_dt)))
        )
        dataset_name = _lafan1_dataset_name(self._config)
        self._dataset_name = dataset_name
        mimic_reward_params = _mimic_reward_params(self._config)
        control_type, control_params = _control_spec(self._config)

        self._loco_env = ImitationFactory.make(
            "MjxUnitreeG1",
            lafan1_dataset_conf={"dataset_name": dataset_name},
            reward_type="MimicReward",
            reward_params=mimic_reward_params,
            control_type=control_type,
            control_params=control_params,
            timestep=float(self._config.sim_dt),
            n_substeps=n_substeps,
            horizon=_episode_length(self._config),
        )

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
