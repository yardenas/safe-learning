"""G1 mocap tracking adapter backed by vendored loco_mujoco MJX env."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco as _vendored_loco_mujoco  # noqa: F401
from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco.task_factories.imitation_factory import ImitationFactory


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
        ),
    )


def _lafan1_dataset_name(config: config_dict.ConfigDict) -> str:
    loco_cfg = getattr(config, "loco", None)
    if loco_cfg is None:
        return "dance1_subject3"

    name = str(getattr(loco_cfg, "dataset_name", "dance1_subject3")).strip()
    if not name:
        return "dance1_subject3"

    name = Path(name).name
    if name.endswith(".npz"):
        name = name[:-4]
    if name.endswith(".csv"):
        name = name[:-4]
    return name or "dance1_subject3"


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

        self._loco_env = ImitationFactory.make(
            "MjxUnitreeG1",
            lafan1_dataset_conf={"dataset_name": dataset_name},
            timestep=float(self._config.sim_dt),
            n_substeps=n_substeps,
            horizon=int(self._config.episode_length),
        )

        self._mj_model = self._loco_env.model
        self._mjx_model = self._loco_env.sys
        self._action_size = int(np.prod(self._loco_env.info.action_space.shape))

    def _to_playground_state(
        self,
        loco_state: Any,
        *,
        rng: jax.Array | None = None,
    ) -> mjx_env.State:
        info = dict(loco_state.info)
        info["_loco_state"] = loco_state
        if rng is not None and "rng" not in info:
            info["rng"] = rng

        reward = jp.asarray(loco_state.reward)
        done = jp.asarray(loco_state.done)

        return mjx_env.State(
            data=loco_state.data,
            obs=loco_state.observation,
            reward=reward,
            done=done,
            metrics={},
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
        return self._to_playground_state(next_loco_state)

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
