from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

ROOT_PATH = epath.Path(__file__).parent / "assets"
SCENE_XML = ROOT_PATH / "xmls" / "scene_23dof.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                configuration=-1.0,
                foot_position=-5.0,
                foot_orientation=-0.1,
                control=-1.0,
                action_rate=0.0,
                alive=0.0,
                termination=-100.0,
            ),
        ),
        reference_path="ss2r/benchmark_suites/mujoco_playground/g1_mocap_tracking/walk1_subject1.npz",
        random_start=True,
        loop_reference=True,
        reset_noise_scale=0.01,
        termination_height=0.45,
        termination_upvector_z=0.0,
    )


def get_assets() -> Dict[str, bytes]:
    assets: Dict[str, bytes] = {}
    mjx_env.update_assets(assets, ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, ROOT_PATH / "xmls" / "assets")
    path = mjx_env.MENAGERIE_PATH / "unitree_g1"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class G1MocapTracking(mjx_env.MjxEnv):
    """G1 mocap tracking task with direct actuator control."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            SCENE_XML.read_text(),
            assets=self._model_assets,
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = SCENE_XML.as_posix()
        self._post_init()

    def _resolve_reference_path(self, reference_path: str) -> Path:
        if not reference_path:
            raise ValueError(
                "task_params.reference_path must point to a local mocap .npz file."
            )

        raw = Path(reference_path)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            repo_root = Path(__file__).resolve().parents[4]
            candidates.extend(
                [
                    Path.cwd() / raw,
                    repo_root / raw,
                    Path(ROOT_PATH) / raw,
                ]
            )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        tried = "\n".join(str(c) for c in candidates)
        raise ValueError(
            "Could not find local mocap reference file. Set task_params.reference_path "
            f"to an existing .npz. Tried:\n{tried}"
        )

    def _post_init(self) -> None:
        self._nu = int(self._mj_model.nu)
        self._init_q = jp.array(self._mj_model.qpos0)
        self._default_pose = jp.array(self._mj_model.qpos0[7 : 7 + self._nu])

        self._feet_site_id = np.array(
            [
                self._mj_model.site("left_foot").id,
                self._mj_model.site("right_foot").id,
            ]
        )

        reference_path = self._resolve_reference_path(str(self._config.reference_path))
        npz_file = np.load(reference_path, allow_pickle=False)
        reference = np.asarray(npz_file["qpos"], dtype=np.float32)
        if reference.shape[1] != self._mj_model.nq:  # type: ignore
            raise ValueError(
                "Reference qpos width does not match model nq: "
                f"reference={reference.shape[1]}, nq={self._mj_model.nq}."  # type: ignore
            )

        self._reference = jp.array(reference)
        self._reference_fps = float(np.asarray(npz_file["frequency"]).item())

        (
            ref_left_pos,
            ref_left_upvector,
            ref_right_pos,
            ref_right_upvector,
        ) = self._precompute_reference_feet(reference)
        self._ref_left_pos = jp.array(ref_left_pos)
        self._ref_left_upvector = jp.array(ref_left_upvector)
        self._ref_right_pos = jp.array(ref_right_pos)
        self._ref_right_upvector = jp.array(ref_right_upvector)

        cost_weights = np.ones(self._mj_model.nq, dtype=np.float32)
        cost_weights[:7] = 5.0
        self._cost_weights = jp.array(cost_weights)

    def _precompute_reference_feet(
        self, reference: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mj_data = mujoco.MjData(self._mj_model)
        n_frames = reference.shape[0]  # type: ignore
        left_foot_site_id = self._mj_model.site("left_foot").id
        right_foot_site_id = self._mj_model.site("right_foot").id

        ref_left_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_left_upvector = np.zeros((n_frames, 3), dtype=np.float32)
        ref_right_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_right_upvector = np.zeros((n_frames, 3), dtype=np.float32)

        for i in range(n_frames):
            mj_data.qpos[:] = reference[i]
            mujoco.mj_forward(self._mj_model, mj_data)
            ref_left_pos[i] = mj_data.site_xpos[left_foot_site_id]
            ref_right_pos[i] = mj_data.site_xpos[right_foot_site_id]
            left_rot = np.asarray(
                mj_data.site_xmat[left_foot_site_id], dtype=np.float32
            ).reshape(3, 3)
            right_rot = np.asarray(
                mj_data.site_xmat[right_foot_site_id], dtype=np.float32
            ).reshape(3, 3)
            ref_left_upvector[i] = left_rot[:, 2]
            ref_right_upvector[i] = right_rot[:, 2]

        return ref_left_pos, ref_left_upvector, ref_right_pos, ref_right_upvector

    def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)

    def _reference_index(self, t: jax.Array, start_idx: jax.Array) -> jax.Array:
        idx = jp.int32(t * self._reference_fps) + jp.int32(start_idx)
        if self._config.loop_reference:
            return jp.mod(idx, self._reference.shape[0])
        return jp.clip(idx, 0, self._reference.shape[0] - 1)

    def _phase_from_index(self, idx: jax.Array) -> jax.Array:
        theta = (
            2.0
            * jp.pi
            * jp.asarray(idx, dtype=jp.float32)
            / jp.asarray(self._reference.shape[0], dtype=jp.float32)
        )
        return jp.array([theta, theta + jp.pi], dtype=jp.float32)

    def _get_upvector(self, data: mjx.Data, frame: str) -> jax.Array:
        return self._get_sensor_data(data, f"upvector_{frame}")

    def _get_projected_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
        return -self._get_upvector(data, frame)

    def _get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
        return self._get_sensor_data(data, f"local_linvel_{frame}")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, start_rng = jax.random.split(rng)
        reference_start_idx = jax.lax.cond(
            self._config.random_start,
            lambda _: jax.random.randint(start_rng, (), 0, self._reference.shape[0]),
            lambda _: jp.zeros((), dtype=jp.int32),
            operand=None,
        )

        q_ref = self._reference[reference_start_idx]
        qpos = q_ref
        qvel = jp.zeros(self.mjx_model.nv)

        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_noise = self._config.reset_noise_scale * jax.random.normal(
            q_rng, (self._mj_model.nq - 7,)
        )
        v_noise = self._config.reset_noise_scale * jax.random.normal(
            v_rng, (self._mj_model.nv,)
        )
        qpos = qpos.at[7:].set(qpos[7:] + q_noise)
        qvel = qvel + v_noise

        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=q_ref[7 : 7 + self._nu],
        )

        info = {
            "rng": rng,
            "reference_start_idx": reference_start_idx,
            "phase": self._phase_from_index(reference_start_idx),
            "last_act": jp.zeros(self._nu),
            "motor_targets": jp.zeros(self._nu),
        }

        metrics = {
            f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.asarray(action)
        motor_targets = self._default_pose + action * self._config.action_scale

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            motor_targets,
            self.n_substeps,
        )

        feet_pos = data.site_xpos[self._feet_site_id]

        done = self._get_termination(data)
        ref_idx_next = self._reference_index(
            data.time,
            state.info["reference_start_idx"],
        )

        rewards = self._get_reward(
            data,
            action,
            state.info["last_act"],
            motor_targets,
            done,
            ref_idx_next,
            feet_pos,
        )
        rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        state.info["last_act"] = action
        state.info["motor_targets"] = motor_targets
        state.info["phase"] = self._phase_from_index(ref_idx_next)

        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value

        obs = self._get_obs(data, state.info)
        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        # Temporarily disable task terminations while tuning/debugging.
        return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> jax.Array:
        gyro = self._get_sensor_data(data, "gyro_pelvis")
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = self._get_projected_gravity(data, "pelvis")
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7 : 7 + self._nu]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6 : 6 + self._nu]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self._get_local_linvel(data, "pelvis")
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

        state_vec = jp.hstack(
            [
                noisy_linvel,
                noisy_gyro,
                noisy_gravity,
                noisy_joint_angles - self._default_pose,
                noisy_joint_vel,
                info["last_act"],
                phase,
            ]
        )

        return state_vec

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        last_act: jax.Array,
        motor_targets: jax.Array,
        done: jax.Array,
        ref_idx: jax.Array,
        feet_pos: jax.Array,
    ) -> dict[str, jax.Array]:
        q_ref = self._reference[ref_idx]
        q_err = self._cost_weights * (data.qpos - q_ref)
        configuration_cost = jp.sum(jp.square(q_err))

        left_pos_err = feet_pos[0] - self._ref_left_pos[ref_idx]
        right_pos_err = feet_pos[1] - self._ref_right_pos[ref_idx]
        foot_position_cost = jp.sum(jp.square(left_pos_err)) + jp.sum(
            jp.square(right_pos_err)
        )

        left_upvector = self._get_sensor_data(data, "left_foot_upvector")
        right_upvector = self._get_sensor_data(data, "right_foot_upvector")
        left_ori_err = left_upvector - self._ref_left_upvector[ref_idx]
        right_ori_err = right_upvector - self._ref_right_upvector[ref_idx]
        foot_orientation_cost = jp.sum(jp.square(left_ori_err)) + jp.sum(
            jp.square(right_ori_err)
        )

        control_cost = jp.sum(jp.square(motor_targets - q_ref[7 : 7 + self._nu]))
        action_rate_cost = jp.sum(jp.square(action - last_act))

        return {
            "configuration": configuration_cost,
            "foot_position": foot_position_cost,
            "foot_orientation": foot_orientation_cost,
            "control": control_cost,
            "action_rate": action_rate_cost,
            "alive": jp.array(1.0),
            "termination": done,
        }

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
