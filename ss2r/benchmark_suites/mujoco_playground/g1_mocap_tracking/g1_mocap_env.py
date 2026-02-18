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
            level=0.0,
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
                # Imitation reward: r^I = w^p r^p + w^v r^v + w^e r^e + w^c r^c.
                pose=0.65,
                velocity=0.1,
                end_effector=0.15,
                com=0.1,
                control=0.0,
                # Debug-only terms (not part of total reward by default).
                left_pos=0.0,
                right_pos=0.0,
                left_ori=0.0,
                right_ori=0.0,
                alive=0.0,
                termination=-10.0,
            ),
            # Exponential shaping coefficients used in exp(-coef * ||error||^2).
            pose_exp_coeff=2.0,
            velocity_exp_coeff=0.1,
            end_effector_exp_coeff=40.0,
            com_exp_coeff=10.0,
            control_exp_coeff=0.01,
            foot_position_exp_coeff=10.0,
            foot_orientation_exp_coeff=2.0,
        ),
        reference_path="ss2r/benchmark_suites/mujoco_playground/g1_mocap_tracking/walk1_subject1.npz",
        random_start=True,
        loop_reference=True,
        reset_noise_scale=0.0,
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
        key_q = np.asarray(
            self._mj_model.keyframe("mocap_default").qpos, dtype=np.float32
        )
        if key_q.shape[0] != self._mj_model.nq:
            raise ValueError(
                "Keyframe 'mocap_default' qpos width does not match model nq: "
                f"keyframe={key_q.shape[0]}, nq={self._mj_model.nq}."
            )
        self._init_q = jp.array(key_q)
        self._default_pose = jp.array(key_q[7 : 7 + self._nu])
        self._left_hand_body_id = int(self._mj_model.body("left_wrist_yaw_link").id)
        self._right_hand_body_id = int(self._mj_model.body("right_wrist_yaw_link").id)

        self._feet_site_id = np.array(
            [
                self._mj_model.site("left_foot").id,
                self._mj_model.site("right_foot").id,
            ]
        )
        actuator_joint_ids = np.asarray(
            self._mj_model.actuator_trnid[:, 0], dtype=np.int32
        )
        pose_body_ids: list[int] = []
        for joint_id in actuator_joint_ids.tolist():
            body_id = int(self._mj_model.jnt_bodyid[joint_id])
            if body_id not in pose_body_ids:
                pose_body_ids.append(body_id)
        self._pose_body_ids = jp.array(pose_body_ids, dtype=jp.int32)
        body_mass = np.asarray(self._mj_model.body_mass, dtype=np.float32)
        self._body_mass = jp.array(body_mass)
        self._body_mass_total = jp.asarray(max(float(np.sum(body_mass)), 1e-8))

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
            ref_ee_pos,
            ref_joint_vel,
            ref_com,
            ref_pose_body_quat,
        ) = self._precompute_reference_terms(reference)
        self._ref_left_pos = jp.array(ref_left_pos)
        self._ref_left_upvector = jp.array(ref_left_upvector)
        self._ref_right_pos = jp.array(ref_right_pos)
        self._ref_right_upvector = jp.array(ref_right_upvector)
        self._ref_ee_pos = jp.array(ref_ee_pos)
        self._ref_joint_vel = jp.array(ref_joint_vel)
        self._ref_com = jp.array(ref_com)
        self._ref_pose_body_quat = jp.array(ref_pose_body_quat)

    def _precompute_reference_terms(
        self, reference: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        mj_data = mujoco.MjData(self._mj_model)
        n_frames = reference.shape[0]  # type: ignore
        left_foot_site_id = self._mj_model.site("left_foot").id
        right_foot_site_id = self._mj_model.site("right_foot").id
        body_mass = np.asarray(self._mj_model.body_mass, dtype=np.float32)
        body_mass_total = max(float(np.sum(body_mass)), 1e-8)
        pose_body_ids = np.asarray(self._pose_body_ids, dtype=np.int32)

        ref_left_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_left_upvector = np.zeros((n_frames, 3), dtype=np.float32)
        ref_right_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_right_upvector = np.zeros((n_frames, 3), dtype=np.float32)
        ref_left_hand_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_right_hand_pos = np.zeros((n_frames, 3), dtype=np.float32)
        ref_com = np.zeros((n_frames, 3), dtype=np.float32)
        ref_pose_body_quat = np.zeros(
            (n_frames, pose_body_ids.shape[0], 4), dtype=np.float32
        )

        for i in range(n_frames):
            mj_data.qpos[:] = reference[i]
            mujoco.mj_forward(self._mj_model, mj_data)
            ref_left_pos[i] = mj_data.site_xpos[left_foot_site_id]
            ref_right_pos[i] = mj_data.site_xpos[right_foot_site_id]
            ref_left_hand_pos[i] = mj_data.xpos[self._left_hand_body_id]
            ref_right_hand_pos[i] = mj_data.xpos[self._right_hand_body_id]
            ref_com[i] = (
                np.sum(mj_data.xipos * body_mass[:, None], axis=0) / body_mass_total
            )
            ref_pose_body_quat[i] = mj_data.xquat[pose_body_ids]
            left_rot = np.asarray(
                mj_data.site_xmat[left_foot_site_id], dtype=np.float32
            ).reshape(3, 3)
            right_rot = np.asarray(
                mj_data.site_xmat[right_foot_site_id], dtype=np.float32
            ).reshape(3, 3)
            ref_left_upvector[i] = left_rot[:, 2]
            ref_right_upvector[i] = right_rot[:, 2]

        ref_ee_pos = np.stack(
            [ref_left_pos, ref_right_pos, ref_left_hand_pos, ref_right_hand_pos], axis=1
        )
        idx_next = np.arange(n_frames) + 1
        if self._config.loop_reference:
            idx_next = idx_next % n_frames
        else:
            idx_next = np.clip(idx_next, 0, n_frames - 1)
        joint_curr = reference[:, 7 : 7 + self._nu]
        joint_next = reference[idx_next, 7 : 7 + self._nu]
        joint_delta = np.arctan2(
            np.sin(joint_next - joint_curr), np.cos(joint_next - joint_curr)
        )
        ref_joint_vel = joint_delta * self._reference_fps

        return (
            ref_left_pos,
            ref_left_upvector,
            ref_right_pos,
            ref_right_upvector,
            ref_ee_pos.astype(np.float32),
            ref_joint_vel.astype(np.float32),
            ref_com.astype(np.float32),
            ref_pose_body_quat.astype(np.float32),
        )

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

    def _get_center_of_mass(self, body_com_pos: jax.Array) -> jax.Array:
        return (
            jp.sum(body_com_pos * self._body_mass[:, None], axis=0)
            / self._body_mass_total
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, start_rng = jax.random.split(rng)
        max_start_frames = min(self._reference.shape[0], 5000)
        reference_start_idx = jax.lax.cond(
            self._config.random_start,
            lambda _: jax.random.randint(start_rng, (), 0, max_start_frames),
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
            "last_last_act": jp.zeros(self._nu),
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
        # Joint position control around mocap reference: u = u_ref + action_scale * a.
        ref_idx_target = self._reference_index(
            state.data.time + self.dt,
            state.info["reference_start_idx"],
        )
        u_ref = self._reference[ref_idx_target][7 : 7 + self._nu]
        motor_targets = u_ref + self._config.action_scale * action

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            motor_targets,
            self.n_substeps,
        )

        feet_pos = data.site_xpos[self._feet_site_id]

        done = self._get_termination(data)
        rewards = self._get_reward(
            data,
            motor_targets,
            done,
            ref_idx_target,
            feet_pos,
        )
        rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = sum(rewards.values())

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["motor_targets"] = motor_targets
        state.info["phase"] = self._phase_from_index(ref_idx_target)

        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value

        obs = self._get_obs(data, state.info)
        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        root_height = data.qpos[2]
        torso_up_z = self._get_sensor_data(data, "upvector_torso")[-1]
        fall_by_height = root_height < self._config.termination_height
        fall_by_orientation = torso_up_z < self._config.termination_upvector_z
        invalid_state = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return fall_by_height | fall_by_orientation | invalid_state

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
        motor_targets: jax.Array,
        done: jax.Array,
        ref_idx: jax.Array,
        feet_pos: jax.Array,
    ) -> dict[str, jax.Array]:
        q_ref = self._reference[ref_idx]
        pose_quat = data.xquat[self._pose_body_ids]
        pose_quat_ref = self._ref_pose_body_quat[ref_idx]
        pose_angle_err = self._quat_angle_error(pose_quat_ref, pose_quat)
        pose_err_sq = jp.sum(jp.square(pose_angle_err))

        joint_vel = data.qvel[6 : 6 + self._nu]
        joint_vel_ref = self._ref_joint_vel[ref_idx]
        vel_err_sq = jp.sum(jp.square(joint_vel_ref - joint_vel))

        left_pos_err = feet_pos[0] - self._ref_left_pos[ref_idx]
        right_pos_err = feet_pos[1] - self._ref_right_pos[ref_idx]
        left_pos_sq = jp.sum(jp.square(left_pos_err))
        right_pos_sq = jp.sum(jp.square(right_pos_err))
        ee_pos = jp.stack(
            [
                feet_pos[0],
                feet_pos[1],
                data.xpos[self._left_hand_body_id],
                data.xpos[self._right_hand_body_id],
            ],
            axis=0,
        )
        ee_err_sq = jp.sum(jp.square(ee_pos - self._ref_ee_pos[ref_idx]))

        left_upvector = self._get_sensor_data(data, "left_foot_upvector")
        right_upvector = self._get_sensor_data(data, "right_foot_upvector")
        left_ori_err = left_upvector - self._ref_left_upvector[ref_idx]
        right_ori_err = right_upvector - self._ref_right_upvector[ref_idx]
        left_ori_sq = jp.sum(jp.square(left_ori_err))
        right_ori_sq = jp.sum(jp.square(right_ori_err))
        com = self._get_center_of_mass(data.xipos)
        com_err_sq = jp.sum(jp.square(com - self._ref_com[ref_idx]))

        u_ref = q_ref[7 : 7 + self._nu]
        control_err_sq = jp.sum(jp.square(motor_targets - u_ref))

        return {
            "pose": self._exp_reward(
                pose_err_sq, self._config.reward_config.pose_exp_coeff
            ),
            "velocity": self._exp_reward(
                vel_err_sq, self._config.reward_config.velocity_exp_coeff
            ),
            "end_effector": self._exp_reward(
                ee_err_sq, self._config.reward_config.end_effector_exp_coeff
            ),
            "com": self._exp_reward(
                com_err_sq, self._config.reward_config.com_exp_coeff
            ),
            "control": self._exp_reward(
                control_err_sq, self._config.reward_config.control_exp_coeff
            ),
            "left_pos": self._exp_reward(
                left_pos_sq, self._config.reward_config.foot_position_exp_coeff
            ),
            "right_pos": self._exp_reward(
                right_pos_sq, self._config.reward_config.foot_position_exp_coeff
            ),
            "left_ori": self._exp_reward(
                left_ori_sq, self._config.reward_config.foot_orientation_exp_coeff
            ),
            "right_ori": self._exp_reward(
                right_ori_sq, self._config.reward_config.foot_orientation_exp_coeff
            ),
            "alive": jp.array(1.0),
            "termination": done,
        }

    def _exp_reward(self, err_sq: jax.Array, coeff: float) -> jax.Array:
        coeff_val = jp.maximum(jp.asarray(coeff, dtype=err_sq.dtype), 1e-8)
        return jp.exp(-coeff_val * err_sq)

    def _quat_angle_error(self, quat_a: jax.Array, quat_b: jax.Array) -> jax.Array:
        dot = jp.sum(quat_a * quat_b, axis=-1)
        dot = jp.clip(jp.abs(dot), 0.0, 1.0)
        return 2.0 * jp.arccos(dot)

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
