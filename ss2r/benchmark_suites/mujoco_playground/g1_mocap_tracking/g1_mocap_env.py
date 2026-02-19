"""DeepMimic mocap tracking task for Unitree G1."""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import collision, mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


def _atleast_3d(
    x: Union[np.ndarray, jax.Array], backend: Any
) -> Union[np.ndarray, jax.Array]:
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 1:
        return x[None, None, ...]
    raise ValueError(f"Expected tensor with <=3 dims, got shape={x.shape}.")


def _quat_scalarfirst_to_scalarlast(
    quat: Union[np.ndarray, jax.Array],
) -> Union[np.ndarray, jax.Array]:
    return quat[..., [1, 2, 3, 0]]


def _quaternion_angular_distance(
    q1: Union[np.ndarray, jax.Array],
    q2: Union[np.ndarray, jax.Array],
    backend: Any,
) -> Union[np.ndarray, jax.Array]:
    q1_norm = backend.linalg.norm(q1, axis=-1, keepdims=True)
    q2_norm = backend.linalg.norm(q2, axis=-1, keepdims=True)
    q1 = q1 / backend.maximum(q1_norm, 1e-8)
    q2 = q2 / backend.maximum(q2_norm, 1e-8)
    dots = backend.sum(q1 * q2, axis=-1)
    dots = backend.clip(backend.abs(dots), -1.0, 1.0)
    return 2.0 * backend.arccos(dots)


def _rotmat_to_rotvec(
    rot: Union[np.ndarray, jax.Array], backend: Any
) -> Union[np.ndarray, jax.Array]:
    trace = backend.trace(rot, axis1=-2, axis2=-1)
    cos_theta = backend.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = backend.arccos(cos_theta)

    vx = rot[..., 2, 1] - rot[..., 1, 2]
    vy = rot[..., 0, 2] - rot[..., 2, 0]
    vz = rot[..., 1, 0] - rot[..., 0, 1]
    v = backend.stack([vx, vy, vz], axis=-1)

    sin_theta = backend.sin(theta)
    denom = 2.0 * sin_theta
    denom_safe = backend.where(
        backend.abs(denom) < 1e-8,
        backend.ones_like(denom) * 1e-8,
        denom,
    )
    axis = v / denom_safe[..., None]
    rotvec = axis * theta[..., None]

    small = backend.abs(sin_theta) < 1e-6
    small_rotvec = 0.5 * v
    return backend.where(small[..., None], small_rotvec, rotvec)


def _transform_motion(
    vel: Union[np.ndarray, jax.Array],
    new_pos: Union[np.ndarray, jax.Array],
    old_pos: Union[np.ndarray, jax.Array],
    rot_mat_new2old: Union[np.ndarray, jax.Array],
    backend: Any,
    flg_local: bool,
) -> Union[np.ndarray, jax.Array]:
    vel = backend.atleast_2d(vel)
    new_pos = backend.atleast_2d(new_pos)
    old_pos = backend.atleast_2d(old_pos)
    rot_mat_new2old = _atleast_3d(rot_mat_new2old, backend)

    lin_vel = vel[:, 3:]
    rot_vel = vel[:, :3]
    rel_pos = new_pos - old_pos

    lin_vel = lin_vel - backend.cross(rel_pos, rot_vel, axis=-1)
    if flg_local:
        lin_vel = backend.einsum(
            "bij,bj->bi", rot_mat_new2old.transpose(0, 2, 1), lin_vel
        )
        rot_vel = backend.einsum(
            "bij,bj->bi", rot_mat_new2old.transpose(0, 2, 1), rot_vel
        )

    return backend.hstack([rot_vel, lin_vel])


def _calc_site_velocities(
    site_ids: Union[np.ndarray, jax.Array],
    data: Union[mujoco.MjData, mjx.Data],
    parent_body_id: Union[np.ndarray, jax.Array],
    root_body_id: Union[np.ndarray, jax.Array],
    backend: Any,
    flg_local: bool,
) -> Union[np.ndarray, jax.Array]:
    site_xpos = data.site_xpos[site_ids]
    site_xmat = data.site_xmat[site_ids].reshape((-1, 3, 3))
    parent_body_cvel = data.cvel[parent_body_id]
    root_subtree_com = data.subtree_com[root_body_id]
    return _transform_motion(
        parent_body_cvel,
        site_xpos,
        root_subtree_com,
        site_xmat,
        backend,
        flg_local=flg_local,
    )


def _calculate_relative_site_quantities(
    data: Union[mujoco.MjData, mjx.Data],
    rel_site_ids: Union[np.ndarray, jax.Array],
    rel_body_ids: Union[np.ndarray, jax.Array],
    body_rootid: Union[np.ndarray, jax.Array],
    backend: Any,
) -> tuple[Union[np.ndarray, jax.Array], ...]:
    site_xpos = data.site_xpos[rel_site_ids]
    site_xmat = data.site_xmat[rel_site_ids].reshape((-1, 3, 3))

    site_root_body_id = body_rootid[rel_body_ids]
    site_xvel = _calc_site_velocities(
        rel_site_ids,
        data,
        rel_body_ids,
        site_root_body_id,
        backend,
        flg_local=False,
    )

    main_id = 0
    main_xpos = site_xpos[main_id]
    main_xmat = site_xmat[main_id]
    main_xvel = site_xvel[main_id]

    other_xpos = site_xpos[1:]
    other_xmat = site_xmat[1:]
    other_xvel = site_xvel[1:]

    rel_rot_mat = backend.einsum("ij,njk->nik", main_xmat.T, other_xmat)

    rpos = other_xpos - main_xpos
    rangles = _rotmat_to_rotvec(rel_rot_mat, backend)

    ang_a = main_xvel[:3]
    lin_a = main_xvel[3:]
    ang_b = other_xvel[:, :3]
    lin_b = other_xvel[:, 3:]

    rlin = backend.einsum("jk,ik->ij", main_xmat, lin_a - lin_b)
    ang_b_in_a = backend.einsum("ikj,ik->ij", rel_rot_mat, ang_b)
    rang = ang_b_in_a - ang_a
    rvel = backend.hstack([rang, rlin])

    return rpos, rangles, rvel


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        reward_config=config_dict.create(
            scales=config_dict.create(
                qpos_reward=1.0,
                qvel_reward=1.0,
                rpos_reward=1.0,
                rquat_reward=1.0,
                rvel_reward=1.0,
                out_of_bounds_reward=1.0,
                joint_acc_reward=1.0,
                joint_torque_reward=1.0,
                action_rate_reward=1.0,
                total_penalties=1.0,
                mimic_total=1.0,
            ),
        ),
        deepmimic=config_dict.create(
            reference_source="hf",
            reference_filename="Lafan1/mocap/UnitreeG1/dance1_subject3.npz",
            reference_repo_id="robfiras/loco-mujoco-datasets",
            reference_repo_type="dataset",
            reference_max_steps=2000,
            action_residual_scale=0.35,
            mimic_site_regex=r".*_mimic$",
            reward_sums=config_dict.create(
                qpos=0.4,
                qvel=0.2,
                rpos=0.5,
                rquat=0.3,
                rvel=0.1,
            ),
            reward_exponents=config_dict.create(
                qpos=10.0,
                qvel=2.0,
                rpos=100.0,
                rquat=10.0,
                rvel=0.1,
            ),
            penalties=config_dict.create(
                action_out_of_bounds=0.01,
                joint_acc=0.0,
                joint_torque=0.0,
                action_rate=0.0,
            ),
        ),
    )


class G1MocapTracking(g1_base.G1Env):
    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("knees_bent").qpos[7:])

        self._apply_mocap_actuator_params()
        self._mjx_model = mjx.put_model(self._mj_model)

        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )
        self._left_foot_geom_id = self._mj_model.geom("left_foot").id
        self._right_foot_geom_id = self._mj_model.geom("right_foot").id
        self._left_shin_geom_id = self._mj_model.geom("left_shin").id
        self._right_shin_geom_id = self._mj_model.geom("right_shin").id

        self._load_mocap_reference()
        self._setup_deepmimic_indices()
        self._setup_action_mapping()
        self._setup_mimic_sites()
        self._precompute_reference_features()
        self._validate_indices_and_shapes()
        self._init_grounded_reset_from_reference()

        self._reward_metric_keys = [
            "qpos_reward",
            "qvel_reward",
            "rpos_reward",
            "rquat_reward",
            "rvel_reward",
            "out_of_bounds_reward",
            "joint_acc_reward",
            "joint_torque_reward",
            "action_rate_reward",
            "total_penalties",
            "mimic_total",
        ]

    def _warn_if_duplicates(self, name: str, values: np.ndarray) -> None:
        unique_count = len(np.unique(values))
        if unique_count != len(values):
            warnings.warn(
                f"[G1MocapTracking] Duplicate values detected in {name}: "
                f"total={len(values)} unique={unique_count}",
                RuntimeWarning,
            )

    def _validate_indices_and_shapes(self) -> None:
        expected_qpos_dim = 5 + (self._mj_model.njnt - 1)
        actual_qpos_dim = int(self._deepmimic_qpos_ind_np.shape[0])
        if actual_qpos_dim != expected_qpos_dim:
            warnings.warn(
                "[G1MocapTracking] DeepMimic qpos index size mismatch: "
                f"expected={expected_qpos_dim}, actual={actual_qpos_dim}",
                RuntimeWarning,
            )

        expected_qvel_dim = 6 + (self._mj_model.njnt - 1)
        actual_qvel_dim = int(self._deepmimic_qvel_ind_np.shape[0])
        if actual_qvel_dim != expected_qvel_dim:
            warnings.warn(
                "[G1MocapTracking] DeepMimic qvel index size mismatch: "
                f"expected={expected_qvel_dim}, actual={actual_qvel_dim}",
                RuntimeWarning,
            )

        quat_dims = int(self._qpos_quat_mask_np.sum())
        if quat_dims != 4:
            warnings.warn(
                "[G1MocapTracking] Unexpected quaternion dims in DeepMimic qpos mask: "
                f"expected=4, actual={quat_dims}",
                RuntimeWarning,
            )

        if np.any(self._deepmimic_qpos_ind_np < 0) or np.any(
            self._deepmimic_qpos_ind_np >= self._mj_model.nq
        ):
            warnings.warn(
                "[G1MocapTracking] qpos indices out of range for model.nq.",
                RuntimeWarning,
            )

        if np.any(self._deepmimic_qvel_ind_np < 0) or np.any(
            self._deepmimic_qvel_ind_np >= self._mj_model.nv
        ):
            warnings.warn(
                "[G1MocapTracking] qvel indices out of range for model.nv.",
                RuntimeWarning,
            )

        self._warn_if_duplicates("deepmimic_qpos_ind", self._deepmimic_qpos_ind_np)
        self._warn_if_duplicates("deepmimic_qvel_ind", self._deepmimic_qvel_ind_np)
        self._warn_if_duplicates("actuator_qpos_adr", self._actuator_qpos_adr_np)
        self._warn_if_duplicates("mimic_site_ids", self._mimic_site_ids_np)

        if self._actuator_qpos_adr_np.shape[0] != self._mj_model.nu:
            warnings.warn(
                "[G1MocapTracking] Actuator-to-qpos mapping size mismatch: "
                f"expected nu={self._mj_model.nu}, "
                f"actual={self._actuator_qpos_adr_np.shape[0]}",
                RuntimeWarning,
            )

        if np.any(self._actuator_qpos_adr_np < 0) or np.any(
            self._actuator_qpos_adr_np >= self._mj_model.nq
        ):
            warnings.warn(
                "[G1MocapTracking] actuator_qpos_adr contains out-of-range values.",
                RuntimeWarning,
            )

        ref_frames = int(self._reference.shape[0])
        if self._reference_goal_qpos.shape != (ref_frames, actual_qpos_dim):
            warnings.warn(
                "[G1MocapTracking] reference_goal_qpos shape mismatch: "
                f"expected={(ref_frames, actual_qpos_dim)}, "
                f"actual={tuple(self._reference_goal_qpos.shape)}",
                RuntimeWarning,
            )
        if self._reference_goal_qvel.shape != (ref_frames, actual_qvel_dim):
            warnings.warn(
                "[G1MocapTracking] reference_goal_qvel shape mismatch: "
                f"expected={(ref_frames, actual_qvel_dim)}, "
                f"actual={tuple(self._reference_goal_qvel.shape)}",
                RuntimeWarning,
            )
        if self._reference_motor_targets.shape != (ref_frames, self.action_size):
            warnings.warn(
                "[G1MocapTracking] reference_motor_targets shape mismatch: "
                f"expected={(ref_frames, self.action_size)}, "
                f"actual={tuple(self._reference_motor_targets.shape)}",
                RuntimeWarning,
            )

        first_targets = np.asarray(self._reference_motor_targets[0])
        first_qpos_indexed = np.asarray(self._reference[0, self._actuator_qpos_adr_np])
        max_abs_diff = float(np.max(np.abs(first_targets - first_qpos_indexed)))
        if max_abs_diff > 1e-5:
            warnings.warn(
                "[G1MocapTracking] reference motor target indexing mismatch: "
                f"max_abs_diff={max_abs_diff:.6e}",
                RuntimeWarning,
            )

        print(
            "[G1MocapTracking] Index validation summary: "
            f"qpos_dim={actual_qpos_dim}, qvel_dim={actual_qvel_dim}, "
            f"quat_dims={quat_dims}, action_size={self.action_size}, "
            f"mimic_sites={len(self._mimic_site_ids_np)}, ref_frames={ref_frames}, "
            f"ref_fps={self._reference_fps:.3f}"
        )

    def _setup_deepmimic_indices(self) -> None:
        root_free_joint_id = None
        for i in range(self._mj_model.njnt):
            if int(self._mj_model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE):
                root_free_joint_id = i
                break
        if root_free_joint_id is None:
            raise ValueError("Could not find free root joint for DeepMimic indices.")

        qpos_ind: list[np.ndarray] = []
        qvel_ind: list[np.ndarray] = []
        quat_qpos_ids: list[int] = []
        nonfree_qvel_ids: list[int] = []

        for jnt_id in range(self._mj_model.njnt):
            qpos_adr = int(self._mj_model.jnt_qposadr[jnt_id])
            dof_adr = int(self._mj_model.jnt_dofadr[jnt_id])
            jnt_type = int(self._mj_model.jnt_type[jnt_id])

            if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
                qpos_ids = np.arange(qpos_adr, qpos_adr + 7, dtype=np.int32)
                qvel_ids = np.arange(dof_adr, dof_adr + 6, dtype=np.int32)
                if jnt_id == root_free_joint_id:
                    qpos_ind.append(qpos_ids[2:])
                    quat_qpos_ids.extend(qpos_ids[3:].tolist())
                else:
                    qpos_ind.append(qpos_ids)
                qvel_ind.append(qvel_ids)
            elif jnt_type in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                qpos_ids = np.array([qpos_adr], dtype=np.int32)
                qvel_ids = np.array([dof_adr], dtype=np.int32)
                qpos_ind.append(qpos_ids)
                qvel_ind.append(qvel_ids)
                nonfree_qvel_ids.extend(qvel_ids.tolist())

        self._deepmimic_qpos_ind_np = np.concatenate(qpos_ind, axis=0)
        self._deepmimic_qvel_ind_np = np.concatenate(qvel_ind, axis=0)

        quat_set = set(quat_qpos_ids)
        self._qpos_quat_mask_np = np.array(
            [idx in quat_set for idx in self._deepmimic_qpos_ind_np], dtype=bool
        )
        self._qpos_quat_indices_np = np.where(self._qpos_quat_mask_np)[0].astype(
            np.int32
        )
        self._qpos_nonquat_indices_np = np.where(~self._qpos_quat_mask_np)[0].astype(
            np.int32
        )

        self._deepmimic_qpos_ind = jp.array(self._deepmimic_qpos_ind_np)
        self._deepmimic_qvel_ind = jp.array(self._deepmimic_qvel_ind_np)
        self._qpos_quat_mask = jp.array(self._qpos_quat_mask_np)
        self._qpos_quat_indices = jp.array(self._qpos_quat_indices_np)
        self._qpos_nonquat_indices = jp.array(self._qpos_nonquat_indices_np)

        nonfree_qvel_mask = np.zeros((self._mj_model.nv,), dtype=bool)
        if nonfree_qvel_ids:
            nonfree_qvel_mask[np.array(nonfree_qvel_ids, dtype=np.int32)] = True
        self._nonfree_qvel_mask = jp.array(nonfree_qvel_mask)

    def _setup_action_mapping(self) -> None:
        actuator_qpos_adr = []
        for actuator_id in range(self._mj_model.nu):
            joint_id = int(self._mj_model.actuator_trnid[actuator_id, 0])
            actuator_qpos_adr.append(int(self._mj_model.jnt_qposadr[joint_id]))

        self._actuator_qpos_adr_np = np.array(actuator_qpos_adr, dtype=np.int32)
        self._actuator_qpos_adr = jp.array(self._actuator_qpos_adr_np)

        ctrl_min = np.asarray(self._mj_model.actuator_ctrlrange[:, 0], dtype=np.float32)
        ctrl_max = np.asarray(self._mj_model.actuator_ctrlrange[:, 1], dtype=np.float32)
        self._actuator_ctrl_min = jp.array(ctrl_min)
        self._actuator_ctrl_max = jp.array(ctrl_max)
        self._actuator_ctrl_half_range = jp.array(0.5 * (ctrl_max - ctrl_min))
        default_motor_targets = np.asarray(self._init_q, dtype=np.float32)[
            self._actuator_qpos_adr_np
        ]
        self._default_motor_targets = jp.array(default_motor_targets)

    def _setup_mimic_sites(self) -> None:
        mimic_site_regex = re.compile(str(self._config.deepmimic.mimic_site_regex))
        available_sites = [
            str(self._mj_model.site(i).name) for i in range(self._mj_model.nsite)
        ]
        mimic_site_names: list[str] = []
        site_ids_list: list[int] = []
        for site_id in range(self._mj_model.nsite):
            site_name = str(self._mj_model.site(site_id).name)
            if mimic_site_regex.fullmatch(site_name):
                mimic_site_names.append(site_name)
                site_ids_list.append(site_id)

        site_source = f"regex='{mimic_site_regex.pattern}'"
        if len(site_ids_list) < 2:
            available_set = set(available_sites)
            preferred_site_names = (
                list(consts.FEET_SITES)
                + list(consts.HAND_SITES)
                + ["imu_in_pelvis", "imu_in_torso"]
            )
            fallback_names = [
                name for name in preferred_site_names if name in available_set
            ]
            fallback_names = list(dict.fromkeys(fallback_names))
            if len(fallback_names) < 2:
                fallback_names = list(available_sites)
                site_source = "all_model_sites"
            else:
                site_source = "preferred_end_effectors_and_imu"

            if len(fallback_names) < 2:
                raise ValueError(
                    "DeepMimic requires at least 2 sites for relative features. "
                    f"Found {len(fallback_names)}. Available sites: {available_sites}"
                )

            mimic_site_names = fallback_names
            site_ids_list = [self._mj_model.site(name).id for name in mimic_site_names]

        site_ids = np.array(site_ids_list, dtype=np.int32)
        site_body_ids = self._mj_model.site_bodyid[site_ids].astype(np.int32)
        body_rootid = np.asarray(self._mj_model.body_rootid, dtype=np.int32)
        body_parentid = np.asarray(self._mj_model.body_parentid, dtype=np.int32)

        def _body_depth(body_id: int) -> int:
            depth = 0
            current = int(body_id)
            while current > 0:
                current = int(body_parentid[current])
                depth += 1
            return depth

        # Make the first site a root-proximal anchor for relative features.
        body_depths = np.array([_body_depth(int(b)) for b in site_body_ids])
        anchor_idx = int(np.argmin(body_depths))
        sorted_indices = np.argsort(site_ids)
        order = [anchor_idx] + [int(i) for i in sorted_indices if int(i) != anchor_idx]
        site_ids = site_ids[np.array(order, dtype=np.int32)]
        site_body_ids = site_body_ids[np.array(order, dtype=np.int32)]
        mimic_site_names = [mimic_site_names[i] for i in order]

        self._mimic_site_names = mimic_site_names
        self._mimic_site_ids_np = site_ids
        self._mimic_site_body_ids_np = site_body_ids
        self._body_rootid_np = body_rootid

        self._mimic_site_ids = jp.array(site_ids)
        self._mimic_site_body_ids = jp.array(site_body_ids)
        self._body_rootid = jp.array(body_rootid)

        print(
            "[G1MocapTracking] Discovered mimic sites from model: "
            f"count={len(mimic_site_names)}, anchor={mimic_site_names[0]}, "
            f"source={site_source}"
        )
        print(f"[G1MocapTracking] mimic_site_names={mimic_site_names}")

    def _precompute_reference_features(self) -> None:
        num_frames = int(self._reference.shape[0])

        reference_goal_qpos = np.asarray(
            self._reference[:, self._deepmimic_qpos_ind_np], dtype=np.float32
        )
        reference_goal_qvel = np.asarray(
            self._reference_qvel[:, self._deepmimic_qvel_ind_np], dtype=np.float32
        )

        n_rel_sites = len(self._mimic_site_ids_np) - 1
        ref_rpos = np.zeros((num_frames, n_rel_sites * 3), dtype=np.float32)
        ref_rquat = np.zeros((num_frames, n_rel_sites * 3), dtype=np.float32)
        ref_rvel = np.zeros((num_frames, n_rel_sites * 6), dtype=np.float32)

        mj_data = mujoco.MjData(self._mj_model)
        for frame_idx in range(num_frames):
            mj_data.qpos[:] = np.asarray(self._reference[frame_idx], dtype=np.float64)
            mj_data.qvel[:] = np.asarray(
                self._reference_qvel[frame_idx], dtype=np.float64
            )
            mujoco.mj_forward(self._mj_model, mj_data)

            rpos_i, rquat_i, rvel_i = _calculate_relative_site_quantities(
                mj_data,
                self._mimic_site_ids_np,
                self._mimic_site_body_ids_np,
                self._body_rootid_np,
                np,
            )
            ref_rpos[frame_idx] = np.asarray(rpos_i, dtype=np.float32).ravel()
            ref_rquat[frame_idx] = np.asarray(rquat_i, dtype=np.float32).ravel()
            ref_rvel[frame_idx] = np.asarray(rvel_i, dtype=np.float32).ravel()

        self._reference_goal_qpos = jp.array(reference_goal_qpos)
        self._reference_goal_qvel = jp.array(reference_goal_qvel)
        self._reference_rpos = jp.array(ref_rpos)
        self._reference_rquat = jp.array(ref_rquat)
        self._reference_rvel = jp.array(ref_rvel)
        self._reference_motor_targets = self._reference[:, self._actuator_qpos_adr_np]
        self._reference_horizon_steps = int(num_frames - 1)

    def _apply_mocap_actuator_params(self) -> None:
        """Applies mocap tracking actuator/joint dynamics parameters."""
        group_specs = [
            (
                [r".*_hip_roll_joint", r".*_knee_joint"],
                99.0984,
                6.3088,
                139.0,
                0.025101925,
            ),
            (
                [r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"],
                40.1792,
                2.5579,
                88.0,
                0.010177520,
            ),
            (
                [r"waist_pitch_joint", r"waist_roll_joint"],
                28.5012,
                1.8144,
                50.0,
                0.007219450,
            ),
            (
                [r".*_ankle_pitch_joint", r".*_ankle_roll_joint"],
                28.5012,
                1.8144,
                50.0,
                0.007219450,
            ),
            (
                [
                    r".*_elbow_joint",
                    r".*_shoulder_pitch_joint",
                    r".*_shoulder_roll_joint",
                    r".*_shoulder_yaw_joint",
                    r".*_wrist_roll_joint",
                ],
                14.2506,
                0.9072,
                25.0,
                0.003609725,
            ),
            (
                [r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"],
                16.7783,
                1.0681,
                5.0,
                0.00425,
            ),
        ]

        compiled_specs = [
            ((tuple(re.compile(pattern) for pattern in patterns)), kp, kd, effort, arm)
            for patterns, kp, kd, effort, arm in group_specs
        ]

        unmatched_actuators: list[str] = []
        matched_count = 0
        for actuator_id in range(self._mj_model.nu):
            actuator_name = str(self._mj_model.actuator(actuator_id).name)
            params = None
            for regexes, kp, kd, effort_limit, armature in compiled_specs:
                if any(regex.fullmatch(actuator_name) for regex in regexes):
                    params = (kp, kd, effort_limit, armature)
                    break

            if params is None:
                unmatched_actuators.append(actuator_name)
                continue

            kp, kd, effort_limit, armature = params
            self._mj_model.actuator_gainprm[actuator_id, 0] = kp
            self._mj_model.actuator_biasprm[actuator_id, 1] = -kp
            self._mj_model.actuator_biasprm[actuator_id, 2] = -kd
            self._mj_model.actuator_forcerange[actuator_id, 0] = -effort_limit
            self._mj_model.actuator_forcerange[actuator_id, 1] = effort_limit

            joint_id = int(self._mj_model.actuator_trnid[actuator_id, 0])
            dof_adr = int(self._mj_model.jnt_dofadr[joint_id])
            self._mj_model.dof_armature[dof_adr] = armature
            matched_count += 1

        if unmatched_actuators:
            raise ValueError(
                f"Missing mocap actuator params for actuators: {unmatched_actuators}."
            )
        print(
            "[G1MocapTracking] Applied mocap actuator params to "
            f"{matched_count} actuators."
        )

    def _foot_geom_lowest_z(self, mj_data: mujoco.MjData, geom_id: int) -> float:
        """Returns the lowest world z point of a foot collision geom."""
        geom_type = int(self._mj_model.geom_type[geom_id])
        geom_xpos = mj_data.geom_xpos[geom_id]
        geom_size = self._mj_model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            rot = mj_data.geom_xmat[geom_id].reshape(3, 3)
            z_extent = (
                abs(rot[2, 0]) * geom_size[0]
                + abs(rot[2, 1]) * geom_size[1]
                + abs(rot[2, 2]) * geom_size[2]
            )
            return float(geom_xpos[2] - z_extent)
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_xpos[2] - geom_size[0])
        return float(geom_xpos[2] - self._mj_model.geom_rbound[geom_id])

    def _init_grounded_reset_from_reference(self) -> None:
        """Precomputes a reset state that starts with feet on ground."""
        qpos0 = np.asarray(self._reference[0], dtype=np.float64).copy()
        qvel0 = np.asarray(self._reference_qvel[0], dtype=np.float64).copy()

        mj_data = mujoco.MjData(self._mj_model)
        mj_data.qpos[:] = qpos0
        mj_data.qvel[:] = qvel0
        mujoco.mj_forward(self._mj_model, mj_data)

        min_foot_z = min(
            self._foot_geom_lowest_z(mj_data, int(geom_id))
            for geom_id in self._feet_geom_id
        )
        qpos0[2] -= min_foot_z

        self._reset_qpos = jp.array(qpos0, dtype=jp.float32)
        self._reset_qvel = jp.array(qvel0, dtype=jp.float32)

    def _resample_reference_to_ctrl_dt(
        self,
        reference_qpos: np.ndarray,
        reference_qvel: np.ndarray,
        source_fps: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        target_dt = float(self.dt)
        target_fps = 1.0 / target_dt

        source_times = np.arange(reference_qpos.shape[0], dtype=np.float64) / source_fps
        target_times = np.arange(
            0.0,
            source_times[-1] + 1e-9,
            target_dt,
            dtype=np.float64,
        )

        qpos_out = np.zeros(
            (target_times.shape[0], reference_qpos.shape[1]), dtype=np.float32
        )
        qvel_out = np.zeros(
            (target_times.shape[0], reference_qvel.shape[1]), dtype=np.float32
        )

        for j in range(reference_qpos.shape[1]):
            qpos_out[:, j] = np.interp(
                target_times, source_times, reference_qpos[:, j]
            ).astype(np.float32)
        for j in range(reference_qvel.shape[1]):
            qvel_out[:, j] = np.interp(
                target_times, source_times, reference_qvel[:, j]
            ).astype(np.float32)

        root_quat = np.asarray(reference_qpos[:, 3:7], dtype=np.float64).copy()
        for i in range(1, root_quat.shape[0]):
            if np.dot(root_quat[i - 1], root_quat[i]) < 0.0:
                root_quat[i] *= -1.0

        quat_interp = np.zeros((target_times.shape[0], 4), dtype=np.float64)
        for k in range(4):
            quat_interp[:, k] = np.interp(target_times, source_times, root_quat[:, k])
        quat_norm = np.linalg.norm(quat_interp, axis=1, keepdims=True)
        quat_interp = quat_interp / np.clip(quat_norm, 1e-8, None)
        qpos_out[:, 3:7] = quat_interp.astype(np.float32)

        return qpos_out, qvel_out, target_fps

    def _resolve_reference_path(self) -> Path:
        deepmimic_cfg = self._config.deepmimic
        source = str(getattr(deepmimic_cfg, "reference_source", "auto")).lower()
        filename = str(
            getattr(deepmimic_cfg, "reference_filename", "walk1_subject1.npz")
        )
        repo_id = str(
            getattr(deepmimic_cfg, "reference_repo_id", "robfiras/loco-mujoco-datasets")
        )
        repo_type = str(getattr(deepmimic_cfg, "reference_repo_type", "dataset"))

        raw_path = Path(filename).expanduser()
        local_candidates: list[Path] = []
        if raw_path.is_absolute():
            local_candidates.append(raw_path)
        else:
            local_candidates.extend(
                [
                    Path.cwd() / raw_path,
                    Path(__file__).parent / raw_path,
                    raw_path,
                    Path(__file__).with_name(raw_path.name),
                ]
            )

        # Keep order and deduplicate.
        dedup_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in local_candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            dedup_candidates.append(candidate)

        if source not in {"auto", "local", "hf"}:
            raise ValueError(
                f"Invalid deepmimic.reference_source='{source}'. "
                "Expected one of {'auto', 'local', 'hf'}."
            )

        if source in {"auto", "local"}:
            for candidate in dedup_candidates:
                if candidate.exists():
                    return candidate

        if source in {"auto", "hf"}:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                if source == "hf":
                    raise ImportError(
                        "huggingface_hub is required when deepmimic.reference_source='hf'."
                    ) from exc
            else:
                try:
                    hf_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=repo_type,
                    )
                    return Path(hf_path)
                except Exception as exc:
                    if source == "hf":
                        raise RuntimeError(
                            "Failed to download reference from Hugging Face: "
                            f"repo_id={repo_id}, repo_type={repo_type}, filename={filename}"
                        ) from exc

        tried = ", ".join(str(p) for p in dedup_candidates)
        raise FileNotFoundError(
            "Could not resolve mocap reference file. "
            f"source={source}, filename={filename}, local_candidates=[{tried}]"
        )

    def _load_mocap_reference(self) -> None:
        reference_path = self._resolve_reference_path()
        print(f"[G1MocapTracking] Loading mocap reference: {reference_path}")
        with np.load(reference_path, allow_pickle=True) as npz_file:
            reference = np.asarray(npz_file["qpos"], dtype=np.float32)
            reference_qvel = np.asarray(npz_file["qvel"], dtype=np.float32)
            reference_joint_names = [str(n) for n in npz_file["joint_names"].tolist()]
            if reference.ndim != 2:
                raise ValueError(
                    f"Expected mocap qpos to be 2D, got shape {reference.shape}."
                )
            if (
                reference_qvel.ndim != 2
                or reference_qvel.shape[0] != reference.shape[0]  # type: ignore
            ):
                raise ValueError(
                    "Expected mocap qvel to align with qpos frames, got "
                    f"shape {reference_qvel.shape} for qpos shape {reference.shape}."
                )

            reference, reference_qvel = self._retarget_reference_to_model(
                reference, reference_qvel, reference_joint_names
            )

            source_fps = float(np.asarray(npz_file["frequency"]).item())
            reference, reference_qvel, target_fps = self._resample_reference_to_ctrl_dt(
                reference,
                reference_qvel,
                source_fps,
            )

            self._reference = jp.array(reference)
            self._reference_qvel = jp.array(reference_qvel)
            self._reference_fps = float(target_fps)
            self._reference_cmd = jp.array(reference_qvel[:, [0, 1, 5]])

    def _retarget_reference_to_model(
        self,
        reference_qpos: np.ndarray,
        reference_qvel: np.ndarray,
        reference_joint_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not reference_joint_names or reference_joint_names[0] != "root":
            raise ValueError(
                "Expected first mocap joint name to be 'root', got "
                f"{reference_joint_names[:1]}."
            )
        expected_joint_count = len(reference_joint_names) - 1
        if reference_qpos.shape[1] != 7 + expected_joint_count:
            raise ValueError(
                "Reference qpos width does not match joint_names. "
                f"qpos width={reference_qpos.shape[1]}, "
                f"expected={7 + expected_joint_count}."
            )
        if reference_qvel.shape[1] != 6 + expected_joint_count:
            raise ValueError(
                "Reference qvel width does not match joint_names. "
                f"qvel width={reference_qvel.shape[1]}, "
                f"expected={6 + expected_joint_count}."
            )

        num_frames = reference_qpos.shape[0]

        # Start from the model default pose so joints missing in the 23-DOF mocap
        # (waist_roll/pitch and wrist pitch/yaw) stay at a valid configuration.
        model_default_qpos = np.asarray(self._init_q, dtype=np.float32)
        retarget_qpos = np.repeat(model_default_qpos[None, :], num_frames, axis=0)
        retarget_qvel = np.zeros((num_frames, self.mj_model.nv), dtype=np.float32)

        # Always copy floating base state directly.
        retarget_qpos[:, :7] = reference_qpos[:, :7]
        retarget_qvel[:, :6] = reference_qvel[:, :6]

        mapped_joint_names: list[str] = []
        missing_joint_names: list[str] = []
        for src_joint_idx, joint_name in enumerate(reference_joint_names[1:]):
            src_qpos_idx = 7 + src_joint_idx
            src_qvel_idx = 6 + src_joint_idx
            try:
                joint = self._mj_model.joint(joint_name)
            except KeyError:
                missing_joint_names.append(joint_name)
                continue
            qpos_adr = int(joint.qposadr)
            dof_adr = int(joint.dofadr)
            retarget_qpos[:, qpos_adr] = reference_qpos[:, src_qpos_idx]
            retarget_qvel[:, dof_adr] = reference_qvel[:, src_qvel_idx]
            mapped_joint_names.append(joint_name)

        if missing_joint_names:
            raise ValueError(
                f"Mocap joint names missing in active model: {missing_joint_names}."
            )

        mapped_joint_names = sorted(set(mapped_joint_names))
        model_joint_names = sorted(
            str(self._mj_model.joint(i).name)
            for i in range(self._mj_model.njnt)
            if str(self._mj_model.joint(i).name) != "world"
        )
        extra_model_joints = sorted(set(model_joint_names) - set(mapped_joint_names))
        print(
            "[G1MocapTracking] Retarget mapping complete: "
            f"mapped={len(mapped_joint_names)} reference joints, "
            f"model_extra={extra_model_joints}"
        )

        return retarget_qpos, retarget_qvel

    def _reference_index(self, t: jax.Array, start_idx: jax.Array) -> jax.Array:
        idx = jp.int32(jp.floor(t * self._reference_fps)) + jp.int32(start_idx)
        return jp.mod(idx, self._reference.shape[0])

    def _phase_from_time(self, t: jax.Array) -> jax.Array:
        episode_duration = jp.asarray(
            self._config.episode_length * self.dt, dtype=jp.float32
        )
        progress = jp.mod(
            jp.asarray(t, dtype=jp.float32), episode_duration
        ) / jp.maximum(episode_duration, 1e-6)
        return jp.array([progress], dtype=jp.float32)

    def _command_from_reference_index(self, idx: jax.Array) -> jax.Array:
        return self._reference_cmd[idx]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        min_future_steps = 1000
        max_start_idx = max(0, int(self._reference.shape[0]) - min_future_steps - 1)
        rng, start_rng = jax.random.split(rng)
        if max_start_idx > 0:
            reference_start_idx = jax.random.randint(
                start_rng,
                shape=(),
                minval=0,
                maxval=max_start_idx + 1,
                dtype=jp.int32,
            )
        else:
            reference_start_idx = jp.int32(0)

        qpos = self._reference[reference_start_idx]
        qvel = self._reference_qvel[reference_start_idx]

        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_motor_targets,
        )

        ref_idx = self._reference_index(data.time, reference_start_idx)
        info = {
            "rng": rng,
            "reference_start_idx": reference_start_idx,
            "phase": self._phase_from_time(data.time),
            "command": self._command_from_reference_index(ref_idx),
            "last_action": jp.zeros((self.action_size,)),
            "last_qvel": data.qvel,
            "motor_targets": jp.zeros((self.action_size,)),
        }

        metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_metric_keys}

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        residual = (
            action
            * self._config.deepmimic.action_residual_scale
            * self._actuator_ctrl_half_range
        )
        motor_targets = jp.clip(
            self._default_motor_targets + residual,
            self._actuator_ctrl_min,
            self._actuator_ctrl_max,
        )

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            motor_targets,
            self.n_substeps,
        )

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )

        ref_idx = self._reference_index(data.time, state.info["reference_start_idx"])
        state.info["phase"] = self._phase_from_time(data.time)
        state.info["command"] = self._command_from_reference_index(ref_idx)
        state.info["motor_targets"] = motor_targets
        state.info["last_action"] = action

        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
            first_contact=jp.zeros((2,)),
            contact=contact,
        )
        reward = rewards["mimic_total"] * self.dt

        state.info["last_qvel"] = data.qvel

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        obs = self._get_obs(data, state.info, contact)
        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
        contact_termination = collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_foot_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._left_foot_geom_id,
            self._right_shin_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_shin_geom_id,
        )
        return (
            fall_termination
            | contact_termination
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    def _current_relative_site_features(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rpos, rquat, rvel = _calculate_relative_site_quantities(
            data,
            self._mimic_site_ids,
            self._mimic_site_body_ids,
            self._body_rootid,
            jp,
        )
        return rpos.ravel(), rquat.ravel(), rvel.ravel()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> jax.Array:
        ref_idx = self._reference_index(data.time, info["reference_start_idx"])

        qpos_obs = data.qpos[self._deepmimic_qpos_ind]
        qvel_obs = data.qvel[self._deepmimic_qvel_ind]
        rpos_obs, rquat_obs, rvel_obs = self._current_relative_site_features(data)

        ref_goal = jp.hstack(
            [
                self._reference_goal_qpos[ref_idx],
                self._reference_goal_qvel[ref_idx],
                self._reference_rpos[ref_idx],
                self._reference_rquat[ref_idx],
                self._reference_rvel[ref_idx],
            ]
        )

        obs = jp.hstack(
            [
                qpos_obs,
                qvel_obs,
                rpos_obs,
                rquat_obs,
                rvel_obs,
                ref_goal,
            ]
        )
        return obs

    def _out_of_bounds_action_cost(self, action: jax.Array) -> jax.Array:
        lower_bound = -jp.ones_like(action)
        upper_bound = jp.ones_like(action)
        lower_cost = jp.where(action < lower_bound, lower_bound - action, 0.0)
        upper_cost = jp.where(action > upper_bound, action - upper_bound, 0.0)
        action_dim = jp.asarray(action.shape[0], dtype=jp.float32)
        return jp.sum(jp.square(lower_cost + upper_cost)) / jp.maximum(action_dim, 1.0)

    def _qpos_distance(self, qpos: jax.Array, qpos_ref: jax.Array) -> jax.Array:
        qpos = qpos[self._deepmimic_qpos_ind]
        qpos_ref = qpos_ref[self._deepmimic_qpos_ind]

        qpos_nonquat = jp.take(qpos, self._qpos_nonquat_indices, axis=0)
        qpos_ref_nonquat = jp.take(qpos_ref, self._qpos_nonquat_indices, axis=0)
        qpos_dist = jp.mean(jp.square(qpos_nonquat - qpos_ref_nonquat))

        qpos_quat = jp.take(qpos, self._qpos_quat_indices, axis=0).reshape(-1, 4)
        qpos_quat_ref = jp.take(qpos_ref, self._qpos_quat_indices, axis=0).reshape(
            -1, 4
        )
        qpos_dist += jp.mean(
            _quaternion_angular_distance(
                _quat_scalarfirst_to_scalarlast(qpos_quat),
                _quat_scalarfirst_to_scalarlast(qpos_quat_ref),
                jp,
            )
        )
        return qpos_dist

    def _qvel_distance(self, qvel: jax.Array, qvel_ref: jax.Array) -> jax.Array:
        qvel = qvel[self._deepmimic_qvel_ind]
        qvel_ref = qvel_ref[self._deepmimic_qvel_ind]
        return jp.mean(jp.square(qvel - qvel_ref))

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics
        del done
        del first_contact
        del contact

        ref_idx = self._reference_index(data.time, info["reference_start_idx"])
        qpos_ref = self._reference[ref_idx]
        qvel_ref = self._reference_qvel[ref_idx]

        qpos_dist = self._qpos_distance(data.qpos, qpos_ref)
        qvel_dist = self._qvel_distance(data.qvel, qvel_ref)

        rpos, rquat, rvel = self._current_relative_site_features(data)
        rpos_dist = jp.mean(jp.square(rpos - self._reference_rpos[ref_idx]))
        rquat_dist = jp.mean(jp.square(rquat - self._reference_rquat[ref_idx]))
        rvel_dist = jp.mean(jp.square(rvel - self._reference_rvel[ref_idx]))

        qpos_reward = jp.exp(-self._config.deepmimic.reward_exponents.qpos * qpos_dist)
        qvel_reward = jp.exp(-self._config.deepmimic.reward_exponents.qvel * qvel_dist)
        rpos_reward = jp.exp(-self._config.deepmimic.reward_exponents.rpos * rpos_dist)
        rquat_reward = jp.exp(
            -self._config.deepmimic.reward_exponents.rquat * rquat_dist
        )
        rvel_reward = jp.exp(-self._config.deepmimic.reward_exponents.rvel * rvel_dist)

        action_out_of_bounds_coeff = (
            self._config.deepmimic.penalties.action_out_of_bounds
        )
        joint_acc_coeff = self._config.deepmimic.penalties.joint_acc
        joint_torque_coeff = self._config.deepmimic.penalties.joint_torque
        action_rate_coeff = self._config.deepmimic.penalties.action_rate

        if action_out_of_bounds_coeff > 0.0:
            out_of_bounds_reward = -self._out_of_bounds_action_cost(action)
        else:
            out_of_bounds_reward = jp.array(0.0)

        if joint_acc_coeff > 0.0:
            last_joint_vel = info["last_qvel"][self._nonfree_qvel_mask]
            joint_vel = data.qvel[self._nonfree_qvel_mask]
            acceleration_norm = jp.sum(
                jp.square((joint_vel - last_joint_vel) / self.dt)
            )
            joint_acc_reward = joint_acc_coeff * -acceleration_norm
        else:
            joint_acc_reward = jp.array(0.0)

        if joint_torque_coeff > 0.0:
            torque_norm = jp.sum(jp.square(data.actuator_force))
            joint_torque_reward = joint_torque_coeff * -torque_norm
        else:
            joint_torque_reward = jp.array(0.0)

        if action_rate_coeff > 0.0:
            action_rate_norm = jp.sum(jp.square(action - info["last_action"]))
            action_rate_reward = action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = jp.array(0.0)

        total_penalties = (
            action_out_of_bounds_coeff * out_of_bounds_reward
            + joint_acc_coeff * joint_acc_reward
            + joint_torque_coeff * joint_torque_reward
            + action_rate_coeff * action_rate_reward
        )
        total_penalties = jp.maximum(total_penalties, -1.0)

        mimic_total = (
            self._config.deepmimic.reward_sums.qpos * qpos_reward
            + self._config.deepmimic.reward_sums.qvel * qvel_reward
            + self._config.deepmimic.reward_sums.rpos * rpos_reward
            + self._config.deepmimic.reward_sums.rquat * rquat_reward
            + self._config.deepmimic.reward_sums.rvel * rvel_reward
            + total_penalties
        )
        mimic_total = jp.maximum(mimic_total, 0.0)
        mimic_total = jp.nan_to_num(mimic_total, nan=0.0)

        return {
            "qpos_reward": jp.nan_to_num(qpos_reward, nan=0.0),
            "qvel_reward": jp.nan_to_num(qvel_reward, nan=0.0),
            "rpos_reward": jp.nan_to_num(rpos_reward, nan=0.0),
            "rquat_reward": jp.nan_to_num(rquat_reward, nan=0.0),
            "rvel_reward": jp.nan_to_num(rvel_reward, nan=0.0),
            "out_of_bounds_reward": jp.nan_to_num(out_of_bounds_reward, nan=0.0),
            "joint_acc_reward": jp.nan_to_num(joint_acc_reward, nan=0.0),
            "joint_torque_reward": jp.nan_to_num(joint_torque_reward, nan=0.0),
            "action_rate_reward": jp.nan_to_num(action_rate_reward, nan=0.0),
            "total_penalties": jp.nan_to_num(total_penalties, nan=0.0),
            "mimic_total": jp.nan_to_num(mimic_total, nan=0.0),
        }

    def sample_command(self, rng: jax.Array) -> jax.Array:
        del rng
        return jp.zeros((3,))
