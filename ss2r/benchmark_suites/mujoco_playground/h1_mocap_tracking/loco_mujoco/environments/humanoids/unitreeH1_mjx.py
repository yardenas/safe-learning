import mujoco
from mujoco import MjSpec

from .unitreeH1 import UnitreeH1, _find_body


class MjxUnitreeH1(UnitreeH1):
    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(
                iterations=2,
                ls_iterations=4,
                disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            )
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            **kwargs,
        )

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """

        # --- 1. remove old feet and add new ones ---
        # remove original foot meshes
        for g in spec.geoms:
            if g.name in ["right_foot", "left_foot"]:
                g.delete()

        # --- 2. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 3. add primitive foot shapes ---
        back_foot_attr = dict(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            quat=[1.0, 0.0, 1.0, 0.0],
            pos=[-0.03, 0.0, -0.05],
            size=[0.015, 0.025, 0.0],
            rgba=[1.0, 1.0, 1.0, 0.2],
            contype=0,
            conaffinity=0,
        )
        front_foot_attr = dict(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            quat=[1.0, 1.0, 0.0, 0.0],
            pos=[0.15, 0.0, -0.054],
            size=[0.02, 0.025, 0.0],
            rgba=[1.0, 1.0, 1.0, 0.2],
            contype=0,
            conaffinity=0,
        )

        r_foot_b = _find_body(spec, "right_ankle_link")
        r_foot_b.add_geom(name="right_foot1", **back_foot_attr)
        r_foot_b.add_geom(name="right_foot2", **front_foot_attr)

        l_foot_b = _find_body(spec, "left_ankle_link")
        l_foot_b.add_geom(name="left_foot1", **back_foot_attr)
        l_foot_b.add_geom(name="left_foot2", **front_foot_attr)

        # --- 4. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="right_foot1")
        spec.add_pair(geomname1="floor", geomname2="right_foot2")
        spec.add_pair(geomname1="floor", geomname2="left_foot1")
        spec.add_pair(geomname1="floor", geomname2="left_foot2")
        spec.add_pair(geomname1="right_foot1", geomname2="left_foot1")
        spec.add_pair(geomname1="right_foot1", geomname2="left_foot2")
        spec.add_pair(geomname1="right_foot2", geomname2="left_foot1")
        spec.add_pair(geomname1="right_foot2", geomname2="left_foot2")

        return spec
