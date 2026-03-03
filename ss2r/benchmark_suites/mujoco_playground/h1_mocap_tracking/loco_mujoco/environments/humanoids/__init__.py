from .unitreeH1 import UnitreeH1
from .unitreeH1_mjx import MjxUnitreeH1

UnitreeH1.register()
MjxUnitreeH1.register()

try:
    from gymnasium import register
except Exception:
    register = None

if register is not None:
    register(
        "LocoMujoco",
        entry_point="ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.core.wrappers.gymnasium:GymnasiumWrapper",
    )
