from .unitreeG1 import UnitreeG1
from .unitreeG1_mjx import MjxUnitreeG1

UnitreeG1.register()
MjxUnitreeG1.register()

try:
    from gymnasium import register
except Exception:
    register = None

if register is not None:
    register(
        "LocoMujoco",
        entry_point="loco_mujoco.core.wrappers.gymnasium:GymnasiumWrapper",
    )
