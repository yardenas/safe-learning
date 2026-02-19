import sys
from pathlib import Path

__version__ = "1.0.1"

# Alias this vendored package to the top-level name expected by upstream imports.
sys.modules.setdefault("loco_mujoco", sys.modules[__name__])

PATH_TO_MODELS = Path(__file__).resolve().parent / "models"
PATH_TO_VARIABLES = Path(__file__).resolve().parent / "LOCOMUJOCO_VARIABLES.yaml"
PATH_TO_SMPL_ROBOT_CONF = Path(__file__).resolve().parent / "smpl" / "robot_confs"

from .core import Mjx, Mujoco
from .environments import LocoEnv
from .task_factories import ImitationFactory, TaskFactory


def get_registered_envs():
    return LocoEnv.registered_envs
