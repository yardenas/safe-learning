from typing import Any

from hydrax.task_base import Task
from mujoco_playground._src import mjx_env


class MujocoPlaygroundTask(Task):
    def __init__(
        self,
        env: mjx_env.MjxEnv,
        dt: float,
        *,
        randomization_cfg: Any | None = None,
    ) -> None:
        self.env = env
        self._randomization_cfg = randomization_cfg
        mj_model = env.mj_model
        super().__init__(mj_model)
        self._mj_model = mj_model
        self.dt = dt
