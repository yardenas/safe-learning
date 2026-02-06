from typing import Any

import jax
import jax.numpy as jnp
from hydrax.task_base import Task
from mujoco import mjx
from mujoco_playground._src import mjx_env


class MujocoPlaygroundTask(Task):
    def __init__(
        self,
        env: mjx_env.MjxEnv,
        dt: float,
        *,
        running_cost_scale: float = 1.0,
        terminal_cost_scale: float = 0.0,
        task_name: str | None = None,
        randomization_cfg: Any | None = None,
    ) -> None:
        self.env = env
        self._task_name = task_name
        self._randomization_cfg = randomization_cfg
        self._running_cost_scale = float(running_cost_scale)
        self._terminal_cost_scale = float(terminal_cost_scale)
        mj_model = env.mj_model
        super().__init__(mj_model)
        self._mj_model = mj_model
        self.dt = dt

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        del u
        if isinstance(x, dict):
            reward = x.get("reward", None)
            if reward is None:
                raise ValueError(
                    "Task payload missing reward; ensure env.step provides reward."
                )
            return -reward * self._running_cost_scale
        if hasattr(x, "reward"):
            return -x.reward * self._running_cost_scale
        raise ValueError(
            "Task running_cost expects reward in state payload or mjx.State."
        )

    def terminal_cost(self, x: jax.Array) -> float:
        del x
        return jnp.asarray(0.0) * self._terminal_cost_scale
