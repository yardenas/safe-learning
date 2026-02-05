from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from hydrax.task_base import Task
from mujoco import mjx


class MujocoPlaygroundTask(Task):
    def __init__(
        self,
        env: Any,
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
        mj_model = _get_mj_model(env)
        super().__init__(mj_model)
        self._mj_model = mj_model

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


def _get_mj_model(env: Any) -> Any:
    if hasattr(env, "mj_model"):
        return env.mj_model
    if hasattr(env, "_mj_model"):
        return env._mj_model
    raise ValueError("Environment does not expose an mj_model.")
