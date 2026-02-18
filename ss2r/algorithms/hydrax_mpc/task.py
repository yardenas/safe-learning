from typing import Any

import jax
from mujoco import mjx
from mujoco_playground._src import mjx_env


class MujocoPlaygroundTask:
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
            return -reward
        if hasattr(x, "reward"):
            return -x.reward
        raise ValueError(
            "Task running_cost expects reward in state payload or mjx.State."
        )

    def terminal_cost(self, x: jax.Array) -> float:
        del x
        return 0.0
