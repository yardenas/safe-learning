from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from hydrax.task_base import Task
from mujoco import mjx

from ss2r.benchmark_suites import randomization_fns


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


class MujocoPlaygroundDomainRandomizedTask(MujocoPlaygroundTask):
    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        model = _task_model(self)
        randomized = _randomize_model_with_task(
            model, rng, self._task_name, self._randomization_cfg
        )
        if randomized is None:
            return _randomize_friction(model, rng)
        return _model_update_dict(model, randomized)

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        model = _task_model(self)
        return _randomize_data_shift(model, data, rng)


def _get_mj_model(env: Any) -> Any:
    if hasattr(env, "mj_model"):
        return env.mj_model
    if hasattr(env, "_mj_model"):
        return env._mj_model
    raise ValueError("Environment does not expose an mj_model.")


def _task_model(task: MujocoPlaygroundTask) -> Any:
    return getattr(task, "model", task._mj_model)


def _randomize_model_with_task(
    model: Any, rng: jax.Array, task_name: str | None, cfg: Any | None
):
    if task_name is None or cfg is None:
        return None
    randomize_fn = randomization_fns.get(task_name)
    if randomize_fn is None:
        return None
    rng_in = rng
    if getattr(rng, "ndim", 0) == 1:
        rng_in = rng[None, :]
    randomized, in_axes, _ = randomize_fn(model, rng_in, cfg)
    if getattr(rng_in, "shape", (0,))[0] == 1:
        randomized = jax.tree_map(
            lambda x, ax: x[0] if ax == 0 else x, randomized, in_axes
        )
    return randomized


def _model_update_dict(base_model: Any, randomized_model: Any) -> Dict[str, jax.Array]:
    updates: Dict[str, jax.Array] = {}
    items = getattr(base_model, "__dict__", None)
    if items is None:
        raise ValueError("Cannot extract model fields for domain randomization.")
    for key, value in items.items():
        if isinstance(value, (jax.Array, np.ndarray)):
            updates[key] = getattr(randomized_model, key)
    return updates


def _randomize_friction(model: Any, rng: jax.Array) -> Dict[str, jax.Array]:
    n_geoms = model.geom_friction.shape[0]
    multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
    new_frictions = model.geom_friction.at[:, 0].set(
        model.geom_friction[:, 0] * multiplier
    )
    return {"geom_friction": new_frictions}


def _randomize_data_shift(
    model: Any, data: mjx.Data, rng: jax.Array
) -> Dict[str, jax.Array]:
    shift = 0.005 * jax.random.normal(rng, (model.nq,))
    return {"qpos": data.qpos + shift}
