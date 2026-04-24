"""Interactive MuJoCo viewer for H1MocapTracking.

Usage from Python:
    from ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.viewer import run_viewer
    run_viewer()
"""

import argparse
import ast
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint as sac_checkpoint
from mujoco import MjData, mjx
from orbax import checkpoint as ocp

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.h1_mocap_env import (
    H1MocapTracking,
)
from ss2r.benchmark_suites.mujoco_playground.h1_mocap_tracking.loco_mujoco.trajectory import (
    Trajectory,
)

DEFAULT_WANDB_RUN_ID = "gytlhnyk"
SUPPORTED_ACTION_MODES = ("reference", "replay", "zero", "random", "policy")


def _random_action(rng: jax.Array, action_size: int) -> tuple[jax.Array, jax.Array]:
    rng, key = jax.random.split(rng)
    if action_size == 0:
        return rng, jp.zeros((0,))
    action = jax.random.uniform(key, (action_size,), minval=-1.0, maxval=1.0)
    return rng, action


def _reference_action(env: H1MocapTracking, state: Any) -> jax.Array:
    del state
    # Actions are residuals around the mocap reference target.
    return jp.zeros((env.action_size,))


@dataclass(frozen=True)
class _ReplayReference:
    name: str
    source: str
    qpos: jax.Array
    qvel: jax.Array

    @property
    def n_frames(self) -> int:
        return int(self.qpos.shape[0])


@dataclass(frozen=True)
class _ViewerRuntime:
    env: H1MocapTracking
    replay_reference: _ReplayReference | None
    reset_fn: Callable[[jax.Array], Any]
    step_fn: Callable[[Any, jax.Array], Any]
    policy_action_fn: Callable[[Any, jax.Array], jax.Array] | None


@dataclass(frozen=True)
class _RolloutSequenceResult:
    states: list[Any]
    done_step: int | None
    min_root_height: float
    noise_std: float
    fell: bool


def _normalize_reference_name(reference_name: str) -> str:
    name = reference_name.strip().replace("\\", "/")
    if not name:
        raise ValueError("Empty mocap reference name.")
    if name.endswith(".csv"):
        name = name[:-4]
    if not name.endswith(".npz"):
        name = f"{name}.npz"
    return name


def _repo_filename_from_reference_name(reference_name: str, reference_dir: str) -> str:
    normalized = _normalize_reference_name(reference_name)
    if "/" in normalized:
        return normalized
    base_dir = reference_dir.strip().strip("/")
    return f"{base_dir}/{normalized}" if base_dir else normalized


def _resolve_local_reference_path(
    reference_name: str, reference_dir: str
) -> Path | None:
    module_dir = Path(__file__).resolve().parent
    normalized = _normalize_reference_name(reference_name)
    repo_filename = _repo_filename_from_reference_name(reference_name, reference_dir)
    candidates = [
        Path(normalized),
        Path(repo_filename),
        module_dir / normalized,
        module_dir / repo_filename,
    ]
    raw_path = Path(reference_name)
    if raw_path.is_absolute():
        candidates.insert(0, Path(_normalize_reference_name(str(raw_path))))
        candidates.insert(0, raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _download_hf_reference(
    *,
    reference_name: str,
    reference_dir: str,
    repo_id: str,
    repo_type: str,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise ImportError(
            "huggingface_hub is required to fetch replay mocap files from HF."
        ) from exc

    filename = _repo_filename_from_reference_name(reference_name, reference_dir)
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    return Path(file_path).resolve()


def _load_replay_reference(env: H1MocapTracking) -> _ReplayReference:
    loco_cfg = getattr(env._config, "loco", None)
    if loco_cfg is None:
        raise ValueError("Missing `loco` config for replay mode.")

    reference_name = str(getattr(loco_cfg, "dataset_name", "dance1_subject3"))
    source_mode = str(getattr(loco_cfg, "reference_source", "hf")).strip().lower()
    reference_dir = str(
        getattr(loco_cfg, "reference_dir", "Lafan1/mocap/UnitreeG1")
    ).strip()
    repo_id = str(
        getattr(loco_cfg, "reference_repo_id", "robfiras/loco-mujoco-datasets")
    ).strip()
    repo_type = str(getattr(loco_cfg, "reference_repo_type", "dataset")).strip()

    if source_mode not in {"auto", "local", "hf"}:
        raise ValueError(
            f"Unsupported loco.reference_source='{source_mode}'. Use auto/local/hf."
        )

    file_path: Path | None
    if source_mode == "local":
        file_path = _resolve_local_reference_path(reference_name, reference_dir)
        if file_path is None:
            raise FileNotFoundError(
                f"Could not find local mocap reference '{reference_name}'."
            )
    elif source_mode == "hf":
        file_path = _download_hf_reference(
            reference_name=reference_name,
            reference_dir=reference_dir,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        file_path = _resolve_local_reference_path(reference_name, reference_dir)
        if file_path is None:
            file_path = _download_hf_reference(
                reference_name=reference_name,
                reference_dir=reference_dir,
                repo_id=repo_id,
                repo_type=repo_type,
            )

    traj = Trajectory.load(str(file_path), backend=np)
    qpos = jp.asarray(np.asarray(traj.data.qpos), dtype=jp.float32)
    qvel = jp.asarray(np.asarray(traj.data.qvel), dtype=jp.float32)
    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError(f"Invalid reference trajectory shape in {file_path}.")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError(f"Mismatched qpos/qvel length in {file_path}.")
    if qpos.shape[0] <= 0:
        raise ValueError(f"Empty reference trajectory in {file_path}.")

    loaded_from = str(file_path)
    return _ReplayReference(
        name=reference_name, source=loaded_from, qpos=qpos, qvel=qvel
    )


@dataclass(frozen=True)
class _SacRunConfig:
    policy_hidden_layer_sizes: tuple[int, ...]
    value_hidden_layer_sizes: tuple[int, ...]
    activation: str
    normalize_observations: bool
    use_bro: bool
    n_critics: int
    n_heads: int
    policy_obs_key: str
    value_obs_key: str


def _cfg_get(config: Mapping[str, Any], path: str, default: Any) -> Any:
    if path in config:
        return config[path]
    current: Any = config
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def _as_int_tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return fallback


def _flatten_nested_dict(
    source: Mapping[str, Any],
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if out is None:
        out = {}
    for key, value in source.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_nested_dict(value, prefix=path, out=out)
        else:
            out[path] = value
    return out


def _extract_sac_run_config(run_config: Mapping[str, Any]) -> _SacRunConfig:
    policy_privileged = bool(_cfg_get(run_config, "training.policy_privileged", False))
    value_privileged = bool(_cfg_get(run_config, "training.value_privileged", False))
    return _SacRunConfig(
        policy_hidden_layer_sizes=_as_int_tuple(
            _cfg_get(run_config, "agent.policy_hidden_layer_sizes", [128, 128]),
            (128, 128),
        ),
        value_hidden_layer_sizes=_as_int_tuple(
            _cfg_get(run_config, "agent.value_hidden_layer_sizes", [512, 512]),
            (512, 512),
        ),
        activation=str(_cfg_get(run_config, "agent.activation", "swish")),
        normalize_observations=bool(
            _cfg_get(run_config, "agent.normalize_observations", True)
        ),
        use_bro=bool(_cfg_get(run_config, "agent.use_bro", True)),
        n_critics=int(_cfg_get(run_config, "agent.n_critics", 2)),
        n_heads=int(_cfg_get(run_config, "agent.n_heads", 1)),
        policy_obs_key="privileged_state" if policy_privileged else "state",
        value_obs_key="privileged_state" if value_privileged else "state",
    )


def _extract_env_overrides_from_run_config(
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    task_params = _cfg_get(run_config, "environment.task_params", {})
    if not isinstance(task_params, Mapping):
        return {}
    flattened = _flatten_nested_dict(task_params)
    allowed_roots = {"ctrl_dt", "sim_dt", "episode_length"}
    return {
        k: v
        for k, v in flattened.items()
        if k in allowed_roots or k.startswith("loco.")
    }


def _download_wandb_checkpoint_and_config(
    run_id: str,
    *,
    entity: str | None,
    project: str,
    timeout: int | None = None,
) -> tuple[str, dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise ImportError("wandb is required for action_mode='policy'.") from exc

    api_kwargs = {"timeout": timeout} if timeout is not None else {}
    api = (
        wandb.Api(overrides={"entity": entity}, **api_kwargs)
        if entity
        else wandb.Api(**api_kwargs)
    )
    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(run_path)
    run_config = dict(run.config)

    artifact_path = (
        f"{entity}/{project}/checkpoint:{run_id}"
        if entity
        else f"{project}/checkpoint:{run_id}"
    )
    artifact = api.artifact(artifact_path)
    download_dir = Path.cwd() / "ckpt" / f"{run_id}_step_latest"
    checkpoint_path = artifact.download(str(download_dir))
    return checkpoint_path, run_config


def _build_sac_policy_action_fn(
    env: H1MocapTracking,
    checkpoint_path: str,
    run_config: Mapping[str, Any],
    *,
    deterministic: bool,
) -> Callable[[Any, jax.Array], jax.Array]:
    sac_cfg = _extract_sac_run_config(run_config)
    try:
        activation_fn = getattr(jax.nn, sac_cfg.activation)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown SAC activation '{sac_cfg.activation}' in W&B run config."
        ) from exc

    normalize_fn = (
        running_statistics.normalize
        if sac_cfg.normalize_observations
        else (lambda obs, _: obs)
    )
    sac_network = sac_networks.make_sac_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=normalize_fn,
        policy_hidden_layer_sizes=sac_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=sac_cfg.value_hidden_layer_sizes,
        activation=activation_fn,
        value_obs_key=sac_cfg.value_obs_key,
        policy_obs_key=sac_cfg.policy_obs_key,
        use_bro=sac_cfg.use_bro,
        n_critics=sac_cfg.n_critics,
        n_heads=sac_cfg.n_heads,
    )
    params = _restore_sac_checkpoint(
        checkpoint_path=checkpoint_path,
        env=env,
        sac_network=sac_network,
        run_config=run_config,
    )
    inference_fn = sac_networks.make_inference_fn(sac_network)(
        (params[0], params[1]),
        deterministic=deterministic,
    )

    @jax.jit
    def _policy_action(obs: Any, key: jax.Array) -> jax.Array:
        batched_obs = jax.tree.map(lambda x: jp.expand_dims(x, axis=0), obs)
        action, _ = inference_fn(batched_obs, key)
        return action[0]

    return _policy_action


def _build_restore_item_from_metadata(checkpoint_path: str) -> Any:
    metadata_path = Path(checkpoint_path) / "_METADATA"
    metadata = json.loads(metadata_path.read_text())
    tree_metadata = metadata.get("tree_metadata")
    if not isinstance(tree_metadata, dict):
        custom = metadata.get("custom")
        if isinstance(custom, dict):
            tree_metadata = custom.get("tree_metadata")
    if not isinstance(tree_metadata, dict):
        raise ValueError(
            f"Could not find tree_metadata in checkpoint metadata: {metadata_path}"
        )

    root: dict[Any, Any] = {}
    for entry in tree_metadata.values():
        key_path: list[Any] = []
        for key_entry in entry["key_metadata"]:
            key = key_entry["key"]
            if key_entry.get("key_type", 2) == 1:
                key = int(key)
            key_path.append(key)

        cursor = root
        for part in key_path[:-1]:
            cursor = cursor.setdefault(part, {})
        # Leaf value is only a structure placeholder.
        cursor[key_path[-1]] = np.array(0.0, dtype=np.float32)

    def _to_container(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        converted = {k: _to_container(v) for k, v in node.items()}
        if converted and all(isinstance(k, int) for k in converted):
            keys = sorted(converted)
            if keys == list(range(len(keys))):
                return [converted[i] for i in keys]
        return converted

    return _to_container(root)


def _restore_sac_checkpoint_numpy(checkpoint_path: str) -> Any:
    restore_item = _build_restore_item_from_metadata(checkpoint_path)
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), restore_item
    )
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(
        str(Path(checkpoint_path).resolve()),
        args=ocp.args.PyTreeRestore(item=restore_item, restore_args=restore_args),
    )
    restored = jax.tree.map(
        lambda x: jp.array(x) if isinstance(x, np.ndarray) else x, restored
    )
    if isinstance(restored, list) and restored and isinstance(restored[0], dict):
        restored[0] = running_statistics.RunningStatisticsState(**restored[0])
    return restored


def _restore_sac_checkpoint(
    checkpoint_path: str,
    env: H1MocapTracking,
    sac_network: sac_networks.SafeSACNetworks,
    run_config: Mapping[str, Any],
) -> Any:
    del env, sac_network, run_config
    try:
        return sac_checkpoint.load(checkpoint_path)
    except Exception as load_error:
        print(
            "[viewer] brax sac checkpoint.load failed; "
            "falling back to metadata-driven numpy restore. "
            f"error={load_error}"
        )
    return _restore_sac_checkpoint_numpy(checkpoint_path)


def _reference_replay_step(
    env: H1MocapTracking,
    state: Any,
    reference: _ReplayReference,
    frame_index: int,
) -> tuple[Any, int]:
    """Override simulator state from an explicit mocap trajectory frame."""
    idx = frame_index % reference.n_frames
    q_ref = reference.qpos[idx]
    qvel_ref = reference.qvel[idx]
    ctrl_ref = q_ref[7 : 7 + env.action_size]

    data = state.data.replace(
        qpos=q_ref,
        qvel=qvel_ref,
        ctrl=ctrl_ref,
        time=state.data.time + env.dt,
    )
    data = mjx.forward(env.mjx_model, data)

    done_bool = (frame_index + 1) >= reference.n_frames
    next_frame_index = 0 if done_bool else frame_index + 1

    info = dict(state.info)
    info["replay_frame"] = jp.asarray(idx, dtype=jp.int32)

    next_state = state.replace(
        data=data,
        reward=jp.zeros_like(state.reward),
        done=jp.asarray(done_bool, dtype=state.done.dtype),
        info=info,
    )
    return next_state, next_frame_index


def _mocap_progress(
    state: Any,
    reference: _ReplayReference | None,
) -> tuple[int, int, int, float]:
    if reference is None:
        return 0, 0, 1, 0.0
    curr_idx = int(np.asarray(state.info.get("replay_frame", 0)).item())
    n_frames = int(reference.n_frames)
    rel_idx = curr_idx % n_frames
    frac = float(rel_idx) / float(max(n_frames - 1, 1))
    return curr_idx, rel_idx, n_frames, frac


def _progress_bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def _print_status(
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    replay_reference: _ReplayReference | None,
) -> None:
    reward = float(np.asarray(state.reward).item())
    done = bool(np.asarray(state.done).item())
    line = (
        f"[ep={episode_idx} step={episode_step} global={step_count}] "
        f"r={reward:+.5f} R={episode_return:+.3f} done={done}"
    )
    if replay_reference is not None:
        curr_idx, rel_idx, n_frames, frac = _mocap_progress(state, replay_reference)
        bar = _progress_bar(frac)
        line += (
            f" mocap={curr_idx}/{n_frames - 1} rel={rel_idx} "
            f"{frac * 100:5.1f}% [{bar}]"
        )
    print(line)


def _add_text(
    data: MjData,
    viewer: mujoco.viewer.Handle,
    text: str,
    z_offset: float,
) -> None:
    """Adds an invisible label geom to user_scn."""
    if viewer.user_scn is None:
        return
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),
        pos=np.array(data.qpos[:3]) + np.array([0.0, 0.0, z_offset]),
        mat=np.eye(3).reshape(-1),
        # Keep label visible; full transparency can hide text in some builds.
        rgba=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    geom.label = text
    viewer.user_scn.ngeom += 1


def _hud_lines(
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    replay_reference: _ReplayReference | None,
) -> list[str]:
    reward = float(np.asarray(state.reward).item())
    done = bool(np.asarray(state.done).item())
    sim_time = float(np.asarray(state.data.time).item())
    lines = [
        (
            f"ep={episode_idx} step={episode_step} g={step_count} "
            f"r={reward:+.4f} R={episode_return:+.2f} done={int(done)}"
        ),
        f"time={sim_time:.4f}s",
    ]
    if replay_reference is not None:
        curr_idx, rel_idx, n_frames, frac = _mocap_progress(state, replay_reference)
        bar = _progress_bar(frac, width=16)
        lines.append(
            f"mocap {curr_idx}/{n_frames - 1} rel={rel_idx} {frac * 100:4.1f}% [{bar}]"
        )
    return lines


def _update_hud_labels(
    viewer: mujoco.viewer.Handle,
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    replay_reference: _ReplayReference | None,
) -> None:
    if viewer.user_scn is None:
        return
    # Viewer runs on a separate thread; mutate user_scn under viewer lock.
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        lines = _hud_lines(
            state=state,
            step_count=step_count,
            episode_idx=episode_idx,
            episode_step=episode_step,
            episode_return=episode_return,
            replay_reference=replay_reference,
        )
        for i, line in enumerate(lines):
            base_z = 0.60
            line_spacing = 0.10
            _add_text(
                state.data,
                viewer,
                line,
                z_offset=base_z + line_spacing * (len(lines) - 1 - i),
            )


def _build_runtime(
    *,
    action_mode: str,
    wandb_run_id: str | None,
    wandb_entity: str | None,
    wandb_project: str,
    policy_deterministic: bool,
    config_overrides: dict[str, Any] | None,
) -> _ViewerRuntime:
    if action_mode not in SUPPORTED_ACTION_MODES:
        raise ValueError(
            f"Unsupported action_mode={action_mode}. "
            f"Expected one of {set(SUPPORTED_ACTION_MODES)}."
        )

    run_config: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    effective_config_overrides = dict(config_overrides or {})
    resolved_wandb_run_id = wandb_run_id or DEFAULT_WANDB_RUN_ID
    if action_mode == "policy":
        checkpoint_path, run_config = _download_wandb_checkpoint_and_config(
            resolved_wandb_run_id,
            entity=wandb_entity,
            project=wandb_project,
        )
        wandb_env_overrides = _extract_env_overrides_from_run_config(run_config)
        effective_config_overrides = {
            **wandb_env_overrides,
            **effective_config_overrides,
        }
        print(
            f"[viewer] loaded wandb run={resolved_wandb_run_id} "
            f"(entity={wandb_entity or '<default>'}, project={wandb_project})"
        )
        print(f"[viewer] checkpoint_path={checkpoint_path}")

    env = H1MocapTracking(
        config_overrides=effective_config_overrides
        if effective_config_overrides
        else None
    )
    replay_reference = _load_replay_reference(env) if action_mode == "replay" else None
    if replay_reference is not None:
        print(
            f"[viewer] replay reference='{replay_reference.name}' "
            f"frames={replay_reference.n_frames} source={replay_reference.source}"
        )

    policy_action_fn = None
    if action_mode == "policy":
        assert checkpoint_path is not None
        assert run_config is not None
        policy_action_fn = _build_sac_policy_action_fn(
            env=env,
            checkpoint_path=checkpoint_path,
            run_config=run_config,
            deterministic=policy_deterministic,
        )

    return _ViewerRuntime(
        env=env,
        replay_reference=replay_reference,
        reset_fn=jax.jit(env.reset),
        step_fn=jax.jit(env.step),
        policy_action_fn=policy_action_fn,
    )


def _root_height(state: Any) -> float:
    return float(np.asarray(state.data.qpos[2]).item())


def _state_done(state: Any) -> bool:
    return bool(np.asarray(state.done).item())


def _rollout_sequence(
    *,
    runtime: _ViewerRuntime,
    initial_state: Any,
    rng: jax.Array,
    action_mode: str,
    num_steps: int,
    noise_std: float = 0.0,
    noise_start_step: int = 0,
    failure_height_threshold: float = 0.55,
) -> _RolloutSequenceResult:
    state = initial_state
    states = [state]
    replay_frame_index = 0
    zero_action = jp.zeros((runtime.env.action_size,))
    min_root_height = _root_height(state)
    done_step: int | None = None
    fell = False

    for step in range(num_steps):
        if action_mode == "replay":
            assert runtime.replay_reference is not None
            state, replay_frame_index = _reference_replay_step(
                runtime.env,
                state,
                runtime.replay_reference,
                replay_frame_index,
            )
        else:
            if action_mode == "zero":
                action = zero_action
            elif action_mode == "random":
                rng, action = _random_action(rng, runtime.env.action_size)
            elif action_mode == "policy":
                assert runtime.policy_action_fn is not None
                rng, policy_key = jax.random.split(rng)
                action = runtime.policy_action_fn(state.obs, policy_key)
            else:
                action = _reference_action(runtime.env, state)

            if noise_std > 0.0 and step >= noise_start_step:
                rng, noise_key = jax.random.split(rng)
                action_noise = noise_std * jax.random.normal(noise_key, action.shape)
                action = jp.clip(action + action_noise, -1.0, 1.0)

            state = runtime.step_fn(state, action)

        states.append(state)
        min_root_height = min(min_root_height, _root_height(state))
        fell = _state_done(state) or min_root_height < failure_height_threshold
        if fell:
            done_step = step + 1
            break

    return _RolloutSequenceResult(
        states=states,
        done_step=done_step,
        min_root_height=min_root_height,
        noise_std=noise_std,
        fell=fell,
    )


def _find_failure_sequence(
    *,
    runtime: _ViewerRuntime,
    initial_state: Any,
    rng: jax.Array,
    action_mode: str,
    num_steps: int,
    base_noise_std: float,
    noise_start_step: int,
    failure_height_threshold: float,
    max_attempts: int,
) -> _RolloutSequenceResult:
    result: _RolloutSequenceResult | None = None
    for attempt in range(max_attempts):
        attempt_rng = jax.random.fold_in(rng, attempt)
        attempt_noise_std = base_noise_std * float(attempt + 1)
        result = _rollout_sequence(
            runtime=runtime,
            initial_state=initial_state,
            rng=attempt_rng,
            action_mode=action_mode,
            num_steps=num_steps,
            noise_std=attempt_noise_std,
            noise_start_step=noise_start_step,
            failure_height_threshold=failure_height_threshold,
        )
        if result.fell:
            print(
                f"[viewer] found failure sequence on attempt={attempt + 1} "
                f"noise_std={attempt_noise_std:.3f} step={result.done_step}"
            )
            return result

    assert result is not None
    print(
        "[viewer] warning: noisy rollout did not satisfy the failure criterion; "
        f"using last attempt with noise_std={result.noise_std:.3f}"
    )
    return result


def _parse_camera(camera: str | None) -> int | str:
    if camera is None:
        return -1
    camera = camera.strip()
    if not camera:
        return -1
    try:
        return int(camera)
    except ValueError:
        return camera


def _sequence_steps(num_steps: int, num_frames: int) -> list[int]:
    if num_frames <= 1:
        return [0]
    raw_steps = np.linspace(0, num_steps, num_frames)
    return [int(round(step)) for step in raw_steps]


def _extract_traj_state_info(state: Any) -> dict[str, int] | None:
    loco_state = state.info.get("_loco_state")
    if loco_state is None:
        return None
    additional_carry = getattr(loco_state, "additional_carry", None)
    traj_state = getattr(additional_carry, "traj_state", None)
    if traj_state is None:
        return None
    return {
        "traj_no": int(np.asarray(traj_state.traj_no).item()),
        "subtraj_step_no": int(np.asarray(traj_state.subtraj_step_no).item()),
        "subtraj_step_no_init": int(np.asarray(traj_state.subtraj_step_no_init).item()),
    }


def _clamp_render_size(
    mj_model: mujoco.MjModel, width: int, height: int
) -> tuple[int, int]:
    max_width = int(mj_model.vis.global_.offwidth)
    max_height = int(mj_model.vis.global_.offheight)
    render_width = min(width, max_width)
    render_height = min(height, max_height)
    if render_width != width or render_height != height:
        print(
            "[viewer] requested render size "
            f"{width}x{height} exceeds offscreen framebuffer "
            f"{max_width}x{max_height}; using {render_width}x{render_height} instead"
        )
    return render_width, render_height


def _render_states_with_masks(
    *,
    mj_model: mujoco.MjModel,
    states: list[Any],
    state_steps: list[int],
    width: int,
    height: int,
    camera: str | None,
    transparent_background: bool,
    transparent_ground: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    crop_masks: list[np.ndarray] = []
    camera_arg = _parse_camera(camera)
    mj_data = MjData(mj_model)
    floor_geom_id = -1
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    render_width, render_height = _clamp_render_size(mj_model, width, height)
    with mujoco.Renderer(
        mj_model, height=render_height, width=render_width
    ) as renderer:
        for step in state_steps:
            state = states[min(max(step, 0), len(states) - 1)]
            mjx.get_data_into(mj_data, mj_model, state.data)
            renderer.update_scene(mj_data, camera=camera_arg)
            rgb_frame = np.asarray(renderer.render()).copy()
            renderer.enable_segmentation_rendering()
            segmentation = np.asarray(renderer.render()).copy()
            renderer.disable_segmentation_rendering()
            background_mask = np.logical_and(
                segmentation[..., 0] == -1,
                segmentation[..., 1] == -1,
            )
            ground_mask = np.zeros(segmentation.shape[:2], dtype=bool)
            if floor_geom_id != -1:
                ground_mask = np.logical_and(
                    segmentation[..., 0] == floor_geom_id,
                    segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
                )
            crop_mask = np.logical_and(~background_mask, ~ground_mask)
            crop_masks.append(crop_mask)
            if transparent_background or transparent_ground:
                alpha_mask = np.ones(segmentation.shape[:2], dtype=bool)
                if transparent_background:
                    alpha_mask &= ~background_mask
                if transparent_ground and floor_geom_id != -1:
                    alpha_mask &= ~ground_mask
                alpha = np.where(alpha_mask, 255, 0).astype(np.uint8)
                frame = np.concatenate([rgb_frame, alpha[..., None]], axis=-1)
            else:
                frame = rgb_frame
            frames.append(frame)
    return frames, crop_masks


def _render_states(
    *,
    mj_model: mujoco.MjModel,
    states: list[Any],
    state_steps: list[int],
    width: int,
    height: int,
    camera: str | None,
    transparent_background: bool,
    transparent_ground: bool = False,
) -> list[np.ndarray]:
    frames, _ = _render_states_with_masks(
        mj_model=mj_model,
        states=states,
        state_steps=state_steps,
        width=width,
        height=height,
        camera=camera,
        transparent_background=transparent_background,
        transparent_ground=transparent_ground,
    )
    return frames


def _compute_crop_bounds(
    masks: list[np.ndarray], frame_width: int, frame_height: int, padding: float
) -> tuple[int, int, int, int]:
    if not masks:
        return (0, frame_height, 0, frame_width)
    combined_mask = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(combined_mask)
    if ys.size == 0 or xs.size == 0:
        return (0, frame_height, 0, frame_width)

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    box_height = max(1, y1 - y0)
    box_width = max(1, x1 - x0)
    y_pad = max(1, int(round(box_height * padding)))
    x_pad = max(1, int(round(box_width * padding)))
    y0 = max(0, y0 - y_pad)
    y1 = min(frame_height, y1 + y_pad)
    x0 = max(0, x0 - x_pad)
    x1 = min(frame_width, x1 + x_pad)

    target_aspect = frame_width / max(frame_height, 1)
    crop_height = max(1, y1 - y0)
    crop_width = max(1, x1 - x0)
    crop_aspect = crop_width / crop_height

    if crop_aspect < target_aspect:
        desired_width = int(round(crop_height * target_aspect))
        width_extra = max(0, desired_width - crop_width)
        x0 -= width_extra // 2
        x1 += width_extra - width_extra // 2
    else:
        desired_height = int(round(crop_width / target_aspect))
        height_extra = max(0, desired_height - crop_height)
        y0 -= height_extra // 2
        y1 += height_extra - height_extra // 2

    if x0 < 0:
        x1 = min(frame_width, x1 - x0)
        x0 = 0
    if x1 > frame_width:
        x0 = max(0, x0 - (x1 - frame_width))
        x1 = frame_width
    if y0 < 0:
        y1 = min(frame_height, y1 - y0)
        y0 = 0
    if y1 > frame_height:
        y0 = max(0, y0 - (y1 - frame_height))
        y1 = frame_height

    return (int(y0), int(y1), int(x0), int(x1))


def _crop_frames(
    frames: list[np.ndarray], crop_bounds: tuple[int, int, int, int]
) -> list[np.ndarray]:
    y0, y1, x0, x1 = crop_bounds
    return [frame[y0:y1, x0:x1] for frame in frames]


def _save_frame_set(
    *,
    output_dir: Path,
    frames: list[np.ndarray],
    state_steps: list[int],
    prefix: str,
) -> None:
    import imageio.v3 as iio

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (frame, step) in enumerate(zip(frames, state_steps, strict=True)):
        frame_path = output_dir / f"{prefix}_{i:02d}_step_{step:03d}.png"
        iio.imwrite(frame_path, frame)


def _save_storyboard(
    *,
    output_path: Path,
    clean_frames: list[np.ndarray],
    failure_frames: list[np.ndarray],
    state_steps: list[int],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(state_steps),
        figsize=(2.2 * len(state_steps), 5.0),
        constrained_layout=True,
    )
    if len(state_steps) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    fig.patch.set_alpha(0.0)
    row_titles = ("Policy rollout", "Policy + action noise")
    for row, frames in enumerate((clean_frames, failure_frames)):
        for col, (frame, step) in enumerate(zip(frames, state_steps, strict=True)):
            ax = axes[row, col]
            ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
            ax.imshow(frame)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={step}", fontsize=10)
        axes[row, 0].set_ylabel(row_titles[row], fontsize=11)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _export_sequence_variant(
    *,
    output_root: Path,
    prefix: str,
    mj_model: mujoco.MjModel,
    clean_states: list[Any],
    failure_states: list[Any],
    state_steps: list[int],
    width: int,
    height: int,
    camera: str | None,
    transparent_background: bool,
    transparent_ground: bool,
    enable_crop: bool,
    crop_padding: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int, int, int] | None]:
    clean_frames, clean_masks = _render_states_with_masks(
        mj_model=mj_model,
        states=clean_states,
        state_steps=state_steps,
        width=width,
        height=height,
        camera=camera,
        transparent_background=transparent_background,
        transparent_ground=transparent_ground,
    )
    failure_frames, failure_masks = _render_states_with_masks(
        mj_model=mj_model,
        states=failure_states,
        state_steps=state_steps,
        width=width,
        height=height,
        camera=camera,
        transparent_background=transparent_background,
        transparent_ground=transparent_ground,
    )

    crop_bounds: tuple[int, int, int, int] | None = None
    if enable_crop and clean_frames and failure_frames:
        frame_height, frame_width = clean_frames[0].shape[:2]
        crop_bounds = _compute_crop_bounds(
            clean_masks + failure_masks,
            frame_width=frame_width,
            frame_height=frame_height,
            padding=crop_padding,
        )
        clean_frames = _crop_frames(clean_frames, crop_bounds)
        failure_frames = _crop_frames(failure_frames, crop_bounds)

    _save_frame_set(
        output_dir=output_root / f"{prefix}clean",
        frames=clean_frames,
        state_steps=state_steps,
        prefix=f"{prefix}clean",
    )
    _save_frame_set(
        output_dir=output_root / f"{prefix}failure",
        frames=failure_frames,
        state_steps=state_steps,
        prefix=f"{prefix}failure",
    )
    _save_storyboard(
        output_path=output_root / f"{prefix}storyboard.png",
        clean_frames=clean_frames,
        failure_frames=failure_frames,
        state_steps=state_steps,
    )
    return clean_frames, failure_frames, crop_bounds


def record_policy_sequences(
    *,
    seed: int,
    action_mode: str,
    wandb_run_id: str | None,
    wandb_entity: str | None,
    wandb_project: str,
    policy_deterministic: bool,
    config_overrides: dict[str, Any] | None,
    output_dir: str,
    num_candidates: int,
    num_steps: int,
    num_frames: int,
    width: int,
    height: int,
    camera: str | None,
    transparent_background: bool,
    transparent_ground_variant: bool,
    enable_crop: bool,
    crop_padding: float,
    failure_noise_std: float,
    failure_noise_start_step: int,
    failure_height_threshold: float,
    failure_attempts: int,
) -> None:
    runtime = _build_runtime(
        action_mode=action_mode,
        wandb_run_id=wandb_run_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        policy_deterministic=policy_deterministic,
        config_overrides=config_overrides,
    )
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_rng = jax.random.PRNGKey(seed)
    candidate_metadata: list[dict[str, Any]] = []

    for candidate_index in range(max(1, num_candidates)):
        candidate_rng = jax.random.fold_in(base_rng, candidate_index)
        reset_key = jax.random.fold_in(candidate_rng, 0)
        initial_state = runtime.reset_fn(reset_key)
        clean_result = _rollout_sequence(
            runtime=runtime,
            initial_state=initial_state,
            rng=jax.random.fold_in(candidate_rng, 1),
            action_mode=action_mode,
            num_steps=num_steps,
            failure_height_threshold=-1.0,
        )
        failure_result = _find_failure_sequence(
            runtime=runtime,
            initial_state=initial_state,
            rng=jax.random.fold_in(candidate_rng, 2),
            action_mode=action_mode,
            num_steps=num_steps,
            base_noise_std=failure_noise_std,
            noise_start_step=failure_noise_start_step,
            failure_height_threshold=failure_height_threshold,
            max_attempts=max(1, failure_attempts),
        )
        state_steps = _sequence_steps(num_steps, num_frames)
        initial_traj_info = _extract_traj_state_info(initial_state)
        candidate_name = f"candidate_{candidate_index:02d}"
        if initial_traj_info is not None:
            candidate_name += (
                f"_traj_{initial_traj_info['traj_no']:03d}"
                f"_start_{initial_traj_info['subtraj_step_no_init']:04d}"
            )
        candidate_dir = output_root / "candidates" / candidate_name
        clean_frames, failure_frames, crop_bounds = _export_sequence_variant(
            output_root=candidate_dir,
            prefix="",
            mj_model=runtime.env.mj_model,
            clean_states=clean_result.states,
            failure_states=failure_result.states,
            state_steps=state_steps,
            width=width,
            height=height,
            camera=camera,
            transparent_background=transparent_background,
            transparent_ground=False,
            enable_crop=enable_crop,
            crop_padding=crop_padding,
        )
        candidate_record = {
            "index": candidate_index,
            "directory": str(candidate_dir.relative_to(output_root)),
            "state_steps": state_steps,
            "crop_bounds": list(crop_bounds) if crop_bounds is not None else None,
            "initial_traj_state": initial_traj_info,
            "clean": {
                "done_step": clean_result.done_step,
                "min_root_height": clean_result.min_root_height,
                "fell": clean_result.fell,
            },
            "failure": {
                "done_step": failure_result.done_step,
                "min_root_height": failure_result.min_root_height,
                "fell": failure_result.fell,
                "noise_std": failure_result.noise_std,
            },
        }

        if transparent_ground_variant:
            _, _, crop_bounds_no_ground = _export_sequence_variant(
                output_root=candidate_dir,
                prefix="no_ground_",
                mj_model=runtime.env.mj_model,
                clean_states=clean_result.states,
                failure_states=failure_result.states,
                state_steps=state_steps,
                width=width,
                height=height,
                camera=camera,
                transparent_background=transparent_background,
                transparent_ground=True,
                enable_crop=enable_crop,
                crop_padding=crop_padding,
            )
            candidate_record["no_ground_crop_bounds"] = (
                list(crop_bounds_no_ground)
                if crop_bounds_no_ground is not None
                else None
            )
        (candidate_dir / "metadata.json").write_text(
            json.dumps(candidate_record, indent=2)
        )
        candidate_metadata.append(candidate_record)

    metadata = {
        "seed": seed,
        "action_mode": action_mode,
        "transparent_background": transparent_background,
        "transparent_ground_variant": transparent_ground_variant,
        "num_candidates": len(candidate_metadata),
        "sequence_num_steps": num_steps,
        "sequence_num_frames": num_frames,
        "sequence_crop_enabled": enable_crop,
        "sequence_crop_padding": crop_padding,
        "failure_noise_std": failure_noise_std,
        "failure_noise_start_step": failure_noise_start_step,
        "failure_height_threshold": failure_height_threshold,
        "failure_attempts": failure_attempts,
        "candidates": candidate_metadata,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[viewer] saved sequence frames to {output_root}")


def run_viewer(
    seed: int = 0,
    action_mode: str = "policy",
    wandb_run_id: str | None = DEFAULT_WANDB_RUN_ID,
    wandb_entity: str | None = None,
    wandb_project: str = "ss2r",
    policy_deterministic: bool = True,
    config_overrides: dict[str, Any] | None = None,
    reset_on_done: bool = True,
    print_every: int = 0,
) -> None:
    """Launch an interactive viewer for H1MocapTracking.

    Args:
        seed: PRNG seed for reset and random actions.
        action_mode: One of {"reference", "replay", "zero", "random", "policy"}.
        config_overrides: Optional env config overrides.
        reset_on_done: Reset automatically when episode terminates.
        print_every: If > 0, print reward/done every N control steps.
    """
    runtime = _build_runtime(
        action_mode=action_mode,
        wandb_run_id=wandb_run_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        policy_deterministic=policy_deterministic,
        config_overrides=config_overrides,
    )
    rng = jax.random.PRNGKey(seed)

    rng, reset_key = jax.random.split(rng)
    state = runtime.reset_fn(reset_key)

    mj_model = runtime.env.mj_model
    mj_data = MjData(mj_model)
    mjx.get_data_into(mj_data, mj_model, state.data)

    action_size = runtime.env.action_size
    zero_action = jp.zeros((action_size,))
    step_count = 0
    episode_idx = 0
    episode_step = 0
    episode_return = 0.0
    replay_frame_index = 0
    ui_state = {"paused": False, "request_reset": False}

    def _on_key(keycode: int) -> None:
        # GLFW space key code is 32.
        if keycode in (32, ord(" ")):
            ui_state["paused"] = not ui_state["paused"]
            status = "paused" if ui_state["paused"] else "running"
            print(f"[viewer] {status}")
        # GLFW backspace is typically 259.
        elif keycode in (259, 8):
            ui_state["request_reset"] = True
            print("[viewer] reset requested")

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=_on_key
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            start = time.time()

            # Pull potential perturbations from interactive MuJoCo viewer back into MJX state.
            data = state.data.replace(
                qpos=jp.array(mj_data.qpos),
                qvel=jp.array(mj_data.qvel),
                mocap_pos=jp.array(mj_data.mocap_pos),
                mocap_quat=jp.array(mj_data.mocap_quat),
                xfrc_applied=jp.array(mj_data.xfrc_applied),
            )
            state = state.replace(data=data)

            if ui_state["request_reset"]:
                ui_state["request_reset"] = False
                episode_idx += 1
                episode_step = 0
                episode_return = 0.0
                replay_frame_index = 0
                rng, reset_key = jax.random.split(rng)
                state = runtime.reset_fn(reset_key)

            if not ui_state["paused"]:
                if action_mode == "replay":
                    assert runtime.replay_reference is not None
                    state, replay_frame_index = _reference_replay_step(
                        runtime.env,
                        state,
                        runtime.replay_reference,
                        replay_frame_index,
                    )
                else:
                    if action_mode == "zero":
                        action = zero_action
                    elif action_mode == "random":
                        rng, action = _random_action(rng, action_size)
                    elif action_mode == "policy":
                        assert runtime.policy_action_fn is not None
                        rng, policy_key = jax.random.split(rng)
                        action = runtime.policy_action_fn(state.obs, policy_key)
                    else:
                        action = _reference_action(runtime.env, state)

                    state = runtime.step_fn(state, action)
                step_count += 1
                episode_step += 1
                episode_return += float(np.asarray(state.reward).item())

            if print_every > 0 and step_count % print_every == 0:
                _print_status(
                    state=state,
                    step_count=step_count,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    replay_reference=runtime.replay_reference,
                )

            done_flag = bool(np.asarray(state.done).item())
            if reset_on_done and done_flag:
                _print_status(
                    state=state,
                    step_count=step_count,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    replay_reference=runtime.replay_reference,
                )
                episode_idx += 1
                episode_step = 0
                episode_return = 0.0
                replay_frame_index = 0
                rng, reset_key = jax.random.split(rng)
                state = runtime.reset_fn(reset_key)

            _update_hud_labels(
                viewer=viewer,
                state=state,
                step_count=step_count,
                episode_idx=episode_idx,
                episode_step=episode_step,
                episode_return=episode_return,
                replay_reference=runtime.replay_reference,
            )
            mjx.get_data_into(mj_data, mj_model, state.data)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < runtime.env.dt:
                time.sleep(runtime.env.dt - elapsed)


def _parse_override(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_config_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected format KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key.")
        overrides[key] = _parse_override(raw_value.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Viewer for H1MocapTracking.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        type=str,
        default="policy",
        choices=list(SUPPORTED_ACTION_MODES),
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=DEFAULT_WANDB_RUN_ID,
        help=(
            "W&B run id used for checkpoint and config lookup in policy mode "
            f"(default: {DEFAULT_WANDB_RUN_ID})."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team. If omitted, uses your default entity.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ss2r",
        help="W&B project name.",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample SAC actions stochastically in policy mode.",
    )
    parser.add_argument(
        "--reference-name",
        type=str,
        default=None,
        help=(
            "Mocap reference name/path (e.g. 'dance1_subject3' or "
            "'Lafan1/mocap/UnitreeG1/dance1_subject3.npz')."
        ),
    )
    parser.add_argument(
        "--reference-source",
        type=str,
        default=None,
        choices=["auto", "local", "hf"],
        help="Reference lookup mode for mocap npz: auto, local, or hf.",
    )
    parser.add_argument(
        "--reference-repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id for mocap files.",
    )
    parser.add_argument(
        "--reference-repo-type",
        type=str,
        default=None,
        help="Hugging Face repo type (defaults to dataset).",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Directory prefix inside the HF dataset repository.",
    )
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument(
        "--no-reset-on-done",
        action="store_true",
        help="Disable automatic reset when done=True.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional environment config override. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sequence-output-dir",
        type=str,
        default=None,
        help="If set, export sliding-window clean/failure frame sequences to this directory.",
    )
    parser.add_argument(
        "--sequence-num-steps",
        type=int,
        default=120,
        help="Total rollout steps used for sequence export.",
    )
    parser.add_argument(
        "--sequence-num-candidates",
        type=int,
        default=8,
        help="Number of different reset initializations to export as candidate sequences.",
    )
    parser.add_argument(
        "--sequence-num-frames",
        type=int,
        default=8,
        help="Number of frames to save for each sequence.",
    )
    parser.add_argument(
        "--sequence-width",
        type=int,
        default=640,
        help="Offscreen render width for exported frames.",
    )
    parser.add_argument(
        "--sequence-height",
        type=int,
        default=480,
        help="Offscreen render height for exported frames.",
    )
    parser.add_argument(
        "--sequence-camera",
        type=str,
        default=None,
        help="MuJoCo camera name or integer id for sequence export.",
    )
    parser.add_argument(
        "--opaque-background",
        action="store_true",
        help="Keep the rendered sky/background opaque in exported sequence frames.",
    )
    parser.add_argument(
        "--transparent-ground",
        action="store_true",
        help="Also export an additional variant with the floor removed.",
    )
    parser.add_argument(
        "--no-sequence-crop",
        action="store_true",
        help="Disable the tighter auto-crop around the robot in exported frames.",
    )
    parser.add_argument(
        "--sequence-crop-padding",
        type=float,
        default=0.18,
        help="Fractional padding around the auto-cropped robot bounding box.",
    )
    parser.add_argument(
        "--failure-noise-std",
        type=float,
        default=0.35,
        help="Base action-noise std used to synthesize a failing rollout.",
    )
    parser.add_argument(
        "--failure-noise-start-step",
        type=int,
        default=24,
        help="Step index at which action noise starts in the failing rollout.",
    )
    parser.add_argument(
        "--failure-height-threshold",
        type=float,
        default=0.55,
        help="Root-height threshold used to flag a rollout as fallen.",
    )
    parser.add_argument(
        "--failure-attempts",
        type=int,
        default=6,
        help="Number of noisy-rollout attempts before giving up on finding a fall.",
    )
    args = parser.parse_args()

    config_overrides = _parse_config_overrides(args.config_override)
    if args.reference_name is not None:
        config_overrides["loco.dataset_name"] = args.reference_name
    if args.reference_source is not None:
        config_overrides["loco.reference_source"] = args.reference_source
    if args.reference_repo_id is not None:
        config_overrides["loco.reference_repo_id"] = args.reference_repo_id
    if args.reference_repo_type is not None:
        config_overrides["loco.reference_repo_type"] = args.reference_repo_type
    if args.reference_dir is not None:
        config_overrides["loco.reference_dir"] = args.reference_dir

    if args.sequence_output_dir is not None:
        record_policy_sequences(
            seed=args.seed,
            action_mode=args.action_mode,
            wandb_run_id=args.wandb_run_id,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            policy_deterministic=not args.stochastic_policy,
            config_overrides=config_overrides if config_overrides else None,
            output_dir=args.sequence_output_dir,
            num_candidates=args.sequence_num_candidates,
            num_steps=args.sequence_num_steps,
            num_frames=args.sequence_num_frames,
            width=args.sequence_width,
            height=args.sequence_height,
            camera=args.sequence_camera,
            transparent_background=not args.opaque_background,
            transparent_ground_variant=args.transparent_ground,
            enable_crop=not args.no_sequence_crop,
            crop_padding=args.sequence_crop_padding,
            failure_noise_std=args.failure_noise_std,
            failure_noise_start_step=args.failure_noise_start_step,
            failure_height_threshold=args.failure_height_threshold,
            failure_attempts=args.failure_attempts,
        )
    else:
        run_viewer(
            seed=args.seed,
            action_mode=args.action_mode,
            wandb_run_id=args.wandb_run_id,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            policy_deterministic=not args.stochastic_policy,
            config_overrides=config_overrides if config_overrides else None,
            reset_on_done=not args.no_reset_on_done,
            print_every=args.print_every,
        )


if __name__ == "__main__":
    main()
