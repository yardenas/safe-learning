"""Interactive MuJoCo viewer for G1MocapTracking.

Usage from Python:
    from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.viewer import run_viewer
    run_viewer(action_mode="reference")
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
from ss2r.algorithms.hydrax_mpc.factory import make_task
from ss2r.algorithms.hydrax_mpc.tree_mpc import TreeMPC, TreeMPCParams
from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.g1_mocap_env import (
    G1MocapTracking,
)
from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco.trajectory import (
    Trajectory,
)


def _random_action(rng: jax.Array, action_size: int) -> tuple[jax.Array, jax.Array]:
    rng, key = jax.random.split(rng)
    if action_size == 0:
        return rng, jp.zeros((0,))
    action = jax.random.uniform(key, (action_size,), minval=-1.0, maxval=1.0)
    return rng, action


def _reference_action(env: G1MocapTracking, state: Any) -> jax.Array:
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


def _load_replay_reference(env: G1MocapTracking) -> _ReplayReference:
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
) -> tuple[str, dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise ImportError("wandb is required for action_mode='policy'.") from exc

    api = wandb.Api(overrides={"entity": entity}) if entity else wandb.Api()
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
    env: G1MocapTracking,
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
    normalizer_params, policy_params = params[0], params[1]
    inference_fn = sac_networks.make_inference_fn(sac_network)(
        (normalizer_params, policy_params),
        deterministic=deterministic,
    )

    @jax.jit
    def _policy_action(obs: Any, key: jax.Array) -> jax.Array:
        batched_obs = jax.tree_map(lambda x: jp.expand_dims(x, axis=0), obs)
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
    env: G1MocapTracking,
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


def _build_tree_mpc(
    planner_env: G1MocapTracking,
    seed: int,
    *,
    num_samples: int,
    horizon: int,
    zoh_steps: int,
    gamma: float,
    temperature: float,
    action_noise_std: float,
    iterations: int,
) -> tuple[TreeMPC, TreeMPCParams]:
    task = make_task(planner_env)
    planner = TreeMPC(
        task=task,
        num_samples=num_samples,
        horizon=horizon,
        zoh_steps=zoh_steps,
        gamma=gamma,
        temperature=temperature,
        action_noise_std=action_noise_std,
        iterations=iterations,
        gae_lambda=0.0,
        use_policy=False,
        use_critic=False,
    )
    params = planner.init_params(seed=seed)
    return planner, params


def _reference_action_sequence(
    env: G1MocapTracking,
    state: Any,
    num_actions: int,
    control_dt: float,
    zoh_steps: int,
) -> jax.Array:
    del state, control_dt, zoh_steps
    # Warmstart at zero residuals so the planner starts from pure mocap commands.
    return jp.zeros((num_actions, env.action_size))


def _reference_replay_step(
    env: G1MocapTracking,
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


def run_viewer(
    seed: int = 0,
    action_mode: str = "reference",
    wandb_run_id: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str = "ss2r",
    policy_deterministic: bool = True,
    config_overrides: dict[str, Any] | None = None,
    reset_on_done: bool = True,
    print_every: int = 0,
    tree_num_samples: int = 128,
    tree_horizon: int = 25,
    tree_zoh_steps: int = 1,
    tree_gamma: float = 0.99,
    tree_temperature: float = 0.2,
    tree_action_noise_std: float = 0.3,
    tree_iterations: int = 2,
    tree_planner_sim_dt: float = 0.02,
) -> None:
    """Launch an interactive viewer for G1MocapTracking.

    Args:
        seed: PRNG seed for reset and random actions.
        action_mode: One of {"reference", "replay", "zero", "random", "tree_mpc", "policy"}.
        config_overrides: Optional env config overrides.
        reset_on_done: Reset automatically when episode terminates.
        print_every: If > 0, print reward/done every N control steps.
    """
    if action_mode not in {
        "reference",
        "replay",
        "zero",
        "random",
        "tree_mpc",
        "policy",
    }:
        raise ValueError(
            f"Unsupported action_mode={action_mode}. "
            "Expected one of {'reference', 'replay', 'zero', 'random', 'tree_mpc', 'policy'}."
        )

    run_config: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    effective_config_overrides = dict(config_overrides or {})
    if action_mode == "policy":
        if not wandb_run_id:
            raise ValueError(
                "action_mode='policy' requires --wandb-run-id (or wandb_run_id)."
            )
        checkpoint_path, run_config = _download_wandb_checkpoint_and_config(
            wandb_run_id,
            entity=wandb_entity,
            project=wandb_project,
        )
        wandb_env_overrides = _extract_env_overrides_from_run_config(run_config)
        effective_config_overrides = {
            **wandb_env_overrides,
            **effective_config_overrides,
        }
        print(
            f"[viewer] loaded wandb run={wandb_run_id} "
            f"(entity={wandb_entity or '<default>'}, project={wandb_project})"
        )
        print(f"[viewer] checkpoint_path={checkpoint_path}")

    env = G1MocapTracking(
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
    rng = jax.random.PRNGKey(seed)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng, reset_key = jax.random.split(rng)
    state = reset_fn(reset_key)

    mj_model = env.mj_model
    mj_data = MjData(mj_model)
    mjx.get_data_into(mj_data, mj_model, state.data)

    action_size = env.action_size
    zero_action = jp.zeros((action_size,))
    step_count = 0
    episode_idx = 0
    episode_step = 0
    episode_return = 0.0
    replay_frame_index = 0
    ui_state = {"paused": False, "request_reset": False}
    planner = None
    planner_params = None
    planner_optimize = None
    planner_env = None
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

    if action_mode == "tree_mpc":
        planner_env = G1MocapTracking()
        planner, planner_params = _build_tree_mpc(
            planner_env=planner_env,
            seed=seed,
            num_samples=tree_num_samples,
            horizon=tree_horizon,
            zoh_steps=tree_zoh_steps,
            gamma=tree_gamma,
            temperature=tree_temperature,
            action_noise_std=tree_action_noise_std,
            iterations=tree_iterations,
        )
        warmstart_actions = _reference_action_sequence(
            env=env,
            state=state,
            num_actions=planner.ctrl_steps,
            control_dt=planner.dt,
            zoh_steps=planner.zoh_steps,
        )
        planner_params = planner_params.replace(actions=warmstart_actions)
        planner_optimize = jax.jit(planner.optimize)

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
                state = reset_fn(reset_key)
                if action_mode == "tree_mpc":
                    assert planner is not None
                    assert planner_params is not None
                    warmstart_actions = _reference_action_sequence(
                        env=env,
                        state=state,
                        num_actions=planner.ctrl_steps,
                        control_dt=planner.dt,
                        zoh_steps=planner.zoh_steps,
                    )
                    planner_params = planner_params.replace(actions=warmstart_actions)

            if not ui_state["paused"]:
                if action_mode == "replay":
                    assert replay_reference is not None
                    state, replay_frame_index = _reference_replay_step(
                        env, state, replay_reference, replay_frame_index
                    )
                else:
                    if action_mode == "zero":
                        action = zero_action
                    elif action_mode == "random":
                        rng, action = _random_action(rng, action_size)
                    elif action_mode == "policy":
                        assert policy_action_fn is not None
                        rng, policy_key = jax.random.split(rng)
                        action = policy_action_fn(state.obs, policy_key)
                    elif action_mode == "tree_mpc":
                        assert planner is not None
                        assert planner_params is not None
                        assert planner_optimize is not None
                        planner_params, _ = planner_optimize(
                            state, planner_params, None
                        )
                        action = planner_params.actions[0]
                    else:
                        action = _reference_action(env, state)

                    state = step_fn(state, action)
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
                    replay_reference=replay_reference,
                )

            done_flag = bool(np.asarray(state.done).item())
            if reset_on_done and done_flag:
                _print_status(
                    state=state,
                    step_count=step_count,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    replay_reference=replay_reference,
                )
                episode_idx += 1
                episode_step = 0
                episode_return = 0.0
                replay_frame_index = 0
                rng, reset_key = jax.random.split(rng)
                state = reset_fn(reset_key)
                if action_mode == "tree_mpc":
                    assert planner is not None
                    assert planner_params is not None
                    warmstart_actions = _reference_action_sequence(
                        env=env,
                        state=state,
                        num_actions=planner.ctrl_steps,
                        control_dt=planner.dt,
                        zoh_steps=planner.zoh_steps,
                    )
                    planner_params = planner_params.replace(actions=warmstart_actions)

            _update_hud_labels(
                viewer=viewer,
                state=state,
                step_count=step_count,
                episode_idx=episode_idx,
                episode_step=episode_step,
                episode_return=episode_return,
                replay_reference=replay_reference,
            )
            mjx.get_data_into(mj_data, mj_model, state.data)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)


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
    parser = argparse.ArgumentParser(description="Viewer for G1MocapTracking.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        type=str,
        default="reference",
        choices=["reference", "replay", "zero", "random", "tree_mpc", "policy"],
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run id used for checkpoint and config lookup (policy mode only).",
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
    parser.add_argument("--tree-num-samples", type=int, default=128)
    parser.add_argument("--tree-horizon", type=int, default=25)
    parser.add_argument("--tree-zoh-steps", type=int, default=1)
    parser.add_argument("--tree-gamma", type=float, default=0.99)
    parser.add_argument("--tree-temperature", type=float, default=0.8)
    parser.add_argument("--tree-action-noise-std", type=float, default=0.3)
    parser.add_argument("--tree-iterations", type=int, default=2)
    parser.add_argument("--tree-planner-sim-dt", type=float, default=0.02)
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
        tree_num_samples=args.tree_num_samples,
        tree_horizon=args.tree_horizon,
        tree_zoh_steps=args.tree_zoh_steps,
        tree_gamma=args.tree_gamma,
        tree_temperature=args.tree_temperature,
        tree_action_noise_std=args.tree_action_noise_std,
        tree_iterations=args.tree_iterations,
        tree_planner_sim_dt=args.tree_planner_sim_dt,
    )


if __name__ == "__main__":
    main()
