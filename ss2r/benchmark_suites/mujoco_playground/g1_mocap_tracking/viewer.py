"""Interactive MuJoCo viewer for G1MocapTracking.

Usage from Python:
    from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.viewer import run_viewer
    run_viewer(action_mode="reference")
"""

import argparse
import ast
import time
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, mjx
from mujoco_playground._src import mjx_env

from ss2r.algorithms.hydrax_mpc.factory import make_task
from ss2r.algorithms.hydrax_mpc.tree_mpc import TreeMPC, TreeMPCParams
from ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.g1_mocap_env import (
    G1MocapTracking,
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


def _reference_replay_step(env: G1MocapTracking, state: Any) -> Any:
    """Override the simulator state with the mocap reference at the next control tick."""
    ref_idx_target = env._reference_index(
        state.data.time + env.dt, state.info["reference_start_idx"]
    )
    q_ref = env._reference[ref_idx_target]
    u_ref = q_ref[7 : 7 + env.action_size]

    data = state.data.replace(
        qpos=q_ref,
        qvel=jp.zeros_like(state.data.qvel),
        ctrl=u_ref,
        time=state.data.time + env.dt,
    )
    data = mjx.forward(env.mjx_model, data)

    feet_pos = data.site_xpos[env._feet_site_id]
    done = env._get_termination(data)
    rewards = env._get_reward(
        data=data,
        motor_targets=u_ref,
        done=done,
        ref_idx=ref_idx_target,
        feet_pos=feet_pos,
    )
    rewards = {
        key: value * env._config.reward_config.scales[key]
        for key, value in rewards.items()
    }
    reward = sum(rewards.values())

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = jp.zeros((env.action_size,))
    state.info["motor_targets"] = u_ref
    state.info["phase"] = env._phase_from_index(ref_idx_target)
    for key, value in rewards.items():
        state.metrics[f"reward/{key}"] = value

    obs = env._get_obs(data, state.info)
    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)


def _active_reward_terms(env: G1MocapTracking) -> list[str]:
    terms: list[str] = []
    for key, value in env._config.reward_config.scales.items():
        if abs(float(value)) > 0.0:
            terms.append(str(key))
    return terms


def _mocap_progress(env: G1MocapTracking, state: Any) -> tuple[int, int, int, float]:
    n_frames = int(env._reference.shape[0])
    start_idx = int(np.asarray(state.info["reference_start_idx"]).item())
    curr_idx = int(
        np.asarray(
            env._reference_index(state.data.time, state.info["reference_start_idx"])
        ).item()
    )
    rel_idx = (curr_idx - start_idx) % n_frames
    frac = float(rel_idx) / float(max(n_frames - 1, 1))
    return curr_idx, rel_idx, n_frames, frac


def _progress_bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def _print_status(
    env: G1MocapTracking,
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    reward_terms: list[str],
) -> None:
    reward = float(np.asarray(state.reward).item())
    done = bool(np.asarray(state.done).item())
    curr_idx, rel_idx, n_frames, frac = _mocap_progress(env, state)
    bar = _progress_bar(frac)
    print(
        f"[ep={episode_idx} step={episode_step} global={step_count}] "
        f"r={reward:+.5f} R={episode_return:+.3f} done={done} "
        f"mocap={curr_idx}/{n_frames-1} rel={rel_idx} {frac*100:5.1f}% [{bar}]"
    )

    parts: list[str] = []
    for term in reward_terms:
        metric_key = f"reward/{term}"
        value = float(np.asarray(state.metrics.get(metric_key, 0.0)).item())
        parts.append(f"{term}={value:+.4f}")
    if parts:
        print("  rewards:", " | ".join(parts))


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
    env: G1MocapTracking,
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    reward_terms: list[str],
    paused: bool,
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
    curr_idx, rel_idx, n_frames, frac = _mocap_progress(env, state)
    bar = _progress_bar(frac, width=16)
    lines.extend(
        [
            (
                f"mocap {curr_idx}/{n_frames-1} rel={rel_idx} "
                f"{frac*100:4.1f}% [{bar}]"
            ),
        ]
    )

    motor_targets = np.asarray(
        state.info.get("motor_targets", np.zeros((env.action_size,)))
    )
    qpos_joint = np.asarray(state.data.qpos[7 : 7 + env.action_size])

    track_err = float(np.linalg.norm(qpos_joint - motor_targets))

    torso_up_z = float(
        np.asarray(
            mjx_env.get_sensor_data(env.mj_model, state.data, "upvector_torso")[-1]
        )
    )
    base_h = float(np.asarray(state.data.qpos[2]))
    lines.append(
        f"joint_track_err={track_err:.3f} torso_up_z={torso_up_z:+.3f} h={base_h:.3f}"
    )

    # Show top reward terms in-text (already dt-scaled in metrics).
    term_values: list[tuple[str, float]] = []
    for term in reward_terms:
        metric_key = f"reward/{term}"
        val = float(np.asarray(state.metrics.get(metric_key, 0.0)).item())
        term_values.append((term, val))
    term_values.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for term, value in term_values[:6]:
        lines.append(f"{term}={value:+.4f}")
    return lines


def _update_hud_labels(
    env: G1MocapTracking,
    viewer: mujoco.viewer.Handle,
    state: Any,
    step_count: int,
    episode_idx: int,
    episode_step: int,
    episode_return: float,
    reward_terms: list[str],
    paused: bool,
) -> None:
    if viewer.user_scn is None:
        return
    # Viewer runs on a separate thread; mutate user_scn under viewer lock.
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        lines = _hud_lines(
            env=env,
            state=state,
            step_count=step_count,
            episode_idx=episode_idx,
            episode_step=episode_step,
            episode_return=episode_return,
            reward_terms=reward_terms,
            paused=paused,
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
        action_mode: One of {"reference", "replay", "zero", "random", "tree_mpc"}.
        config_overrides: Optional env config overrides.
        reset_on_done: Reset automatically when episode terminates.
        print_every: If > 0, print reward/done every N control steps.
    """
    if action_mode not in {"reference", "replay", "zero", "random", "tree_mpc"}:
        raise ValueError(
            f"Unsupported action_mode={action_mode}. "
            "Expected one of {'reference', 'replay', 'zero', 'random', 'tree_mpc'}."
        )

    if action_mode in {"reference", "replay", "tree_mpc"} and config_overrides is None:
        # Deterministic reset helps isolate whether reference tracking itself is stable.
        config_overrides = {"random_start": False, "reset_noise_scale": 0.0}

    env = G1MocapTracking(config_overrides=config_overrides)
    rng = jax.random.PRNGKey(seed)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    replay_step_fn = jax.jit(lambda s: _reference_replay_step(env, s))

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
    reward_terms = _active_reward_terms(env)
    ui_state = {"paused": False, "request_reset": False}
    planner = None
    planner_params = None
    planner_optimize = None
    planner_env = None

    if action_mode == "tree_mpc":
        planner_overrides = dict(config_overrides or {})
        # planner_overrides["sim_dt"] = 0.01
        # planner_overrides["ctrl_dt"] = 0.02
        planner_env = G1MocapTracking(config_overrides=planner_overrides)
        # planner_env.mj_model.opt.timestep = 0.01
        # planner_env.mj_model.opt.iterations = 10
        # planner_env.mj_model.opt.ls_iterations = 50
        # planner_env.mj_model.opt.o_solimp[:] = np.array(
        #     [0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float64
        # )
        # planner_env._mjx_model = mjx.put_model(planner_env.mj_model)
        print(
            "[planner_env] "
            f"n_substeps={planner_env.n_substeps} "
            f"dt={planner_env.mj_model.opt.timestep:.4f} "
            f"iters={planner_env.mj_model.opt.iterations} "
            f"ls_iters={planner_env.mj_model.opt.ls_iterations}"
        )
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
                    state = replay_step_fn(state)
                else:
                    if action_mode == "zero":
                        action = zero_action
                    elif action_mode == "random":
                        rng, action = _random_action(rng, action_size)
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
                    env=env,
                    state=state,
                    step_count=step_count,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    reward_terms=reward_terms,
                )

            if reset_on_done and bool(np.asarray(state.done).item()):
                _print_status(
                    env=env,
                    state=state,
                    step_count=step_count,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    reward_terms=reward_terms,
                )
                episode_idx += 1
                episode_step = 0
                episode_return = 0.0
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
                env=env,
                viewer=viewer,
                state=state,
                step_count=step_count,
                episode_idx=episode_idx,
                episode_step=episode_step,
                episode_return=episode_return,
                reward_terms=reward_terms,
                paused=ui_state["paused"],
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
        choices=["reference", "replay", "zero", "random", "tree_mpc"],
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
    run_viewer(
        seed=args.seed,
        action_mode=args.action_mode,
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
