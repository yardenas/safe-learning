"""AWAC-style training with optional TreeMPC actor supervision."""

import time
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.training import replay_buffers
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint
from brax.training.types import Params, PRNGKey
from flax import struct
from ml_collections import config_dict

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.awac_mpc import losses as awac_losses
from ss2r.algorithms.hydrax_mpc.factory import make_controller, make_task
from ss2r.algorithms.hydrax_mpc.tree_mpc import TreeMPCModelParams, TreeMPCParams
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
    float32,
)
from ss2r.rl.evaluation import ConstraintsEvaluator, Evaluator
from ss2r.rl.utils import quantize_images, remove_pixels, restore_state

make_inference_fn = sac_networks.make_inference_fn
make_networks = sac_networks.make_sac_networks


ActorUpdateSource = Literal["planner_replay", "planner_online", "critic_replay"]


@struct.dataclass
class TrainingState:
    policy_optimizer_state: optax.OptState
    policy_params: Params
    qr_optimizer_state: optax.OptState
    qr_params: Params
    target_qr_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    sac_network,
    policy_optimizer: optax.GradientTransformation,
    qr_optimizer: optax.GradientTransformation,
) -> TrainingState:
    key_policy, key_qr = jax.random.split(key)
    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    qr_params = sac_network.qr_network.init(key_qr)
    qr_optimizer_state = qr_optimizer.init(qr_params)
    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
    normalizer_params = running_statistics.init_state(remove_pixels(obs_shape))
    return TrainingState(  # type: ignore
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        qr_optimizer_state=qr_optimizer_state,
        qr_params=qr_params,
        target_qr_params=qr_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params,
    )


def _is_batched(state: envs.State) -> bool:
    return getattr(state.reward, "ndim", 0) >= 1


def _flatten_transition_batch(transitions: Transition) -> Transition:
    return jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim > 1 else x.reshape((-1,)),
        transitions,
    )


def _flatten_time_batch_tree(tree: Any, time_len: int, batch_size: int) -> Any:
    def _flatten_leaf(x):
        if not hasattr(x, "shape"):
            return x
        if x.ndim >= 2 and x.shape[0] == time_len and x.shape[1] == batch_size:
            return x.reshape((time_len * batch_size,) + x.shape[2:])
        if x.ndim >= 1 and x.shape[0] == time_len:
            x = x[:, None, ...]
            x = jnp.broadcast_to(x, (time_len, batch_size) + x.shape[2:])
            return x.reshape((time_len * batch_size,) + x.shape[2:])
        return jnp.broadcast_to(x, (time_len * batch_size,) + x.shape)

    return jax.tree.map(_flatten_leaf, tree)


def _concat_transition_batches(a: Transition, b: Transition) -> Transition:
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), a, b)


def _init_planner_params(
    controller, seed: int, batch_size: int | None
) -> TreeMPCParams:
    base_params = controller.init_params(seed)
    if batch_size is None or batch_size <= 1:
        return base_params
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    actions = jnp.broadcast_to(
        base_params.actions, (batch_size,) + base_params.actions.shape
    )
    return base_params.replace(actions=actions, rng=keys)


def _planner_params_for_batch(
    base_params: TreeMPCParams, key: jax.Array, batch_size: int
) -> TreeMPCParams:
    keys = jax.random.split(key, batch_size)
    actions = jnp.broadcast_to(
        base_params.actions, (batch_size,) + base_params.actions.shape
    )
    return base_params.replace(actions=actions, rng=keys)  # type: ignore


def _planner_model_params(training_state: TrainingState) -> TreeMPCModelParams:
    return TreeMPCModelParams(  # type: ignore
        normalizer_params=training_state.normalizer_params,
        policy_params=training_state.policy_params,
        qr_params=training_state.qr_params,
    )


def _build_extras(state_info: dict[str, Any], done: jax.Array):
    truncation = (
        state_info["truncation"] if "truncation" in state_info else jnp.zeros_like(done)
    )
    return {
        "state_extras": {"truncation": truncation},
        "policy_extras": {},
    }


def _strip_policy_extras(transitions: Transition) -> Transition:
    return transitions._replace(
        extras={
            "state_extras": transitions.extras["state_extras"],
            "policy_extras": {},
        }
    )


def _to_storage_transition(
    transitions: Transition,
    planner_states: Any | None,
) -> Transition:
    transitions = float16(transitions)
    transitions = transitions._replace(
        observation=quantize_images(transitions.observation),
        next_observation=quantize_images(transitions.next_observation),
    )
    policy_extras = (
        {"planner_state": planner_states} if planner_states is not None else {}
    )
    return transitions._replace(
        extras={
            "state_extras": transitions.extras["state_extras"],
            "policy_extras": policy_extras,
        }
    )


def _planner_supervised_batch(
    transitions: Transition,
    controller,
    planner_params_template: TreeMPCParams,
    key: PRNGKey,
    planner_model_params: TreeMPCModelParams | None = None,
    planner_rollout_steps: int = 1,
) -> tuple[Transition, jax.Array]:
    planner_states = transitions.extras["policy_extras"].get("planner_state", None)
    if planner_states is None:
        raise ValueError(
            "Planner-supervised mode requires planner_state in replay data."
        )
    if planner_rollout_steps < 1:
        raise ValueError("planner_rollout_steps must be >= 1.")
    batch_size = transitions.reward.shape[0]
    planner_params = _planner_params_for_batch(planner_params_template, key, batch_size)

    def _rollout_step(carry, _):
        current_states, current_params = carry
        next_params, _ = jax.vmap(
            lambda s, p: controller.optimize(s, p, planner_model_params)
        )(current_states, current_params)
        action = next_params.actions[:, 0, :]
        next_states, rewards = jax.vmap(lambda s, a: controller._step_env_repeat(s, a))(
            current_states, action
        )
        discount = 1.0 - next_states.done.astype(jnp.float32)
        if "truncation" in next_states.info:
            truncation = next_states.info["truncation"]
        else:
            truncation = jnp.zeros_like(rewards)
        step_transition = (
            current_states.obs,
            action,
            rewards,
            next_states.obs,
            discount,
            truncation,
        )
        return (next_states, next_params), step_transition

    (
        (_, _),
        (
            obs,
            actions,
            rewards,
            next_obs,
            discount,
            truncation,
        ),
    ) = jax.lax.scan(
        _rollout_step,
        (planner_states, planner_params),
        (),
        length=planner_rollout_steps,
    )

    # Sum rewards over rollout steps and average over batch.
    avg_rollout_return = jnp.mean(jnp.sum(rewards, axis=0))

    def _flatten_rollout(x):
        if not hasattr(x, "shape"):
            return x
        if x.ndim < 2:
            raise ValueError(
                "Expected rollout tensors with shape [R, B, ...] for planner supervision."
            )
        return x.reshape((-1,) + x.shape[2:])

    obs_flat = jax.tree.map(_flatten_rollout, obs)
    next_obs_flat = jax.tree.map(_flatten_rollout, next_obs)
    action_flat = _flatten_rollout(actions)
    reward_flat = _flatten_rollout(rewards)
    discount_flat = _flatten_rollout(discount)
    truncation_flat = _flatten_rollout(truncation)

    return Transition(
        observation=obs_flat,
        action=action_flat,
        reward=reward_flat,
        discount=discount_flat,
        next_observation=next_obs_flat,
        extras={
            "state_extras": {"truncation": truncation_flat},
            "policy_extras": {},
        },
    ), avg_rollout_return


def _tree_finite_fraction(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    finite_fracs = [
        jnp.mean(jnp.isfinite(leaf).astype(jnp.float32))
        for leaf in leaves
        if hasattr(leaf, "dtype")
    ]
    if not finite_fracs:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(finite_fracs))


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    discounting: float = 0.99,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    sim_prefill_steps: int = 0,
    critic_pretrain_ratio: float = 0.0,
    grad_updates_per_step: int = 1,
    num_critic_updates_per_actor_update: int = 1,
    deterministic_eval: bool = False,
    rollout_length: int = 1,
    awac_lambda: float = 1.0,
    max_weight: float | None = None,
    n_critics: int = 2,
    n_heads: int = 1,
    use_bro: bool = True,
    actor_update_source: ActorUpdateSource = "planner_online",
    planner_batches_per_step: int = 1,
    planner_rollout_steps: int = 1,
    min_actor_replay_size: int | None = None,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    planner_environment: Optional[envs.Env] = None,
    controller_name: str = "tree",
    controller_kwargs: Optional[dict[str, Any]] = None,
    policy_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    value_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    activation: Callable[[jax.Array], jax.Array] = jax.nn.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
):
    valid_actor_sources = {"planner_replay", "planner_online", "critic_replay"}
    if actor_update_source not in valid_actor_sources:
        raise ValueError(
            f"Unknown actor_update_source: {actor_update_source}, expected one of {valid_actor_sources}."
        )
    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )
    if sim_prefill_steps < 0:
        raise ValueError("sim_prefill_steps must be >= 0.")
    if critic_pretrain_ratio < 0:
        raise ValueError("critic_pretrain_ratio must be >= 0.")
    if max_replay_size is None:
        max_replay_size = num_timesteps
    if min_actor_replay_size is None:
        min_actor_replay_size = batch_size

    planner_mode = actor_update_source in {"planner_replay", "planner_online"}
    if planner_batches_per_step < 1:
        raise ValueError("planner_batches_per_step must be >= 1.")
    if planner_rollout_steps < 1:
        raise ValueError("planner_rollout_steps must be >= 1.")
    if planner_mode and controller_name != "tree":
        raise ValueError(
            "Planner-supervised modes only support controller_name='tree'."
        )
    if planner_mode and planner_environment is None:
        raise ValueError(
            "planner_environment is required for planner_replay/planner_online modes."
        )

    env = environment
    if wrap_env_fn is not None:
        env = wrap_env_fn(env)

    rng = jax.random.PRNGKey(seed)
    obs_size = env.observation_size
    if isinstance(obs_size, Mapping):
        for key, value in obs_size.items():
            if key.startswith("pixels/") and len(value) > 3 and value[0] == 1:
                value = value[1:]
                obs_size[key] = value  # type: ignore
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize

    sac_network = make_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        n_critics=n_critics,
        n_heads=n_heads,
        use_bro=use_bro,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )
    make_policy = make_inference_fn(sac_network)

    policy_optimizer = optax.adam(learning_rate=learning_rate)
    qr_optimizer = optax.adam(learning_rate=critic_learning_rate)

    if isinstance(obs_size, Mapping):
        dummy_obs = {k: jnp.zeros(v) for k, v in obs_size.items()}
    else:
        dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    base_dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": jnp.zeros(())}, "policy_extras": {}},
    )

    rng, init_key = jax.random.split(rng)
    training_state = _init_training_state(
        init_key,
        obs_size,
        sac_network,
        policy_optimizer,
        qr_optimizer,
    )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        if len(params) >= 10:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=params[0],
                policy_params=params[1],
                qr_params=params[3],
                target_qr_params=params[3],
                policy_optimizer_state=restore_state(
                    params[6], training_state.policy_optimizer_state
                ),
                qr_optimizer_state=restore_state(
                    params[8], training_state.qr_optimizer_state
                ),
            )
        else:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=params[0],
                policy_params=params[1],
                qr_params=params[2],
                target_qr_params=params[3],
                policy_optimizer_state=restore_state(
                    params[4], training_state.policy_optimizer_state
                ),
                qr_optimizer_state=restore_state(
                    params[5], training_state.qr_optimizer_state
                ),
            )

    env_keys = jax.random.split(rng, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(env_keys)
    if not _is_batched(env_state):
        raise ValueError(
            "AWAC-MPC expects a batched training environment state. "
            "Use a vectorized/wrapped env with batch dimension."
        )

    planner_params_template = None
    controller = None
    if planner_mode:
        planner_env = planner_environment
        assert planner_env is not None
        controller_kwargs = dict(controller_kwargs or {})
        configured_action_repeat = controller_kwargs.get("action_repeat", action_repeat)
        if configured_action_repeat is not None and int(
            configured_action_repeat
        ) != int(action_repeat):
            logging.warning(
                "controller action_repeat=%s with training.action_repeat=%s.",
                configured_action_repeat,
                action_repeat,
            )
        controller_kwargs["n_critics"] = int(n_critics)
        controller_kwargs["n_heads"] = int(n_heads)
        controller_kwargs["gamma"] = discounting
        controller_cfg = SimpleNamespace(
            agent={
                "controller_name": controller_name,
                "controller_kwargs": controller_kwargs,
            },
            training=SimpleNamespace(seed=seed),
        )
        task = make_task(planner_env)  # type: ignore[arg-type]
        controller = make_controller(controller_cfg, task, env=planner_env)  # type: ignore[arg-type]
        if hasattr(controller, "bind_sac_network"):
            controller.bind_sac_network(sac_network)
        planner_params_template = _init_planner_params(controller, seed, None)

    dummy_planner_state = None
    if planner_mode:
        dummy_planner_state = jax.tree.map(
            lambda x: x[0]
            if hasattr(x, "shape") and x.shape and x.shape[0] == num_envs
            else x,
            env_state,
        )

    replay_dummy_transition = _to_storage_transition(
        base_dummy_transition,
        dummy_planner_state,
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=replay_dummy_transition,
        sample_batch_size=batch_size,
    )

    actor_dummy_transition = _to_storage_transition(
        base_dummy_transition,
        planner_states=None,
    )
    actor_replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=actor_dummy_transition,
        sample_batch_size=batch_size,
    )

    rng, rb_key, actor_rb_key = jax.random.split(rng, 3)
    buffer_state = replay_buffer.init(rb_key)
    actor_buffer_state = actor_replay_buffer.init(actor_rb_key)

    critic_loss_fn, actor_loss_fn = awac_losses.make_losses(
        sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        awac_lambda=awac_lambda,
        use_bro=use_bro,
        max_weight=max_weight,
    )
    critic_update = gradients.gradient_update_fn(
        critic_loss_fn, qr_optimizer, pmap_axis_name=None, has_aux=True
    )
    actor_update = gradients.gradient_update_fn(
        actor_loss_fn, policy_optimizer, pmap_axis_name=None, has_aux=True
    )

    def _collect_experience(
        training_state: TrainingState,
        env_local: envs.Env,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        count_env_steps: bool,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        policy = make_policy(
            (training_state.normalizer_params, training_state.policy_params),
            deterministic=False,
        )

        def step_fn(carry, _):
            state, k = carry
            k, action_key = jax.random.split(k)
            action, _ = policy(state.obs, action_key)
            next_state = env_local.step(state, action)
            extras = _build_extras(next_state.info, next_state.done)
            if planner_mode:
                extras["policy_extras"]["planner_state"] = state
            transition = Transition(
                observation=state.obs,
                action=action,
                reward=next_state.reward,
                discount=jnp.asarray(1.0 - next_state.done, dtype=jnp.float32),
                next_observation=next_state.obs,
                extras=extras,
            )
            return (next_state, k), transition

        (env_state, key), transitions = jax.lax.scan(
            step_fn, (env_state, key), (), length=rollout_length
        )

        planner_states = None
        if planner_mode:
            raw_planner_states = transitions.extras["policy_extras"]["planner_state"]
            transitions = _strip_policy_extras(transitions)
            time_len, batch_dim = raw_planner_states.reward.shape[:2]
            planner_states = _flatten_time_batch_tree(
                raw_planner_states, time_len, batch_dim
            )

        transitions = _flatten_transition_batch(transitions)
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            remove_pixels(transitions.observation),
        )
        transitions = _to_storage_transition(transitions, planner_states)

        buffer_state = replay_buffer.insert(buffer_state, transitions)
        env_steps = training_state.env_steps
        if count_env_steps:
            env_steps = env_steps + rollout_length * action_repeat * num_envs
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=env_steps,
        )
        return training_state, env_state, buffer_state, key

    def collect_real_experience(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        return _collect_experience(
            training_state,
            env,
            env_state,
            buffer_state,
            key,
            True,
        )

    def push_planner_batch_to_actor_buffer(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        actor_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[ReplayBufferState, ReplayBufferState, PRNGKey, jax.Array]:
        if not planner_mode:
            return (
                buffer_state,
                actor_buffer_state,
                key,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        assert controller is not None
        assert planner_params_template is not None

        key, planner_key = jax.random.split(key)
        buffer_state, sampled = replay_buffer.sample(buffer_state)
        sampled_real = float32(_strip_policy_extras(sampled))
        planner_model_params = _planner_model_params(training_state)
        sampled_planner, avg_rollout_return = _planner_supervised_batch(
            sampled,
            controller,
            planner_params_template,
            planner_key,
            planner_model_params,
            planner_rollout_steps,
        )
        sampled_planner = float32(_strip_policy_extras(sampled_planner))

        # Keep the original real transition in actor replay in addition to planner data.
        actor_batch = _concat_transition_batches(sampled_planner, sampled_real)
        actor_batch = _to_storage_transition(actor_batch, planner_states=None)
        actor_buffer_state = actor_replay_buffer.insert(actor_buffer_state, actor_batch)
        return buffer_state, actor_buffer_state, key, avg_rollout_return

    def sgd_step(
        carry: Tuple[TrainingState, ReplayBufferState, ReplayBufferState, PRNGKey, int],
        unused_t,
    ) -> Tuple[
        Tuple[TrainingState, ReplayBufferState, ReplayBufferState, PRNGKey, int],
        Metrics,
    ]:
        training_state, buffer_state, actor_buffer_state, key, count = carry
        key, key_critic, key_actor, key_planner = jax.random.split(key, 4)

        buffer_state, sampled = replay_buffer.sample(buffer_state)
        sampled_for_planner = sampled
        sampled = float32(sampled)

        critic_transitions = _strip_policy_extras(sampled)
        planner_avg_rollout_return = jnp.asarray(0.0, dtype=jnp.float32)

        if actor_update_source == "planner_replay":
            actor_buffer_state, actor_transitions = actor_replay_buffer.sample(
                actor_buffer_state
            )
            actor_transitions = float32(actor_transitions)
        elif actor_update_source == "planner_online":
            assert (
                planner_mode
                and controller is not None
                and planner_params_template is not None
            )
            actor_transitions, planner_avg_rollout_return = _planner_supervised_batch(
                sampled_for_planner,
                controller,
                planner_params_template,
                key_planner,
                _planner_model_params(training_state),
                planner_rollout_steps,
            )
            actor_transitions = _strip_policy_extras(actor_transitions)
            actor_transitions = float32(actor_transitions)
            # Include real transitions alongside planner rollouts for actor updates.
            actor_transitions = _concat_transition_batches(
                actor_transitions, critic_transitions
            )
        else:
            actor_transitions = critic_transitions

        (critic_loss, critic_aux), qr_params, qr_optimizer_state = critic_update(
            training_state.qr_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qr_params,
            critic_transitions,
            key_critic,
            optimizer_state=training_state.qr_optimizer_state,
            params=training_state.qr_params,
        )

        polyak = lambda target, new: jax.tree.map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)

        if actor_update_source == "planner_online":
            total_actor_samples = actor_transitions.reward.shape[0]
            if total_actor_samples % batch_size != 0:
                raise ValueError(
                    "planner_online/all_planner_actions expects actor transitions "
                    f"divisible by batch_size. got={total_actor_samples}, batch_size={batch_size}"
                )
            key_actor, key_perm, key_grad = jax.random.split(key_actor, 3)
            num_actor_minibatches = total_actor_samples // batch_size

            def _convert_actor_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (num_actor_minibatches, -1) + x.shape[1:])
                return x

            shuffled_actor_data = jax.tree_util.tree_map(
                _convert_actor_data, actor_transitions
            )

            def _actor_step(carry, minibatch):
                policy_params, policy_optimizer_state, k = carry
                k, step_key = jax.random.split(k)
                (
                    (actor_loss_i, aux_i),
                    new_policy_params_i,
                    new_policy_optimizer_state_i,
                ) = actor_update(
                    policy_params,
                    training_state.normalizer_params,
                    qr_params,
                    minibatch,
                    step_key,
                    optimizer_state=policy_optimizer_state,
                    params=policy_params,
                )
                return (
                    new_policy_params_i,
                    new_policy_optimizer_state_i,
                    k,
                ), (
                    actor_loss_i,
                    aux_i,
                )

            (
                (new_policy_params, new_policy_optimizer_state, _),
                (actor_losses, actor_auxes),
            ) = jax.lax.scan(
                _actor_step,
                (
                    training_state.policy_params,
                    training_state.policy_optimizer_state,
                    key_grad,
                ),
                shuffled_actor_data,
                length=num_actor_minibatches,
            )
            actor_loss = jnp.mean(actor_losses)
            aux = jax.tree.map(jnp.mean, actor_auxes)
        else:
            (
                (actor_loss, aux),
                new_policy_params,
                new_policy_optimizer_state,
            ) = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                qr_params,
                actor_transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
                params=training_state.policy_params,
            )

        should_update_actor = count % num_critic_updates_per_actor_update == 0
        update_if_needed = lambda x, y: jnp.where(should_update_actor, x, y)
        policy_params = jax.tree_map(
            update_if_needed, new_policy_params, training_state.policy_params
        )
        policy_optimizer_state = jax.tree_map(
            update_if_needed,
            new_policy_optimizer_state,
            training_state.policy_optimizer_state,
        )

        new_training_state = training_state.replace(  # type: ignore
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            qr_optimizer_state=qr_optimizer_state,
            qr_params=qr_params,
            target_qr_params=new_target_qr_params,
            gradient_steps=training_state.gradient_steps + 1,
        )
        actor_aux = {f"actor/{k}": v for k, v in aux.items()}
        critic_aux = {f"critic/{k}": v for k, v in critic_aux.items()}
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "planner/avg_rollout_return": planner_avg_rollout_return,
            "actor/loss_is_finite": jnp.isfinite(actor_loss).astype(jnp.float32),
            "critic/loss_is_finite": jnp.isfinite(critic_loss).astype(jnp.float32),
            "batch/reward_mean": jnp.mean(critic_transitions.reward),
            "batch/discount_mean": jnp.mean(critic_transitions.discount),
            "batch/action_abs_mean": jnp.mean(jnp.abs(critic_transitions.action)),
            "batch/truncation_mean": jnp.mean(
                critic_transitions.extras["state_extras"]["truncation"]
            ),
            "batch/observation_finite_fraction": _tree_finite_fraction(
                critic_transitions.observation
            ),
            "batch/action_finite_fraction": _tree_finite_fraction(
                critic_transitions.action
            ),
            **critic_aux,
            **actor_aux,
        }
        return (
            new_training_state,
            buffer_state,
            actor_buffer_state,
            key,
            count + 1,
        ), metrics

    def training_step_jitted(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        actor_buffer_state: ReplayBufferState,
        training_key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, ReplayBufferState, Metrics]:
        (
            (
                training_state,
                buffer_state,
                actor_buffer_state,
                *_,
            ),
            metrics,
        ) = jax.lax.scan(
            sgd_step,
            (training_state, buffer_state, actor_buffer_state, training_key, 0),
            (),
            length=grad_updates_per_step,
        )
        return training_state, buffer_state, actor_buffer_state, metrics

    env_steps_per_experience_call = rollout_length * action_repeat * num_envs
    planner_prefill_batch_size = num_envs

    sim_prefill_steps_effective = 0
    if planner_mode and sim_prefill_steps > 0:
        sim_prefill_steps_effective = sim_prefill_steps
    sim_prefill_calls = 0
    sim_transitions = 0
    if sim_prefill_steps_effective > 0:
        sim_transitions_per_prefill_call = (
            planner_prefill_batch_size * planner_rollout_steps
        )
        sim_prefill_calls = -(
            -sim_prefill_steps_effective // sim_transitions_per_prefill_call
        )
        sim_transitions = sim_prefill_calls * sim_transitions_per_prefill_call
    num_prefill_experience_call = -(-min_replay_size // env_steps_per_experience_call)
    num_prefill_env_steps = num_prefill_experience_call * env_steps_per_experience_call
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_experience_call)
    )

    if not eval_env:
        eval_env = environment
    evaluator = ConstraintsEvaluator(
        eval_env,
        lambda params: make_policy(params, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=jax.random.PRNGKey(seed + 1),
        budget=float("inf"),
        num_episodes=num_eval_episodes,
    )
    planner_eval_enabled = (
        planner_mode and controller is not None and planner_params_template is not None
    )

    planner_evaluator = None
    if planner_eval_enabled:

        def _planner_eval_policy_fn(_):
            def _planner_eval_policy(state: envs.State, rng: PRNGKey, params):
                assert controller is not None
                assert planner_params_template is not None
                del rng
                model_params, planner_params = params
                if _is_batched(state):
                    planner_params_out, _ = jax.vmap(
                        lambda s, p: controller.optimize(s, p, model_params)
                    )(state, planner_params)
                    action = planner_params_out.actions[:, 0, :]
                else:
                    planner_params_out, _ = controller.optimize(
                        state, planner_params, model_params
                    )
                    action = planner_params_out.actions[0]
                return action, (model_params, planner_params_out), {}

            return _planner_eval_policy

        planner_evaluator = Evaluator(
            eval_env,
            _planner_eval_policy_fn,
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=jax.random.PRNGKey(seed + 2),
        )

    def run_planner_evaluation(
        training_state: TrainingState,
        key: PRNGKey,
        prefix: str = "planner",
    ) -> tuple[dict[str, float], PRNGKey]:
        if not planner_eval_enabled or planner_evaluator is None:
            return {}, key
        assert planner_params_template is not None

        model_params = _planner_model_params(training_state)
        key, params_key = jax.random.split(key)
        planner_params = _planner_params_for_batch(
            planner_params_template, params_key, num_eval_envs
        )
        planner_eval_params = (model_params, planner_params)
        raw_metrics = planner_evaluator.run_evaluation(
            planner_eval_params,
            training_metrics={},
        )
        metrics: dict[str, float] = {}
        for name, value in raw_metrics.items():
            if "cost" in name:
                continue
            if name.startswith("eval/"):
                metrics[f"eval/{prefix}/{name[5:]}"] = float(value)
            else:
                metrics[f"{prefix}/{name}"] = float(value)
        return metrics, key

    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.policy_params),
            training_metrics={},
        )
        planner_eval_metrics, rng = run_planner_evaluation(training_state, rng)
        metrics = {**metrics, **planner_eval_metrics}
        logging.info(metrics)
        progress_fn(0, metrics)

    def pretrain_critic_on_dataset(
        training_state: TrainingState,
        dataset: Transition,
        key: PRNGKey,
        steps: int,
    ) -> Tuple[TrainingState, PRNGKey, dict[str, jnp.ndarray]]:
        if steps <= 0:
            return training_state, key, {}

        dataset = float32(_strip_policy_extras(dataset))
        dataset_size = int(dataset.reward.shape[0])
        if dataset_size <= 0:
            return training_state, key, {}

        def step_fn(carry, _):
            ts, k = carry
            k, batch_key, critic_key = jax.random.split(k, 3)
            batch_indices = jax.random.randint(
                batch_key, (batch_size,), 0, dataset_size
            )
            sampled = jax.tree.map(lambda x: x[batch_indices], dataset)
            (critic_loss, _), qr_params, qr_optimizer_state = critic_update(
                ts.qr_params,
                ts.policy_params,
                ts.normalizer_params,
                ts.target_qr_params,
                sampled,
                critic_key,
                optimizer_state=ts.qr_optimizer_state,
                params=ts.qr_params,
            )
            polyak = lambda target, new: jax.tree.map(
                lambda x, y: x * (1 - tau) + y * tau, target, new
            )
            new_target_qr_params = polyak(ts.target_qr_params, qr_params)
            ts = ts.replace(  # type: ignore
                qr_optimizer_state=qr_optimizer_state,
                qr_params=qr_params,
                target_qr_params=new_target_qr_params,
                gradient_steps=ts.gradient_steps + 1,
            )
            return (ts, k), critic_loss

        (training_state, key), losses = jax.lax.scan(
            step_fn,
            (training_state, key),
            (),
            length=steps,
        )
        metrics = {"pretrain/critic_loss": jnp.mean(losses)}
        return training_state, key, metrics

    def prefill_real_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, _):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, _ = collect_real_experience(ts, es, bs, k)
            return (ts, es, bs, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_experience_call,
        )[0]

    t = time.time()
    if sim_prefill_calls > 0:
        if controller is None or planner_params_template is None:
            raise ValueError(
                "Sim prefill requires planner controller and planner params."
            )
        logging.info(
            "prefill from sim only: sim_prefill_steps=%s sim_prefill_calls=%s sim_transitions=%s",
            sim_prefill_steps_effective,
            sim_prefill_calls,
            sim_transitions,
        )

        def collect_sim_prefill_batch(
            training_state: TrainingState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, Transition, PRNGKey]:
            key, reset_key, plan_key = jax.random.split(key, 3)
            real_keys = jax.random.split(reset_key, num_envs)
            real_state = reset_fn(real_keys)
            if not hasattr(controller, "optimize"):
                raise ValueError(
                    "Controller must implement optimize for planner prefill."
                )
            planner_state = real_state
            prefill_batch_size = num_envs
            seed_transitions = Transition(
                observation=planner_state.obs,
                action=jnp.zeros((prefill_batch_size, action_size), dtype=jnp.float32),
                reward=jnp.zeros((prefill_batch_size,), dtype=jnp.float32),
                discount=jnp.ones((prefill_batch_size,), dtype=jnp.float32),
                next_observation=planner_state.obs,
                extras={
                    "state_extras": {
                        "truncation": jnp.zeros(
                            (prefill_batch_size,), dtype=jnp.float32
                        )
                    },
                    "policy_extras": {"planner_state": planner_state},
                },
            )
            assert planner_params_template is not None
            sim_transitions, _ = _planner_supervised_batch(
                seed_transitions,
                controller,
                planner_params_template,
                plan_key,
                _planner_model_params(training_state),
                planner_rollout_steps,
            )
            sim_transitions = _strip_policy_extras(sim_transitions)
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                remove_pixels(sim_transitions.observation),
            )
            training_state = training_state.replace(normalizer_params=normalizer_params)  # type: ignore
            return training_state, sim_transitions, key

        critic_pretrain_steps = int(jnp.ceil(sim_transitions * critic_pretrain_ratio))
        pretrain_losses = []
        base_steps = (
            critic_pretrain_steps // sim_prefill_calls if sim_prefill_calls > 0 else 0
        )
        extra_steps = (
            critic_pretrain_steps % sim_prefill_calls if sim_prefill_calls > 0 else 0
        )

        for prefill_idx in range(sim_prefill_calls):
            rng, collect_key = jax.random.split(rng)
            training_state, sim_batch, _ = collect_sim_prefill_batch(
                training_state, collect_key
            )
            steps_this_batch = (
                base_steps + (1 if prefill_idx < extra_steps else 0)
                if critic_pretrain_steps > 0
                else 0
            )
            if steps_this_batch > 0:
                rng, pretrain_key = jax.random.split(rng)
                training_state, _, pretrain_metrics = pretrain_critic_on_dataset(
                    training_state, sim_batch, pretrain_key, steps_this_batch
                )
                if pretrain_metrics:
                    pretrain_losses.append(pretrain_metrics["pretrain/critic_loss"])

        if critic_pretrain_steps > 0 and pretrain_losses:
            logging.info(
                "critic pretrain steps %s loss %s",
                critic_pretrain_steps,
                jnp.mean(jnp.stack(pretrain_losses)),
            )
    rng, prefill_key = jax.random.split(rng)
    if num_prefill_experience_call > 0:
        logging.info(
            "prefill from real: min_replay_size=%s prefill_calls=%s",
            min_replay_size,
            num_prefill_experience_call,
        )
        training_state, env_state, buffer_state, _ = prefill_real_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

    if actor_update_source == "planner_replay":
        planner_transitions_per_state = planner_rollout_steps if planner_mode else 1
        actor_transitions_per_planner_batch = batch_size * planner_transitions_per_state
        num_actor_prefill_batches = -(
            -min_actor_replay_size // actor_transitions_per_planner_batch
        )
        replay_size_for_actor_prefill = int(jnp.sum(replay_buffer.size(buffer_state)))
        if replay_size_for_actor_prefill == 0:
            logging.info(
                "skipping actor replay prefill because critic replay buffer is empty."
            )
        else:
            for _ in range(num_actor_prefill_batches):
                (
                    buffer_state,
                    actor_buffer_state,
                    rng,
                    _,
                ) = push_planner_batch_to_actor_buffer(
                    training_state,
                    buffer_state,
                    actor_buffer_state,
                    rng,
                )

    replay_size = jnp.sum(replay_buffer.size(buffer_state))
    logging.info("replay size after prefill %s", replay_size)
    if actor_update_source == "planner_replay":
        actor_replay_size = jnp.sum(actor_replay_buffer.size(actor_buffer_state))
        logging.info("actor replay size after prefill %s", actor_replay_size)
    training_walltime = time.time() - t

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        actor_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, envs.State, ReplayBufferState, ReplayBufferState, Metrics
    ]:
        training_state, env_state, buffer_state, key = collect_real_experience(
            training_state, env_state, buffer_state, key
        )
        planner_replay_rollout_return = jnp.asarray(0.0, dtype=jnp.float32)
        if actor_update_source == "planner_replay":
            for _ in range(planner_batches_per_step):
                (
                    buffer_state,
                    actor_buffer_state,
                    key,
                    planner_return,
                ) = push_planner_batch_to_actor_buffer(
                    training_state,
                    buffer_state,
                    actor_buffer_state,
                    key,
                )
                planner_replay_rollout_return = (
                    planner_replay_rollout_return + planner_return
                )
            planner_replay_rollout_return = (
                planner_replay_rollout_return / planner_batches_per_step
            )

        (
            training_state,
            buffer_state,
            actor_buffer_state,
            training_metrics,
        ) = training_step_jitted(
            training_state,
            buffer_state,
            actor_buffer_state,
            key,
        )
        training_metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        if actor_update_source == "planner_replay":
            training_metrics["actor_buffer_current_size"] = actor_replay_buffer.size(
                actor_buffer_state
            )
            training_metrics[
                "planner/avg_rollout_return"
            ] = planner_replay_rollout_return
        return (
            training_state,
            env_state,
            buffer_state,
            actor_buffer_state,
            training_metrics,
        )

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        actor_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState,
        envs.State,
        ReplayBufferState,
        ReplayBufferState,
        Metrics,
    ]:
        def f(carry, _):
            ts, es, bs, abs_, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, abs_, metrics = training_step(ts, es, bs, abs_, k)
            return (ts, es, bs, abs_, new_key), metrics

        (
            (
                training_state,
                env_state,
                buffer_state,
                actor_buffer_state,
                key,
            ),
            metrics,
        ) = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, actor_buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, actor_buffer_state, metrics

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        actor_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState,
        envs.State,
        ReplayBufferState,
        ReplayBufferState,
        Metrics,
    ]:
        nonlocal training_walltime
        t0 = time.time()
        (
            training_state,
            env_state,
            buffer_state,
            actor_buffer_state,
            metrics,
        ) = training_epoch(
            training_state,
            env_state,
            buffer_state,
            actor_buffer_state,
            key,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t0
        training_walltime += epoch_training_time
        sps = (
            env_steps_per_experience_call * num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, actor_buffer_state, metrics

    current_step = 0
    metrics = {}
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)
        rng, epoch_key = jax.random.split(rng)
        (
            training_state,
            env_state,
            buffer_state,
            actor_buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state,
            env_state,
            buffer_state,
            actor_buffer_state,
            epoch_key,
        )
        current_step = int(training_state.env_steps)

        if checkpoint_logdir:
            params = (
                training_state.normalizer_params,
                training_state.policy_params,
                training_state.qr_params,
                training_state.target_qr_params,
                training_state.policy_optimizer_state,
                training_state.qr_optimizer_state,
            )
            dummy_ckpt_config = config_dict.ConfigDict()
            checkpoint.save(checkpoint_logdir, current_step, params, dummy_ckpt_config)

        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.policy_params),
            training_metrics,
        )
        planner_eval_metrics, rng = run_planner_evaluation(training_state, rng)
        metrics = {**metrics, **planner_eval_metrics}
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    params = (
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.qr_params,
        training_state.target_qr_params,
        training_state.policy_optimizer_state,
        training_state.qr_optimizer_state,
    )
    logging.info("total steps: %s", total_steps)
    return make_policy, params, metrics
