"""MPO-style training with optional TreeMPC actor supervision."""

import time
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.envs.base import Wrapper
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint
from brax.training.types import Params, PRNGKey
from flax import struct
from ml_collections import config_dict

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.awac_mpc import losses as awac_losses
from ss2r.algorithms.mpc.tree_mpc import (
    TreeMPC,
    TreeMPCModelParams,
    TreeMPCParams,
    make_task,
)
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.pytree_uniform_sampling_queue import (
    PytreeUniformSamplingQueue,
)
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


@struct.dataclass
class TrainingState:
    policy_optimizer_state: optax.OptState
    policy_params: Params
    target_policy_params: Params
    qr_optimizer_state: optax.OptState
    qr_params: Params
    target_qr_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _copy_pytree(tree: Any) -> Any:
    """Materializes fresh arrays so donated pytrees do not alias leaves."""
    return jax.tree_util.tree_map(jnp.copy, tree)


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
        target_policy_params=_copy_pytree(policy_params),
        qr_optimizer_state=qr_optimizer_state,
        qr_params=qr_params,
        target_qr_params=_copy_pytree(qr_params),
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
        target_policy_params=training_state.target_policy_params,
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


def _float32_training_transition(transitions: Transition) -> Transition:
    """Promotes trainable transition fields without mutating saved planner state."""
    transition_no_policy = _strip_policy_extras(transitions)
    transition_no_policy = float32(transition_no_policy)
    return transition_no_policy._replace(
        extras={
            "state_extras": transition_no_policy.extras["state_extras"],
            "policy_extras": transitions.extras["policy_extras"],
        }
    )


def _flatten_leading_dims(tree: Any, leading_dims: int = 2) -> Any:
    def _flatten_leaf(x):
        if not hasattr(x, "shape") or x.ndim < leading_dims:
            return x
        flattened_dim = 1
        for dim in x.shape[:leading_dims]:
            flattened_dim *= dim
        new_shape = (flattened_dim,) + x.shape[leading_dims:]
        return x.reshape(new_shape)

    return jax.tree.map(_flatten_leaf, tree)


def _shuffle_and_batch_actor_transitions(
    actor_transitions: Transition,
    batch_size: int,
    key: PRNGKey,
) -> Transition:
    total_actor_samples = actor_transitions.reward.shape[0]
    if total_actor_samples % batch_size != 0:
        raise ValueError(
            "planner_online actor transitions must be divisible by batch_size. "
            f"got={total_actor_samples}, batch_size={batch_size}"
        )
    permutation = jax.random.permutation(key, total_actor_samples)
    num_actor_minibatches = total_actor_samples // batch_size

    def _shuffle_and_reshape(x: jnp.ndarray):
        x = x[permutation]
        return x.reshape((num_actor_minibatches, batch_size) + x.shape[1:])

    return jax.tree.map(_shuffle_and_reshape, actor_transitions)


class _PlannerActionRepeatWrapper(Wrapper):
    """Minimal wrapper that repeats each planner env step."""

    def __init__(self, env: envs.Env, action_repeat: int):
        super().__init__(env)
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1.")
        self.env = env
        self.action_repeat = int(action_repeat)

    def step(self, state: envs.State, action: jax.Array) -> envs.State:
        if self.action_repeat == 1:
            return self.env.step(state, action)

        def _repeat_fn(carry, _):
            next_state = self.env.step(carry, action)
            return next_state, next_state.reward

        next_state, rewards = jax.lax.scan(
            _repeat_fn, state, (), length=self.action_repeat
        )
        return next_state.replace(reward=jnp.sum(rewards, axis=0))

    def compress_planner_state(self, state: Any) -> Any:
        if hasattr(self.env, "compress_planner_state"):
            return self.env.compress_planner_state(state)
        return state

    def restore_planner_state(self, planner_state: Any) -> Any:
        if hasattr(self.env, "restore_planner_state"):
            return self.env.restore_planner_state(planner_state)
        return planner_state


def _planner_supervised_batch(
    transitions: Transition,
    controller,
    planner_params_template: TreeMPCParams,
    key: PRNGKey,
    planner_model_params: TreeMPCModelParams | None = None,
    restore_planner_state_fn: Callable[[Any], Any] | None = None,
) -> tuple[Transition, jax.Array]:
    if not getattr(controller, "rollout_actions", True):
        return _strip_policy_extras(transitions), jnp.zeros((), dtype=jnp.float32)

    planner_states = transitions.extras["policy_extras"].get("planner_state", None)
    if planner_states is None:
        raise ValueError(
            "Planner-supervised mode requires planner_state in replay data."
        )
    if not hasattr(controller, "optimize_with_candidates"):
        raise ValueError(
            "Planner-supervised mode requires controller.optimize_with_candidates."
        )
    if restore_planner_state_fn is not None:
        planner_states = jax.vmap(restore_planner_state_fn)(planner_states)
    batch_size = transitions.reward.shape[0]
    planner_params = _planner_params_for_batch(planner_params_template, key, batch_size)
    _, rollouts = jax.vmap(
        lambda planner_state, planner_params_i: controller.optimize_with_candidates(
            planner_state,
            planner_params_i,
            planner_model_params,
        )
    )(
        planner_states,
        planner_params,
    )
    avg_rollout_return = jnp.mean(rollouts.returns)
    actor_transitions = Transition(
        observation=_flatten_leading_dims(rollouts.all_traj_obs, 3),
        action=_flatten_leading_dims(rollouts.all_traj_actions, 3),
        reward=_flatten_leading_dims(rollouts.all_traj_rewards, 3),
        discount=_flatten_leading_dims(rollouts.all_traj_discount, 3),
        next_observation=_flatten_leading_dims(rollouts.all_traj_next_obs, 3),
        extras={
            "state_extras": {
                "truncation": _flatten_leading_dims(rollouts.all_traj_truncation, 3),
            },
            "policy_extras": {
                "baseline_value": _flatten_leading_dims(rollouts.all_traj_returns, 3),
            },
        },
    )
    return actor_transitions, avg_rollout_return


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
    policy_target_tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    num_critic_updates_per_actor_update: int = 1,
    deterministic_eval: bool = False,
    reset_on_eval: bool = True,
    planner_eval: bool = True,
    rollout_length: int = 1,
    mpo_eta: float = 1.0,
    mpo_eta_epsilon: float = 0.1,
    mpo_eta_min: float = 1e-3,
    mpo_eta_opt_maxiter: int = 10,
    mpo_log_prob_min: float = -100.0,
    mpo_num_action_samples: int = 16,
    n_critics: int = 2,
    use_bro: bool = True,
    actor_grad_clip_norm: float = 1.0,
    critic_grad_clip_norm: float = 1.0,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    planner_environment: Optional[envs.Env] = None,
    controller_kwargs: Optional[dict[str, Any]] = None,
    policy_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    value_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    activation: Callable[[jax.Array], jax.Array] = jax.nn.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    env_reset_every_steps: int = 0,
):
    if not 0.0 <= policy_target_tau <= 1.0:
        raise ValueError(
            f"policy_target_tau must be in [0, 1], got {policy_target_tau}."
        )
    if mpo_eta <= 0.0:
        raise ValueError(f"mpo_eta must be > 0, got {mpo_eta}.")
    if mpo_eta_epsilon <= 0.0:
        raise ValueError(f"mpo_eta_epsilon must be > 0, got {mpo_eta_epsilon}.")
    if mpo_eta_min <= 0.0:
        raise ValueError(f"mpo_eta_min must be > 0, got {mpo_eta_min}.")
    if mpo_eta_opt_maxiter < 1:
        raise ValueError(
            f"mpo_eta_opt_maxiter must be >= 1, got {mpo_eta_opt_maxiter}."
        )
    if mpo_log_prob_min > 0.0:
        raise ValueError(f"mpo_log_prob_min must be <= 0, got {mpo_log_prob_min}.")
    if actor_grad_clip_norm <= 0.0:
        raise ValueError(
            f"actor_grad_clip_norm must be > 0, got {actor_grad_clip_norm}."
        )
    if critic_grad_clip_norm <= 0.0:
        raise ValueError(
            f"critic_grad_clip_norm must be > 0, got {critic_grad_clip_norm}."
        )
    if max_replay_size is None:
        max_replay_size = num_timesteps
    if planner_environment is None:
        raise ValueError("planner_environment is required for AWAC-MPC.")
    if env_reset_every_steps < 0:
        raise ValueError(
            f"env_reset_every_steps must be >= 0, got {env_reset_every_steps}."
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
        use_bro=use_bro,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        safe=False,
    )
    make_policy = make_inference_fn(sac_network)

    make_optimizer = lambda lr, grad_clip_norm: optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate=lr),
    )
    policy_optimizer = make_optimizer(learning_rate, actor_grad_clip_norm)
    qr_optimizer = make_optimizer(critic_learning_rate, critic_grad_clip_norm)

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
                target_policy_params=_copy_pytree(params[1]),
                qr_params=params[3],
                target_qr_params=_copy_pytree(params[3]),
                policy_optimizer_state=restore_state(
                    params[6], training_state.policy_optimizer_state
                ),
                qr_optimizer_state=restore_state(
                    params[8], training_state.qr_optimizer_state
                ),
            )
        elif len(params) >= 7:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=params[0],
                policy_params=params[1],
                target_policy_params=_copy_pytree(params[6]),
                qr_params=params[2],
                target_qr_params=_copy_pytree(params[3]),
                policy_optimizer_state=restore_state(
                    params[4], training_state.policy_optimizer_state
                ),
                qr_optimizer_state=restore_state(
                    params[5], training_state.qr_optimizer_state
                ),
            )
        else:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=params[0],
                policy_params=params[1],
                target_policy_params=_copy_pytree(params[1]),
                qr_params=params[2],
                target_qr_params=_copy_pytree(params[3]),
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

    planner_env = _PlannerActionRepeatWrapper(planner_environment, action_repeat)
    compress_planner_state = getattr(
        planner_env, "compress_planner_state", lambda state: state
    )
    restore_planner_state = getattr(
        planner_env, "restore_planner_state", lambda state: state
    )
    controller_kwargs = dict(controller_kwargs or {})
    controller_kwargs["use_bro"] = bool(use_bro)
    controller_kwargs["gamma"] = discounting
    controller_kwargs["reward_scaling"] = reward_scaling
    task = make_task(planner_env)  # type: ignore[arg-type]
    controller = TreeMPC(task=task, sac_network=sac_network, **controller_kwargs)
    planner_params_template = _init_planner_params(controller, seed, None)

    dummy_planner_state = jax.tree.map(
        lambda x: x[0]
        if hasattr(x, "shape") and x.shape and x.shape[0] == num_envs
        else x,
        env_state,
    )
    dummy_planner_state = compress_planner_state(dummy_planner_state)

    replay_dummy_transition = _to_storage_transition(
        base_dummy_transition,
        dummy_planner_state,
    )
    # Planner state is stored in replay extras, so keep the buffer in pytree form.
    replay_buffer = PytreeUniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=replay_dummy_transition,
        sample_batch_size=batch_size,
    )

    rng, rb_key = jax.random.split(rng)
    buffer_state = replay_buffer.init(rb_key)

    critic_loss_fn, actor_loss_fn = awac_losses.make_losses(
        sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        mpo_eta_init=mpo_eta,
        mpo_eta_epsilon=mpo_eta_epsilon,
        mpo_eta_min=mpo_eta_min,
        mpo_eta_opt_maxiter=mpo_eta_opt_maxiter,
        mpo_log_prob_min=mpo_log_prob_min,
        mpo_num_action_samples=mpo_num_action_samples,
        use_bro=use_bro,
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
        start_step = jnp.asarray(training_state.env_steps, dtype=jnp.int32) // (
            action_repeat * num_envs
        )

        def step_fn(carry, step_offset):
            state, k = carry
            k, action_key, reset_key = jax.random.split(k, 3)
            action, _ = policy(state.obs, action_key)
            next_state = env_local.step(state, action)
            discount = jnp.asarray(1.0 - next_state.done, dtype=jnp.float32)
            should_force_reset = False
            if count_env_steps and env_reset_every_steps > 0:
                should_force_reset = (
                    (start_step + step_offset + 1) % env_reset_every_steps
                ) == 0
                discount = jnp.where(
                    should_force_reset,
                    jnp.zeros_like(discount),
                    discount,
                )
            extras = _build_extras(next_state.info, next_state.done)
            extras["policy_extras"]["planner_state"] = compress_planner_state(state)
            transition = Transition(
                observation=state.obs,
                action=action,
                reward=next_state.reward,
                discount=discount,
                next_observation=next_state.obs,
                extras=extras,
            )
            if count_env_steps and env_reset_every_steps > 0:
                reset_keys = jax.random.split(reset_key, num_envs)
                reset_state = reset_fn(reset_keys)
                next_carry_state = jax.lax.cond(
                    should_force_reset,
                    lambda _: reset_state,
                    lambda _: next_state,
                    operand=None,
                )
            else:
                next_carry_state = next_state
            return (next_carry_state, k), transition

        (env_state, key), transitions = jax.lax.scan(
            step_fn,
            (env_state, key),
            jnp.arange(rollout_length, dtype=jnp.int32),
        )

        raw_planner_states = transitions.extras["policy_extras"]["planner_state"]
        transitions = _strip_policy_extras(transitions)
        time_len, batch_dim = transitions.reward.shape[:2]
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

    def sgd_step(
        carry: Tuple[TrainingState, ReplayBufferState, PRNGKey, int],
        unused_t,
    ) -> Tuple[Tuple[TrainingState, ReplayBufferState, PRNGKey, int], Metrics]:
        training_state, buffer_state, key, count = carry
        key, key_critic, key_perm, key_planner, key_actor = jax.random.split(key, 5)

        buffer_state, sampled = replay_buffer.sample(buffer_state)
        sampled = _float32_training_transition(sampled)

        critic_transitions = _strip_policy_extras(sampled)
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

        polyak = lambda target, new, coeff: jax.tree.map(
            lambda x, y: x * (1 - coeff) + y * coeff, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params, tau)
        actor_transitions, planner_avg_rollout_return = _planner_supervised_batch(
            sampled,
            controller,
            planner_params_template,
            key_planner,
            _planner_model_params(training_state.replace(qr_params=qr_params)),
            restore_planner_state_fn=restore_planner_state,
        )
        shuffled_actor_data = _shuffle_and_batch_actor_transitions(
            actor_transitions,
            batch_size,
            key_perm,
        )
        num_actor_minibatches = shuffled_actor_data.reward.shape[0]

        def _actor_step(carry, minibatch):
            policy_params, policy_optimizer_state, key = carry
            key, key_loss = jax.random.split(key)
            (
                (actor_loss_i, aux_i),
                new_policy_params_i,
                new_policy_optimizer_state_i,
            ) = actor_update(
                policy_params,
                training_state.target_policy_params,
                training_state.normalizer_params,
                qr_params,
                minibatch,
                key_loss,
                optimizer_state=policy_optimizer_state,
                params=policy_params,
            )
            return (
                new_policy_params_i,
                new_policy_optimizer_state_i,
                key,
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
                key_actor,
            ),
            shuffled_actor_data,
            length=num_actor_minibatches,
        )
        actor_loss = jnp.mean(actor_losses)
        aux = jax.tree.map(jnp.mean, actor_auxes)

        should_update_actor = count % num_critic_updates_per_actor_update == 0
        update_if_needed = lambda x, y: jnp.where(should_update_actor, x, y)
        policy_params = jax.tree.map(
            update_if_needed, new_policy_params, training_state.policy_params
        )
        policy_optimizer_state = jax.tree.map(
            update_if_needed,
            new_policy_optimizer_state,
            training_state.policy_optimizer_state,
        )
        new_target_policy_params = polyak(
            training_state.target_policy_params,
            policy_params,
            policy_target_tau,
        )
        new_training_state = training_state.replace(  # type: ignore
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            target_policy_params=new_target_policy_params,
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
            **critic_aux,
            **actor_aux,
        }
        return (
            new_training_state,
            buffer_state,
            key,
            count + 1,
        ), metrics

    def training_step_jitted(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        training_key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        (
            (
                training_state,
                buffer_state,
                *_,
            ),
            metrics,
        ) = jax.lax.scan(
            sgd_step,
            (training_state, buffer_state, training_key, 0),
            (),
            length=grad_updates_per_step,
        )
        return training_state, buffer_state, metrics

    env_steps_per_experience_call = rollout_length * action_repeat * num_envs
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

    planner_evaluator = None
    if planner_eval:

        def _planner_eval_policy_fn(_):
            def _planner_eval_policy(state: envs.State, rng: PRNGKey, params):
                del rng
                model_params, planner_params = params
                planner_params_out, _ = jax.vmap(
                    lambda s, p: controller.optimize_with_candidates(s, p, model_params)
                )(state, planner_params)
                action = controller.action_sequence(planner_params_out.actions)[:, 0, :]
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
        if not planner_eval:
            return {}, key
        assert planner_evaluator is not None
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

    prefill_real_replay_buffer = jax.jit(
        prefill_real_replay_buffer,
        donate_argnames=("training_state", "env_state", "buffer_state"),
    )

    t = time.time()
    rng, prefill_key = jax.random.split(rng)
    training_state, env_state, buffer_state, _ = prefill_real_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    replay_size = jnp.sum(replay_buffer.size(buffer_state))
    logging.info("replay size after prefill %s", replay_size)
    training_walltime = time.time() - t

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        training_state, env_state, buffer_state, key = collect_real_experience(
            training_state, env_state, buffer_state, key
        )

        (
            training_state,
            buffer_state,
            training_metrics,
        ) = training_step_jitted(
            training_state,
            buffer_state,
            key,
        )
        training_metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        )

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, _):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (
            (
                training_state,
                env_state,
                buffer_state,
                key,
            ),
            metrics,
        ) = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.jit(
        training_epoch,
        donate_argnames=("training_state", "env_state", "buffer_state"),
    )

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t0 = time.time()
        (
            training_state,
            env_state,
            buffer_state,
            metrics,
        ) = training_epoch(
            training_state,
            env_state,
            buffer_state,
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
        return training_state, env_state, buffer_state, metrics

    current_step = 0
    metrics = {}
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)
        rng, epoch_key = jax.random.split(rng)
        (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state,
            env_state,
            buffer_state,
            epoch_key,
        )
        if reset_on_eval:
            reset_keys = jax.random.split(epoch_key, num_envs)
            env_state = reset_fn(reset_keys)
        current_step = int(training_state.env_steps)

        if checkpoint_logdir:
            params = (
                training_state.normalizer_params,
                training_state.policy_params,
                training_state.qr_params,
                training_state.target_qr_params,
                training_state.policy_optimizer_state,
                training_state.qr_optimizer_state,
                training_state.target_policy_params,
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
        training_state.target_policy_params,
    )
    logging.info("total steps: %s", total_steps)
    return make_policy, params, metrics
