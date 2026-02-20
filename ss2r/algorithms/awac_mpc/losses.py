"""AWAC-style losses for tree-MPC rollouts."""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.networks import SafeSACNetworks
from ss2r.algorithms.sac.q_transforms import QTransformation

Transition: TypeAlias = types.Transition

AWAC_VALUE_SAMPLES = 16


def atanh_with_slack(y: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """
    Stable inverse tanh with configurable slack.
    Clips y to [-1+eps, 1-eps] before applying atanh.
    """
    y = y.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=jnp.float32)

    # Clip with "extra" margin away from ±1
    y = jnp.clip(y, -1.0 + eps_arr, 1.0 - eps_arr)

    # Equivalent to jnp.arctanh(y) but explicitly uses log1p (stable near 0 and 1)
    return 0.5 * (jnp.log1p(y) - jnp.log1p(-y))


def _reduce_q(q_values: jax.Array, use_bro: bool) -> jax.Array:
    if use_bro:
        return jnp.mean(q_values, axis=-1)
    return jnp.min(q_values, axis=-1)


def make_losses(
    sac_network: SafeSACNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    awac_lambda: float,
    normalize_advantage: bool,
    use_bro: bool,
    target_entropy: float | None = None,
    max_weight: float | None = None,
):
    target_entropy = -0.5 * action_size if target_entropy is None else target_entropy
    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(loss)

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
        target_q_fn: QTransformation,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        action = transitions.action
        q_old_action = qr_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        key, another_key = jax.random.split(key)

        def policy(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_dist_params = policy_network.apply(
                normalizer_params, policy_params, obs
            )
            next_action = parametric_action_distribution.sample_no_postprocessing(
                next_dist_params, key
            )
            next_log_prob = parametric_action_distribution.log_prob(
                next_dist_params, next_action
            )
            next_action = parametric_action_distribution.postprocess(next_action)
            return next_action, next_log_prob

        q_fn = lambda obs, action: qr_network.apply(
            normalizer_params, target_q_params, obs, action
        )
        target_q = target_q_fn(
            transitions,
            q_fn,
            policy,
            discounting,
            alpha,
            reward_scaling,
            another_key,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)
        loss = 0.5 * jnp.mean(jnp.square(q_error))
        aux = {
            "q_data_mean": jnp.mean(q_old_action),
            "q_data_std": jnp.std(q_old_action),
            "q_target_mean": jnp.mean(target_q),
            "q_target_std": jnp.std(target_q),
            "td_error_mean": jnp.mean(q_error),
            "td_error_abs_mean": jnp.mean(jnp.abs(q_error)),
            "td_error_abs_max": jnp.max(jnp.abs(q_error)),
        }
        return loss, aux

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        precomputed_advantage = transitions.extras["policy_extras"].get(
            "advantage", None
        )

        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        key_v, key_exploration = jax.random.split(key)
        v_sample_keys = jax.random.split(key_v, AWAC_VALUE_SAMPLES)
        v_actions_raw = jax.vmap(
            lambda sample_key: parametric_action_distribution.sample_no_postprocessing(
                dist_params, sample_key
            )
        )(v_sample_keys)
        v_actions = jax.vmap(parametric_action_distribution.postprocess)(v_actions_raw)
        q_pi_samples = jax.vmap(
            lambda sampled_action: qr_network.apply(
                normalizer_params, q_params, transitions.observation, sampled_action
            )
        )(v_actions)
        v = jnp.mean(_reduce_q(q_pi_samples, use_bro), axis=0)

        pi_action_raw = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key_exploration
        )
        pi_action_log_prob = parametric_action_distribution.log_prob(
            dist_params, pi_action_raw
        )
        q = qr_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        q = _reduce_q(q, use_bro)
        if precomputed_advantage is not None:
            advantage = precomputed_advantage
        else:
            advantage = q - v
        if normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        unclipped_weights = advantage
        weights = unclipped_weights
        if max_weight is not None:
            weights = jnp.clip(weights, -max_weight, max_weight)
        weights = jax.lax.stop_gradient(weights)
        log_prob = parametric_action_distribution.log_prob(
            dist_params, atanh_with_slack(transitions.action)
        )
        loss = -(log_prob * weights).mean()
        exploration_loss = (alpha * pi_action_log_prob).mean()
        loss += exploration_loss
        clip_fraction = (
            jnp.mean((unclipped_weights > max_weight).astype(jnp.float32))
            if max_weight is not None
            else jnp.zeros(())
        )
        aux = {
            "advantage_mean": jnp.mean(advantage),
            "advantage_std": jnp.std(advantage),
            "advantage_min": jnp.min(advantage),
            "advantage_max": jnp.max(advantage),
            "weight_mean": jnp.mean(weights),
            "weight_std": jnp.std(weights),
            "weight_min": jnp.min(weights),
            "weight_max": jnp.max(weights),
            "weight_clip_fraction": clip_fraction,
            "max_weight_limit": jnp.asarray(
                -1.0 if max_weight is None else max_weight, dtype=jnp.float32
            ),
            "log_prob_mean": jnp.mean(log_prob),
            "log_prob_std": jnp.std(log_prob),
            "log_prob_min": jnp.min(log_prob),
            "log_prob_max": jnp.max(log_prob),
            "v_pi_mean": jnp.mean(v),
            "q_data_mean": jnp.mean(q),
            "exploration_loss": exploration_loss,
            "awac_loss": loss - exploration_loss,
            "alpha": jnp.asarray(alpha, dtype=jnp.float32),
        }
        return loss, aux

    return alpha_loss, critic_loss, actor_loss
