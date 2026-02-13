"""AWAC-style losses for tree-MPC rollouts."""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.networks import SafeSACNetworks

Transition: TypeAlias = types.Transition


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
    awac_lambda: float,
    use_bro: bool,
    max_weight: float | None = None,
):
    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        action = transitions.action
        q_old_action = qr_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        key, next_key = jax.random.split(key)
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample(next_dist_params, next_key)
        next_q = qr_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        next_v = _reduce_q(next_q, use_bro)
        target_q = transitions.reward * reward_scaling + transitions.discount * (
            discounting * next_v
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
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        pi_action = parametric_action_distribution.sample(dist_params, key)
        q_pi = qr_network.apply(
            normalizer_params, q_params, transitions.observation, pi_action
        )
        v = _reduce_q(q_pi, use_bro)
        q = qr_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        q = _reduce_q(q, use_bro)
        advantage = q - v
        unclipped_weights = jnp.exp(advantage / awac_lambda)
        weights = unclipped_weights
        if max_weight is not None:
            weights = jnp.minimum(weights, max_weight)
        weights = jax.lax.stop_gradient(weights)
        log_prob = parametric_action_distribution.log_prob(
            dist_params, atanh_with_slack(transitions.action)
        )
        loss = -(log_prob * weights).mean()
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
        }
        return loss, aux

    return critic_loss, actor_loss
