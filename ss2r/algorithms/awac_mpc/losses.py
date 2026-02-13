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
    normalize_advantage: bool,
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
        if transitions.reward.ndim == 2:
            # Sequence advantage for committing to planner rollout and then
            # following pi, against baseline V_pi(s_t).
            first_obs = jax.tree.map(lambda x: x[0], transitions.observation)
            first_action = transitions.action[0]
            terminal_next_obs = jax.tree.map(
                lambda x: x[-1], transitions.next_observation
            )

            key_v0, key_vt = jax.random.split(key)
            dist_params_first = policy_network.apply(
                normalizer_params, policy_params, first_obs
            )
            pi_action_first = parametric_action_distribution.sample(
                dist_params_first, key_v0
            )
            v0 = _reduce_q(
                qr_network.apply(
                    normalizer_params, q_params, first_obs, pi_action_first
                ),
                use_bro,
            )
            dist_params_terminal = policy_network.apply(
                normalizer_params, policy_params, terminal_next_obs
            )
            pi_action_terminal = parametric_action_distribution.sample(
                dist_params_terminal, key_vt
            )
            v_terminal = _reduce_q(
                qr_network.apply(
                    normalizer_params, q_params, terminal_next_obs, pi_action_terminal
                ),
                use_bro,
            )

            truncation = transitions.extras["state_extras"]["truncation"]
            rewards = transitions.reward * reward_scaling
            step_discount = transitions.discount * discounting * (1 - truncation)
            discount_prefix = jnp.cumprod(step_discount, axis=0)
            reward_weights = jnp.concatenate(
                [jnp.ones_like(step_discount[:1]), discount_prefix[:-1]], axis=0
            )
            seq_return = jnp.sum(reward_weights * rewards, axis=0) + (
                discount_prefix[-1] * v_terminal
            )
            advantage = seq_return - v0
            if normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            unclipped_weights = jnp.exp(advantage / awac_lambda)
            weights = unclipped_weights
            if max_weight is not None:
                weights = jnp.minimum(weights, max_weight)
            weights = jax.lax.stop_gradient(weights)
            log_prob = parametric_action_distribution.log_prob(
                dist_params_first, atanh_with_slack(first_action)
            )
            loss = -(log_prob * weights).mean()
            q_first = _reduce_q(
                qr_network.apply(
                    normalizer_params,
                    q_params,
                    first_obs,
                    first_action,
                ),
                use_bro,
            )
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
                "v_pi_mean": jnp.mean(v0),
                "q_data_mean": jnp.mean(q_first),
                "seq_return_mean": jnp.mean(seq_return),
            }
            return loss, aux

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
        if normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
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
