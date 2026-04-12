"""Q-based MPO actor losses and TD critic losses for TreeMPC rollouts."""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey
from jax.scipy import optimize as jax_optimize

from ss2r.algorithms.sac.networks import SafeSACNetworks

Transition: TypeAlias = types.Transition


def _reduce_q(q_values: jax.Array, use_bro: bool) -> jax.Array:
    if use_bro:
        return jnp.mean(q_values, axis=-1)
    return jnp.min(q_values, axis=-1)


def _logmeanexp(x: jax.Array, axis: int = -1) -> jax.Array:
    x_max = jnp.max(x, axis=axis, keepdims=True)
    lme = jnp.log(jnp.mean(jnp.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return jnp.squeeze(lme, axis=axis)


def _solve_eta_dual(
    q_values: jax.Array,
    mpo_eta_epsilon: float,
    mpo_eta_init: float,
    mpo_eta_opt_maxiter: int,
) -> jax.Array:
    q_values = jax.lax.stop_gradient(q_values)
    x0 = jnp.asarray([jnp.log(mpo_eta_init)], dtype=jnp.float32)

    def eta_dual(log_eta_vec: jax.Array) -> jax.Array:
        eta = jnp.exp(log_eta_vec[0]) + 1e-8
        scaled_q = q_values / eta
        lme = _logmeanexp(scaled_q, axis=-1)
        return eta * (mpo_eta_epsilon + jnp.mean(lme))

    result = jax_optimize.minimize(
        eta_dual,
        x0,
        method="BFGS",
        options={"maxiter": mpo_eta_opt_maxiter},
    )
    return jnp.exp(result.x[0]) + 1e-8


def make_losses(
    sac_network: SafeSACNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    mpo_eta_init: float,
    mpo_eta_epsilon: float,
    mpo_eta_min: float,
    mpo_eta_opt_maxiter: int,
    mpo_log_prob_min: float,
    use_bro: bool,
):
    if mpo_eta_init <= 0.0:
        raise ValueError(f"mpo_eta_init must be > 0, got {mpo_eta_init}.")
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

    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def critic_loss(
        qr_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_qr_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        q_old_action = qr_network.apply(
            normalizer_params,
            qr_params,
            transitions.observation,
            transitions.action,
        )
        next_dist_params = policy_network.apply(
            normalizer_params,
            policy_params,
            transitions.next_observation,
        )
        next_action = parametric_action_distribution.sample(next_dist_params, key)
        next_q = qr_network.apply(
            normalizer_params,
            target_qr_params,
            transitions.next_observation,
            next_action,
        )
        target_q = transitions.reward * reward_scaling + transitions.discount * (
            discounting * _reduce_q(next_q, use_bro)
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
        transitions: Transition,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        policy_extras = transitions.extras["policy_extras"]
        candidate_raw_actions = policy_extras["candidate_raw_actions"]
        candidate_q_values = policy_extras["candidate_q_values"]

        eta = _solve_eta_dual(
            candidate_q_values,
            mpo_eta_epsilon=mpo_eta_epsilon,
            mpo_eta_init=mpo_eta_init,
            mpo_eta_opt_maxiter=mpo_eta_opt_maxiter,
        )
        mpo_scores = candidate_q_values / eta
        mpo_scores = mpo_scores - jnp.max(mpo_scores, axis=-1, keepdims=True)
        mpo_weights = jax.nn.softmax(mpo_scores, axis=-1)
        mpo_weights = jax.lax.stop_gradient(mpo_weights)

        current_dist_params = policy_network.apply(
            normalizer_params,
            policy_params,
            transitions.observation,
        )

        def _log_prob_for_state(dist_params_b, raw_actions_na):
            return jax.vmap(
                lambda raw_action: parametric_action_distribution.log_prob(
                    dist_params_b, raw_action
                )
            )(raw_actions_na)

        sampled_log_probs_current = jax.vmap(_log_prob_for_state)(
            current_dist_params, candidate_raw_actions
        )
        finite_log_probs_current = jnp.nan_to_num(
            sampled_log_probs_current,
            nan=mpo_log_prob_min,
            neginf=mpo_log_prob_min,
            posinf=0.0,
        )
        clipped_log_probs_current = jnp.maximum(
            finite_log_probs_current,
            mpo_log_prob_min,
        )

        nll_loss_per_state = -jnp.sum(mpo_weights * clipped_log_probs_current, axis=-1)
        nll_loss = jnp.mean(nll_loss_per_state)
        aux = {
            "eta": eta,
            "nll_loss": nll_loss,
            "nll_loss_max": jnp.max(nll_loss_per_state),
            "weight_min": jnp.min(mpo_weights),
            "weight_max": jnp.max(mpo_weights),
            "weight_mean": jnp.mean(mpo_weights),
            "q_sample_mean": jnp.mean(candidate_q_values),
            "q_sample_std": jnp.std(candidate_q_values),
        }
        return nll_loss, aux

    return critic_loss, actor_loss
