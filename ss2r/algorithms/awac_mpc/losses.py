"""MPO actor losses and TD critic losses for tree-MPC rollouts."""

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


def _sample_raw_actions(
    dist_params: jax.Array,
    key: PRNGKey,
    num_action_samples: int,
    parametric_action_distribution,
) -> tuple[jax.Array, jax.Array]:
    sample_keys = jax.random.split(key, num_action_samples)
    raw_actions_nba = jax.vmap(
        lambda k: parametric_action_distribution.sample_no_postprocessing(
            dist_params, k
        )
    )(sample_keys)
    actions_nba = jax.vmap(parametric_action_distribution.postprocess)(raw_actions_nba)
    return raw_actions_nba, actions_nba


def _dist_params_to_mean_logstd(
    dist_params: jax.Array,
    parametric_action_distribution,
) -> tuple[jax.Array, jax.Array]:
    """Extract pre-squash Gaussian mean/log-std using Brax's distribution path."""
    dist = parametric_action_distribution.create_dist(dist_params)
    mean = dist.loc
    logstd = jnp.log(dist.scale)
    return mean, logstd


def _gaussian_diag_log_prob(
    actions: jax.Array,
    mean: jax.Array,
    logstd: jax.Array,
) -> jax.Array:
    var = jnp.exp(2.0 * logstd)
    log2pi = jnp.log(2.0 * jnp.pi)
    quad = jnp.square(actions - mean) / (var + 1e-8)
    return -0.5 * jnp.sum(quad + 2.0 * logstd + log2pi, axis=-1)


def _gaussian_kl_target_current(
    target_mean: jax.Array,
    target_logstd: jax.Array,
    current_mean: jax.Array,
    current_logstd: jax.Array,
) -> jax.Array:
    """KL(N_target || N_current), summed over action dimensions."""
    target_var = jnp.exp(2.0 * target_logstd)
    current_var = jnp.exp(2.0 * current_logstd)
    kl_vec = (
        (target_var + jnp.square(target_mean - current_mean))
        / (2.0 * current_var + 1e-8)
        + current_logstd
        - target_logstd
        - 0.5
    )
    return jnp.sum(kl_vec, axis=-1)


def _gaussian_kl_mean_only(
    target_mean: jax.Array,
    target_logstd: jax.Array,
    current_mean: jax.Array,
) -> jax.Array:
    """KL where only mean changes: KL(N_t(mu_t,s_t) || N(mu,s_t))."""
    target_var = jnp.exp(2.0 * target_logstd)
    kl_vec = 0.5 * jnp.square(target_mean - current_mean) / (target_var + 1e-8)
    return jnp.sum(kl_vec, axis=-1)


def _gaussian_kl_std_only(
    target_logstd: jax.Array,
    current_logstd: jax.Array,
) -> jax.Array:
    """KL where only std changes: KL(N_t(mu_t,s_t) || N(mu_t,s))."""
    target_var = jnp.exp(2.0 * target_logstd)
    current_var = jnp.exp(2.0 * current_logstd)
    kl_vec = (
        current_logstd - target_logstd + target_var / (2.0 * current_var + 1e-8) - 0.5
    )
    return jnp.sum(kl_vec, axis=-1)


def _positive_dual(raw: jax.Array) -> jax.Array:
    return jax.nn.softplus(raw) + 1e-8


def make_losses(
    sac_network: SafeSACNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    mpo_eta_init: float,
    mpo_eta_epsilon: float,
    mpo_eta_opt_maxiter: int,
    mpo_num_action_samples: int,
    mpo_kl_mean_epsilon: float,
    mpo_kl_std_epsilon: float,
    use_bro: bool,
):
    if mpo_eta_init <= 0.0:
        raise ValueError(f"mpo_eta_init must be > 0, got {mpo_eta_init}.")
    if mpo_eta_epsilon <= 0.0:
        raise ValueError(f"mpo_eta_epsilon must be > 0, got {mpo_eta_epsilon}.")
    if mpo_eta_opt_maxiter < 1:
        raise ValueError(
            f"mpo_eta_opt_maxiter must be >= 1, got {mpo_eta_opt_maxiter}."
        )
    if mpo_num_action_samples < 1:
        raise ValueError(
            f"mpo_num_action_samples must be >= 1, got {mpo_num_action_samples}."
        )
    if mpo_kl_mean_epsilon < 0.0:
        raise ValueError(
            f"mpo_kl_mean_epsilon must be >= 0, got {mpo_kl_mean_epsilon}."
        )
    if mpo_kl_std_epsilon < 0.0:
        raise ValueError(f"mpo_kl_std_epsilon must be >= 0, got {mpo_kl_std_epsilon}.")

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
        target_policy_params: Params,
        dual_params: Params,
        normalizer_params: Any,
        q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        target_dist_params = policy_network.apply(
            normalizer_params, target_policy_params, transitions.observation
        )
        raw_actions_nba, actions_nba = _sample_raw_actions(
            target_dist_params,
            key,
            mpo_num_action_samples,
            parametric_action_distribution,
        )

        sampled_q_values = jax.vmap(
            lambda sampled_actions_ba: _reduce_q(
                qr_network.apply(
                    normalizer_params,
                    q_params,
                    transitions.observation,
                    sampled_actions_ba,
                ),
                use_bro,
            )
        )(actions_nba)
        # [N, B] -> [B, N]
        sampled_q_values = jnp.swapaxes(sampled_q_values, 0, 1)

        eta = _solve_eta_dual(
            sampled_q_values,
            mpo_eta_epsilon=mpo_eta_epsilon,
            mpo_eta_init=mpo_eta_init,
            mpo_eta_opt_maxiter=mpo_eta_opt_maxiter,
        )
        mpo_scores = sampled_q_values / eta
        mpo_scores = mpo_scores - jnp.max(mpo_scores, axis=-1, keepdims=True)
        mpo_weights = jax.nn.softmax(mpo_scores, axis=-1)
        mpo_weights = jax.lax.stop_gradient(mpo_weights)

        current_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        sampled_log_probs_current = jax.vmap(
            lambda raw_actions_ba: parametric_action_distribution.log_prob(
                current_dist_params, raw_actions_ba
            )
        )(raw_actions_nba)
        # [N, B] -> [B, N]
        sampled_log_probs_current = jnp.swapaxes(sampled_log_probs_current, 0, 1)
        nll_loss = jnp.mean(-jnp.sum(mpo_weights * sampled_log_probs_current, axis=-1))

        raw_actions_bna = jnp.swapaxes(raw_actions_nba, 0, 1)
        target_mean, target_logstd = _dist_params_to_mean_logstd(
            target_dist_params, parametric_action_distribution
        )
        current_mean, current_logstd = _dist_params_to_mean_logstd(
            current_dist_params, parametric_action_distribution
        )

        # Standard MPO M-step decomposition.
        log_prob_mean = _gaussian_diag_log_prob(
            raw_actions_bna,
            current_mean[:, None, :],
            target_logstd[:, None, :],
        )
        log_prob_std = _gaussian_diag_log_prob(
            raw_actions_bna,
            target_mean[:, None, :],
            current_logstd[:, None, :],
        )
        nll_mean = jnp.mean(-jnp.sum(mpo_weights * log_prob_mean, axis=-1))
        nll_std = jnp.mean(-jnp.sum(mpo_weights * log_prob_std, axis=-1))
        nll_decoupled = nll_mean + nll_std

        kl_mean = jnp.mean(
            _gaussian_kl_mean_only(
                target_mean=target_mean,
                target_logstd=target_logstd,
                current_mean=current_mean,
            )
        )
        kl_std = jnp.mean(
            _gaussian_kl_std_only(
                target_logstd=target_logstd,
                current_logstd=current_logstd,
            )
        )
        alpha_mean = _positive_dual(dual_params["log_alpha_mean"])
        alpha_std = _positive_dual(dual_params["log_alpha_std"])

        kl_mean_penalty = jax.lax.stop_gradient(alpha_mean) * (
            kl_mean - mpo_kl_mean_epsilon
        )
        kl_std_penalty = jax.lax.stop_gradient(alpha_std) * (
            kl_std - mpo_kl_std_epsilon
        )

        loss = nll_decoupled + kl_mean_penalty + kl_std_penalty
        kl_target_current = jnp.mean(
            _gaussian_kl_target_current(
                target_mean=target_mean,
                target_logstd=target_logstd,
                current_mean=current_mean,
                current_logstd=current_logstd,
            )
        )
        weight_entropy = -jnp.mean(
            jnp.sum(mpo_weights * jnp.log(mpo_weights + 1e-8), axis=-1)
        )
        aux = {
            "eta": eta,
            "nll_loss": nll_loss,
            "nll_decoupled": nll_decoupled,
            "nll_mean": nll_mean,
            "nll_std": nll_std,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "kl_target_current_mean": kl_target_current,
            "kl_mean": kl_mean,
            "kl_std": kl_std,
            "kl_mean_penalty": kl_mean_penalty,
            "kl_std_penalty": kl_std_penalty,
            "weight_entropy": weight_entropy,
            "weight_min": jnp.min(mpo_weights),
            "weight_max": jnp.max(mpo_weights),
            "weight_mean": jnp.mean(mpo_weights),
            "q_sample_mean": jnp.mean(sampled_q_values),
            "q_sample_std": jnp.std(sampled_q_values),
        }
        return loss, aux

    def dual_loss(
        dual_params: Params,
        kl_mean: jax.Array,
        kl_std: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        alpha_mean = _positive_dual(dual_params["log_alpha_mean"])
        alpha_std = _positive_dual(dual_params["log_alpha_std"])
        loss = alpha_mean * (mpo_kl_mean_epsilon - jax.lax.stop_gradient(kl_mean))
        loss += alpha_std * (mpo_kl_std_epsilon - jax.lax.stop_gradient(kl_std))
        aux = {
            "dual_loss": loss,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "kl_mean_target": jnp.asarray(mpo_kl_mean_epsilon, dtype=kl_mean.dtype),
            "kl_std_target": jnp.asarray(mpo_kl_std_epsilon, dtype=kl_std.dtype),
        }
        return loss, aux

    return critic_loss, actor_loss, dual_loss
