from typing import Mapping, Sequence, Tuple

import jax.nn as jnn
import jax.numpy as jnp
from brax.training import networks, types
from flax import linen

from ss2r.algorithms.sac.networks import ActivationFn
from ss2r.algorithms.sac.vision_networks import Encoder, make_q_vision_network
from ss2r.algorithms.td3.networks import TD3Networks


def make_policy_vision_network(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    state_obs_key: str = "",
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
):
    class Policy(linen.Module):
        @linen.compact
        def __call__(self, obs):
            hidden = Encoder(name="SharedEncoder")(obs)
            hidden = linen.Dense(encoder_hidden_dim)(hidden)
            hidden = linen.LayerNorm()(hidden)
            if tanh:
                hidden = jnn.tanh(hidden)
            additional_obs = []
            for k, x in obs.items():
                if not k.startswith("pixels/"):
                    additional_obs.append(x)
            if additional_obs:
                additional_obs = jnp.concatenate(additional_obs, axis=-1)
                hidden = jnp.concatenate([hidden, additional_obs], axis=-1)
            action = networks.MLP(
                layer_sizes=list(hidden_layer_sizes) + [action_size],
                activation=activation,
            )(hidden)
            return jnp.tanh(action)

    pi_module = Policy()

    def apply(processor_params, params, obs):
        if state_obs_key:
            state_obs = preprocess_observations_fn(
                obs[state_obs_key],
                networks.normalizer_select(processor_params, state_obs_key),
            )
            obs = {**obs, state_obs_key: state_obs}
        return pi_module.apply(params, obs)

    dummy_obs = {
        key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
    }
    return networks.FeedForwardNetwork(
        init=lambda key: pi_module.init(key, dummy_obs), apply=apply
    )


def make_td3_vision_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    activation: ActivationFn = linen.swish,
    state_obs_key: str = "",
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
) -> TD3Networks:
    policy_network = make_policy_vision_network(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        state_obs_key=state_obs_key,
        encoder_hidden_dim=encoder_hidden_dim,
        tanh=tanh,
    )
    qr_network = make_q_vision_network(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
        state_obs_key=state_obs_key,
        encoder_hidden_dim=encoder_hidden_dim,
        tanh=tanh,
    )
    return TD3Networks(  # type: ignore
        policy_network=policy_network,
        qr_network=qr_network,
    )
