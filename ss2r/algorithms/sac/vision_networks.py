from typing import Mapping, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.sac.networks import (
    MLP,
    ActivationFn,
    BroNet,
    SafeSACNetworks,
)


class Encoder(linen.Module):
    features: Sequence[int] = (32, 64, 128, 256)
    strides: Sequence[int] = (2, 2, 2, 2)
    padding: str = "SAME"

    @linen.compact
    def __call__(self, data) -> jnp.ndarray:
        pixels_hidden = {k: v for k, v in data.items() if k.startswith("pixels/")}
        cnn_outs = []
        for x in pixels_hidden.values():
            for features, stride in zip(self.features, self.strides):
                x = linen.Conv(
                    features,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    kernel_init=jnn.initializers.orthogonal(jnp.sqrt(2)),
                    padding=self.padding,
                )(x)
                x = jnn.relu(x)

            if len(x.shape) == 4:
                x = x.reshape([x.shape[0], -1])
            else:
                x = x.reshape([-1])
            cnn_outs.append(x)
        return jnp.concatenate(cnn_outs, axis=-1)


def make_policy_vision_network(
    observation_size: Mapping[str, Tuple[int, ...]],
    output_size: int,
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
            hidden = jax.lax.stop_gradient(hidden)
            hidden = linen.Dense(encoder_hidden_dim)(hidden)
            hidden = linen.LayerNorm()(hidden)
            if tanh:
                hidden = jnn.tanh(hidden)
            outs = networks.MLP(
                layer_sizes=list(hidden_layer_sizes) + [output_size],
                activation=activation,
            )(hidden)
            return outs

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


def make_q_vision_network(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    n_critics: int = 2,
    state_obs_key: str = "",
    use_bro: bool = True,
    n_heads: int = 1,
    head_size: int = 1,
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
):
    class QModule(linen.Module):
        n_critics: int

        @linen.compact
        def __call__(self, obs, actions):
            hidden = Encoder(name="SharedEncoder")(obs)
            hidden = linen.Dense(encoder_hidden_dim)(hidden)
            hidden = linen.LayerNorm()(hidden)
            if tanh:
                hidden = jnn.tanh(hidden)
            hidden = jnp.concatenate([hidden, actions], axis=-1)
            res = []
            net = BroNet if use_bro else MLP
            for _ in range(self.n_critics):
                q = net(  # type: ignore
                    layer_sizes=list(hidden_layer_sizes) + [head_size],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                    num_heads=n_heads,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, params, obs, actions):
        if state_obs_key:
            state_obs = preprocess_observations_fn(
                obs[state_obs_key],
                networks.normalizer_select(processor_params, state_obs_key),
            )
            obs = {**obs, state_obs_key: state_obs}
        return q_module.apply(params, obs, actions)

    dummy_obs = {
        key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
    }
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_sac_vision_networks(
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
    *,
    safe: bool = False,
) -> SafeSACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_vision_network(
        observation_size=observation_size,
        output_size=parametric_action_distribution.param_size,
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
        encoder_hidden_dim=encoder_hidden_dim,
        tanh=tanh,
    )
    if safe:
        qc_network = make_q_vision_network(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            use_bro=use_bro,
            n_critics=n_critics,
            n_heads=n_heads,
            encoder_hidden_dim=encoder_hidden_dim,
            tanh=tanh,
        )
        old_apply = qc_network.apply
        qc_network.apply = lambda *args, **kwargs: jnn.softplus(
            old_apply(*args, **kwargs)
        )
    else:
        qc_network = None
    return SafeSACNetworks(
        policy_network=policy_network,
        qr_network=qr_network,
        qc_network=qc_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
