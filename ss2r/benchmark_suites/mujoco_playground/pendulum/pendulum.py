import jax
import jax.numpy as jnp

_POLE_GEOM_ID = -2
_MASS_GEOM_ID = -1
_POLE_BODY_ID = -1


def domain_randomization(sys, rng, cfg):
    nominal_half_length = sys.geom_size[_POLE_GEOM_ID, 1]
    nominal_tip = nominal_half_length * 2.0

    @jax.vmap
    def randomize(rng):
        length_sample = jax.random.uniform(
            rng, minval=cfg.pendulum_length[0], maxval=cfg.pendulum_length[1]
        )
        new_half_length = nominal_half_length + 0.5 * length_sample
        new_tip = nominal_tip + length_sample
        geom_size = sys.geom_size.copy()
        geom_size = geom_size.at[_POLE_GEOM_ID, 1].set(new_half_length)
        geom_pos = sys.geom_pos.copy()
        geom_pos = geom_pos.at[_POLE_GEOM_ID, 2].set(new_half_length)
        geom_pos = geom_pos.at[_MASS_GEOM_ID, 2].set(new_tip)
        body_ipos = sys.body_ipos.copy()
        body_ipos = body_ipos.at[_POLE_BODY_ID, 2].add(length_sample)
        return geom_size, geom_pos, body_ipos, length_sample

    geom_size, geom_pos, body_ipos, length_sample = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_ipos": 0,
            "geom_size": 0,
            "geom_pos": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_ipos": body_ipos,
            "geom_size": geom_size,
            "geom_pos": geom_pos,
        }
    )
    return sys, in_axes, jnp.expand_dims(length_sample, axis=-1)
