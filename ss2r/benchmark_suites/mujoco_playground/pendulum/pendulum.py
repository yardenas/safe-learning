import jax
import jax.numpy as jnp

_POLE_ID = -1


def domain_randomization(sys, rng, cfg):
    nominal_length = sys.geom_size[_POLE_ID, 1]

    @jax.vmap
    def randomize(rng):
        length_sample = jax.random.uniform(
            rng, minval=cfg.pendulum_length[0], maxval=cfg.pendulum_length[1]
        )
        length = nominal_length + length_sample
        scale_factor = length / nominal_length
        geom = sys.geom_size.copy()
        geom = geom.at[_POLE_ID, 1].set(length)
        mass = sys.body_mass.at[_POLE_ID].multiply(scale_factor)
        inertia = sys.body_inertia.at[_POLE_ID].multiply(scale_factor**3)
        inertia_pos = sys.body_ipos.copy()
        inertia_pos = inertia_pos.at[_POLE_ID, -1].add(length_sample / 2.0)
        return inertia_pos, mass, inertia, geom, length_sample

    inertia_pos, mass, inertia, geom, length_sample = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "geom_size": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_mass": mass,
            "body_inertia": inertia,
            "body_ipos": inertia_pos,
            "geom_size": geom,
        }
    )
    return sys, in_axes, jnp.expand_dims(length_sample, axis=-1)
