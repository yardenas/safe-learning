import functools

from ss2r.algorithms.hydrax_mpc import render as render_lib
from ss2r.algorithms.hydrax_mpc import train


def get_train_fn(cfg, checkpoint_path=None, restore_checkpoint_path=None):
    del checkpoint_path, restore_checkpoint_path
    training_cfg = cfg.training
    return functools.partial(
        train.train,
        episode_length=training_cfg.episode_length,
        seed=training_cfg.seed,
        cfg=cfg,
    )


def render(*args, **kwargs):
    return render_lib.render(*args, **kwargs)
