import functools


def get_train_fn(cfg, checkpoint_path, restore_checkpoint_path):
    import jax.nn as jnn

    import ss2r.algorithms.awac_mpc.train as awac_mpc

    agent_cfg = dict(cfg.agent)
    training_cfg = {
        k: v
        for k, v in cfg.training.items()
        if k
        not in [
            "render_episodes",
            "train_domain_randomization",
            "eval_domain_randomization",
            "render",
            "store_checkpoint",
            "value_privileged",
            "policy_privileged",
            "wandb_id",
            "hard_resets",
            "nonepisodic",
            "action_delay",
            "safe",
            "safety_budget",
        ]
    }

    policy_hidden_layer_sizes = tuple(agent_cfg.pop("policy_hidden_layer_sizes"))
    value_hidden_layer_sizes = tuple(agent_cfg.pop("value_hidden_layer_sizes"))
    activation = getattr(jnn, agent_cfg.pop("activation"))

    del agent_cfg["name"]

    train_fn = functools.partial(
        awac_mpc.train,
        **agent_cfg,
        **training_cfg,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        checkpoint_logdir=checkpoint_path,
        restore_checkpoint_path=restore_checkpoint_path,
    )
    return train_fn
