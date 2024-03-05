import jax
import torch
import hydra
import os
import flax.linen as nn
import time

import utils.dlpack
from models.s4wm.s4_wm import S4WorldModel
from functools import partial


@partial(jax.jit, static_argnums=(0))
def _forward_world_model(
    model, params, prime, cache, imgs: jax.Array, actions: jax.Array
) -> jax.Array:
    imgs = jax.lax.stop_gradient(imgs)
    actions = jax.lax.stop_gradient(actions)
    preds, vars = model.apply(
        {
            "params": params,
            "prime": prime,
            "cache": cache,
        },
        imgs,
        actions,
        mutable=["cache"],
    )
    return preds, vars["cache"]


def forward_world_model_torch(
    model, params, prime, cache, imgs: torch.tensor, actions: torch.tensor
) -> jax.Array:
    jax_imgs, jax_actions = utils.dlpack.from_torch_to_jax(
        imgs
    ), utils.dlpack.from_torch_to_jax(actions)
    jax_preds, _ = _forward_world_model(
        model, params, prime, cache, jax_imgs, jax_actions
    )

    return jax_preds


def init_recurrence(model, params, init_x, rng):
    variables = model.init(rng, init_x[0], init_x[1])
    vars = {
        "params": params,
        "cache": variables["cache"],
        "prime": variables["prime"],
    }
    _, prime_vars = model.apply(vars, init_x[0], init_x[1], mutable=["prime", "cache"])
    return vars["params"], prime_vars["prime"], vars["cache"]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    init_rng = jax.random.PRNGKey(0)
    init_depth = jax.lax.stop_gradient(
        jax.numpy.ones((128, 2, 270, 480, 1), dtype=jax.numpy.complex64)
    )
    init_actions = jax.lax.stop_gradient(
        jax.numpy.ones((128, 2, 4), dtype=jax.numpy.complex64)
    )

    world_model = S4WorldModel(
        S4_config=cfg.model, training=False, **cfg.wm, rnn_mode=True
    )

    params = world_model.restore_checkpoint_state(
        "scripts/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_14"
    )["params"]

    params, prime, cache = init_recurrence(
        world_model, params, (init_depth, init_actions), init_rng
    )

    preds, vars = world_model.apply(
        {"params": params, "prime": prime, "cache": cache},
        init_depth,
        init_actions,
        mutable=["cache"],
    )
    cache = vars["cache"]

    print(cache)


if __name__ == "__main__":
    main()
