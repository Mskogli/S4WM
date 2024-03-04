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
def forward_world_model(
    model, params, imgs: jax.Array, actions: jax.Array
) -> jax.Array:
    imgs = jax.lax.stop_gradient(imgs)
    actions = jax.lax.stop_gradient(actions)

    preds = model.apply({"params": params}, imgs, actions)

    return preds[2].mean()


def forward_world_model_torch(
    model, params, imgs: torch.tensor, actions: torch.tensor
) -> jax.Array:
    jax_imgs, jax_actions = utils.dlpack.from_torch_to_jax(
        imgs
    ), utils.dlpack.from_torch_to_jax(actions)

    jax_preds = forward_world_model(model, params, jax_imgs, jax_actions)
    torch_preds = utils.dlpack.from_jax_to_torch(jax_preds)
    return torch_preds


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    world_model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    params = world_model.init_from_checkpoint(
        ckpt_dir="/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_37"
    )

    imgs = jax.numpy.zeros((1, 4, 270, 480, 1))
    actions = jax.numpy.zeros((1, 4, 4))

    torch_imgs = torch.rand((1, 100, 270, 480, 1), device="cuda:0", requires_grad=False)
    torch_actions = torch.rand((1, 100, 4), device="cuda:0", requires_grad=False)

    start = time.time()
    jax_imgs = utils.dlpack.from_torch_to_jax(torch_imgs)
    jax_actions = utils.dlpack.from_torch_to_jax(torch_actions)
    preds = forward_world_model(world_model, params, jax_imgs, jax_actions)
    torch_preds = utils.dlpack.from_jax_to_torch(preds)
    end = time.time()
    print("World Model forward pass with compilation overhead: ", end - start)

    start = time.time()
    jax_imgs = utils.dlpack.from_torch_to_jax(torch_imgs)
    jax_actions = utils.dlpack.from_torch_to_jax(torch_actions)

    preds = forward_world_model(world_model, params, jax_imgs, jax_actions)
    torch_preds = utils.dlpack.from_jax_to_torch(preds)
    end = time.time()
    print("World Model forward pass: ", end - start)

    start = time.time()
    torch_preds = forward_world_model_torch(
        world_model, params, torch_imgs, torch_actions
    )
    end = time.time()
    print("World Model forward pass: ", end - start)


if __name__ == "__main__":
    main()
