import jax
import torch
import hydra
import os
import time
import jax.numpy as jnp

import utils.dlpack
from models.s4wm.s4_wm import S4WorldModel
from functools import partial


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(model, params, imgs: jax.Array, actions: jax.Array) -> jax.Array:
    imgs = jax.lax.stop_gradient(imgs)
    actions = jax.lax.stop_gradient(actions)
    return model.forward_RNN_mode(params, imgs, actions)


def forward_world_model_torch(
    model, params, prime, cache, imgs: torch.tensor, actions: torch.tensor
) -> jax.Array:
    jax_imgs, jax_actions = utils.dlpack.from_torch_to_jax(
        imgs
    ), utils.dlpack.from_torch_to_jax(actions)
    jax_preds = _jitted_forward(model, params, prime, cache, jax_imgs, jax_actions)
    return jax_preds


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    NUM_ENVS = 128

    S4wm = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)

    # World Model initialization
    init_depth = jnp.zeros((NUM_ENVS, 1, 270, 480, 1))
    init_actions = jnp.zeros((NUM_ENVS, 1, 4))
    params = S4wm.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/models/s4wm/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_41"
    )["params"]
    S4wm.init_RNN_mode(params, init_depth, init_actions)

    torch_inputs_imgs = torch.tensor(
        (NUM_ENVS, 1, 270, 480, 1), "cuda:0", requires_grad=False
    )
    torch_inputs_actions = torch.tensor((NUM_ENVS, 1, 4), "cuda:0", requires_grad=False)

    # First call jit compiles the function
    torch_outputs = forward_world_model_torch(
        S4wm, params, torch_inputs_imgs, torch_inputs_actions
    )

    start = time.time()
    torch_outputs = forward_world_model_torch(
        S4wm, params, torch_inputs_imgs, torch_inputs_actions
    )
    end = time.time()
    print("Forward pass took: ", end - start)

    # TODO: Add "single step forward function"


if __name__ == "__main__":
    main()
