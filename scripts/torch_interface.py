import jax
import torch
import hydra
import os

import utils.dlpack
from models.s4wm.s4_wm import S4WorldModel
from functools import partial


class S4WMTorchWrapper:

    def __init__():
        pass


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

    # TODO: Add "single step forward function"


if __name__ == "__main__":
    main()
