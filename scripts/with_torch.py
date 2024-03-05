import jax
import torch
import hydra
import os
import flax.linen as nn
import time
import orbax

from models.s4wm.s4_wm import S4WorldModel
from functools import partial
from utils.dlpack import from_jax_to_torch, from_torch_to_jax


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, imgs, actions):
    imgs = jax.lax.stop_gradient(imgs)
    actions = jax.lax.stop_gradient(actions)
    out, _ = model.apply(params, imgs, actions, mutable=["cache"])
    return out["z_posterior"]["sample"], out["hidden"]


def forward_torch(model, params, imgs, actions):
    imgs, actions = from_torch_to_jax(imgs), from_torch_to_jax(actions)
    out = jitted_forward(model, params, imgs, actions)
    return from_jax_to_torch(out[0]), from_jax_to_torch(out[1])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    world_model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0), num=2)

    init_depth = jax.random.normal(init_rng, (2, 75, 270, 480, 1))
    init_actions = jax.random.normal(init_rng, (2, 75, 4))

    _ = world_model.init(
        init_rng,
        init_depth,
        init_actions,
    )
    params = world_model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_7"
    )["params"]

    vars = {"params": params, "prime": _["prime"], "cache": _["cache"]}

    imgs = torch.rand((100, 2, 270, 480, 1), device="cuda:0", requires_grad=False)
    actions = torch.rand((100, 2, 4), device="cuda:0", requires_grad=False)

    out = forward_torch(world_model, vars, imgs, actions)

    start = time.time()
    out_2 = forward_torch(world_model, vars, imgs, actions)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
