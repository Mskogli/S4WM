import jax
import hydra
import os
import torch

import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(0)

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)
    _, trainloader = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    init_depth = jnp.zeros_like(test_depth_imgs)
    init_actions = jnp.zeros_like(test_actions)

    params = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2-latent_type=disc/checkpoint_13"
    )["params"]
    cache, prime = model.init_RNN_mode(params, init_depth, init_actions)

    ctx_l = 70
    dream_l = 10
    context_imgs = test_depth_imgs[:, :ctx_l, :]
    context_actions = test_actions[:, 1 : ctx_l + 1, :]
    dream_actions = test_actions[:, ctx_l + 1 : ctx_l + dream_l + 1, :]
    dream_actions = jnp.zeros_like(context_actions)

    out, _ = model.apply(
        {"params": params, "cache": cache, "prime": prime},
        context_imgs,
        context_actions,
        dream_actions,
        10,
        mutable=["cache"],
        method="dream",
    )

    for i in range(dream_l):
        plt.imsave(f"imgs/draum_{i}.png", out[0][i][1, 0].reshape(270, 480))
        plt.imsave(
            f"imgs/gt_dream{i}.png", test_depth_imgs[1, ctx_l + i].reshape(270, 480)
        )


if __name__ == "__main__":
    main()
