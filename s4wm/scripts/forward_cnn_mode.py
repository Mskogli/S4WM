import jax
import hydra
import os
import torch
import numpy

import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WorldModel
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.manual_seed(28)
    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=False, **cfg.wm)
    _, trainloader = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    init_depth = jnp.zeros_like(test_depth_imgs)
    init_actions = jnp.zeros_like(test_actions)

    params = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_97"
    )["params"]

    model.init(jax.random.PRNGKey(0), init_depth, init_actions)

    out = model.apply(
        {"params": params}, test_depth_imgs, test_actions, compute_reconstructions=True
    )

    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()
    for i in range(75):
        plt.imsave(f"imgs/pred_{i}.png", pred_depth[1, i, :].reshape(270, 480))
        plt.imsave(f"imgs/recon_{i}.png", recon_depth[1, i, :].reshape(270, 480))
        plt.imsave(f"imgs/label_{i}.png", test_depth_imgs[1, i, :].reshape(270, 480))


if __name__ == "__main__":
    main()
