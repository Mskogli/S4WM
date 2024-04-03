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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=False, **cfg.wm)
    torch.manual_seed(0)

    _, trainloader = create_depth_dataset(batch_size=4)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    state = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=4-128x32_latent-3_blocks/checkpoint_19"
    )
    params = state["params"]

    print(test_depth_imgs.shape, test_actions.shape)
    model.init(jax.random.PRNGKey(0), test_depth_imgs, test_actions)

    out = model.apply(
        {"params": params},
        test_depth_imgs,
        test_actions,
        compute_reconstructions=True,
        sample_mean=True,
    )

    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()
    print(pred_depth.shape)
    for i in range(99):
        plt.imsave(
            f"imgs/pred_{i}.png", pred_depth[3, i, :].reshape(135, 240), vmin=0, vmax=1
        )
        plt.imsave(
            f"imgs/recon_{i}.png",
            recon_depth[3, i, :].reshape(135, 240),
            vmin=0,
            vmax=1,
        )
        plt.imsave(
            f"imgs/label_{i}.png",
            test_depth_imgs[3, i + 1, :].reshape(135, 240),
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    main()
