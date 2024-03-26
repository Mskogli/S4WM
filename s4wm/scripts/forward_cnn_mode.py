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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=False, **cfg.wm)
    torch.manual_seed(0)

    _, trainloader = create_depth_dataset(batch_size=4)
    test_depth_imgs, test_actions, _ = next(iter(trainloader))

    test_depth_imgs = from_torch_to_jax(test_depth_imgs)
    test_actions = from_torch_to_jax(test_actions)

    init_depth = jnp.zeros_like(test_depth_imgs)
    init_actions = jnp.zeros_like(test_actions)

    state = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=4-latent_type=disc/checkpoint_8"
    )
    params = state["params"]

    model.init(jax.random.PRNGKey(0), test_depth_imgs, test_actions)

    out = model.apply(
        {"params": params},
        test_depth_imgs,
        test_actions,
        compute_reconstructions=True,
        sample_mean=False,
    )

    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()
    for i in range(100):
        plt.imsave(f"imgs/pred_{i}.png", pred_depth[2, i, :].reshape(135, 240))
        plt.imsave(f"imgs/recon_{i}.png", recon_depth[2, i, :].reshape(135, 240))
        plt.imsave(f"imgs/label_{i}.png", test_depth_imgs[2, i, :].reshape(135, 240))


if __name__ == "__main__":
    main()
