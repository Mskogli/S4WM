import jax
import hydra
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint
import torch

from omegaconf import DictConfig
from models.s4wm.s4_wm import S4WorldModel
from data.dataloaders import create_depth_dataset
import numpy


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    SEED = 29
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)
    trainloader, _ = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions = next(iter(trainloader))

    test_depth_imgs = jax.lax.stop_gradient(jnp.expand_dims(test_depth_imgs, axis=-1))
    test_actions = jax.lax.stop_gradient(test_actions)

    init_depth = jnp.zeros_like(test_depth_imgs)
    init_actions = jnp.zeros_like(test_actions)

    params = model.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/models/s4wm/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_41"
    )["params"]

    model.init_RNN_mode(params, init_depth, init_actions)
    out = model.forward_RNN_mode(
        params, test_depth_imgs, test_actions, compute_reconstructions=True
    )

    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()
    for i in range(10):
        plt.imsave(f"imgs/pred_rnn_{i}.png", pred_depth[1, i + 60, :].reshape(270, 480))
        plt.imsave(f"imgs/recon_{i}.png", recon_depth[1, i + 60, :].reshape(270, 480))
        plt.imsave(
            f"imgs/label_{i}.png", test_depth_imgs[1, i + 60, :].reshape(270, 480)
        )


if __name__ == "__main__":
    main()
