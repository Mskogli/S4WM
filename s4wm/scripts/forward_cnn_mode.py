import jax
import hydra
import os
import torch

import matplotlib.pyplot as plt

from omegaconf import DictConfig

from s4wm.nn.s4_wm import S4WM
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    model = S4WM(S4_config=cfg.model, training=False, **cfg.wm)
    torch.manual_seed(0)  # Dataloader order

    _, val_loader = create_depth_dataset(file_path=cfg.train.dataset_path, batch_size=4)
    val_depth_images, val_actions, _ = next(iter(val_loader))
    val_depth_images, val_actions = map(
        from_torch_to_jax, (val_depth_images, val_actions)
    )

    state = model.restore_checkpoint_state(
        "/home/mathias/dev/rl_checkpoints/gaussian_128"
    )
    params = state["params"]
    model.init(
        jax.random.PRNGKey(0), val_depth_images, val_actions, jax.random.PRNGKey(1)
    )

    out = model.apply(
        {"params": params},
        val_depth_images,
        val_actions,
        jax.random.PRNGKey(2),
        reconstruct_priors=True,
    )
    pred_depth = out["depth"]["pred"].mean()
    recon_depth = out["depth"]["recon"].mean()

    batch_id = 2
    c_map = "magma"

    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    for i in range(99):
        plt.imsave(
            f"imgs/pred_{i}.png",
            pred_depth[batch_id, i, :].reshape(135, 240),
            cmap=c_map,
            vmin=0,
            vmax=1,
        )
        plt.imsave(
            f"imgs/recon_{i}.png",
            recon_depth[batch_id, i, :].reshape(135, 240),
            cmap=c_map,
            vmin=0,
            vmax=1,
        )
        plt.imsave(
            f"imgs/label_{i}.png",
            val_depth_images[batch_id, i + 1, :].reshape(135, 240),
            cmap=c_map,
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    main()
