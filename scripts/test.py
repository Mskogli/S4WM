import jax
import hydra
import os
import orbax

import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from models.s4wm.s4_wm import S4WorldModel
from data.dataloaders import create_depth_dataset
from flax.training import checkpoints
import torch
import numpy


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    SEED = 93
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)

    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    trainloader, _ = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions = next(iter(trainloader))
    test_depth_imgs = jax.lax.stop_gradient(test_depth_imgs)
    test_actions = jax.lax.stop_gradient(test_actions)

    rng = jax.random.PRNGKey(0)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    init_depth = jax.random.normal(init_rng, (2, 75, 270, 480, 1))
    init_actions = jax.random.normal(init_rng, (2, 75, 4))

    params = model.init(
        init_rng,
        init_depth,
        init_actions,
    )
    params = params["params"]

    params = jax.lax.stop_gradient(params)

    NOTARGET_CKPT_DIR = f"/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2_discrete/checkpoint_66"
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt_state = ckptr.restore(NOTARGET_CKPT_DIR, item=None)
    params = ckpt_state["params"]

    preds = model.apply(
        {"params": params}, jnp.expand_dims(test_depth_imgs, axis=-1), test_actions
    )
    pred_depth = preds[2].mean()
    pred_pred_depth = preds[3].mean()

    for i in range(74):
        pred = pred_depth[1, i, :].reshape((270, 480))
        pred_p = pred_pred_depth[1, i, :].reshape((270, 480))
        plt.imsave(f"imgs/test_{i}.png", pred)
        plt.imsave(f"imgs/test_hat_{i}.png", pred_p)
        plt.imsave(
            f"imgs/test_label{i}.png", test_depth_imgs[1, i + 1, :].reshape((270, 480))
        )

    plt.show()


if __name__ == "__main__":
    main()
