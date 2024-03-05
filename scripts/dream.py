import jax
import hydra
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint
import torch

from omegaconf import DictConfig, OmegaConf
from models.s4wm.s4_wm import S4WorldModel
from data.dataloaders import create_depth_dataset
from flax.training import checkpoints
import numpy


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    torch.manual_seed(101)
    numpy.random.seed(101)
    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)
    trainloader, _ = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions = next(iter(trainloader))

    test_depth_imgs = jax.lax.stop_gradient(jnp.expand_dims(test_depth_imgs, axis=-1))
    test_actions = jax.lax.stop_gradient(test_actions)

    rng = jax.random.PRNGKey(0)
    init_rng, _ = jax.random.split(rng, num=2)

    init_depth = jnp.zeros((2, 75, 270, 480, 1))
    init_actions = jnp.zeros((2, 75, 4))

    NOTARGET_CKPT_DIR = f"/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_97"
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt_state = ckptr.restore(NOTARGET_CKPT_DIR, item=None)
    params = ckpt_state["params"]

    model._init_RNN_mode(init_rng, init_depth, init_actions)

    preds = model._forward_RNN_mode(params, test_depth_imgs, test_actions)

    pred_depth = preds[2].mean()
    pred_pred_depth = preds[3].mean()
    pred_posteriors = preds[0].mean()
    pred_priors = preds[1].mean()

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
