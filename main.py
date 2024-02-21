import jax
import hydra
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from models.s4.s4_wm import S4WorldModel
from utils.datasets import create_quad_depth_trajectories_datasets
from flax.training import checkpoints


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    trainloader, testloader, _, _, _ = create_quad_depth_trajectories_datasets(bsz=2)
    test_depth_imgs, test_actions = next(iter(trainloader))

    ckpt_state = checkpoints.restore_checkpoint(
        f"/home/mathias/dev/structured-state-space-wm/checkpoints/quad_depth_trajectories/s4-d_model=256-lr=0.001-bsz=100/checkpoint_9",
        target=None,
    )

    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(1), num=2)
    _ = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.expand_dims(test_depth_imgs, axis=-3),
        test_actions,
    )

    params = ckpt_state["params"]
    preds = model.apply(
        {"params": params}, jnp.expand_dims(test_depth_imgs, axis=-3), test_actions
    )

    pred_depth_images = preds[2].mean()

    for i in range(20):    
        example = pred_depth_images[1, i, :].reshape(270, 480)
        plt.imshow(example)
        plt.imsave(f"pred_{i}.png", example)


if __name__ == "__main__":
    main()
