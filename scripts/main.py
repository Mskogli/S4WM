import jax
import hydra
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from models.s4wm.s4_wm import S4WorldModel
from data.dataloaders import create_depth_dataset
from flax.training import checkpoints


def init_S4_RNN_mode(model, params, x0, rng) -> None:
    variables = model.init({"params": rng[0], "dropout": rng[1]}, x0[0], x0[1])
    vars = {
        "params": params,
        "cache": variables["cache"],
        "prime": variables["prime"],
    }

    _, prime_vars = model.apply(vars, x0[0], x0[1], mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    model = S4WorldModel(S4_config=cfg.model, training=False, **cfg.wm)
    trainloader, _ = create_depth_dataset(batch_size=2)
    test_depth_imgs, test_actions = next(iter(trainloader))
    ckpt_state = checkpoints.restore_checkpoint(
        f"/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2/checkpoint_52",
        target=None,
    )

    params = ckpt_state["params"]

    pred_imgs = model.dream(
        params,
        jnp.expand_dims(test_depth_imgs[:, :-1], axis=-1),
        test_actions[:, 1:],
        jnp.zeros_like(test_actions),
        19,
    )

    for i in range(19):
        pred = pred_imgs[0, i, :].reshape((270, 480))
        print(pred.shape)
        plt.imsave(f"imgs/pred_{i}.png", pred)

    plt.show()


if __name__ == "__main__":
    main()
