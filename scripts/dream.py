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


def init_S4_RNN_mode(model, params, x0, rng_init, rng_drop) -> None:
    variables = model.init(rng_init, x0[0], x0[1])
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
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    torch.random.manual_seed(0)

    model = S4WorldModel(S4_config=cfg.model, training=False, rnn_mode=True, **cfg.wm)

    ckpt_dir = "/home/mathias/dev/structured-state-space-wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_55"
    ckpt_mngr = orbax.checkpoint.Checkpointer(
        orbax.checkpoint.PyTreeCheckpointHandler()
    )
    ckpt_state = ckpt_mngr.restore(ckpt_dir, item=None)
    params = ckpt_state["params"]

    trainloader, _ = create_depth_dataset(batch_size=40)
    test_depth_imgs, test_actions = next(iter(trainloader))
    test_depth_imgs = jnp.expand_dims(test_depth_imgs, axis=-1)
    imgs = test_depth_imgs[22].reshape(1, 75, 270, 480, 1)
    actions = test_actions[22].reshape(1, 75, 4)
    rng_init = jax.random.PRNGKey(0)
    rng_drop = jax.random.PRNGKey(1)

    x0 = (jnp.zeros_like(imgs), jnp.zeros_like(actions))
    params, prime, cache = init_S4_RNN_mode(model, params, x0, rng_init, rng_drop)

    (_, _, pred_imgs, ppred_imgs), vars = model.apply(
        {"params": ckpt_state["params"], "prime": prime, "cache": cache},
        imgs,
        actions,
        mutable=["cache"],
    )
    cache = vars["cache"]

    for i in range(40):
        pred = pred_imgs.mean()[0, i, :].reshape((270, 480))
        ppred = ppred_imgs.mean()[0, i, :].reshape((270, 480))
        plt.imsave(f"imgs/dream_pred_{i}.png", pred)
        plt.imsave(f"imgs/dream_ppred_{i}.png", ppred)
        plt.imsave(f"imgs/dream_label_{i}.png", imgs[0, i, :].reshape((270, 480)))
    plt.show()


if __name__ == "__main__":
    main()
