import hydra
import os
import jax.numpy as jnp
import torch
import jax
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from s4wm.nn.s4_wm import S4WM, PyTree, PRNGKey
from s4wm.data.dataloaders import create_depth_dataset
from s4wm.utils.dlpack import from_torch_to_jax
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model: S4WM,
    params: PyTree,
    cache: PyTree,
    prime: PyTree,
    imgs: jnp.ndarray,
    actions: jnp.ndarray,
    key: PRNGKey,
) -> jnp.ndarray:
    out, vars = model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        imgs,
        actions,
        key,
        True,
        mutable=["cache"],
    )
    return (
        out["depth"]["recon"].mean(),
        out["depth"]["pred"].mean(),
        out["z_prior"]["sample"],
        vars,
    )


@partial(jax.jit, static_argnums=(0))
def dream(
    model: S4WM,
    params: PyTree,
    cache: PyTree,
    prime: PyTree,
    pred_posterior: jnp.ndarray,
    action: jnp.ndarray,
    key: PRNGKey,
) -> jnp.ndarray:
    out, vars = model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        pred_posterior,
        action,
        key,
        mutable=["cache"],
        method="forward_open_loop",
    )
    return out["depth_pred"].mean(), out["z_post_pred"]["sample"], vars


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    CTX_LENGTH = 20
    DREAM_LENGTH = 5
    VIZ_BATCH = 3
    BATCH_SIZE = 4

    key = jax.random.PRNGKey(0)
    model = S4WM(S4_config=cfg.model, training=False, **cfg.wm)
    torch.manual_seed(0)  # Dataloader order

    _, val_loader = create_depth_dataset(
        file_path=cfg.train.dataset_path, batch_size=BATCH_SIZE
    )
    val_depth_imgs, val_actions, _ = next(iter(val_loader))

    val_depth_imgs = from_torch_to_jax(val_depth_imgs)
    val_actions = from_torch_to_jax(val_actions)

    init_depth = jnp.zeros((BATCH_SIZE, 1, 135, 240, 1))
    init_actions = jnp.zeros((BATCH_SIZE, 1, 4))

    state = model.restore_checkpoint_state(
        "/home/mathias/dev/rl_checkpoints/gaussian_128"
    )
    params = state["params"]

    cache, prime = model.init_RNN_mode(params, init_depth, init_actions)

    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    # Build context
    z_post = None

    for i in range(CTX_LENGTH):
        sample_key, key = jax.random.split(key, num=2)
        depth = jnp.expand_dims(val_depth_imgs[:, i], axis=1)
        action = jnp.expand_dims(val_actions[:, i], axis=1)

        depth_recon, depth_pred, z_post, variables = _jitted_forward(
            model,
            params,
            cache,
            prime,
            depth,
            action,
            sample_key,
        )
        cache = variables["cache"]

        plt.imsave(
            f"imgs/recon_rnn_{i}.png",
            depth_recon[VIZ_BATCH].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        if i == CTX_LENGTH - 1:
            plt.imsave(
                f"imgs/dream_rnn_0.png",
                depth_pred[VIZ_BATCH].reshape(135, 240),
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            plt.imsave(
                f"imgs/dream_label_0.png",
                val_depth_imgs[VIZ_BATCH, i + 1].reshape(135, 240),
                cmap="magma",
                vmin=0,
                vmax=1,
            )

    # Open loop predictions

    for i in range(DREAM_LENGTH):
        sample_key, key = jax.random.split(key, num=2)
        action = jnp.expand_dims(val_actions[:, i + CTX_LENGTH], axis=1)

        # Override actions

        # action = action.at[:, :, 3].set(-1)
        # action = action.at[:, :, 0].set(0)
        # action = action.at[:, :, 1].set(0)
        # action = action.at[:, :, 2].set(1)

        depth_recon, z_post, variables = dream(
            model, params, cache, prime, z_post, action, key
        )
        cache = variables["cache"]
        plt.imsave(
            f"imgs/dream_rnn_{i+1}.png",
            depth_recon[VIZ_BATCH].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        plt.imsave(
            f"imgs/dream_label_{i+1}.png",
            val_depth_imgs[VIZ_BATCH, i + CTX_LENGTH + 1].reshape(135, 240),
            cmap="magma",
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    main()
