import jax
import optax
import jax.numpy as jnp
import flax.linen as nn

from tqdm import tqdm
from flax.training import checkpoints, train_state
from .nets import AutoEncoder
from functools import partial
from data.dataloaders import create_depth_dataset

import matplotlib.pyplot as plt


def mse_recon_loss(model: nn.Module, params: dict, batch: jnp.ndarray) -> float:
    imgs, _ = batch
    recon_imgs = model.apply({"params": params}, jnp.expand_dims(imgs, axis=-1))
    loss = ((recon_imgs - imgs) ** 2).mean(axis=1).sum()
    return loss


def init_model(model: nn.Module, seed: int):
    # Initialize model

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    init_data = jax.random.normal(rng, (2, 10, 270, 480, 1))

    params = model.init(init_rng, init_data)["params"]
    # Initialize learning rate schedule and optimizer
    optimizer = optax.chain(
        optax.clip(1.0), optax.adam(learning_rate=0.0001)  # Clip gradients at 1
    )
    # Initialize training state
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    return state


@partial(jax.jit, static_argnums=2)
def train_step(state, batch, model):
    loss_fn = lambda params: mse_recon_loss(model, params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params
    )  # Get loss and gradients for loss
    state = state.apply_gradients(grads=grads)  # Optimizer update step
    return state, loss


def train_epoch(state, model, train_loader):
    # Train model for one epoch, and log avg loss
    losses = []
    for batch in tqdm(train_loader):
        state, loss = train_step(state, batch, model)
        losses.append(loss)
    return jnp.mean(jnp.array(losses), axis=-1), state


if __name__ == "__main__":
    import os

    train = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = AutoEncoder(latent_dim=128)
    state = init_model(model, 42)
    train_loader, val_loader = create_depth_dataset(batch_size=1)

    train_imgs, _ = next(iter(train_loader))

    if train:
        for epoch_id in range(40):
            train_loss, state = train_epoch(state, model, train_loader)
            print("epoch loss: ", train_loss)

            run_id = (
                f"{os.path.dirname(os.path.realpath(__file__))}/checkpoints/autoencoder"
            )
            _ = checkpoints.save_checkpoint(
                run_id,
                state,
                epoch_id,
                keep=40,
            )
    else:
        print("batman")
        train_imgs, _ = next(iter(train_loader))
        ckpt_state = checkpoints.restore_checkpoint(
            f"/home/mathias/dev/structured-state-space-wm/models/autoencoder/checkpoints/autoencoder/checkpoint_21",
            target=None,
        )
        params = ckpt_state["params"]
        preds = model.apply({"params": params}, jnp.expand_dims(train_imgs, axis=-1))
        for i in range(30):
            plt.imsave(f"preds_{i}.png", preds[0, i, :].reshape(270, 480))
        print(preds)
