import os
import hydra
import jax
import jax.numpy as np
import optax

from functools import partial
from flax.training import checkpoints, train_state
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models.s4 import S4WorldModel, S4Layer
from utils.datasets import Datasets

try:
    # Slightly nonstandard import name to make config easier - see example_train()
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(
    rng,
    model_cls,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    init_depth = jax.random.normal(init_rng, (4, 20, 1, 270, 480))
    init_actions = jax.random.normal(init_rng, (4, 20, 4))

    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        init_depth,
        init_actions,
    )

    params = params["params"]

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: lr
    # lr_layer is a dictionary from parameter name to LR multiplier
    if lr_layer is None:
        lr_layer = {}

    optimizers = {
        k: optax.adam(learning_rate=schedule_fn(v * lr)) for k, v in lr_layer.items()
    }

    optimizers["__default__"] = optax.adamw(
        learning_rate=schedule_fn(lr),
        weight_decay=weight_decay,
    )
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)

    # Check that all special parameter names are actually parameters
    extra_keys = set(lr_layer.keys()) - set(jax.tree_leaves(name_map(params)))
    assert (
        len(extra_keys) == 0
    ), f"Special params {extra_keys} do not correspond to actual params"

    # Print parameter count
    _is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: (
            param.size * (2 if _is_complex(param) else 1)
            if lr_layer.get(k, lr) > 0.0
            else 0
        )
    )(params)

    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_epoch(state, rng, model_cls, trainloader):
    model = model_cls(training=True)
    batch_losses = []

    for batch_depth, batch_actions in tqdm(trainloader):
        rng, drop_rng = jax.random.split(rng)
        batch_depth_labels = batch_depth[:, 1:, ...].reshape(
            batch_depth.shape[0], batch_depth.shape[1] - 1, -1
        )
        state, loss = train_step(
            state,
            drop_rng,
            batch_depth,
            batch_actions,
            batch_depth_labels,
            model,
        )
        batch_losses.append(loss)
        print("Loss: ", loss)

    return (
        state,
        np.mean(np.array(batch_losses)),
    )


def validate(params, model_cls, testloader):
    losses = []
    model = model_cls(training=False)

    for batch_depth, batch_actions in tqdm(testloader):
        batch_depth_labels = batch_depth[:, 1:, ...].reshape(
            batch_depth.shape[0], batch_depth.shape[1] - 1, -1
        )

        loss = eval_step(
            batch_depth,
            batch_actions,
            batch_depth_labels,
            params,
            model,
        )

        losses.append(loss)
    return np.mean(np.array(losses))


@partial(jax.jit, static_argnums=5)
def train_step(state, rng, batch_depth, batch_actions, batch_depth_labels, model):

    def loss_fn(params):

        z_posterior, z_prior, img_prior = model.apply(
            {"params": params},
            batch_depth,  # Depth images
            batch_actions,  # Actions
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )[0]

        loss = model.compute_loss(
            img_prior_dist=img_prior,
            img_posterior=batch_depth_labels,
            z_posterior_dist=z_posterior[:, 1:],
            z_prior_dist=z_prior,
        )

        return np.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@partial(jax.jit, static_argnums=4)
def eval_step(batch_depth, batch_actions, batch_depth_labels, params, model):

    z_posterior, z_prior, img_prior = model.apply(
        {"params": params}, batch_depth, batch_actions
    )

    loss = model.compute_loss(
        img_prior_dist=img_prior,
        img_posterior=batch_depth_labels,
        z_posterior_dist=z_posterior[:, 1:],
        z_prior_dist=z_prior,
    )

    loss = np.mean(loss)

    return loss


def example_train(
    dataset: str,
    layer: str,
    seed: int,
    wm: DictConfig,
    model: DictConfig,
    train: DictConfig,
):
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Create dataset
    create_dataset_fn = Datasets[dataset]
    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(bsz=4)

    # Get model class and arguments
    layer_cls = S4Layer
    model.layer.l_max = l_max

    # Extract custom hyperparameters from model class
    lr_layer = getattr(layer_cls, "lr", None)

    print(f"[*] Starting `{layer}` Training on `{dataset}` =>> Initializing...")

    model_cls = partial(S4WorldModel, S4_config=model, **wm)

    state = create_train_state(
        rng,
        model_cls,
        lr=train.lr,
        lr_layer=lr_layer,
        lr_schedule=train.lr_schedule,
        weight_decay=train.weight_decay,
        total_steps=len(trainloader) * train.epochs,
    )

    # Loop over epochs
    best_loss, best_epoch = 10000000, 0
    for epoch in range(train.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        print(f"[*] Running Epoch {epoch + 1} Validation...")

        state, train_loss = train_epoch(
            state,
            train_rng,
            model_cls,
            trainloader,
        )

        test_loss = validate(state.params, model_cls, testloader)

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f} -- Train Loss:")
        print(f"\tVal Loss: {test_loss:.5f} -- Train Loss:")

        if test_loss < best_loss:
            best_loss, best_epoch = test_loss, epoch

            suf = f"-{train.suffix}" if train.suffix is not None else ""
            run_id = f"{os.path.dirname(os.path.realpath(__file__))}/checkpoints/{dataset}/{layer}-d_model={256}-lr={train.lr}-bsz={train.bsz}{suf}"
            _ = checkpoints.save_checkpoint(
                run_id,
                state,
                epoch,
                keep=train.epochs,
            )

        print(f"\tBest Test Loss: {best_loss:.5f}")

        if wandb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "test/loss": test_loss,
                },
                step=epoch,
            )
            wandb.run.summary["Best Test Loss"] = best_loss
            wandb.run.summary["Best Epoch"] = best_epoch


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # Track with wandb
    if wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(**wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    example_train(**cfg)


if __name__ == "__main__":
    main()
