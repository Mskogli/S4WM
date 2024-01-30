import os
import shutil
import hydra
import jax
import jax.numpy as np
import optax
import torch

from functools import partial
from flax.training import checkpoints, train_state
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models.s4 import BatchStackedModel, S4Layer
from utils.datasets import Datasets

try:
    # Slightly nonstandard import name to make config easier - see example_train()
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


# ### Utilities
# We define a couple of utility functions below to compute a standard cross-entropy loss, and compute
# "token"-level prediction accuracy.

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer. Note that for S4 models,
# we use a custom learning rate for parameters of the S4 kernel (lr = 0.001, no weight decay).
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
    trainloader,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[:, :-1, :].numpy()),
    )

    # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!
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
        k: optax.adam(learning_rate=schedule_fn(v * lr))
        for k, v in lr_layer.items()
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
        lambda k, param: param.size * (2 if _is_complex(param) else 1)
        if lr_layer.get(k, lr) > 0.0
        else 0
    )(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def train_epoch(state, rng, model, trainloader, classification=False):
    # Store Metrics
    model = model(training=True)
    batch_losses, batch_accuracies = [], []

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs = np.array(batch[:, :-1, :].numpy())
        labels = np.array(batch[:, 1:, :128].numpy())  # Not the most efficient...

        print("labels", labels.shape)
        print("inputs", inputs.shape)
        
        rng, drop_rng = jax.random.split(rng)
        state, loss, acc = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            classification=classification,
        )
        batch_losses.append(loss)
        batch_accuracies.append(acc)

    # Return average loss over batches
    return (
        state,
        np.mean(np.array(batch_losses)),
        np.mean(np.array(batch_accuracies)),
    )


def validate(params, model, testloader, classification=False):
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        loss, acc = eval_step(
            inputs, labels, params, model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))



@partial(jax.jit, static_argnums=(4, 5))
def train_step(
    state, rng, batch_inputs, batch_labels, model, classification=False
):
    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        return loss, (logits, acc)

    if not classification:
        batch_labels = batch_inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if not classification:
        batch_labels = batch_inputs[:, :, 0]
    logits = model.apply({"params": params}, batch_inputs)
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc


def example_train(
    dataset: str,
    layer: str,
    seed: int,
    model: DictConfig,
    train: DictConfig,
):
    # Warnings and sanity checks
    if not train.checkpoint:
        print("[*] Warning: models are not being checkpoint")

    print("[*] Setting Randomness...")
    torch.random.manual_seed(seed)  # For dataloader order
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Check if classification dataset
    classification = "classification" in dataset

    # Create dataset
    create_dataset_fn = Datasets[dataset]
    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(
        bsz=train.bsz
    )

    # Get model class and arguments
    layer_cls = S4Layer
    model.layer.l_max = l_max

    # Extract custom hyperparameters from model class
    lr_layer = getattr(layer_cls, "lr", None)

    print(f"[*] Starting `{layer}` Training on `{dataset}` =>> Initializing...")

    model_cls = partial(
        BatchStackedModel,
        layer_cls=layer_cls,
        d_output=n_classes,
        classification=classification,
        **model,
    )

    state = create_train_state(
        rng,
        model_cls,
        trainloader,
        lr=train.lr,
        lr_layer=lr_layer,
        lr_schedule=train.lr_schedule,
        weight_decay=train.weight_decay,
        total_steps=len(trainloader) * train.epochs,
    )

    # Loop over epochs
    best_loss, best_acc, best_epoch = 10000, 0, 0
    for epoch in range(train.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss, train_acc = train_epoch(
            state,
            train_rng,
            model_cls,
            trainloader,
            classification=classification,
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(
            state.params, model_cls, testloader, classification=classification
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_acc:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_acc:.4f}"
        )

        # Save a checkpoint each epoch & handle best (test loss... not "copacetic" but ehh)
        if train.checkpoint:
            suf = f"-{train.suffix}" if train.suffix is not None else ""
            run_id = f"checkpoints/{dataset}/{layer}-d_model={model.d_model}-lr={train.lr}-bsz={train.bsz}{suf}"
            ckpt_path = checkpoints.save_checkpoint(
                run_id,
                state,
                epoch,
                keep=train.epochs,
            )

        if (classification and test_acc > best_acc) or (
            not classification and test_loss < best_loss
        ):
            if train.checkpoint:
                shutil.copy(ckpt_path, f"{run_id}/best_{epoch}")
                if os.path.exists(f"{run_id}/best_{best_epoch}"):
                    os.remove(f"{run_id}/best_{best_epoch}")

            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch

        print(
            f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if wandb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                },
                step=epoch,
            )
            wandb.run.summary["Best Test Loss"] = best_loss
            wandb.run.summary["Best Test Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch


@hydra.main(version_base=None, config_path="models/s4", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # Track with wandb
    if wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    example_train(**cfg)


if __name__ == "__main__":
    main()
