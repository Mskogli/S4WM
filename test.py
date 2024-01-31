import torch
import hydra
import jax
import jax.numpy as jnp
import numpy as np

from omegaconf import DictConfig
from flax.training import checkpoints
from utils.ag_traj_dataset import AerialGymTrajDataset
from models.s4 import BatchStackedModel, S4Layer
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface
import matplotlib.pyplot as plt


def test_inference(model_conf: DictConfig) -> None:
    test_dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cpu",
        actions=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=True
    )
    test_batch = next(iter(test_loader))
    test_batch = jnp.array(test_batch[:, :-1, :].numpy())

    vae = VAENetworkInterface()
    torch.random.manual_seed(0)
    key = jax.random.PRNGKey(1)
    init_rng, dropout_rng = jax.random.split(key, num=2)
    model_conf.layer.l_max = 96

    model = BatchStackedModel(
        layer_cls=S4Layer,
        d_output=128,
        classification=False,
        training=False,
        **model_conf,
    )
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.array(next(iter(test_loader))[:, :-1, :].numpy()),
    )

    ckpt_dir = "/home/mathias/dev/structured-state-space-wm/checkpoints/quad_depth_trajectories/s4-d_model=256-lr=0.001-bsz=256/checkpoint_77"
    ckpt_state = checkpoints.restore_checkpoint(ckpt_dir, target=None)
    params = ckpt_state["params"]
    preds = model.apply({"params": params}, test_batch)

    # Decode latents to attain depth images from the predictions
    pred_latent_tensor = torch.from_numpy(np.asarray(preds[0, 65, :])).to(
        torch.device("cuda:0")
    )
    current_latent_tensor = torch.from_numpy(np.asarray(test_batch[0, 65, :128])).to(
        torch.device("cuda:0")
    )
    gt_latent_tensor = torch.from_numpy(np.asarray(test_batch[0, 66, :128])).to(
        torch.device("cuda:0")
    )

    pred_depth_img = vae.decode(pred_latent_tensor.view(1, -1))
    current_depth_img = vae.decode(current_latent_tensor.view(1, -1))
    gt_depth_img = vae.decode(gt_latent_tensor.view(1, -1))

    # Plot results
    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor="w", edgecolor="k")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.set_title("$l_{t+1}$")
    plt.imshow(gt_depth_img.reshape(270, 480))
    plt.axis("off")

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("$\hat l_{t+1}$")
    plt.imshow(pred_depth_img.reshape(270, 480))
    plt.axis("off")

    ax3 = fig.add_subplot(1, 3, 1)
    ax3.set_title("$l_t$")
    plt.imshow(current_depth_img.reshape(270, 480))
    plt.axis("off")

    plt.savefig("s4_inference_test.png")


def init_recurrence(model, params, init_x, rng):
    variables = model.init(rng, init_x)
    vars = {
        "params": params,
        "cache": variables["cache"].unfreeze(),
        "prime": variables["prime"].unfreeze(),
    }
    print("[*] Priming")
    _, prime_vars = model.apply(vars, init_x, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


def init_S4_RNN_mode(model, params, x0, rng) -> None:
    variables = model.init(rng, x0)

    vars = {
        "params": params,
        "cache": variables["cache"],
        "prime": variables["prime"],
    }

    _, prime_vars = model.apply(vars, x0, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


def sample_S4_RNN() -> None:
    pass


def sample(model, params, prime, cache, x, start, end, rng):
    def loop(i, cur):
        x, rng, cache = cur
        r, rng = jax.random.split(rng)
        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            x[:, np.arange(1, 2) * i],
            mutable=["cache"],
        )

        def update(x, out):
            p = jax.random.categorical(r, out[0])
            x = x.at[i + 1, 0].set(p)
            return x

        x = jax.vmap(update)(x, out)
        return x, rng, vars["cache"].unfreeze()

    return jax.lax.fori_loop(start, end, jax.jit(loop), (x, rng, cache))[0]


def test_dreaming(model_conf: DictConfig) -> None:
    test_dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cpu",
        actions=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )
    test_batch = next(iter(test_loader))
    test_batch = jnp.array(test_batch[:, :-1, :].numpy())
    torch.random.manual_seed(0)

    init_rng = jax.random.PRNGKey(1)

    model = BatchStackedModel(
        layer_cls=S4Layer,
        d_output=128,
        classification=False,
        training=False,
        decode=True,
        **model_conf,
    )

    ckpt_dir = "/home/mathias/dev/structured-state-space-wm/checkpoints/quad_depth_trajectories/s4-d_model=256-lr=0.001-bsz=256/checkpoint_77"
    ckpt_state = checkpoints.restore_checkpoint(ckpt_dir, target=None)
    params = ckpt_state["params"]

    x0 = jnp.zeros_like(test_batch)
    params, prime, cache = init_S4_RNN_mode(model, params, x0, init_rng)

    # Build the model context
    L_CONTEXT = 80
    L_DREAM = 5
    preds, vars = model.apply(
        {"params": params, "prime": prime, "cache": cache},
        test_batch[:, 0:L_CONTEXT, :],
        mutable=["cache"],
    )
    cache = vars["cache"]

    dream_actions = np.zeros((128, 1, 4))
    dream_actions[:, :, :] = [0, 0, 0, 7.7]
    dream = [preds[:, -1, :]]

    vae = VAENetworkInterface()
    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor="w", edgecolor="k")

    for i in range(L_DREAM):
        preds, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            jnp.concatenate((dream[-1].reshape(128, 1, 128), dream_actions), axis=2),
            mutable=["cache"],
        )
        cache = vars["cache"]
        dream.append(preds)

        # Convert to torch and decode
        pred_latent_tensor = torch.from_numpy(np.asarray(preds[13, :, :])).to(
            torch.device("cuda:0")
        )
        pred_depth_img = vae.decode(pred_latent_tensor)

        # Vizualise
        ax = fig.add_subplot(1, L_DREAM, i + 1)
        ax.set_title(f"$l_{i+1}$")
        plt.imshow(pred_depth_img.reshape(270, 480))
        plt.axis("off")

    plt.savefig("s4_dreaming_test.png")


@hydra.main(version_base=None, config_path="models/s4", config_name="config")
def main(cfg: DictConfig) -> None:
    # test_inference(cfg.model)
    test_dreaming(cfg.model)


if __name__ == "__main__":
    main()
