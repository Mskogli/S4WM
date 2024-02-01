import os
import torch
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from flax.training import checkpoints
from utils.ag_traj_dataset import AerialGymTrajDataset
from models.s4 import BatchStackedModel, S4Layer
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface


def init_S4_RNN_mode(model, params, x0, rng) -> None:
    variables = model.init(rng, x0)

    vars = {
        "params": params,
        "cache": variables["cache"],
        "prime": variables["prime"],
    }

    _, prime_vars = model.apply(vars, x0, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


def s4_RNN_inference(model_conf: DictConfig, test_conf: DictConfig) -> None:
    CONTEXT_LENGTH = test_conf.context_length
    PREDICTION_LENGTH = test_conf.prediction_length

    test_dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cpu",
        actions=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_data = next(iter(test_loader))
    test_data = jnp.array(test_data[:, :-1, :].numpy())

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

    ckpt_state = checkpoints.restore_checkpoint(
        f"{os.path.dirname(os.path.realpath(__file__))}/{test_conf.ckpt_dir}",
        target=None,
    )
    params = ckpt_state["params"]

    x0 = jnp.zeros_like(test_data)
    params, prime, cache = init_S4_RNN_mode(model, params, x0, init_rng)

    # Feed the model context
    pred, vars = model.apply(
        {"params": params, "prime": prime, "cache": cache},
        test_data[:, 0:CONTEXT_LENGTH, :],
        mutable=["cache"],
    )
    cache = vars["cache"]

    dream_actions = np.zeros((1, 1, 4))
    dream_actions[:, :, :] = [0, 0, 0, 1.5]
    preds = [pred[:, -1, :]]

    vae = VAENetworkInterface()
    fig, axs = plt.subplots(nrows=2, ncols=PREDICTION_LENGTH + 1)
    axs = axs.flatten()

    pred_latent_tensor = torch.from_numpy(np.asarray(pred[:, -1, :])).to(
        torch.device("cuda:0")
    )

    gt_latent_tensor = (
        torch.from_numpy(np.asarray(test_data[:, CONTEXT_LENGTH, :128]))
        .to(torch.device("cuda:0"))
        .view(1, 1, 128)
    )

    gt_depth_img = vae.decode(gt_latent_tensor)
    pred_depth_img = vae.decode(pred_latent_tensor.view(1, 1, 128))

    axs[0].set_title(f"$l_{1}$")
    axs[0].imshow(pred_depth_img.reshape(270, 480))
    axs[0].set_axis_off()

    axs[PREDICTION_LENGTH + 1].set_title(f"$l_{1}$")
    axs[PREDICTION_LENGTH + 1].imshow(gt_depth_img.reshape(270, 480))
    axs[PREDICTION_LENGTH + 1].set_axis_off()

    for i in range(PREDICTION_LENGTH):
        print(cache)
        pred, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            jnp.concatenate(
                (
                    preds[-1].reshape(1, 1, 128),
                    dream_actions.reshape(1, 1, 4),
                ),
                axis=2,
            ),
            mutable=["cache"],
        )
        cache = vars["cache"]
        preds.append(pred)

        # Convert to torch and decode
        pred_latent_tensor = torch.from_numpy(np.asarray(pred[:, :, :])).to(
            torch.device("cuda:0")
        )

        gt_latent_tensor = (
            torch.from_numpy(np.asarray(test_data[:, CONTEXT_LENGTH + i + 1, :128]))
            .to(torch.device("cuda:0"))
            .view(1, 1, 128)
        )

        gt_depth_img = vae.decode(gt_latent_tensor)
        pred_depth_img = vae.decode(pred_latent_tensor.view(1, 1, 128))

        # Vizualise
        axs[i + 1].set_title(f"$l_{i+1}$")
        axs[i + 1].imshow(pred_depth_img.reshape(270, 480))
        axs[i + 1].set_axis_off()

        axs[i + 2 + PREDICTION_LENGTH].set_title(f"$l_{i+1}$")
        axs[i + 2 + PREDICTION_LENGTH].imshow(gt_depth_img.reshape(270, 480))
        axs[i + 2 + PREDICTION_LENGTH].set_axis_off()

    fig.tight_layout()
    plt.savefig("s4_predictions.png")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    s4_RNN_inference(cfg.model, cfg.test)


if __name__ == "__main__":
    main()
