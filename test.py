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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    test_batch = next(iter(test_loader))
    test_batch = jnp.array(test_batch[:, :-1, :].numpy())

    vae = VAENetworkInterface()

    torch.random.manual_seed(0)  # For dataloader order
    key = jax.random.PRNGKey(1)
    init_rng, dropout_rng = jax.random.split(key, num=2)
    model_conf.layer.l_max = 96

    model = BatchStackedModel(layer_cls=S4Layer,  d_output=128, classification=False, training=False, **model_conf)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.array(next(iter(test_loader))[:, :-1, :].numpy()),
    )

    ckpt_dir = '/home/mathias/dev/structured-state-space-wm/checkpoints/quad_depth_trajectories/s4-d_model=256-lr=0.001-bsz=256/checkpoint_77'  # Replace with your checkpoint directory
    ckpt_state = checkpoints.restore_checkpoint(ckpt_dir, target=None)

    params = ckpt_state['params']
    preds = model.apply({"params": params}, test_batch)
    
    pred_latent_tensor = torch.from_numpy(np.asarray(preds[0, 65, :])).to(torch.device("cuda:0"))
    current_latent_tensor = torch.from_numpy(np.asarray(test_batch[0, 65, :128])).to(torch.device("cuda:0"))
    gt_latent_tensor = torch.from_numpy(np.asarray(test_batch[0, 66, :128])).to(torch.device("cuda:0"))
    
    pred_depth_img = vae.decode(pred_latent_tensor.view(1, -1))
    current_depth_img = vae.decode(current_latent_tensor.view(1, -1))
    gt_depth_img = vae.decode(gt_latent_tensor.view(1, -1))

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor="w", edgecolor="k")

    ax1 = fig.add_subplot(1,3,2)
    ax1.set_title("$l_{t+1}$")
    plt.imshow(gt_depth_img.reshape(270, 480))
    plt.axis("off")

    ax2 = fig.add_subplot(1,3,3)
    ax2.set_title("$\hat l_{t+1}$")
    plt.imshow(pred_depth_img.reshape(270, 480))
    plt.axis("off")
    
    ax3 = fig.add_subplot(1,3,1)
    ax3.set_title("$l_t$")
    plt.imshow(current_depth_img.reshape(270, 480))
    plt.axis("off")

    plt.savefig("s4_test.png")

@hydra.main(version_base=None, config_path="models/s4", config_name="config")
def main(cfg: DictConfig) -> None:
    test_inference(cfg.model)


if __name__ == "__main__":

    main()
