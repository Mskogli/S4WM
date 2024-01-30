import torch
import hydra
import jax
import flax
import jax.numpy as jnp

from omegaconf import DictConfig, OmegaConf
from flax.training import checkpoints, train_state
from utils.ag_traj_dataset import AerialGymTrajDataset
from models.s4 import BatchStackedModel, S4Layer


def test_inference(model_conf: DictConfig) -> None:
    test_dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cpu",
        actions=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_batch = next(iter(test_loader))
    test_batch = jnp.array(test_batch[:, :-1, :].numpy())


    torch.random.manual_seed(0)  # For dataloader order
    key = jax.random.PRNGKey(1)
    init_rng, dropout_rng = jax.random.split(key, num=2)
    model_conf.layer.l_max = 96

    model = BatchStackedModel(layer_cls=S4Layer,  d_output=128, classification=False, training=False, **model_conf)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.array(next(iter(test_loader))[:, :-1, :].numpy()),
    )


    ckpt_dir = '/home/mathias/dev/structured-state-space-wm/checkpoints/quad_depth_trajectories/s4-d_model=256-lr=0.001-bsz=128/checkpoint_26'  # Replace with your checkpoint directory
    ckpt_state = checkpoints.restore_checkpoint(ckpt_dir, target=None)
    print("Batman")

    params = ckpt_state['params']

    preds = model.apply({"params": params}, test_batch)

    
    print("Preds: ", preds[0, 85, 0:10])
    print("GT: ", test_batch[0, 86, 0:10])


@hydra.main(version_base=None, config_path="models/s4", config_name="config")
def main(cfg: DictConfig) -> None:
    test_inference(cfg.model)


if __name__ == "__main__":

    main()
