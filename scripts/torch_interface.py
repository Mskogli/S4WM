import jax
import torch
import hydra
import os
import time
import jax.numpy as jnp

import utils.dlpack
from models.s4wm.s4_wm import S4WorldModel
from functools import partial


tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(
    jax.lax.stop_gradient, x
)  # stop gradient - used for KL balancing


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model, params, cache, prime, imgs: jax.Array, actions: jax.Array
) -> jax.Array:
    return model.apply(
        {"params": jax.lax.stop_gradient(params), "cache": cache, "prime": prime},
        jax.lax.stop_gradient(imgs),
        jax.lax.stop_gradient(actions),
        single_step=True,
        mutable=["cache"],
        method="forward_RNN_mode",
    )


def forward_world_model_torch(
    model, params, cache, prime, imgs: torch.tensor, actions: torch.tensor
) -> torch.tensor:
    jax_imgs, jax_actions = utils.dlpack.from_torch_to_jax(
        imgs
    ), utils.dlpack.from_torch_to_jax(actions)

    jax_preds, vars = _jitted_forward(
        model, params, cache, prime, jax_imgs, jax_actions
    )
    return (
        utils.dlpack.from_jax_to_torch(jax_preds["hidden"]),
        utils.dlpack.from_jax_to_torch(jax_preds["z_posterior"]["dist"].mean()),
        vars["cache"],
    )


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    NUM_ENVS = 1

    S4wm = S4WorldModel(
        S4_config=cfg.model,
        training=False,
        process_in_chunks=False,
        rnn_mode=True,
        **cfg.wm
    )

    # World Model initialization
    init_depth = jnp.zeros((NUM_ENVS, 1, 270, 480, 1))
    init_actions = jnp.zeros((NUM_ENVS, 1, 4))
    params = S4wm.restore_checkpoint_state(
        "models/s4wm/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_0"
    )["params"]
    cache, prime = S4wm.init_RNN_mode(params, init_depth, init_actions)

    torch_inputs_imgs = torch.zeros(
        (NUM_ENVS, 1, 270, 480, 1), device="cuda:0", requires_grad=False
    )

    torch_inputs_actions = torch.zeros(
        (NUM_ENVS, 1, 4), device="cuda:0", requires_grad=False
    )

    # First call jit compiles the function
    _ = forward_world_model_torch(
        S4wm, params, cache, prime, torch_inputs_imgs, torch_inputs_actions
    )

    fwp_times = []
    for _ in range(1000):
        start = time.time()
        _ = forward_world_model_torch(
            S4wm, params, cache, prime, torch_inputs_imgs, torch_inputs_actions
        )
        end = time.time()
        fwp_times.append(end - start)

    fwp_times = jnp.array(fwp_times)
    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))


if __name__ == "__main__":
    main()
