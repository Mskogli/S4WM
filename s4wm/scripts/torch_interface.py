import jax
import torch
import hydra
import os
import time
import jax.numpy as jnp

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from s4wm.nn.s4_wm import S4WorldModel
from functools import partial
from omegaconf import DictConfig


tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(
    jax.lax.stop_gradient, x
)  # stop gradient - used for KL balancing


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model, params, cache, prime, imgs: jax.Array, actions: jax.Array
) -> jax.Array:
    return model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        imgs,
        actions,
        single_step=True,
        mutable=["cache"],
        method="forward_RNN_mode",
    )


def forward_world_model_torch(
    model, params, cache, prime, imgs: torch.tensor, actions: torch.tensor
) -> torch.tensor:
    jax_imgs, jax_actions = from_torch_to_jax(imgs), from_torch_to_jax(actions)

    jax_preds, vars = _jitted_forward(
        model,
        jax.lax.stop_gradient(params),
        jax.lax.stop_gradient(cache),
        jax.lax.stop_gradient(prime),
        jax.lax.stop_gradient(jax_imgs),
        jax.lax.stop_gradient(jax_actions),
    )
    return (
        from_jax_to_torch(jax_preds["hidden"]),
        from_jax_to_torch(jax_preds["z_posterior"]["dist"].mean()),
        vars["cache"],
    )


@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    NUM_ENVS = 1

    S4wm = S4WorldModel(
        S4_config=DictConfig(
            {
                "d_model": 1024,
                "n_layers": 2,
                "n_blocks": 4,
                "dropout": 0.05,
                "layer": {"l_max": 74, "N": 128},
            }
        ),
        training=False,
        process_in_chunks=False,
        rnn_mode=True,
        **cfg.wm
    )

    print(type(cfg.model))
    print(cfg.wm)
    # World Model initialization
    init_depth = jnp.zeros((NUM_ENVS, 1, 270, 480, 1))
    init_actions = jnp.zeros((NUM_ENVS, 1, 4))
    params = S4wm.restore_checkpoint_state(
        "/home/mathias/dev/structured-state-space-wm/s4wm/scripts/checkpoints/depth_dataset/d_model=1024-lr=0.0001-bsz=2/checkpoint_97"
    )["params"]
    cache, prime = S4wm.init_RNN_mode(params, init_depth, init_actions)

    torch_inputs_imgs = torch.rand(
        (NUM_ENVS, 1, 270, 480, 1), device="cuda:0", requires_grad=False
    )

    torch_inputs_actions = torch.rand(
        (NUM_ENVS, 1, 4), device="cuda:0", requires_grad=False
    )

    jitted = jax.jit(
        lambda p, c, pr, i, a: S4wm.apply(
            sg({"params": p, "cache": c, "prime": pr}),
            i,
            a,
            method="forward_RNN_mode",
            mutable=["cache"],
        )
    )

    params, cache, prime, init_depth, init_actions = (
        sg(params),
        sg(cache),
        sg(prime),
        sg(init_depth),
        sg(init_actions),
    )

    print("jitted")
    _ = jitted(params, cache, prime, init_depth, init_actions)
    fwp_times = []
    for _ in range(100):
        start = time.time()
        _ = jitted(params, cache, prime, init_depth, init_actions)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)

    fwp_times = jnp.array(fwp_times)
    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))


if __name__ == "__main__":
    main()