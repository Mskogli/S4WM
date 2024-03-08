import jax
import torch
import hydra
import os
import time
import jax.numpy as jnp

from functools import partial
from omegaconf import DictConfig
from typing import Tuple

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from s4wm.nn.s4_wm import S4WorldModel


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
            "params": sg(params),
            "cache": sg(cache),
            "prime": sg(prime),
        },
        imgs,
        actions,
        single_step=True,
        mutable=["cache"],
        method="forward_RNN_mode",
    )


class TorchWrapper:
    def __init__(
        self,
        batch_dim: int,
        ckpt_path: str,
        d_latent: int = 128,
        d_pssm_block: int = 512,
        d_ssm: int = 128,
    ) -> None:
        self.d_pssm_block = 512
        self.model = S4WorldModel(
            S4_config=DictConfig(
                {
                    "d_model": d_pssm_block,
                    "layer": {"l_max": 1, "N": d_ssm},
                }
            ),
            training=False,
            process_in_chunks=False,
            rnn_mode=True,
            **DictConfig(
                {
                    "latent_dim": d_latent,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            jnp.zeros((batch_dim, 1, 270, 480, 1)),
            jnp.zeros((batch_dim, 1, 4)),
        )

        self.rnn_cache, self.prime, self.params = (
            sg(self.rnn_cache),
            sg(self.prime),
            sg(self.params),
        )

        return

    @partial(jax.jit, static_argnums=(0))
    def _jitted_forward(
        model,
        params: dict,
        cache: dict,
        prime: dict,
        depth_imgs: jax.Array,
        actions: jax.Array,
    ) -> dict:
        return model.apply(
            {
                "params": params,
                "cache": cache,
                "prime": prime,
            },
            depth_imgs,
            actions,
            single_step=True,
            mutable=["cache"],
            method="forward_RNN_mode",
        )

    def forward(
        self, depth_imgs: torch.tensor, actions: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        jax_imgs, jax_actions = from_torch_to_jax(depth_imgs), from_torch_to_jax(
            actions
        )

        jax_preds, vars = _jitted_forward(
            self.model, self.params, self.rnn_cache, self.prime, jax_imgs, jax_actions
        )
        self.rnn_cache = vars["cache"]

        return (
            from_jax_to_torch(jax_preds["hidden"]),
            from_jax_to_torch(jax_preds["z_posterior"]["dist"].mean()),
        )

    def reset_cache(self, batch_idx: int) -> None:
        for i in range(self.model.PSSM_blocks.n_blocks):
            for j in range(2):
                self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ][batch_idx] = jnp.zeros(
                    (self.N, self.d_pssm_block), dtype=jnp.complex64
                )

        return


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

    NUM_ENVS = 68

    torch_wm = TorchWrapper(
        NUM_ENVS,
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=2-latent_type=cont/checkpoint_2",
        d_latent=256,
        d_pssm_block=512,
    )

    torch_inputs_imgs = torch.rand(
        (NUM_ENVS, 1, 270, 480, 1), device="cuda:0", requires_grad=False
    )
    torch_inputs_actions = torch.rand(
        (NUM_ENVS, 1, 4), device="cuda:0", requires_grad=False
    )

    _ = torch_wm.forward(torch_inputs_imgs, torch_inputs_actions)

    fwp_times = []
    for _ in range(500):
        start = time.time()
        _ = torch_wm.forward(torch_inputs_imgs, torch_inputs_actions)
        end = time.time()
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)

    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))
