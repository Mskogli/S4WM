import jax
import torch
import os
import time
import jax.numpy as jnp

from functools import partial
from omegaconf import DictConfig
from typing import Tuple, Sequence

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
        d_pssm_blocks: int = 512,
        d_ssm: int = 32,
        num_pssm_blocks: int = 4,
        discrete_latent_state: bool = True,
        l_max: int = 99,
    ) -> None:
        self.d_pssm_block = d_pssm_blocks
        self.d_ssm = d_ssm
        self.num_pssm_blocks = num_pssm_blocks

        self.model = S4WorldModel(
            S4_config=DictConfig(
                {
                    "d_model": d_pssm_blocks,
                    "layer": {"l_max": l_max, "N": d_ssm},
                    "n_blocks": num_pssm_blocks,
                }
            ),
            training=False,
            process_in_chunks=False,
            rnn_mode=True,
            discrete_latent_state=discrete_latent_state,
            **DictConfig(
                {
                    "latent_dim": d_latent,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        init_depth = jnp.zeros((batch_dim, 1, 135, 240, 1))
        init_actions = jnp.zeros((batch_dim, 1, 4))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.rnn_cache, self.prime, self.params = (
            sg(self.rnn_cache),
            sg(self.prime),
            sg(self.params),
        )

        # Force compilation
        _ = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_depth,
            init_actions,
        )
        return

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
            from_jax_to_torch(jax_preds["z_posterior"]["sample"]),
        )

    def reset_cache(self, batch_idx: Sequence) -> None:
        for i in range(self.num_pssm_blocks):
            for j in range(2):
                self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ] = (
                    self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                        "cache_x_k"
                    ]
                    .at[jnp.array([batch_idx])]
                    .set(jnp.ones((self.d_ssm, self.d_pssm_block), dtype=jnp.complex64))
                )
        return


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

    NUM_ENVS = 1

    torch_wm = TorchWrapper(
        NUM_ENVS,
        "/home/mathias/dev/structured-state-space-wm/s4wm/nn/checkpoints/depth_dataset/d_model=512-lr=0.0001-bsz=4-latent_type=disc/checkpoint_10",
        d_latent=1024,
        d_pssm_blocks=512,
    )

    init_depth = torch.zeros(
        (NUM_ENVS, 1, 135, 240, 1), requires_grad=False, device="cuda:0"
    )
    init_actions = torch.ones((NUM_ENVS, 1, 4), requires_grad=False, device="cuda:0")

    fwp_times = []
    for _ in range(200):
        start = time.time()
        _ = torch_wm.forward(init_depth, init_actions)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)

    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))
