import torch
import os
import time
import jax.numpy as jnp

from s4wm.nn.s4_wm import S4WMTorchWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if __name__ == "__main__":
    NUM_ENVS = 256
    LATENT_DIM = 128

    S4WM = S4WMTorchWrapper(
        NUM_ENVS,
        "/home/mathias/dev/rl_checkpoints/gaussian_128",
        latent_dim=2 * LATENT_DIM,
        S4_block_dim=512,
        num_S4_blocks=3,
        ssm_dim=128,
    )

    depth_imgs = torch.zeros((NUM_ENVS, 1, 135, 240, 1), device="cuda:0")
    actions = torch.ones((NUM_ENVS, 1, 4), device="cuda:0")
    latents = torch.zeros((NUM_ENVS, 1, LATENT_DIM), device="cuda:0")

    fwp_times = []
    for _ in range(2000):
        start = time.time()
        _ = S4WM.forward(depth_imgs, latents, actions)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)

    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))
