import torch
import os
import time
import jax.numpy as jnp

from s4wm.nn.s4_wm import S4WMTorchWrapper


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    NUM_ENVS = 1024
    torch_wm = S4WMTorchWrapper(
        NUM_ENVS,
        "/home/mihir/dev-mathias/structured-state-space-wm/weights/512_resnet_encoder_decoder",
        d_latent=1024,
        d_pssm_blocks=512,
        num_pssm_blocks=3,
        d_ssm=100,
    )

    init_depth = torch.zeros(
        (NUM_ENVS, 1, 135, 240, 1), requires_grad=False, device="cuda:0"
    )
    init_actions = torch.ones((NUM_ENVS, 1, 4), requires_grad=False, device="cuda:0")
    latent = jnp.zeros((NUM_ENVS, 1, 1024))

    fwp_times = []
    for _ in range(2000):
        start = time.time()
        _ = torch_wm.forward(init_depth, init_actions, latent)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)

    print("Forward pass avg: ", jnp.mean(fwp_times))
    print("Forward pass std: ", jnp.std(fwp_times))
