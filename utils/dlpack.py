import jax.dlpack
import torch.utils.dlpack

"""
Utility functions to transfer jax arrays to torch tensors and vice versa without unloading the data from the GPU
"""


def from_jax_to_torch(jax_array: jax.Array) -> torch.tensor:
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_array))


def from_torch_to_jax(x_torch: torch.tensor) -> jax.Array:
    shape = x_torch.shape
    x_torch_flat = torch.flatten(x_torch)
    x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch_flat))
    return x_jax.reshape(shape)


if __name__ == "__main__":
    jax_array = jax.numpy.array([1, 2, 3, 4, 5])
    torch_tensor = torch.tensor([1, 2, 3, 4, 5], device="cuda:0")

    torch_from_jax = from_jax_to_torch(jax_array)
    print(torch_from_jax.device)

    jax_from_torch = from_torch_to_jax(torch_tensor)
    print(jax_from_torch.devices())
