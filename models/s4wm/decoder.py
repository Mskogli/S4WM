import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros


class ImageDecoder(nn.Module):
    latent_dim: int
    act: str = "elu"

    def setup(self) -> None:

        if self.act == "elu":
            self.act_fn = nn.elu
        elif self.act == "gelu":
            self.act_fn = nn.gelu
        elif self.act == "silu":
            self.act_fn = nn.silu
        else:
            self.act_fn = lambda x: x

        glorot_init = (
            glorot_uniform()
        )  # Equivalent to Pytorch's Xavier Uniform https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.glorot_uniform.html

        self.dense_00 = nn.Dense(features=512, kernel_init=glorot_init, bias_init=zeros)
        self.dense_01 = nn.Dense(
            features=9 * 15 * 128, kernel_init=glorot_init, bias_init=zeros
        )

        self.deconv_1 = nn.ConvTranspose(
            128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_2 = nn.ConvTranspose(
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_3 = nn.ConvTranspose(
            32,
            kernel_size=(6, 6),
            strides=(5, 4),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_4 = nn.ConvTranspose(
            16,
            kernel_size=(4, 4),
            strides=(3, 4),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_5 = nn.ConvTranspose(
            1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )

    def _upsample(self, latent: jnp.ndarray) -> jnp.ndarray:
        x = self.dense_00(latent)
        x = self.act_fn(x)
        x = self.dense_01(x)
        x = x.reshape(x.shape[0], x.shape[1], 9, 15, 128)

        x = self.deconv_1(x)
        x = self.act_fn(x)

        x = self.deconv_2(x)
        x = self.act_fn(x)

        x = self.deconv_3(x)
        x = self.act_fn(x)

        x = self.deconv_4(x)
        x = self.act_fn(x)

        x = self.deconv_5(x)
        x = nn.sigmoid(x)

        return jnp.squeeze(x, axis=-1)

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        # Running the forward pass in chunks requires less contiguous memory
        chunks = jnp.array_split(latents, 4, axis=1)
        downsampled_chunks = [self._upsample(chunk) for chunk in chunks]

        return jnp.concatenate(downsampled_chunks, axis=1)


if __name__ == "__main__":
    # Test decoder implementation
    key = random.PRNGKey(0)
    decoder = ImageDecoder(latent_dim=128)

    random_latent_batch = random.normal(key, (2, 10, 128))
    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)
    print("Output shape: ", output.shape)
