import jax.numpy as jnp
import os
import jax
import time

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros
from functools import partial


class ImageDecoder(nn.Module):
    act: str = "silu"
    process_in_chunks: bool = False

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
            features=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_2 = nn.ConvTranspose(
            features=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="SAME",
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_3 = nn.ConvTranspose(
            features=32,
            kernel_size=(7, 6),
            strides=(4, 4),
            padding=(3, 4),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.deconv_4 = nn.ConvTranspose(
            features=16,
            kernel_size=(3, 4),
            strides=(2, 2),
            padding=(0, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.deconv_5 = nn.ConvTranspose(
            features=1,
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

        if self.process_in_chunks:
            chunks = jnp.array_split(latents, 4, axis=1)
            downsampled_chunks = [self._upsample(chunk) for chunk in chunks]

            return jnp.concatenate(downsampled_chunks, axis=1)
        else:
            return self._upsample(latents)


class Decoder(nn.Module):
    c_out: int
    c_hid: int
    discrete_latent_state: bool

    @nn.compact
    def __call__(self, x):
        if self.discrete_latent_state:
            x = nn.Dense(features=512)
            x = nn.silu(x)
            x = nn.Dense(features=8 * 15 * self.c_hid)(x)
            x = nn.silu(x)
        else:
            x = nn.Dense(features=8 * 15 * self.c_hid)(x)
            x = nn.silu(x)

        x = x.reshape(x.shape[0], x.shape[1], 8, 15, -1)

        x = nn.ConvTranspose(
            features=2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2), padding=(2, 2)
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2), padding=(2, 1)
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_hid, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_out, kernel_size=(3, 3), strides=(2, 2), padding=(0, 1)
        )(x)
        x = nn.sigmoid(jnp.squeeze(x[:, :, :, :-1], axis=-1))
        return x


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, latent):
    return model.apply(
        {
            "params": jax.lax.stop_gradient(params),
        },
        jax.lax.stop_gradient(latent),
    )


if __name__ == "__main__":
    # Test decoder implementation
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    key = random.PRNGKey(0)
    decoder = Decoder(c_hid=32, c_out=1, discrete_latent_state=False)

    random_latent_batch = random.normal(key, (1, 2, 128))
    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)
    print("Output shape: ", output.shape)

    # _ = jitted_forward(decoder, params, random_latent_batch)

    # fnc = jax.vmap(jitted_forward)
    # fwp_times = []
    # for _ in range(200):
    #     start = time.time()
    #     _ = jitted_forward(decoder, params, random_latent_batch)
    #     end = time.time()
    #     print(end - start)
    #     fwp_times.append(end - start)
    # fwp_times = jnp.array(fwp_times)
