import jax.numpy as jnp
import jax
import time
import os

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros
from functools import partial


class ImageEncoder(nn.Module):
    latent_dim: int
    act: str = "elu"
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

        self.conv_00 = nn.Conv(
            features=32, kernel_size=(5, 5), strides=1, padding=(2, 2)
        )
        self.conv_01 = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_10 = nn.Conv(
            features=32, kernel_size=(5, 5), strides=2, padding=(1, 1)
        )
        self.conv_11 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=1,
            padding=(2, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_20 = nn.Conv(
            features=64,
            kernel_size=(5, 5),
            strides=2,
            padding=(2, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_21 = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_30 = nn.Conv(
            features=128,
            kernel_size=(5, 5),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_skip_0 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=2,
            padding=(3, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_skip_1 = nn.Conv(
            features=128,
            kernel_size=(5, 5),
            strides=(4, 4),
            padding=(2, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.dense_40 = nn.Dense(features=512, kernel_init=glorot_init, bias_init=zeros)
        self.dense_41 = nn.Dense(
            features=self.latent_dim, kernel_init=glorot_init, bias_init=zeros
        )

    def _downsample(self, img: jnp.ndarray) -> jnp.ndarray:
        # First stage
        x_00 = self.conv_00(img)
        x_01 = self.conv_01(x_00)
        x_01 = self.act_fn(x_01)

        # Second Stage
        x_10 = self.conv_10(x_01)
        x_11 = self.conv_11(x_10)

        # Add first skip connection
        x_skip_0 = self.conv_skip_0(x_01)
        x_11 = x_11 + x_skip_0
        x_11 = self.act_fn(x_11)

        # Third stage
        x_20 = self.conv_20(x_11)
        x_21 = self.conv_21(x_20)

        # Add second skip connection
        x_skip_1 = self.conv_skip_1(x_11)
        x_21 = x_21 + x_skip_1
        x_21 = self.act_fn(x_21)

        # Fourth stage
        x_30 = self.conv_30(x_21)
        x_30 = x_30.reshape(x_30.shape[0], x_30.shape[1], -1)

        # Fifth stage
        x_40 = self.dense_40(x_30)
        x_40 = self.act_fn(x_40)
        x_41 = self.dense_41(x_40)

        return x_41

    def __call__(self, imgs: jnp.ndarray) -> jnp.ndarray:
        # Running the forward pass in chunks requires less contiguous memory
        if self.process_in_chunks:
            chunks = jnp.array_split(imgs, 4, axis=0)
            downsampled_chunks = [self._downsample(chunk) for chunk in chunks]

            return jnp.concatenate(downsampled_chunks, axis=1)
        else:
            return self._downsample(imgs)


class Encoder(nn.Module):
    c_hid: int
    embedding_dim: 512
    latent_dim: 4096

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.silu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.silu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.silu(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Dense(features=self.embedding_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(features=self.latent_dim)(x)
        return x


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, image):
    return model.apply(
        {
            "params": jax.lax.stop_gradient(params),
        },
        jax.lax.stop_gradient(image),
    )


if __name__ == "__main__":
    # Test Encoder Implementation
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    key = random.PRNGKey(0)
    encoder = Encoder(latent_dim=1024, c_hid=16, embedding_dim=2048)

    random_img_batch = random.normal(key, (128, 135, 240, 1))
    params = encoder.init(key, random_img_batch)["params"]
    output = encoder.apply({"params": params}, random_img_batch)
    print("Output shape: ", output.shape)

    _ = jitted_forward(encoder, params, random_img_batch)

    random_img_batch = random.normal(key, (128, 135, 240, 1))
    fnc = jax.vmap(jitted_forward)
    fwp_times = []
    for _ in range(200):
        start = time.time()
        _ = jitted_forward(encoder, params, random_img_batch)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)
