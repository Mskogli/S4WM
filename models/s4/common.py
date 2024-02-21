import jax
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros


class ImageEncoder(nn.Module):
    latent_dim: int
    seq_len: int = 150
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

        self.conv_00 = nn.Conv(
            features=32, kernel_size=(5, 5), strides=2, padding=(2, 2)
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

        self.conv_20 = nn.Conv(64, kernel_size=(5, 5), strides=2, padding=(2, 2))
        self.conv_21 = nn.Conv(
            128,
            kernel_size=(128),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_30 = nn.Conv(128, kernel_size=(5, 5), strides=2)

        self.conv_skip_0 = nn.Conv(64, kernel_size=(4, 4), strides=2, padding=(3, 3))
        self.conv_skip_1 = nn.Conv(128, kernel_size=(5, 5), strides=4, padding=(2, 2))

        self.dense_40 = nn.Dense(features=512)
        self.dense_41 = nn.Dense(features=self.latent_dim)

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
        return self._downsample(imgs)


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

        self.dense_00 = nn.Dense(features=512)
        self.dense_01 = nn.Dense(features=9 * 15 * 128)

        self.deconv_1 = nn.ConvTranspose(
            128, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )
        self.deconv_2 = nn.ConvTranspose(
            64, kernel_size=(5, 5), strides=(2, 2), padding="SAME"
        )
        self.deconv_3 = nn.ConvTranspose(
            32, kernel_size=(6, 6), strides=(5, 4), padding="SAME"
        )
        self.deconv_4 = nn.ConvTranspose(
            16, kernel_size=(4, 4), strides=(3, 4), padding="SAME"
        )
        self.deconv_5 = nn.ConvTranspose(
            1, kernel_size=(4, 4), strides=(1, 1), padding="SAME"
        )

    def _upsample(self, latent: jnp.ndarray) -> jnp.ndarray:
        x = self.dense_00(latent)
        x = nn.relu(x)
        x = self.dense_01(x)
        x = x.reshape(x.shape[0], x.shape[1], 9, 15, 128)

        x = self.deconv_1(x)
        x = nn.relu(x)

        x = self.deconv_2(x)
        x = nn.relu(x)

        x = self.deconv_3(x)
        x = nn.relu(x)

        x = self.deconv_4(x)
        x = nn.relu(x)

        x = self.deconv_5(x)
        x = nn.tanh(x)

        return jnp.squeeze(x, axis=-1)

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        return self._upsample(latents)


if __name__ == "__main__":
    # Test Encoder Implementation
    key = random.PRNGKey(0)
    encoder = ImageEncoder(latent_dim=128, act="silu")
    random_img_batch = random.normal(key, (2, 150, 1, 270, 480))

    params = encoder.init(key, random_img_batch)["params"]
    output = encoder.apply({"params": params}, random_img_batch)
    print("Image decoder output shape: ", output.shape)

    del encoder, params, output

    decoder = ImageDecoder(latent_dim=128)
    random_latent_batch = random.normal(key, (2, 10, 128))
    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)
    print("output", output)
