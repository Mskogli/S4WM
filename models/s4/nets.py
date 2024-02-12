import jax.numpy as jnp

from jax import random
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from jax.nn.initializers import glorot_uniform, zeros

tfd = tfp.distributions


class ImageEncoder(nn.Module):
    latent_dim: int
    act: str = "elu"
    c_hid: int = 32

    def setup(self):

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
        zero_init = zeros

        self.conv_1 = nn.Conv(
            features=self.c_hid,
            kernel_size=(5, 5),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_2 = nn.Conv(
            features=self.c_hid,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_3 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(5, 5),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_4 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_5 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(5, 5),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.dense = nn.Dense(features=self.latent_dim)

    def __call__(self, img):
        x = self.conv_1(img)
        x = self.act_fn(x)

        x = self.conv_2(x)
        x = self.act_fn(x)

        x = self.conv_3(x)
        x = self.act_fn(x)

        x = self.conv_4(x)
        x = self.act_fn(x)

        x = self.conv_5(x)
        x = self.act_fn(x)

        x = x.reshape(x.shape[0], -1)  # Flatten grid to feature vector
        x = self.dense(x)
        return x


class ImageDecoder(nn.Module):
    latent_dim: int
    output_channels: int = 1
    hidden_channels: int = 32
    act: str = "elu"

    def setup(self):

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
        zero_init = zeros

        self.dense_1 = nn.Dense(features=512)
        self.dense_2 = nn.Dense(features=9 * 15 * 128)
        self.deconv_1 = nn.ConvTranspose(
            features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )
        self.deconv_2 = nn.ConvTranspose(
            features=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="SAME",
        )
        self.deconv_3 = nn.ConvTranspose(
            features=32,
            kernel_size=(6, 6),
            strides=(5, 4),
            padding="SAME",
        )
        self.deconv_4 = nn.ConvTranspose(
            features=16, kernel_size=(4, 4), strides=(3, 4), padding="SAME"
        )
        self.deconv_5 = nn.ConvTranspose(
            features=self.output_channels,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="SAME",
        )

    def __call__(self, latent):
        x = self.dense_1(latent)
        x = self.act_fn(x)
        x = self.dense_2(x)
        x = x.reshape(x.shape[0], 9, 15, 128)

        x = self.deconv_1(x)
        x = self.act_fn(x)

        x = self.deconv_2(x)
        x = self.act_fn(x)

        x = self.deconv_3(x)
        x = self.act_fn(x)

        x = self.deconv_4(x)
        x = self.act_fn(x)

        x = self.deconv_5(x)
        x = nn.tanh(x)
        x = jnp.squeeze(x, axis=-1)

        return x


if __name__ == "__main__":
    # Test Encoder Implementation
    key = random.PRNGKey(0)
    input_img = random.normal(key, (1, 270, 480))
    img_encoder = ImageEncoder(c_hid=32, latent_dim=128, act="silu")

    params = img_encoder.init(random.PRNGKey(1), input_img)["params"]
    output = img_encoder.apply({"params": params}, input_img)
    print("Decoder Output Shape: ", output.shape)

    # Test Decoder Implementation
    input_latent = random.normal(random.PRNGKey(2), (1, 128))
    img_decoder = ImageDecoder(latent_dim=128)

    params = img_decoder.init(random.PRNGKey(3), input_latent)["params"]
    output = img_decoder.apply({"params": params}, input_latent)
    print("Encoder Output Shape: ", output.shape)
