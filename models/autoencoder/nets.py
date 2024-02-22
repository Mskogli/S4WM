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

        self.conv_20 = nn.Conv(
            64,
            kernel_size=(5, 5),
            strides=2,
            padding=(2, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_21 = nn.Conv(
            128,
            kernel_size=(128),
            strides=2,
            kernel_init=glorot_init,
            bias_init=zeros,
        )

        self.conv_30 = nn.Conv(
            128, kernel_size=(5, 5), strides=2, kernel_init=glorot_init, bias_init=zeros
        )

        self.conv_skip_0 = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=2,
            padding=(3, 2),
            kernel_init=glorot_init,
            bias_init=zeros,
        )
        self.conv_skip_1 = nn.Conv(
            128,
            kernel_size=(5, 5),
            strides=(2, 4),
            padding=(2, 3),
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
        chunks = jnp.array_split(imgs, 4, axis=1)
        downsampled_chunks = [self._downsample(chunk) for chunk in chunks]

        return jnp.concatenate(downsampled_chunks, axis=1)


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
        x = nn.tanh(x)

        return jnp.squeeze(x, axis=-1)

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        # Running the forward pass in chunks requires less contiguous memory
        chunks = jnp.array_split(latents, 4, axis=1)
        downsampled_chunks = [self._upsample(chunk) for chunk in chunks]

        return jnp.concatenate(downsampled_chunks, axis=1)


class Encoder(nn.Module):
    @nn.compact
    def _downsample(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        return x

    def __call__(self, imgs):
        chunks = jnp.array_split(imgs, 4, axis=1)
        downsampled_chunks = [self._downsample(chunk) for chunk in chunks]

        return jnp.concatenate(downsampled_chunks, axis=1)


class Decoder(nn.Module):
    @nn.compact
    def _upsample(self, z):
        z = nn.Dense(features=34 * 60 * 128)(
            z
        )  # Corresponds to the flattened size before the final layer of the encoder
        z = z.reshape(
            (z.shape[0], z.shape[1], 34, 60, 128)
        )  # Reshape to the feature map size before flattening
        z = nn.ConvTranspose(
            features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            features=1,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(z)

        return jnp.squeeze(z[:, :, :-2, ...], axis=-1)

    def __call__(self, z):
        chunks = jnp.array_split(z, 4, axis=1)
        downsampled_chunks = [self._upsample(chunk) for chunk in chunks]

        return jnp.concatenate(downsampled_chunks, axis=1)


class AutoEncoder(nn.Module):
    latent_dim: int

    def setup(self) -> None:
        self.encoder = ImageEncoder(self.latent_dim)
        self.decoder = ImageDecoder(self.latent_dim)

    def __call__(self, imgs: jnp.ndarray) -> jnp.ndarray:
        latents = self.encoder(imgs)
        preds = self.decoder(latents)
        return preds


if __name__ == "__main__":
    # Test Encoder Implementation
    key = random.PRNGKey(0)
    encoder = ImageEncoder(latent_dim=256)
    random_img_batch = random.normal(key, (2, 10, 270, 480, 1))

    params = encoder.init(key, random_img_batch)["params"]
    output = encoder.apply({"params": params}, random_img_batch)

    del encoder, params, output

    decoder = Decoder()
    random_latent_batch = random.normal(key, (2, 10, 128))
    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)
    print(output.shape)
