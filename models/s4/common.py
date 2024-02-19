import jax
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros


class ImageEncoder(nn.Module):
    latent_dim: int
    seq_len: int = 150
    chunk_size: int = 15
    act: str = "elu"
    c_hid: int = 32

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

        self.conv_1 = nn.Conv(
            features=self.c_hid,
            kernel_size=(5, 5),
            strides=2,
        )
        self.conv_2 = nn.Conv(
            features=self.c_hid,
            kernel_size=(3, 3),
            strides=1,
        )
        self.conv_3 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(5, 5),
            strides=2,
        )
        self.conv_4 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(3, 3),
            strides=1,
        )
        self.conv_5 = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(5, 5),
            strides=2,
        )
        self.dense = nn.Dense(features=self.latent_dim)

    def _downsample(self, img: jnp.ndarray) -> jnp.ndarray:
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

        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten grid to feature vector
        x = self.dense(x)
        return x

    def __call__(self, imgs: jnp.ndarray) -> jnp.ndarray:
        # output = []

        # num_chunks = (self.seq_len + self.chunk_size - 1) // self.chunk_size
        # for i in range(num_chunks):
        #     start = i * self.chunk_size
        #     end = start + self.chunk_size
        #     # Adjust end for the last chunk to not exceed seq_len
        #     end = min(end, self.seq_len)
        #     input_chunk = imgs[:, start:end, :]
        #     output.append(self._downsample(input_chunk))

        # output = jnp.concatenate(output, axis=1)
        # return output.reshape(-1, self.seq_len, self.latent_dim)
        return self._downsample(imgs)


class ImageDecoder(nn.Module):
    latent_dim: int
    seq_len: int = 149
    chunk_size: int = 15
    img_h: int = 270
    img_w: int = 480
    output_channels: int = 1
    hidden_channels: int = 32
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

    def _upsample(self, latent: jnp.ndarray) -> jnp.ndarray:
        x = self.dense_1(latent)
        x = self.act_fn(x)
        x = self.dense_2(x)
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
        # output = []

        # num_chunks = (self.seq_len + self.chunk_size - 1) // self.chunk_size
        # for i in range(num_chunks):
        #     start = i * self.chunk_size
        #     end = start + self.chunk_size
        #     Adjust end for the last chunk to not exceed seq_len
        #     end = min(end, self.seq_len)
        #     input_chunk = latents[:, start:end, :]
        #     output.append(jnp.array(self._upsample(input_chunk)))

        # output = jnp.concatenate(output, axis=1)
        # return output.reshape(-1, self.seq_len, self.img_h, self.img_w)
        return self._upsample(latents)


class ResNetEncoder(nn.Module):
    embedding_dim: int = 128

    @nn.compact
    def __call__(self, depth_imgs: jnp.ndarray) -> jnp.ndarray:

        # First layer group
        x_00 = nn.Conv(
            features=32, kernel_size=(5, 5), strides=(2, 2), padding=((2, 2), (2, 2))
        )(depth_imgs)
        x_01 = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            kernel_init=jax.nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )(x_00)
        x_01 = nn.elu(x_01)
        print(x_01.shape)

        # Second layer group
        x_10 = nn.Conv(
            features=32, kernel_size=(5, 5), strides=(2, 2), padding=((1, 1), (1, 1))
        )(x_01)
        x_11 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=jax.nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )(x_10)

        print(x_11.shape)
        # First skip connection
        x_01_jump = nn.Conv(
            features=64, kernel_size=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1))
        )(x_01)
        x_11 = x_11 + x_01_jump
        x_11 = nn.elu(x_11)

        # Third layer group
        x_20 = nn.Conv(
            features=64, kernel_size=(5, 5), strides=(2, 2), padding=((2, 2), (2, 2))
        )(x_11)
        x_21 = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            kernel_init=jax.nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )(x_20)

        # Second skip connection
        x_11_jump = nn.Conv(
            features=128, kernel_size=(5, 5), strides=(4, 4), padding=((2, 2), (1, 1))
        )(x_11)
        x_21 = x_21 + x_11_jump
        x_21 = nn.elu(x_21)

        # Fourth layer
        x_30 = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2))(x_21)
        x_30 = x_30.reshape(x_30.shape[0], x_30.shape[1], -1)
        x_31 = nn.Dense(features=512)(x_30)
        x_31 = nn.elu(x_31)
        x_31 = nn.Dense(features=self.embedding_dim)


# Conv initialized with kaiming int, but uses fan-out instead of fan-in mode
# Fan-out focuses on the gradient distribution, and is commonly used in ResNets
resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="normal"
)


class ResNetBlock(nn.Module):
    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNetEncoder(nn.Module):
    num_classes: int
    act_fn: callable
    block_class: nn.Module
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        if (
            self.block_class == ResNetBlock
        ):  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x, train=train)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Dense(self.num_classes)(x)
        return x


if __name__ == "__main__":
    # Test Encoder Implementation
    key = random.PRNGKey(0)
    img_encoder = ResNetEncoder(
        num_classes=128, act_fn=nn.silu, block_class=ResNetBlock
    )
    input_img_1 = random.normal(key, (2, 149, 1, 270, 480))
    input_img = random.normal(key, (2, 149, 1, 270, 480))

    params = img_encoder.init(random.PRNGKey(1), input_img)["params"]
    print("initated")
    output = img_encoder.apply({"params": params}, input_img, train=False)
    print(output.shape)
    # print("Encoder Output Shape: ", output.shape)

    # del output, input_img, params, img_encoder

    # # Test Decoder Implementation
    # input_latent = random.normal(random.PRNGKey(2), (8, 149, 128))
    # img_decoder = ImageDecoder(latent_dim=128)

    # params = img_decoder.init(random.PRNGKey(3), input_latent)["params"]
    # output = img_decoder.apply({"params": params}, input_latent)
    # print("Decoder Output Shape: ", output.shape)
