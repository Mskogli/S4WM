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

kernel_init = (
            glorot_uniform()
        )
class Encoder(nn.Module):
    c_hid: int
    embedding_dim: 512
    latent_dim: 1024
    discrete_latent_state: True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(4, 4), strides=2, kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(4, 4), strides=2, kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.silu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(4, 4), strides=2, kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.silu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(4, 4), strides=2, kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.silu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(4, 4), strides=2, kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.silu(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.discrete_latent_state:
            x = nn.Dense(features=self.latent_dim)(x)
        else:
            x = nn.Dense(features=2 * self.latent_dim)(x)
        return x


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, image):
    return model.apply(
        {
            "params": jax.lax.stop_gradient(params),
        },
        jax.lax.stop_gradient(image),
    )


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
            kernel_size=(2, 2),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(2, 2),
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
    act_fn: callable
    block_class: nn.Module
    latent_dim: int
    num_blocks: tuple = (0, 1, 1)
    c_hidden: tuple = (16, 16, 32)

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

        # Mapping to classification output
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Dense(features=256)(x)
        return x


if __name__ == "__main__":
    # Test Encoder Implementation
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    key = random.PRNGKey(0)
    encoder = Encoder(
        c_hid=32, embedding_dim=512, discrete_latent_state=True, latent_dim=1024
    )

    random_img_batch = random.normal(key, (4, 100, 135, 240, 1))
    params = encoder.init(key, random_img_batch)
    output, _ = encoder.apply(
        jax.lax.stop_gradient(params), random_img_batch, mutable=["batch_stats"]
    )
    print(
        "Output shape: ",
        output.shape,
    )

    # _ = jitted_forward(encoder, params, random_img_batch)

    # random_img_batch = random.normal(key, (128, 135, 240, 1))
    # fnc = jax.vmap(jitted_forward)
    # fwp_times = []
    # for _ in range(200):
    #     start = time.time()
    #     _ = jitted_forward(encoder, params, random_img_batch)
    #     end = time.time()
    #     print(end - start)
    #     fwp_times.append(end - start)
    # fwp_times = jnp.array(fwp_times)
