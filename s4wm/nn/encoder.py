import jax
import os
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros
from functools import partial


class SimpleEncoder(nn.Module):
    c_hid: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.c_hid,
            kernel_size=(4, 4),
            strides=2,
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=self.c_hid,
            kernel_size=(4, 4),
            strides=2,
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=2,
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=2,
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=2,
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return x


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    key = random.PRNGKey(0)
    encoder = SimpleEncoder(
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
