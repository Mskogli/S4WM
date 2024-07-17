import jax.numpy as jnp
import os
import jax
import time

from jax import random
from flax import linen as nn
from jax.nn.initializers import glorot_uniform, zeros
from functools import partial


class SimpleDecoder(nn.Module):
    c_out: int
    c_hid: int
    discrete_latent_state: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=4 * 8 * 2 * self.c_hid)(x)
        x = nn.silu(x)
        x = x.reshape(x.shape[0], x.shape[1], 4, 8, -1)

        x = nn.ConvTranspose(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=2 * self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 1),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_hid,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(3, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_out,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=(2, 2),
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.silu(x)
        x = nn.ConvTranspose(
            features=self.c_out,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=glorot_uniform(),
            bias_init=zeros,
        )(x)
        x = nn.sigmoid(jnp.squeeze(x[:, :, :-1, :], axis=-1))
        return x


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="normal"
)

# Code below adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial5/Inception_ResNet_DenseNet.html


class ResNetBlockDecoder(nn.Module):
    act_fn: callable
    c_out: int
    subsample: bool = False

    @nn.compact
    def __call__(self, x):
        z = nn.ConvTranspose(
            self.c_out,
            kernel_size=(2, 2),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        if self.subsample:
            x = nn.ConvTranspose(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNetDecoder(nn.Module):
    act_fn: callable = nn.silu
    block_class: nn.Module = ResNetBlockDecoder
    num_blocks: tuple = (1, 1, 1)
    c_hidden: tuple = (64, 32, 16)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5 * 8 * self.c_hidden[0])(x)
        x = self.act_fn(x)
        x = x.reshape(x.shape[0], x.shape[1], 5, 8, self.c_hidden[0])

        x = nn.ConvTranspose(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)
        x = self.act_fn(x)
        x = nn.ConvTranspose(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)

        x = self.act_fn(x)

        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group
                subsample = bc == 0
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x)

        x = nn.ConvTranspose(
            1,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(x)

        return nn.sigmoid(jnp.squeeze(x[:, :, :-1, :-8], axis=-1))


@partial(jax.jit, static_argnums=(0))
def jitted_forward(model, params, latent):
    return model.apply(
        {
            "params": params,
        },
        latent,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    key = random.PRNGKey(0)
    decoder = ResNetDecoder(act_fn=nn.silu, block_class=ResNetBlockDecoder)

    random_latent_batch = random.normal(key, (1, 2, 2048))

    params = decoder.init(key, random_latent_batch)["params"]
    output = decoder.apply({"params": params}, random_latent_batch)

    _ = jitted_forward(decoder, params, random_latent_batch)

    fnc = jax.vmap(jitted_forward)
    fwp_times = []
    for _ in range(2000):
        start = time.time()
        _ = jitted_forward(decoder, params, random_latent_batch)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
    fwp_times = jnp.array(fwp_times)
