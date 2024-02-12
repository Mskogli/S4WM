import jax
import jax.numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from .common import ImageEncoder, ImageDecoder
from .utils import OneHotDist

tfd = tfp.distributions


class S4WorldModel(nn.module):

    def setup(self) -> None:
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

    def get_embedding_from_image(self):
        pass

    def get_statistics_from_embedding(self):
        pass

    def get_latent_distribution_from_statistics(self):
        pass

    def loss(self):
        pass

    def get_image_from_hidden_state(self):
        pass

    def compute_hidden_state(self):
        pass


def get_latent_distribution_from_statistics(latent_stats, discrete: bool = True):
    if discrete:
        return tfd.Independent(OneHotDist(latent_stats["logits"]), 1)
    else:
        mean = latent_stats["mean"]
        std = latent_stats["std"]
        return tfd.MultivariateNormalDiag(mean, std)


def get_statistics_from_embedding(
    embedding, discrete: bool = False, unimix: float = 0.0
):
    if discrete:
        logits = embedding.reshape(embedding.shape[:1] + (32, 32))

        if unimix:
            probs = jax.nn.softmax(logits, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = jnp.log(probs)
        return {"logits": logits}

    mean, std = jnp.split(embedding, 2, -1)
    std = 2 * jax.nn.sigmoid(std / 2) + 0.1
    return {"mean": mean, "std": std}


if __name__ == "__main__":
    pass
