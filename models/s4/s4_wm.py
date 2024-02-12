import jax
import jax.numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from common import ImageEncoder, ImageDecoder
from utils import OneHotDist

from typing import Dict

tfd = tfp.distributions
f32 = jnp.float32


class S4WorldModel(nn.Module):
    latent_dim: int = 128
    discrete_latent_state: bool = True

    def setup(self) -> None:
        self.num_classes = self.latent_dim
        self.encoder = ImageEncoder(latent_dim=self.latent_dim, act="silu")
        self.decoder = ImageDecoder(latent_dim=self.latent_dim, act="silu")

        self.embedding_to_discrete_stats_fc = nn.Dense(
            features=self.latent_dim * self.num_classes
        )
        self.embedding_to_continious_stats_fc = nn.Dense(features=2 * self.latent_dim)

    def get_latent_state_from_image(self, image: jnp.ndarray) -> jnp.ndarray:
        embedding = self.encoder(image)
        statistics = self._get_statistics_from_embedding(
            embedding=embedding, discrete=self.discrete_latent_state, unimix=0.01
        )
        latent_distribution = self._get_distribution_from_statistics(
            statistics=statistics, discrete=self.discrete_latent_state
        )
        z = latent_distribution.sample(seed=jax.random.PRNGKey(1))
        return z

    def get_image_from_hidden_state(self, hidden: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(hidden)

    def get_statistics_from_embedding(self):
        pass

    def get_latent_distribution_from_statistics(self):
        pass

    def loss(self):
        pass

    def compute_hidden_state(self):
        pass

    def _get_distribution_from_statistics(
        self, statistics: Dict[str, jnp.ndarray], discrete: bool = True
    ) -> tfd.Distribution:
        if discrete:
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        else:
            mean = statistics["mean"].astype(f32)
            std = statistics["std"].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)

    def _get_statistics_from_embedding(
        self, embedding: jnp.ndarray, discrete: bool = False, unimix: float = 0.0
    ) -> Dict[str, jnp.ndarray]:
        if discrete:
            x = self.embedding_to_discrete_stats_fc(embedding)
            logits = x.reshape(x.shape[:1] + (self.latent_dim, self.num_classes))

            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}

        x = self.embedding_to_continious_stats_fc(embedding)
        mean, std = jnp.split(x, 2, -1)
        std = 2 * jax.nn.sigmoid(std / 2) + 0.1
        return {"mean": mean, "std": std}

    def __call__(self, img):
        z = self.get_latent_state_from_image(img)
        print(z.shape)
        print(z)


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    input_img = jax.random.normal(key, (1, 270, 480))

    model = S4WorldModel(discrete_latent_state=False)
    params = model.init(jax.random.PRNGKey(1), input_img)["params"]
