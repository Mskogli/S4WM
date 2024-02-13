import jax
import jax.numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from common import ImageEncoder, ImageDecoder
from s4_nn import S4Block
from utils import OneHotDist, sg

from typing import Dict, Union, Tuple

tfd = tfp.distributions
f32 = jnp.float32


class S4WorldModel(nn.Module):
    latent_dim: int = 128
    action_dim: int = 4
    hidden_dim: int = 128
    discrete_latent_state: bool = True
    batch_size: int = 1

    def setup(self) -> None:
        self.num_classes = self.latent_dim

        self.encoder = ImageEncoder(latent_dim=self.latent_dim, act="silu")
        self.decoder = ImageDecoder(latent_dim=self.latent_dim, act="silu")
        self.sequence_block = S4Block

        embedding_to_stats_head = nn.Dense(
            features=(
                self.latent_dim * self.num_classes
                if self.discrete_latent_state
                else 2 * self.latent_dim
            )
        )
        hidden_to_stats_head = nn.Dense(
            features=(
                self.latent_dim * self.num_classes
                if self.discrete_latent_state
                else 2 * self.latent_dim
            )
        )

        self.statistic_heads = {
            "embedding": embedding_to_stats_head,
            "hidden": hidden_to_stats_head,
        }

        self.input_head = nn.Dense(features=self.latent_dim + self.action_dim)

    def get_latent_posterior_from_image(
        self, image: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        """_summary_

        Args:
            image (jnp.ndarray): _description_

        Returns:
            Tuple[jnp.ndarray, tfd.Distribution]: _description_
        """
        embedding = self.encoder(image)
        statistics = self._get_statistics(
            x=embedding,
            name="embedding",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )
        z_posterior_dist = self._get_distribution_from_statistics(
            statistics=statistics, discrete=self.discrete_latent_state
        )
        z_posterior = z_posterior_dist.sample(seed=jax.random.PRNGKey(1))
        return z_posterior, z_posterior_dist

    def get_latent_prior_from_hidden(
        self, hidden: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        """_summary_

        Args:
            hidden (jnp.ndarray): _description_

        Returns:
            Tuple[jnp.ndarray, tfd.Distribution]: _description_
        """
        statistics = self._get_statistics(
            x=hidden,
            name="hidden",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )
        z_prior_dist = self._get_distribution_from_statistics(
            statistics=statistics, discrete=self.discrete_latent_state
        )
        z_prior = z_prior_dist.sample(seed=jax.random.PRNGKey(1))
        return z_prior, z_prior_dist

    def get_image_prior_from_hidden(
        self, hidden: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        """_summary_

        Args:
            hidden (jnp.ndarray): _description_

        Returns:
            Tuple[jnp.ndarray, tfd.Distribution]: _description_
        """
        x = self.decoder(hidden)
        img_prior_dist = self._get_distribution_from_statistics(
            statistics=x, image=True
        )
        img_prior = img_prior_dist.sample(seed=jax.random.PRNGKey(1))

        return img_prior, img_prior_dist

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.array,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distriubtion,
    ) -> None:
        """_summary_

        Args:
            img_prior_dist (tfd.Distribution): _description_
            img_posterior (jnp.array): _description_
            z_posterior_dist (tfd.Distribution): _description_
            z_prior_dist (tfd.Distriubtion): _description_

        Returns:
            jnp.float32: _description_
        """
        BETA_REC = 0.8
        BETA_KL = 0.2
        ALPHA = 0.8  # KL Balancing parameter

        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf
        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))
        kl_loss = ALPHA * dynamics_loss + (1 - ALPHA) * representation_loss

        reconstruction_loss = -img_prior_dist.log_prob(img_posterior.astype(f32))
        total_loss = BETA_REC * reconstruction_loss + BETA_KL * kl_loss

        return total_loss

    def _get_distribution_from_statistics(
        self,
        statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        discrete: bool = True,
        image: bool = False,
    ) -> tfd.Distribution:
        """_summary_

        Args:
            statistics (Union[Dict[str, jnp.ndarray], jnp.ndarray]): _description_
            discrete (bool, optional): _description_. Defaults to True.
            image (bool, optional): _description_. Defaults to False.

        Returns:
            tfd.Distribution: _description_
        """
        if image:
            mean = statistics.reshape(1, -1).astype(f32)
            std = jnp.ones_like(mean).astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)
        elif discrete:
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        else:
            mean = statistics["mean"].astype(f32)
            std = statistics["std"].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)

    def _get_statistics(
        self, x: jnp.ndarray, name: str, discrete: bool = False, unimix: float = 0.0
    ) -> Dict[str, jnp.ndarray]:
        """_summary_

        Args:
            x (jnp.ndarray): _description_
            name (str): _description_
            discrete (bool, optional): _description_. Defaults to False.
            unimix (float, optional): _description_. Defaults to 0.0.

        Returns:
            Dict[str, jnp.ndarray]: _description_
        """
        if discrete:
            x = self.statistic_heads[name](x)
            logits = x.reshape(x.shape[:1] + (self.latent_dim, self.num_classes))

            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}

        x = self.statistic_heads[name](x)
        mean, std = jnp.split(x, 2, -1)
        std = 2 * jax.nn.sigmoid(std / 2) + 0.1
        return {"mean": mean, "std": std}

    def __call__(self, img: jnp.ndarray, action: jnp.ndarray):
        """_summary_

        Args:
            img (jnp.ndarray): _description_
            action (jnp.ndarray): _description_
        """

        # Compute the latent state z_t and the posterior distribution z_t ~ q(z | x)
        z_posterior, z_posterior_dist = self.get_latent_posterior_from_image(img)

        if self.discrete_latent_state:
            shape = (self.batch_size, self.latent_dim * self.num_classes)
            z_posterior = z_posterior.reshape(shape)

        g = self.input_head(jnp.concatenate((z_posterior, action), axis=-1))
        hidden = self.sequence_block(g.reshape(1, 1, -1))

        # Compute prior \hat z_{t+1} ~ p(\hat z | h) and predict next depth image
        z_prior, z_prior_dist = self.get_latent_prior_from_hidden(hidden)
        x_prior = self.get_image_prior_from_hidden(hidden)


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    input_img = jax.random.normal(key, (1, 270, 480))

    model = S4WorldModel(discrete_latent_state=False, latent_dim=128)
    params = model.init(jax.random.PRNGKey(1), input_img, jnp.zeros((1, 4)))["params"]
