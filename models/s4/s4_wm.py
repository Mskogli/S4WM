import jax
import jax.numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .common import ImageEncoder, ImageDecoder
from .s4_nn import S4Block
from .utils import OneHotDist, sg

from typing import Dict, Union, Tuple

tfd = tfp.distributions
f32 = jnp.float32


class S4WorldModel(nn.Module):
    """Structured State Space Sequence (S4) based world model

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    S4_config: DictConfig

    latent_dim: int = 128
    hidden_dim: int = 512
    num_actions: int = 4
    alpha: float = 0.8
    beta_rec: float = 0.8
    beta_kl: float = 0.2
    discrete_latent_state: bool = False
    training: bool = False

    def setup(self) -> None:
        self.num_classes = (
            jnp.sqrt(self.latent_dim) if self.discrete_latent_state else None
        )
        self.seed = jax.random.PRNGKey(0)

        self.encoder = ImageEncoder(latent_dim=self.latent_dim, act="silu")
        self.decoder = ImageDecoder(latent_dim=self.latent_dim, act="silu")

        self.sequence_block = S4Block(**self.S4_config, training=self.training)

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

        self.input_head = nn.Dense(features=self.latent_dim + self.num_actions)

    def get_latent_posteriors_from_images(
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
        z_posterior = z_posterior_dist.sample(seed=self.seed)
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
        z_prior = z_prior_dist.sample(seed=self.seed)
        return z_prior, z_prior_dist

    def get_image_prior_dists(
        self, hidden: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        """_summary_

        Args:
            hidden (jnp.ndarray): _description_

        Returns:
            Tuple[jnp.ndarray, tfd.Distribution]: _description_
        """
        x = self.decoder(hidden)
        img_prior_dists = self._get_distribution_from_statistics(
            statistics=x, image=True
        )
        img_priors = img_prior_dists.sample(seed=self.seed)

        return img_priors, img_prior_dists

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.array,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distribution,
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

        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf
        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))
        kl_loss = self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss

        reconstruction_loss = -img_prior_dist.log_prob(img_posterior.astype(f32))
        total_loss = self.beta_rec * reconstruction_loss + self.beta_kl * kl_loss

        return total_loss

    def _get_distribution_from_statistics(
        self,
        statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        discrete: bool = True,
        image: bool = False,
    ) -> tfd.Distribution:

        if image:
            mean = statistics.reshape(
                statistics.shape[0], statistics.shape[1], -1
            ).astype(f32)
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

        if discrete:
            x = self.statistic_heads[name](x)
            logits = x.reshape(x.shape[:2] + (self.num_classes, self.num_classes))

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

    def __call__(self, imgs: jnp.ndarray, actions: jnp.ndarray):
        """_summary_

        Args:
            img (jnp.ndarray): (batch_size, seq_length, H, W)
            action (jnp.ndarray): (batch_size, seq_length, num_actions)
        """

        batch_size, seq_length = imgs.shape[:2]

        # Compute the latent state z_t and the posterior distribution z_t ~ q(z | x)
        z_posteriors, z_posterior_dists = self.get_latent_posteriors_from_images(
            imgs[:, :-1]
        )

        if self.discrete_latent_state:
            z_posteriors = z_posteriors.reshape(
                (batch_size, seq_length, self.num_classes * self.num_classes)
            )

        g = self.input_head(jnp.concatenate((z_posteriors, actions[:, :-1]), axis=-1))
        hidden = self.sequence_block(g)

        # Compute prior \hat z_{t+1} ~ p(\hat z | h) and predict next depth image
        z_priors, z_prior_dists = self.get_latent_prior_from_hidden(hidden)
        img_priors, img_prior_dists = self.get_image_prior_dists(
            jnp.concatenate((hidden, z_priors), axis=-1)
        )

        ret = (
            (z_priors, z_prior_dists),
            (z_posteriors, z_posterior_dists),
            (img_priors, img_prior_dists),
        )
        return ret


if __name__ == "__main__":
    batch_size, seq_length = 8, 150

    # Setup
    key = jax.random.PRNGKey(0)
    dummy_input_img = jax.random.normal(key, (batch_size, seq_length, 1, 270, 480))
    dummy_input_actions = jax.random.normal(key, (batch_size, seq_length, 4))

    model = S4WorldModel(discrete_latent_state=False, latent_dim=128)
    params = model.init(jax.random.PRNGKey(1), dummy_input_img, dummy_input_actions)
    print("Model initialized")

    model_loss = model.apply(params, dummy_input_img, dummy_input_actions)
    print("Loss: ", model_loss)
