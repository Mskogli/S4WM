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
    beta_rec: float = 1.0
    beta_kl: float = 0.0
    discrete_latent_state: bool = False
    training: bool = True

    def setup(self) -> None:
        self.num_classes = (
            jnp.sqrt(self.latent_dim) if self.discrete_latent_state else None
        )
        self.seed = jax.random.PRNGKey(0)

        self.encoder = ImageEncoder(latent_dim=self.latent_dim)
        self.decoder = ImageDecoder(latent_dim=self.latent_dim)

        self.sequence_block = S4Block(**self.S4_config, training=self.training)

        embedding_to_stats_head = nn.Dense(
            features=(
                self.latent_dim if self.discrete_latent_state else 2 * self.latent_dim
            )
        )
        hidden_to_stats_head = nn.Dense(
            features=(
                self.latent_dim if self.discrete_latent_state else 2 * self.latent_dim
            )
        )

        self.statistic_heads = {
            "embedding": embedding_to_stats_head,
            "hidden": hidden_to_stats_head,
        }

        self.input_head = nn.Dense(features=1024)

    def get_latent_posteriors_from_images(
        self, image: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        embedding = self.encoder(image)
        statistics = self.get_statistics(
            x=embedding,
            name="embedding",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )

        z_posterior_dist = self.get_distribution_from_statistics(
            statistics=statistics, discrete=self.discrete_latent_state
        )
        z_posterior = z_posterior_dist.sample(seed=self.seed)

        return z_posterior, z_posterior_dist

    def get_latent_prior_from_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        statistics = self.get_statistics(
            x=hidden,
            name="hidden",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )
        z_prior_dist = self.get_distribution_from_statistics(
            statistics=statistics, discrete=self.discrete_latent_state
        )
        return z_prior_dist

    def get_image_prior_dists(
        self, hidden: jnp.ndarray, z_posterior: jnp.ndarray
    ) -> tfd.Distribution:
        x = self.decoder(jnp.concatenate((hidden, z_posterior), axis=-1))
        print("decoded image:",  x)   
        img_prior_dists = self.get_distribution_from_statistics(
            statistics=x, image=True
        )

        return img_prior_dists

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.ndarray,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distribution,
    ) -> jnp.ndarray:

        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf

        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        dynamics_loss = jnp.maximum(dynamics_loss, 1.0)

        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))
        representation_loss = jnp.maximum(representation_loss, 1.0)

        kl_loss = self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss
        reconstruction_loss = (
            -img_prior_dist.log_prob(img_posterior.astype(f32)) / (507)
        )
        return jnp.sum(
            self.beta_rec * reconstruction_loss + self.beta_kl * kl_loss, axis=-1
        )

    def get_distribution_from_statistics(
        self,
        statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        discrete: bool = True,
        image: bool = False,
    ) -> tfd.Distribution:

        if image:
            mean = statistics.reshape(
                statistics.shape[0], statistics.shape[1], -1
            ).astype(f32)
            return tfd.MultivariateNormalDiag(mean, jnp.ones_like(mean))
        elif discrete:
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        else:
            mean = statistics["mean"].astype(f32)
            std = statistics["std"].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)

    def get_statistics(
        self, x: jnp.ndarray, name: str, discrete: bool = False, unimix: float = 0.01
    ) -> Dict[str, jnp.ndarray]:

        if discrete:
            logits = self.statistic_heads[name](x)
            logits = logits.reshape(logits.shape[0], logits.shape[1], 32, 32)

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

    def __call__(
        self, imgs: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[tfd.Distribution, ...]:  # 3 tuple

        batch_size, seq_length = imgs.shape[:2]

        # Compute the latent posteriors from the input images
        z_posteriors, z_posterior_dists = self.get_latent_posteriors_from_images(imgs)

        # Reshape the posterior if the latent embedding is discrete (e.g. 32x32)
        if self.discrete_latent_state:
            z_posteriors = z_posteriors.reshape(
                (batch_size, seq_length, self.latent_dim)
            )

        # Concatenate and mix the latent posteriors and the actions, compute the dynamics embedding by forward passing the stacked PSSM blocks
        g = self.input_head(
            jnp.concatenate((z_posteriors[:, :-1], actions[:, 1:]), axis=-1)
        )
        hidden = self.sequence_block(g)

        # Compute the latent prior distributions from the hidden state
        z_prior_dists = self.get_latent_prior_from_hidden(hidden)

        # Compute the image priors trough the hidden states and the latent posteriors
        img_prior_dists  = self.get_image_prior_dists(hidden, z_posteriors[:, 1:])

        return z_posterior_dists, z_prior_dists, img_prior_dists


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
