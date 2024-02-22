import jax
import jax.numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .decoder import ImageDecoder
from .encoder import ImageEncoder
from .s4_nn import S4Block
from .dists import OneHotDist, MSEDist, sg

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
    seed: int = 42

    def setup(self) -> None:
        self.num_classes = (
            jnp.sqrt(self.latent_dim) if self.discrete_latent_state else None
        )
        self.rng = jax.random.PRNGKey(self.seed)

        self.encoder = ImageEncoder(latent_dim=2 * self.latent_dim)
        self.decoder = ImageDecoder(latent_dim=self.latent_dim)

        self.PSSM_blocks = S4Block(**self.S4_config, training=self.training)

        self.statistic_heads = {
            "embedding": nn.Dense(
                features=(
                    self.latent_dim
                    if self.discrete_latent_state
                    else 2 * self.latent_dim
                )
            ),
            "hidden": nn.Dense(
                features=(
                    self.latent_dim
                    if self.discrete_latent_state
                    else 2 * self.latent_dim
                )
            ),
        }

        self.input_head = nn.Dense(features=self.latent_dim)

    def get_latent_posteriors_from_images(
        self, image: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        embedding = self.encoder(image)

        posterior_statistics = self.get_statistics(
            x=embedding,
            statistics_head="embedding",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )

        dist_type = "OneHot" if self.discrete_latent_state else "NormalDiag"
        z_posterior_dist = self.get_distribution_from_statistics(
            statistics=posterior_statistics, dist_type=dist_type
        )
        z_posterior = z_posterior_dist.sample(seed=self.rng)

        return z_posterior, z_posterior_dist

    def get_latent_prior_from_hidden(self, hidden: jnp.ndarray) -> jnp.ndarray:
        statistics = self.get_statistics(
            x=hidden,
            statistics_head="hidden",
            discrete=self.discrete_latent_state,
            unimix=0.01,
        )

        dist_type = "OneHot" if self.discrete_latent_state else "NormalDiag"
        z_prior_dist = self.get_distribution_from_statistics(
            statistics=statistics, dist_type=dist_type
        )
        return z_prior_dist

    def get_image_prior_dists(
        self, hidden: jnp.ndarray, z_posterior: jnp.ndarray
    ) -> tfd.Distribution:
        x = self.decoder(jnp.concatenate((hidden, z_posterior), axis=-1))
        img_prior_dists = self.get_distribution_from_statistics(
            statistics=x, dist_type="MSE"
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
        dynamics_loss = jnp.maximum(dynamics_loss, 0.1)

        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))
        representation_loss = jnp.maximum(representation_loss, 0.1)

        kl_loss = self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss
        kl_loss = jnp.sum(kl_loss, axis=-1)

        reconstruction_loss = -img_prior_dist.log_prob(img_posterior.astype(f32))
        reconstruction_loss = jnp.sum(reconstruction_loss, axis=-1)

        return self.beta_rec * reconstruction_loss + self.beta_kl * kl_loss

    def get_distribution_from_statistics(
        self,
        statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        dist_type: str,
    ) -> tfd.Distribution:
        if dist_type == "MSE":
            mean = statistics.reshape(
                statistics.shape[0], statistics.shape[1], -1
            ).astype(f32)
            return MSEDist(mean, 1)
        elif dist_type == "OneHot":
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        elif dist_type == "NormalDiag":
            mean = statistics["mean"].astype(f32)
            std = statistics["std"].astype(f32)
            return tfd.Independent(tfd.Normal(mean, std), 1)
        else:
            raise (NotImplementedError)

    def get_statistics(
        self,
        x: jnp.ndarray,
        statistics_head: str,
        discrete: bool = False,
        unimix: float = 0.01,
    ) -> Dict[str, jnp.ndarray]:

        if discrete:
            logits = self.statistic_heads[statistics_head](x)
            logits = logits.reshape(logits.shape[0], logits.shape[1], 32, 32)

            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}

        x = self.statistic_heads[statistics_head](x)
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
        hidden = self.PSSM_blocks(g)

        # Compute the latent prior distributions from the hidden state
        z_prior_dists = self.get_latent_prior_from_hidden(hidden)

        # Compute the image priors trough the hidden states and the latent posteriors
        img_prior_dists = self.get_image_prior_dists(hidden, z_posteriors[:, 1:])

        return z_posterior_dists, z_prior_dists, img_prior_dists


if __name__ == "__main__":
    # TODO: Add init code

    pass
