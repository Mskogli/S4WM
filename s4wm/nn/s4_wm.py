import jax
import orbax
import jax.numpy as jnp
import orbax.checkpoint

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
    beta_kl: float = 1.0

    discrete_latent_state: bool = False
    training: bool = True
    seed: int = 47

    use_with_torch: bool = False
    rnn_mode: bool = False
    process_in_chunks: bool = True

    S4_vars = {
        "hidden": None,  # x_k-1
        "matrcies": None,  # discrete time state space matrices
    }

    def setup(self) -> None:
        self.num_classes = (
            jnp.sqrt(self.latent_dim) if self.discrete_latent_state else None
        )
        self.rng = jax.random.PRNGKey(self.seed)

        self.encoder = ImageEncoder(
            latent_dim=(
                self.latent_dim if self.discrete_latent_state else 2 * self.latent_dim
            ),
            process_in_chunks=self.process_in_chunks,
            act="elu",
        )
        self.decoder = ImageDecoder(
            latent_dim=self.latent_dim,
            process_in_chunks=self.process_in_chunks,
            act="elu",
        )

        self.PSSM_blocks = S4Block(
            **self.S4_config, rnn_mode=self.rnn_mode, training=self.training
        )

        self.statistic_heads = {
            "embedding": lambda x: x,
            "hidden": nn.Dense(
                features=(
                    self.latent_dim
                    if self.discrete_latent_state
                    else 2 * self.latent_dim
                )
            ),
        }

        self.input_head = nn.Dense(features=self.S4_config["d_model"])

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
        batch_size, seq_length = image.shape[:2]
        z_posterior = (
            z_posterior.reshape(batch_size, seq_length, self.latent_dim)
            if self.discrete_latent_state
            else z_posterior
        )

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
        z_prior = z_prior_dist.sample(seed=self.rng)
        batch_size, seq_length = hidden.shape[:2]
        z_prior = (
            z_prior.reshape(batch_size, seq_length, self.latent_dim)
            if self.discrete_latent_state
            else z_prior
        )
        return z_prior, z_prior_dist

    def reconstruct_depth(
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
        clip: bool = False,
    ) -> jnp.ndarray:

        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf

        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))

        if clip:
            dynamics_loss = jnp.maximum(dynamics_loss, 1.0)
            representation_loss = jnp.maximum(representation_loss, 1.0)

        kl_loss = (
            self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss
        ) / self.latent_dim
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
            return MSEDist(mean, 1, agg="mean")
        elif dist_type == "OneHot":
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        elif dist_type == "NormalDiag":
            mean = statistics["mean"].astype(f32)
            std = statistics["std"].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)
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
            logits = logits.reshape(logits.shape[0], logits.shape[1], 90, 160)

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
        self,
        imgs: jnp.ndarray,
        actions: jnp.ndarray,
        compute_reconstructions: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 tuple

        out = {
            "z_posterior": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }

        # Compute the latent posteriors from the input images
        out["z_posterior"]["sample"], out["z_posterior"]["dist"] = (
            self.get_latent_posteriors_from_images(imgs)
        )

        # Concatenate and mix the latent posteriors and the actions, compute the dynamics embedding by forward passing the stacked PSSM blocks
        g = self.input_head(
            jnp.concatenate(
                (
                    out["z_posterior"]["sample"][:, :-1],
                    actions[:, 1:],
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.PSSM_blocks(g)

        # Compute the latent prior distributions from the hidden state
        out["z_prior"]["sample"], out["z_prior"]["dist"] = (
            self.get_latent_prior_from_hidden(out["hidden"])
        )

        if compute_reconstructions:
            # Reconstruct depth images by decoding the hidden and latent posterior states

            out["depth"]["recon"] = self.reconstruct_depth(
                out["hidden"], out["z_posterior"]["sample"][:, 1:]
            )

            # Predict depth images by decoding the hidden and latent prior states
            # out["depth"]["pred"] = self.reconstruct_depth(
            #     out["hidden"], out["z_prior"]["dist"].mean()
            # )

        return out

    def forward_single_step(
        self, image: jax.Array, action: jax.Array, compute_recon: bool = True
    ) -> None:

        out = {
            "z_posterior": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }

        # Compute the latent posteriors from the input images
        out["z_posterior"]["sample"], out["z_posterior"]["dist"] = (
            self.get_latent_posteriors_from_images(image)
        )

        # Concatenate and mix the latent posteriors and the actions, compute the dynamics embedding by forward passing the stacked PSSM blocks
        g = self.input_head(
            jnp.concatenate(
                (
                    out["z_posterior"]["sample"],
                    action,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.PSSM_blocks(g)
        # Compute the latent prior distributions from the hidden state
        out["z_prior"]["sample"], out["z_prior"]["dist"] = (
            self.get_latent_prior_from_hidden(out["hidden"])
        )

        if compute_recon:
            # Reconstruct depth images by decoding the hidden and latent posterior states

            out["depth"]["recon"] = self.reconstruct_depth(
                out["hidden"], out["z_posterior"]["sample"]
            )

            # Predict depth images by decoding the hidden and latent prior states
            out["depth"]["pred"] = self.reconstruct_depth(
                out["hidden"], out["z_prior"]["sample"]
            )

        return out

    def init_RNN_mode(self, params, init_imgs, init_actions) -> None:
        assert self.rnn_mode
        variables = self.init(jax.random.PRNGKey(0), init_imgs, init_actions)
        vars = {
            "params": params,
            "cache": variables["cache"],
            "prime": variables["prime"],
        }
        _, prime_vars = self.apply(
            vars, init_imgs, init_actions, mutable=["prime", "cache"]
        )
        return vars["cache"], prime_vars["prime"]

    def forward_RNN_mode(
        self,
        imgs,
        actions,
        compute_reconstructions: bool = False,
        single_step: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 Tuple
        assert self.rnn_mode
        preds = {}
        if not single_step:
            preds = self.__call__(
                imgs,
                actions,
                compute_reconstructions,
            )
        else:
            preds = self.forward_single_step(
                imgs,
                actions,
                compute_reconstructions,
            )
        return preds

    def init_CNN_mode(self, init_imgs, init_actions) -> None:
        assert not self.rnn_mode
        variables = self.init(jax.random.PRNGKey(0), init_imgs, init_actions)

    def forward_CNN_mode(self, imgs, actions, compute_reconstructions: bool = False):
        return self._call_(actions, imgs, compute_reconstructions)

    def restore_checkpoint_state(self, ckpt_dir: str) -> dict:
        ckptr = orbax.checkpoint.Checkpointer(
            orbax.checkpoint.PyTreeCheckpointHandler()
        )
        ckpt_state = ckptr.restore(ckpt_dir, item=None)

        return ckpt_state

    # Dreaming utils
    def _build_context(
        self, context_imgs: jnp.ndarray, context_actions: jnp.ndarray
    ) -> None:
        context_posteriors, _ = self.get_latent_posteriors_from_images(context_imgs)
        g = self.input_head(
            jnp.concatenate((context_posteriors, context_actions), axis=-1)
        )
        last_hidden = self.PSSM_blocks(g)[:, -1, :]
        last_prior, _ = self.get_latent_prior_from_hidden(last_hidden).mean()
        return last_prior, last_hidden

    def _open_loop_prediction(
        self, prev_prior: jnp.ndarray, next_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        g = self.input_head(
            jnp.concatenate(
                (
                    prev_prior.reshape((-1, 1, self.latent_dim)),
                    next_action.reshape((-1, 1, self.num_actions)),
                ),
                axis=-1,
            )
        )
        hidden = self.PSSM_blocks(g)
        next_prior, _ = self.get_latent_prior_from_hidden(hidden).mean()

        return next_prior, hidden

    def _decode_predictions(
        self, hiddens: jnp.ndarray, priors: jnp.ndarray
    ) -> jnp.ndarray:
        hiddens = jnp.array(hiddens)
        priors = jnp.array(priors)

        hiddens = jnp.reshape(hiddens, (-1, len(hiddens), self.latent_dim))
        priors = jnp.reshape(priors, (-1, len(priors), self.latent_dim))
        img_post = self.reconstruct_depth(hiddens, priors)
        return img_post.mean()

    def dream(
        self,
        context_imgs: jnp.ndarray,
        context_actions: jnp.ndarray,
        dream_actions: jnp.ndarray,
        dream_length: int = 10,
        viz: bool = False,
    ) -> jnp.ndarray:

        last_prior, last_hidden = self._build_context(context_imgs, context_actions)
        priors, hiddens = [last_prior], [last_hidden]

        for i in range(dream_length):
            pass

        return


if __name__ == "__main__":
    # TODO: Add init code

    pass
