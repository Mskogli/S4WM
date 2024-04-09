import jax
import orbax
import jax.numpy as jnp
import orbax.checkpoint
import torch

from functools import partial
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .decoder import ImageDecoder, Decoder
from .encoder import ImageEncoder, SimpleEncoder, ResNetEncoder, ResNetBlock
from .s4_nn import S4Block
from .dists import OneHotDist, MSEDist, sg, LogCoshDist

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from typing import Dict, Union, Tuple, Sequence, Literal

tfd = tfp.distributions
f32 = jnp.float32

ImageDistribution = Literal["MSE", "LogCosh"]
LatentDistribution = Literal["Gaussian", "Categorical"]
LossReduction = Literal["sum", "mean"]


class S4WorldModel(nn.Module):
    """Structured State Space Sequence (S4) based world model

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    S4_config: DictConfig

    seed: int = 22

    latent_dim: int = 128
    img_dim: int = 32400
    num_classes: int = 32
    num_modes: int = 32

    alpha: float = 0.8
    beta_rec: float = 1.0
    beta_kl: float = 1.0
    kl_lower_bound: float = 1.0

    discrete_latent_state: bool = False
    training: bool = True
    rnn_mode: bool = False
    sample_mean: bool = True
    clip_kl_loss: bool = True

    image_dist_type: ImageDistribution = "MSE"
    latent_dist_type: LatentDistribution = "Gaussian"
    loss_reduction: LossReduction = "mean"

    def setup(self) -> None:
        self.rng_post, self.rng_prior = jax.random.split(
            jax.random.PRNGKey(self.seed), num=2
        )

        self.encoder = SimpleEncoder(
            embedding_dim=512,
            discrete_latent_state=self.discrete_latent_state,
            c_hid=32,
            latent_dim=self.latent_dim,
        )
        self.decoder = Decoder(
            c_out=1, c_hid=64, discrete_latent_state=self.discrete_latent_state
        )

        self.S4_blocks = S4Block(
            **self.S4_config, rnn_mode=self.rnn_mode, training=self.training
        )

        self.statistic_heads = {
            "embedding": nn.Sequential(
                [
                    nn.Dense(features=self.latent_dim),
                    nn.silu,
                    nn.Dense(features=self.latent_dim),
                ]
            ),
            "hidden": nn.Sequential(
                [
                    nn.Dense(features=self.S4_config["d_model"]),
                    nn.silu,
                    nn.Dense(features=self.S4_config["d_model"]),  # 1x 2024
                    nn.silu,
                    nn.Dense(features=self.latent_dim),
                ]
            ),
        }

        self.input_head = nn.Sequential(
            [
                nn.Dense(features=self.S4_config["d_model"]),
                nn.silu,
                nn.Dense(features=self.S4_config["d_model"]),
            ]
        )

    def get_latent_posteriors(
        self, embedding: jnp.ndarray
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        post_stats = self.get_statistics(x=embedding, statistics_head="embedding")
        pass

    def get_latent_posteriors_from_images(
        self, embedding: jnp.ndarray, sample_mean: bool = False
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
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
        if not sample_mean:
            z_posterior = z_posterior_dist.sample(seed=self.rng_post)
        else:
            z_posterior = (
                z_posterior_dist.mode()
                if self.discrete_latent_state
                else z_posterior_dist.mean()
            )

        batch_size, seq_length = embedding.shape[:2]
        z_posterior = (
            z_posterior.reshape(batch_size, seq_length, self.latent_dim)
            if self.discrete_latent_state
            else z_posterior
        )

        return z_posterior, z_posterior_dist

    def get_latent_prior_from_hidden(
        self, hidden: jnp.ndarray, sample_mean: bool = False
    ) -> jnp.ndarray:
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

        if not sample_mean:
            z_prior = z_prior_dist.sample(seed=self.rng_prior)
        else:
            z_prior = (
                z_prior_dist.mode()
                if self.discrete_latent_state
                else z_prior_dist.mean()
            )

        batch_size, seq_length = hidden.shape[:2]
        z_prior = (
            z_prior.reshape(batch_size, seq_length, self.latent_dim)
            if self.discrete_latent_state
            else z_prior
        )
        return z_prior, z_prior_dist

    def reconstruct_depth(
        self, hidden: jnp.ndarray, embed: jnp.ndarray
    ) -> tfd.Distribution:
        x = self.decoder(jnp.concatenate((hidden, embed), axis=-1))
        img_prior_dists = self.get_distribution_from_statistics(
            statistics=x, dist_type="Image"
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
        # in order to focus on learning the posterior rather than the prior

        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))

        if self.clip_kl_loss:
            dynamics_loss = jnp.maximum(dynamics_loss, self.kl_lower_bound)
            representation_loss = jnp.maximum(representation_loss, self.kl_lower_bound)

        kl_loss = self.beta_kl * jnp.sum(
            (self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss),
            axis=-1,
        )
        recon_loss = self.beta_rec * (
            -jnp.sum(img_prior_dist.log_prob(img_posterior.astype(f32)), axis=-1)
        )

        if self.loss_reduction == "mean":
            kl_loss = kl_loss / self.num_classes
            recon_loss = recon_loss / self.img_dim

        total_loss = recon_loss + kl_loss

        return total_loss, (
            recon_loss,
            kl_loss,
        )

    def get_latent_distribution(
        self, statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> tfd.Distribution:
        if self.latent_dist_type == "Categorical":
            return tfd.Independent(OneHotDist(statistics["logits"].astype(f32)), 1)
        elif self.latent_dist_type == "Gaussian":
            mean = statistics["mean"]
            std = statistics["std"]
            return tfd.MultivariateNormalDiag(mean, std)
        else:
            raise NotImplementedError("Latent distribution type not defined")

    def get_image_distribution(self, statistics: jnp.ndarray) -> tfd.Distribution:
        mode = statistics.reshape(statistics.shape[0], statistics.shape[1], -1).astype(
            f32
        )
        if self.image_dist_type == "MSE":
            return MSEDist(mode, 1, agg=self.loss_reduction)
        elif self.image_dist_type == "LogCosh":
            return LogCoshDist(mode, 1, agg=self.loss_reduction)
        else:
            raise NotImplementedError("Image distribution type not defined")

    def get_statistics(
        self,
        x: jnp.ndarray,
        statistics_head: str,
        discrete: bool = False,
        unimix: float = 0.01,
    ) -> Dict[str, jnp.ndarray]:
        if discrete:
            logits = self.statistic_heads[statistics_head](x)
            logits = logits.reshape(
                logits.shape[0], logits.shape[1], self.num_modes, self.num_classes
            )
            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}
        else:
            x = self.statistic_heads[statistics_head](x)
            mean, std = jnp.split(x, 2, -1)
            std = nn.softplus(std) + 0.1

            return {"mean": mean, "std": std}

    def encode_and_step(
        self, image: jnp.ndarray, action: jnp.ndarray, latent: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        embedding = self.encoder(image)
        z, _ = self.get_latent_posteriors_from_images(embedding, False)
        h = self.S4_blocks(self.input_head(jnp.concatenate((latent, action), axis=-1)))
        return z, h

    def __call__(
        self,
        imgs: jnp.ndarray,
        actions: jnp.ndarray,
        compute_reconstructions: bool = False,
        sample_mean: bool = False,
        train: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 tuple
        shapes = imgs.shape
        multi_step = shapes[1] > 1

        out = {
            "z_posterior": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }
        embeddings = self.encoder(imgs)
        out["z_posterior"]["sample"], out["z_posterior"]["dist"] = (
            self.get_latent_posteriors_from_images(embeddings, sample_mean)
        )

        g = self.input_head(
            jnp.concatenate(
                (
                    (
                        out["z_posterior"]["sample"][:, :-1]
                        if multi_step
                        else out["z_posterior"]["sample"]
                    ),
                    actions,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.S4_blocks(g)
        out["z_prior"]["sample"], out["z_prior"]["dist"] = (
            self.get_latent_prior_from_hidden(out["hidden"], sample_mean)
        )

        out["depth"]["recon"] = self.reconstruct_depth(
            out["hidden"],
            (
                out["z_posterior"]["sample"][:, 1:]
                if multi_step
                else out["z_posterior"]["sample"]
            ),
        )

        if compute_reconstructions:
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
    ) -> Tuple[tfd.Distribution, ...]:  # 3 Tuple
        assert self.rnn_mode
        return self.__call__(
            imgs,
            actions,
            compute_reconstructions,
        )

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
        posterior, _ = self.get_latent_posteriors_from_images(
            context_imgs, sample_mean=False
        )
        g = self.input_head(jnp.concatenate((posterior, context_actions), axis=-1))
        hidden = jnp.expand_dims(self.S4_blocks(g)[:, -1, :], axis=1)
        prior, _ = self.get_latent_prior_from_hidden(hidden, sample_mean=False)
        return prior, hidden

    def _open_loop_prediction(
        self, predicted_posterior: jnp.ndarray, next_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        g = self.input_head(
            jnp.concatenate(
                (
                    predicted_posterior.reshape((-1, 1, self.latent_dim)),
                    next_action.reshape((-1, 1, self.num_actions)),
                ),
                axis=-1,
            )
        )
        hidden = self.S4_blocks(g)
        prior, _ = self.get_latent_prior_from_hidden(hidden, sample_mean=True)
        return prior, hidden

    def _decode_predictions(
        self, hidden: jnp.ndarray, prior: jnp.ndarray
    ) -> jnp.ndarray:
        img_post = self.reconstruct_depth(hidden, prior)
        return img_post.mean()

    def dream(
        self,
        context_imgs: jnp.ndarray,
        context_actions: jnp.ndarray,
        dream_actions: jnp.ndarray,
        dream_horizon: int = 10,
    ) -> Tuple[jnp.ndarray, ...]:  # 3 Tuple

        prior, hidden = self._build_context(context_imgs, context_actions)
        priors = [prior]
        hiddens = [hidden]
        pred_depths = []

        for i in range(dream_horizon):
            prior, hidden = self._open_loop_prediction(
                predicted_posterior=prior, next_action=dream_actions[:, i]
            )
            priors.append(prior)
            hiddens.append(hidden)

        for x in zip(hiddens, priors):
            pred_depth = self._decode_predictions(
                jnp.expand_dims(x[0], axis=1), jnp.expand_dims(x[1], axis=1)
            )
            pred_depths.append(pred_depth)

        return pred_depths, priors


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model, params, cache, prime, image: jax.Array, action: jax.Array, latent: jax.Array
) -> jax.Array:
    return model.apply(
        {
            "params": params,
            "cache": cache,
            "prime": prime,
        },
        image,
        action,
        latent,
        mutable=["cache"],
        method="encode_and_step",
    )


class S4WMTorchWrapper:
    def __init__(
        self,
        batch_dim: int,
        ckpt_path: str,
        d_latent: int = 128,
        d_pssm_blocks: int = 1024,
        d_ssm: int = 64,
        num_pssm_blocks: int = 4,
        discrete_latent_state: bool = True,
        l_max: int = 99,
    ) -> None:
        self.d_pssm_block = d_pssm_blocks
        self.d_ssm = d_ssm
        self.num_pssm_blocks = num_pssm_blocks

        self.model = S4WorldModel(
            S4_config=DictConfig(
                {
                    "d_model": d_pssm_blocks,
                    "layer": {"l_max": l_max, "N": d_ssm},
                    "n_blocks": num_pssm_blocks,
                }
            ),
            training=False,
            rnn_mode=True,
            discrete_latent_state=discrete_latent_state,
            **DictConfig(
                {
                    "latent_dim": d_latent,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        init_depth = jnp.zeros((batch_dim, 1, 135, 240, 1))
        init_actions = jnp.zeros((batch_dim, 1, 4))
        init_latent = jnp.zeros((batch_dim, 1, 1024))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.rnn_cache, self.prime, self.params = (
            self.rnn_cache,
            self.prime,
            self.params,
        )

        # Force compilation
        _ = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_depth,
            init_actions,
            init_latent,
        )
        return

    def forward(
        self, depth_imgs: torch.tensor, actions: torch.tensor, latent: jax.Array
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        jax_imgs, jax_actions = from_torch_to_jax(depth_imgs), from_torch_to_jax(
            actions
        )
        out, variables = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            jax_imgs,
            jax_actions,
            latent,
        )
        self.rnn_cache = variables["cache"]

        return (
            from_jax_to_torch(out[0]),
            from_jax_to_torch(out[1]),
        )

    def reset_cache(self, batch_idx: Sequence) -> None:
        for i in range(self.num_pssm_blocks):
            for j in range(2):
                self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ] = (
                    self.rnn_cache["PSSM_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                        "cache_x_k"
                    ]
                    .at[jnp.array([batch_idx])]
                    .set(jnp.ones((self.d_ssm, self.d_pssm_block), dtype=jnp.complex64))
                )
        return


if __name__ == "__main__":
    # TODO: Add init code

    pass
