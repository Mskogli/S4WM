import jax
import orbax
import jax.numpy as jnp
import orbax.checkpoint
import torch

from functools import partial
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .decoder import ImageDecoder
from .encoder import ImageEncoder
from .s4_nn import S4Block
from .dists import OneHotDist, MSEDist, sg, LogCoshDist

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from typing import Dict, Union, Tuple, Sequence

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
    img_dim: int = 32400
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

    image_dist_type = "MSE"

    def setup(self) -> None:
        self.num_classes = (
            jnp.sqrt(self.latent_dim) if self.discrete_latent_state else None
        )
        self.rng_post, self.rng_prior = jax.random.split(
            jax.random.PRNGKey(self.seed), num=2
        )

        self.encoder = ImageEncoder(
            latent_dim=(
                self.latent_dim if self.discrete_latent_state else 2 * self.latent_dim
            ),
            process_in_chunks=self.process_in_chunks,
            act="silu",
        )
        self.decoder = ImageDecoder(
            latent_dim=self.latent_dim,
            process_in_chunks=self.process_in_chunks,
            act="silu",
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
        self, image: jnp.ndarray, sample_mean: bool = False
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
        if not sample_mean:
            z_posterior = z_posterior_dist.sample(seed=self.rng_post)
        else:
            z_posterior = z_posterior_dist.mode()

        batch_size, seq_length = image.shape[:2]
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
            z_prior = z_prior_dist.sample(seed=self.rng_post)
        else:
            z_prior = z_prior_dist.mode()

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
            statistics=x, dist_type="Image"
        )
        return img_prior_dists

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.ndarray,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distribution,
        reduction: str = "mean",  # Mean or sum
        clip: bool = True,
        free: float = 0.5,
    ) -> jnp.ndarray:

        # Compute the KL loss with KL balancing https://arxiv.org/pdf/2010.02193.pdf

        dynamics_loss = sg(z_posterior_dist).kl_divergence(z_prior_dist)
        representation_loss = z_posterior_dist.kl_divergence(sg(z_prior_dist))

        if clip:
            dynamics_loss = jnp.maximum(dynamics_loss, free)
            representation_loss = jnp.maximum(representation_loss, free)

        kl_loss = self.alpha * dynamics_loss + (1 - self.alpha) * representation_loss
        reconstruction_loss = -img_prior_dist.log_prob(img_posterior.astype(f32))

        if reduction == "mean":
            kl_loss = self.beta_kl * jnp.sum(kl_loss / self.latent_dim, axis=-1)
            reconstruction_loss = self.beta_rec * jnp.sum(
                reconstruction_loss / self.img_dim, axis=-1
            )
        else:
            kl_loss = self.beta_kl * jnp.sum(kl_loss, axis=-1)
            reconstruction_loss = self.beta_rec * jnp.sum(reconstruction_loss, axis=-1)

        total_loss = reconstruction_loss + kl_loss
        return total_loss, (
            reconstruction_loss,
            kl_loss,
        )

    def get_distribution_from_statistics(
        self,
        statistics: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        dist_type: str,
    ) -> tfd.Distribution:
        if dist_type == "Image":
            mean = statistics.reshape(
                statistics.shape[0], statistics.shape[1], -1
            ).astype(f32)
            img_dist = (
                LogCoshDist(mean, 1, agg="mean")
                if self.image_dist_type == "MSE"
                else tfd.Independent(tfd.Normal(mean, 1), 1)
            )
            return img_dist
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
            # if statistics_head == "hidden":
            # logits = nn.relu(logits)
            logits = logits.reshape(logits.shape[0], logits.shape[1], 128, 32)

            if unimix:
                probs = jax.nn.softmax(logits, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix) * probs + unimix * uniform
                logits = jnp.log(probs)
            return {"logits": logits}

        x = self.statistic_heads[statistics_head](x)
        # x = nn.relu(x)
        mean, std = jnp.split(x, 2, -1)
        std = 2 * jax.nn.sigmoid(std / 2) + 0.1

        return {"mean": mean, "std": std}

    def encode_and_step(
        self, image: jnp.ndarray, action: jnp.ndarray, latent: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        z = self.get_latent_posterior_from_images(image, False)
        h = self.PSSM_blocks(self.input_head(jnp.concatenate(latent, action)))
        return z, h

    def __call__(
        self,
        imgs: jnp.ndarray,
        actions: jnp.ndarray,
        compute_reconstructions: bool = False,
        sample_mean: bool = False,
    ) -> Tuple[tfd.Distribution, ...]:  # 3 tuple

        out = {
            "z_posterior": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }

        # Compute the latent posteriors from the input images
        out["z_posterior"]["sample"], out["z_posterior"]["dist"] = (
            self.get_latent_posteriors_from_images(imgs, sample_mean)
        )

        # Concatenate and mix the latent posteriors and the actions, compute the dynamics embedding by forward passing the stacked PSSM blocks
        g = self.input_head(
            jnp.concatenate(
                (
                    out["z_posterior"]["sample"],
                    actions,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.PSSM_blocks(g)

        # Compute the latent prior distributions from the hidden state
        out["z_prior"]["sample"], out["z_prior"]["dist"] = (
            self.get_latent_prior_from_hidden(out["hidden"], sample_mean)
        )

        if compute_reconstructions:
            # Predict depth images by decoding the hidden and latent prior states
            out["depth"]["pred"] = self.reconstruct_depth(
                out["hidden"], out["z_prior"]["sample"]
            )

            out["depth"]["recon"] = self.reconstruct_depth(
                out["hidden"], out["z_posterior"]["sample"][:, 1:]
            )

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
            self.get_latent_posteriors_from_images(image, sample_mean=True)
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
            self.get_latent_prior_from_hidden(out["hidden"], sample_mean=True)
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
        print(init_imgs.shape, init_actions.shape)
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
        if not single_step:
            return self.__call__(
                imgs,
                actions,
                compute_reconstructions,
            )
        else:
            return self.forward_single_step(
                imgs,
                actions,
                compute_reconstructions,
            )

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
        posterior, _ = self.get_latent_posteriors_from_images(
            context_imgs, sample_mean=False
        )
        g = self.input_head(jnp.concatenate((posterior, context_actions), axis=-1))
        hidden = jnp.expand_dims(self.PSSM_blocks(g)[:, -1, :], axis=1)
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
        hidden = self.PSSM_blocks(g)
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
            "params": sg(params),
            "cache": sg(cache),
            "prime": sg(prime),
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
        d_pssm_blocks: int = 512,
        d_ssm: int = 128,
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
            process_in_chunks=False,
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
        init_actions = jnp.zeros((batch_dim, 1, 20))
        init_latent = jnp.zeros((batch_dim, 1, 4096))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.rnn_cache, self.prime, self.params = (
            sg(self.rnn_cache),
            sg(self.prime),
            sg(self.params),
        )

        # Force compilation
        _ = _jitted_forward(
            self.model,
            sg(self.params),
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
        (z, h), variables = _jitted_forward(
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
            from_jax_to_torch(z),
            from_jax_to_torch(h),
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
