import jax
import orbax
import jax.numpy as jnp
import orbax.checkpoint
import torch

from functools import partial
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from omegaconf import DictConfig

from .decoder import ResNetDecoder
from .encoder import ResNetEncoder
from .s4_nn import S4Blocks
from .dists import OneHotDist, MSEDist, sg, LogCoshDist

from s4wm.utils.dlpack import from_jax_to_torch, from_torch_to_jax
from typing import Dict, Union, Tuple, Sequence, Literal, Any

tfd = tfp.distributions
f32 = jnp.float32

ImageDistribution = Literal["MSE", "LogCosh"]
LatentDistribution = Literal["Gaussian", "Categorical"]
LossReduction = Literal["sum", "mean"]

from jax.tree_util import PyTreeDef

PyTree = Union[Any, tuple, list, dict, PyTreeDef]
PRNGKey = jnp.ndarray


class S4WM(nn.Module):
    S4_config: DictConfig

    latent_dim: int = 128
    num_classes: int = 32
    num_modes: int = 32

    alpha: float = 0.8
    beta_rec: float = 1.0
    beta_kl: float = 1.0
    kl_lower_bound: float = 1.0

    training: bool = True

    rnn_mode: bool = False
    sample_mean: bool = False
    clip_kl_loss: bool = True

    image_dist_type: ImageDistribution = "MSE"
    latent_dist_type: LatentDistribution = "Categorical"
    loss_reduction: LossReduction = "mean"

    def setup(self) -> None:
        self.rng_post, self.rng_prior = jax.random.split(jax.random.PRNGKey(0), num=2)
        self.discrete_latent_state = self.latent_dist_type == "Categorical"

        self.encoder = ResNetEncoder(act_fn=nn.silu)
        self.decoder = ResNetDecoder(act_fn=nn.silu)

        self.S4_blocks = S4Blocks(
            **self.S4_config, rnn_mode=self.rnn_mode, training=self.training
        )

        self.statistic_heads = {
            "embedding": lambda x: x,
            "hidden": nn.Sequential(
                [
                    nn.Dense(features=self.S4_config["d_model"]),
                    nn.silu,
                    nn.Dense(features=512),  # 1x 2024
                    nn.silu,
                    nn.Dense(features=self.latent_dim),
                ]
            ),
        }

        self.input_head = nn.Sequential(
            [
                nn.Dense(features=256),
                nn.silu,
                nn.Dense(features=self.S4_config["d_model"]),
            ]
        )

    def compute_latent(
        self, statistics: jnp.ndarray, rng_seed: PRNGKey
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        dists = self.get_latent_distribution(statistics)

        if not self.sample_mean:
            sample = dists.sample(seed=rng_seed)
        else:
            sample = dists.mode() if self.discrete_latent_state else dists.mean()

        if self.discrete_latent_state:
            sample = jax.lax.collapse(
                sample,
                start_dimension=2,
                stop_dimension=4,  # Flatten sample from a categorical with shape (batch, seq_l, modes, num_classes)
            )

        return sample, dists

    def compute_posteriors(
        self, embedding: jnp.ndarray, rng_seed: PRNGKey
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        post_stats = self.get_statistics(embedding, statistics_head="embedding")
        return self.compute_latent(post_stats, rng_seed)

    def compute_priors(
        self, hidden: jnp.ndarray, rng_seed: PRNGKey
    ) -> Tuple[jnp.ndarray, tfd.Distribution]:
        prior_stats = self.get_statistics(hidden, statistics_head="hidden")
        return self.compute_latent(prior_stats, rng_seed)

    def reconstruct_depth(
        self, hidden: jnp.ndarray, latent_sample: jnp.ndarray
    ) -> tfd.Distribution:
        x = self.decoder(jnp.concatenate((hidden, latent_sample), axis=-1))
        return self.get_image_distribution(x)

    def compute_loss(
        self,
        img_prior_dist: tfd.Distribution,
        img_posterior: jnp.ndarray,
        z_posterior_dist: tfd.Distribution,
        z_prior_dist: tfd.Distribution,
    ) -> jnp.ndarray:
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
        unimix: float = 0.01,
    ) -> Dict[str, jnp.ndarray]:
        if self.discrete_latent_state:
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

    def __call__(
        self,
        depth_imgs: jnp.ndarray,
        actions: jnp.ndarray,
        rng_seed: PRNGKey,
        reconstruct_priors: bool = False,
    ) -> Dict[str, Tuple[tfd.Distribution, jnp.ndarray]]:
        out = {
            "z_post": {"dist": None, "sample": None},
            "z_prior": {"dist": None, "sample": None},
            "depth": {"recon": None, "pred": None},
            "hidden": None,
        }

        post_key, prior_key = jax.random.split(rng_seed)
        multi_step = depth_imgs.shape[1] > 1

        embeddings = self.encoder(depth_imgs)
        out["z_post"]["sample"], out["z_post"]["dist"] = self.compute_posteriors(
            embeddings, post_key
        )

        g = self.input_head(
            jnp.concatenate(
                (
                    (
                        out["z_post"]["sample"][:, :-1]
                        if multi_step
                        else out["z_post"]["sample"]
                    ),
                    actions,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.S4_blocks(g)

        out["z_prior"]["sample"], out["z_prior"]["dist"] = self.compute_priors(
            out["hidden"], prior_key
        )

        out["depth"]["recon"] = self.reconstruct_depth(
            out["hidden"],
            (out["z_post"]["sample"][:, 1:] if multi_step else out["z_post"]["sample"]),
        )

        if reconstruct_priors:
            out["depth"]["pred"] = self.reconstruct_depth(
                out["hidden"], out["z_prior"]["sample"]
            )

        return out

    def forward_open_loop(
        self, predicted_posterior: jnp.ndarray, action: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        out = {
            "z_post_pred": {"dist": None, "sample": None},
            "depth_pred": None,
            "hidden": None,
        }
        g = self.input_head(
            jnp.concatenate(
                (
                    predicted_posterior,
                    action,
                ),
                axis=-1,
            )
        )
        out["hidden"] = self.S4_blocks(g)
        out["z_post_pred"]["sample"], out["z_post_pred"]["dist"] = self.compute_priors(
            out["hidden"], rng_seed=key
        )
        out["depth_pred"] = self.reconstruct_depth(
            out["hidden"], out["z_post_pred"]["sample"]
        )
        return out

    def encode_and_step(
        self, image: jnp.ndarray, action: jnp.ndarray, latent: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, ...]:  # 2 Tuple
        z, _ = self.compute_posteriors(self.encoder(image), key)
        h = self.S4_blocks(self.input_head(jnp.concatenate((latent, action), axis=-1)))
        return z, h

    def encode_and_step_open_loop(
        self, action: jnp.ndarray, latent: jnp.ndarray, key
    ) -> Tuple[jnp.ndarray, ...]:  # 2 tuple
        h = self.S4_blocks(self.input_head(jnp.concatenate((latent, action), axis=-1)))
        z, _ = self.compute_priors(h, key)
        return z, h

    def init_RNN_mode(
        self, params: PyTree, init_imgs: jnp.ndarray, init_actions: jnp.ndarray
    ) -> Tuple[PyTree, ...]:  # 2 tuple
        assert self.rnn_mode

        variables = self.init(
            jax.random.PRNGKey(0), init_imgs, init_actions, jax.random.PRNGKey(1)
        )
        vars = {
            "params": params,
            "cache": variables["cache"],
            "prime": variables["prime"],
        }

        _, prime_vars = self.apply(
            vars,
            init_imgs,
            init_actions,
            jax.random.PRNGKey(2),
            mutable=["prime", "cache"],
        )

        return vars["cache"], prime_vars["prime"]

    def restore_checkpoint_state(self, ckpt_dir: str) -> PyTree:
        ckptr = orbax.checkpoint.Checkpointer(
            orbax.checkpoint.PyTreeCheckpointHandler()
        )
        ckpt_state = ckptr.restore(ckpt_dir, item=None)

        return ckpt_state


# ---- Torch utilities and wrappers ----


@partial(jax.jit, static_argnums=(0))
def _jitted_forward(
    model: S4WM,
    params: PyTree,
    cache: PyTree,
    prime: PyTree,
    image: jnp.ndarray,
    action: jnp.ndarray,
    latent: jnp.ndarray,
    rng_seed: PRNGKey,
) -> jnp.ndarray:
    return model.apply(
        {
            "params": params,
            "cache": cache,  # The internal hidden state of the state space models
            "prime": prime,  # The DPLR system matrices, Lambda, P, Q, ...
        },
        image,
        action,
        latent,
        rng_seed,
        mutable=["cache"],
        method="encode_and_step",
    )


@partial(jax.jit, static_argnums=(0))
def _jitted_open_loop_predict(
    model: S4WM,
    params: PyTree,
    cache: PyTree,
    prime: PyTree,
    action: jnp.ndarray,
    latent: jnp.ndarray,
    rng_seed: PRNGKey,
) -> jax.Array:
    return model.apply(
        {"params": params, "cache": cache, "prime": prime},
        action,
        latent,
        rng_seed,
        mutable=["cache"],
        method="encode_and_step_open_loop",
    )


class S4WMTorchWrapper:
    def __init__(
        self,
        batch_dim: int,
        ckpt_path: str,
        latent_dim: int = 128,
        S4_block_dim: int = 1024,
        ssm_dim: int = 64,
        num_S4_blocks: int = 4,
        l_max: int = 99,
        sample_mean: bool = True,
    ) -> None:
        self.d_pssm_block = S4_block_dim
        self.d_ssm = ssm_dim
        self.num_pssm_blocks = num_S4_blocks

        self.model = S4WM(
            S4_config=DictConfig(
                {
                    "d_model": S4_block_dim,
                    "layer": {"l_max": l_max, "N": ssm_dim},
                    "n_blocks": num_S4_blocks,
                }
            ),
            training=False,
            rnn_mode=True,
            sample_mean=sample_mean,
            latent_dist_type="Gaussian",
            **DictConfig(
                {
                    "latent_dim": latent_dim,
                }
            ),
        )

        self.params = self.model.restore_checkpoint_state(ckpt_path)["params"]

        init_depth = jnp.zeros((batch_dim, 1, 135, 240, 1))
        init_actions = jnp.zeros((batch_dim, 1, 4))
        init_latent = jnp.zeros((batch_dim, 1, 128))

        self.rnn_cache, self.prime = self.model.init_RNN_mode(
            self.params,
            init_depth,
            init_actions,
        )

        self.key = jax.random.PRNGKey(0)

        # Dummy calls to the jitted functions in order to invoke jit compilation

        _ = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_depth,
            init_actions,
            init_latent,
            self.key,
        )

        _ = _jitted_open_loop_predict(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            init_actions,
            init_latent,
            self.key,
        )

        return

    def forward(
        self, depth_imgs: torch.tensor, actions: torch.tensor, latents: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        self.key, subkey = jax.random.split(self.key)

        jax_imgs, jax_actions, jax_latent = (
            from_torch_to_jax(depth_imgs),
            from_torch_to_jax(actions),
            from_torch_to_jax(latents),
        )

        out, variables = _jitted_forward(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            jax_imgs,
            jax_actions,
            jax_latent,
            subkey,
        )

        self.rnn_cache = variables["cache"]

        return (
            from_jax_to_torch(out[0]),
            from_jax_to_torch(out[1]),
        )

    def open_loop_predict(
        self, actions: torch.tensor, latents: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 tuple

        self.key, subkey = jax.random.split(self.key)

        jax_action, jax_latent = (
            from_torch_to_jax(actions),
            from_torch_to_jax(latents),
        )

        out, variables = _jitted_open_loop_predict(
            self.model,
            self.params,
            self.rnn_cache,
            self.prime,
            jax_action,
            jax_latent,
            subkey,
        )

        self.rnn_cache = variables["cache"]

        return (
            from_jax_to_torch(out[0]),
            from_jax_to_torch(out[1]),
        )

    def reset_cache(self, batch_idx: Sequence) -> None:
        batch_idx = from_torch_to_jax(batch_idx)
        for i in range(self.num_pssm_blocks):
            for j in range(2):
                self.rnn_cache["S4_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                    "cache_x_k"
                ] = (
                    self.rnn_cache["S4_blocks"][f"blocks_{i}"][f"layers_{j}"]["seq"][
                        "cache_x_k"
                    ]
                    .at[jnp.array([batch_idx])]
                    .set(
                        jnp.zeros((self.d_ssm, self.d_pssm_block), dtype=jnp.complex64)
                    )
                )
        return
