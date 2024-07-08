import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from s4wm.nn.dists import MSEDist, OneHotDist

tfd = tfp.distributions
tfm = tfp.math

if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    key_1 = jax.random.PRNGKey(1)
    mean = jax.random.normal(key, (2, 100, 270 * 480))
    mean_2 = jax.random.normal(key_1, (2, 100, 270 * 480))

    mean_kl_1 = jax.random.normal(key, (2, 100, 1024))
    mean_kl_2 = jax.random.normal(key_1, (2, 100, 1024))

    onehot_logits = jax.random.normal(key, (2, 100, 32, 128))
    onehot_logits_2 = jax.random.normal(key_1, (2, 100, 32, 128))

    onehot_dist = tfd.Independent(OneHotDist(onehot_logits), 1)
    onehot_dist_2 = tfd.Independent(OneHotDist(onehot_logits_2), 1)

    print(jnp.sum(onehot_dist.kl_divergence(onehot_dist_2), axis=-1).shape)

    dist_log_prob_1 = tfd.Independent(tfd.Normal(mean, 1), 1)
    dist_log_prob_2 = tfd.MultivariateNormalDiag(mean_2, jnp.ones_like(mean_2))

    dist_kl_1 = tfd.MultivariateNormalDiag(mean_kl_1, 3 * jnp.ones_like(mean_kl_1))
    dist_kl_2 = tfd.MultivariateNormalDiag(
        mean_kl_1 + 0.1, 3 * jnp.ones_like(mean_kl_1)
    )

    kl_div = onehot_dist.kl_divergence(onehot_dist_2)
    print(kl_div.shape)

    img_dist = MSEDist(mean, 1, agg="mean")
    log_cosh_loss = jnp.mean(
        jnp.sum(tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1),
        axis=-1,
    )

    print(
        jnp.sum(
            tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1
        ).shape
    )
    log_cosh = jnp.mean(
        tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1
    )

    print(kl_div)
