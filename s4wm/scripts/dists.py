import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from s4wm.nn.dists import MSEDist

tfd = tfp.distributions
tfm = tfp.math

if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    key_1 = jax.random.PRNGKey(1)
    mean = jax.random.normal(key, (2, 100, 270 * 480))
    mean_2 = jax.random.normal(key_1, (2, 100, 270 * 480))

    mean_kl_1 = jax.random.normal(key, (2, 100, 1024))
    mean_kl_2 = jax.random.normal(key_1, (2, 100, 1024))

    dist_log_prob_1 = tfd.Independent(tfd.Normal(mean, 1), 1)
    dist_log_prob_2 = tfd.MultivariateNormalDiag(mean_2, jnp.ones_like(mean_2))

    dist_kl_1 = tfd.MultivariateNormalDiag(mean_kl_1, jnp.ones_like(mean_kl_1))
    dist_kl_2 = tfd.MultivariateNormalDiag(mean_kl_1, jnp.ones_like(mean_kl_1) + 0.5)

    kl_div = jnp.sum(dist_kl_1.kl_divergence(dist_kl_2) / 1024, axis=-1)

    img_dist = MSEDist(mean, 1, agg="mean")
    log_cosh_loss = jnp.sum(
        jnp.sum(tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1),
        axis=-1,
    )
    log_cosh = jnp.mean(
        tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1
    )
    print(jnp.sum(dist_log_prob_1.log_prob(mean_2) / (270 * 480), axis=-1))

    print(kl_div / 256)
