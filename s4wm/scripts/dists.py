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

    mean_kl_1 = jax.random.normal(key, (2, 100, 512))
    mean_kl_2 = jax.random.normal(key_1, (2, 100, 512))

    dist_log_prob_1 = tfd.MultivariateNormalDiag(mean, jnp.ones_like(mean))
    dist_log_prob_2 = tfd.MultivariateNormalDiag(mean_2, jnp.ones_like(mean_2))

    dist_kl_1 = tfd.MultivariateNormalDiag(mean_kl_1, jnp.ones_like(mean_kl_1))
    dist_kl_2 = tfd.MultivariateNormalDiag(mean_kl_2, jnp.ones_like(mean_kl_2))

    kl_div = jnp.sum(dist_kl_1.kl_divergence(dist_kl_2), axis=-1)

    log_cosh_loss = jnp.sum(
        jnp.sum(tfm.log_cosh(dist_log_prob_1.mean() - dist_log_prob_2.mean()), axis=-1),
        axis=-1,
    )

    print(kl_div)
    print(log_cosh_loss * (512 / (270 * 480)))
    print(log_cosh_loss / kl_div)
