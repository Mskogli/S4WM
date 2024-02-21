import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from models.s4.utils import OneHotDist

tfd = tfp.distributions


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key_1 = jax.random.PRNGKey(1)
    mean = jax.random.normal(key, (4, 45, 270*480))
    mean_2 = jax.random.normal(key_1, (4, 45, 270*480))
    
    dist_1 = tfd.MultivariateNormalDiag(mean, jnp.ones_like(mean))
    
    log_probs = jnp.sum(-dist_1.log_prob(jnp.zeros_like(mean)), axis=-1) / (507)
    
    print(log_probs)
