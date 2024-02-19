import jax
from tensorflow_probability.substrates import jax as tfp
from models.s4.utils import OneHotDist

tfd = tfp.distributions


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (8, 10, 270 * 480))
    dists = tfd.Independent(tfd.Normal(logits, 1), 1)
    print(dists[:, :-1])
