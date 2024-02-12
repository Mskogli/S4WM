import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


class OneHotDist(tfd.OneHotCategorical):

    def __init__(self, logits=None, probs=None, dtype=jnp.float32):
        super().__init__(logits, probs, dtype)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=(), seed=None):
        sample = sg(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor
