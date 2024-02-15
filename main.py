import jax
import jax.numpy as jnp

from models.s4.s4_wm import S4WorldModel


if __name__ == "__main__":
    batch_size, seq_length = 2, 5

    # Setup
    key = jax.random.PRNGKey(0)
    dummy_input_img = jax.random.normal(key, (batch_size, seq_length, 1, 270, 480))
    dummy_input_actions = jax.random.normal(key, (batch_size, seq_length, 4))

    world_model = S4WorldModel(discrete_latent_state=False, latent_dim=128)
    params = world_model.init(
        jax.random.PRNGKey(1), dummy_input_img, dummy_input_actions
    )
    params = params["params"]

    print("Model initialized")

    z_prior, z_posterior, img_prior = world_model.apply(
        {"params": params}, dummy_input_img, dummy_input_actions
    )

    img_posts = jnp.squeeze(dummy_input_img[:, 1:], axis=-3)
    img_posts = img_posts.reshape((batch_size, seq_length - 1, -1))

    loss = world_model.compute_loss(
        img_prior_dist=img_prior[-1],
        img_posterior=img_posts,
        z_posterior_dist=z_posterior[-1],
        z_prior_dist=z_prior[-1],
    )
    print("Loss: ", loss.shape)
    print(loss)
