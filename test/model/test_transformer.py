import functools

import jax

import haiku as hk
import jax.numpy as jnp

from DEQModel.model.train import build_forward_fn, lm_loss_fn



def test_model_forward_pass():
    vocab_size, d_model, num_heads, num_layers, dropout_rate = 512, 20, 4, 1, 0.5
    max_iter, eps = 10, 1e-6
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward_fn(vocab_size,
                                  d_model,
                                  num_heads,
                                  num_layers,
                                  dropout_rate)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.ones((2,5), dtype=jnp.int32), 'target': jnp.ones((2,5))}
    params = forward_fn.init(rng, data)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size, is_training=True)
    jit_loss = jax.jit(loss_fn)

    loss = jit_loss(params, rng, data)
    print(loss)