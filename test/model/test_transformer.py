from typing import Mapping

import jax

import haiku as hk
import jax.numpy as jnp
import numpy as np

from src.model import model
from src.model.train import build_forward_fn, lm_loss_fn


def test_simple_rootfind():
    def build_forward_rootfind(max_iter):
        """Create the model's forward pass."""
        def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
            """Forward pass."""
            tokens = data['obs']
            rootfind = model.EquilibriumLayer(max_iter)
            def g(x):
                return x ** 2
            return rootfind(tokens, g)
        return forward_fn

    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward_rootfind(50)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.asarray(1 + np.random.rand(2,5,6))}
    params = forward_fn.init(rng, data)
    jit_fwd = jax.jit(forward_fn.apply)
    result = jit_fwd(params, rng, data)
    np.testing.assert_almost_equal(result, np.ones((2,5,6)),decimal=6)

def test_transform_with_rng_update_fails():
    def build_forward_rootfind(vocab_size, d_model, max_iter):
        """Create the model's forward pass."""

        def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
            """Forward pass."""
            tokens = data['obs']

            # Embed the input tokens and positions.
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
            token_embs = token_embedding_map(tokens)

            def transformer(x):
                # Mock Transformer with dropout
                h = hk.Linear(d_model)(x)
                hk.next_rng_key()
                return h

            output_embedding = transformer(token_embs)

            # Apply rootfind
            rootfind = model.EquilibriumLayer(max_iter)
            hidden = rootfind(transformer, output_embedding)
            return hk.Linear(vocab_size)(hidden)

        return forward_fn

    max_iter, vocab_size, d_model = 30, 10, 5
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward_rootfind(vocab_size, d_model, max_iter)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.asarray(np.random.rand(8, 5), dtype=jnp.int32)}
    params = forward_fn.init(rng, data)
    forward_fn = jax.jit(forward_fn.apply)
    forward_fn(params, rng, data)

def test_transform_deq():
    vocab_size, d_model, num_heads, num_layers, dropout_rate = 50, 20, 4, 1, 0.1
    forward_fn = build_forward_fn(vocab_size,
                                  d_model,
                                  num_heads,
                                  num_layers,
                                  dropout_rate,
                                  30)

    rng = jax.random.PRNGKey(42)
    forward_fn = hk.transform(forward_fn)
    data = {'obs': jnp.asarray(np.random.rand(8, 5), dtype=jnp.int32)}

    params = forward_fn.init(rng, data)
    forward_fn = jax.jit(forward_fn.apply)
    forward_fn(params, rng, data)
