from functools import partial
from typing import Mapping, Optional

import jax

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

from src.model.train import build_forward_fn, lm_loss_fn
from src.modules.rootfind import rootfind


def test_simple_rootfind():
    def build_forward_rootfind(max_iter):
        """Create the model's forward pass."""
        def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
            """Forward pass."""
            tokens = data['obs']
            def g(x):
                return x ** 2
            return rootfind(tokens, g, max_iter)
        return forward_fn

    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward_rootfind(50)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.asarray(np.random.rand(2,5,6))}
    params = forward_fn.init(rng, data)
    jit_fwd = jax.jit(forward_fn.apply)
    result = jit_fwd(params, rng, data)
    np.testing.assert_almost_equal(result, np.ones((2,5,6)),decimal=6)

def test_transform_with_rng_update_fails():
    class MockTransformer(hk.Module):
        def __init__(self, output_shape: int, name: Optional[str] = None):
            super().__init__(name=name)
            self._output_shape = output_shape

        def __call__(self, x):
            layer = hk.Linear(self._output_shape)
            hk.next_rng_key()
            return layer(x)

    def build_forward(vocab_size, d_model, max_iter):
        def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
            tokens = data['obs']

            # Mock Transformer with dropout
            transformer = MockTransformer(d_model)

            def func(x):
                return transformer(x) - x

            output_embedding = transformer(tokens)
            # Apply rootfind
            hidden = rootfind(output_embedding, func, max_iter)
            return hk.Linear(vocab_size)(hidden)

        return forward_fn

    max_iter, vocab_size, d_model = 30, 10, 5
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(vocab_size, d_model, max_iter)
    forward_fn = hk.transform(forward_fn)

    data = {'obs': jnp.asarray(np.random.rand(8, 5, 5), dtype=jnp.float64)}
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


def test_loss_grad():
    vocab_size, d_model, num_heads, num_layers, dropout_rate = 50, 20, 4, 1, 0.1
    forward_fn = build_forward_fn(vocab_size,
                                  d_model,
                                  num_heads,
                                  num_layers,
                                  dropout_rate,
                                  30)

    rng = jax.random.PRNGKey(42)
    data = {'obs': jnp.asarray(np.random.rand(8, 5), dtype=jnp.int32),
            'target': jnp.ones((8, 5))}


    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, data)
    loss_fn = partial(lm_loss_fn, forward_fn.apply, vocab_size)
    jit_loss = jax.jit(loss_fn)

    loss, grad = value_and_grad(jit_loss)(params, rng, data)