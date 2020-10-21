from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax import value_and_grad

from deq_jax.src.modules.rootfind import rootfind


def test_rootfind_1D():
    @jax.jit
    def func(x):
        return x ** 2

    jit_rootfind = jax.jit(rootfind, static_argnums=[0, 1])

    def toy_model(x):
        return jnp.sum(rootfind(func, 30, x))

    root = jit_rootfind(func, 30, jnp.ones((1,2,3)))
    value, grad = value_and_grad(toy_model)(jnp.ones((1, 2, 3)))

    assert (6.0 == value or 0. == value)


def test_rootfind_multi_variable():
    def layer(w, x):
        return w * x

    def toy_model(data, params):
        fun = lambda z: layer(params, z) - z
        return jnp.sum(rootfind(fun, 30, data))

    params = -2 * jnp.ones((1, 2, 3))
    data = jnp.ones((1, 2, 3))
    value, grad = value_and_grad(toy_model)(data, params)

    assert (-6.0 == value or 0. == value)
    assert (grad == 1 / 3 * jnp.ones((1, 2, 3))).all()

def test_rootfind_in_haiku_module():
    # currently failing as g is not haiku transformed
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            # mock transformer
            linear_1 = hk.Linear(output_size, name='l1')
            transformed_linear = hk.without_apply_rng(
                hk.transform(linear_1)
            )
            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            fun = lambda z: transformed_linear.apply(inner_params, z) - z
            y = rootfind(fun, max_iter, x)
            return hk.Linear(output_size)(y)

        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    @jax.jit
    def loss_fn(params, rng, x):
        h = forward_fn.apply(params, rng, x)
        return jnp.sum(h)

    value = loss_fn(params, rng, jnp.ones((1, 2, 3)))
    value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))
