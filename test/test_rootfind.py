from typing import Optional

import jax
from jax import value_and_grad

from deq_jax.src.modules.rootfind import rootfind, h
import jax.numpy as jnp
import haiku as hk


def test_rootfind_1D():
    @jax.jit
    def func(x):
        return x ** 2

    jit_rootfind = jax.jit(rootfind, static_argnums=[0, 2])

    def toy_model(x):
        y = func(x)
        return jnp.sum(rootfind(func, y, 30))

    root = jit_rootfind(func, jnp.ones((1, 2, 3)), 30)
    value, grad = value_and_grad(toy_model)(jnp.ones((1, 2, 3)))

    assert (6.0 == value or 0. == value)


def test_rootfind_multi_variable():
    def layer(x, w):
        return x * w

    def toy_model(params, data):
        y = layer(data, params)
        return jnp.sum(rootfind(layer, y, 30, params))

    params = -2 * jnp.ones((1, 2, 3))
    data = jnp.ones((1, 2, 3))
    value, grad = value_and_grad(toy_model)(params, data)

    assert (-6.0 == value or 0. == value)
    assert (grad == 1 / 3 * jnp.ones((1, 2, 3))).all()


def test_rootfind_in_haiku_fn():
    # currently failing as g is not haiku transformed
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear = hk.Linear(output_size, name='l1')
            h = linear(x)
            y = rootfind(linear, h, max_iter)
            return hk.Linear(output_size, name='l2')(y)

        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    def loss_fn(params, rng, x):
        h = forward_fn.apply(params, rng, x)
        return jnp.sum(h)

    value = loss_fn(params, rng, jnp.ones((1, 2, 3)))
    value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))


def test_rootfind_in_haiku_module():
    # currently failing as g is not haiku transformed
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = hk.Linear(output_size, name='l1')
            linear_2 = hk.Linear(output_size, name='l2')
            h = linear_2(x)
            y = rootfind(linear_1, h, max_iter)
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


def test_passing_bk_haiku_fn():
    Rootfind = hk.to_module(h)

    def build_forward(output_size):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            # mock embeddings
            linear_1 = hk.Linear(output_size, name='l1')
            # mock transformer
            linear_2 = hk.Linear(output_size, name='l2')
            h_layer = Rootfind(name='rf')
            z = linear_1(x)
            y = h_layer(linear_2, z)
            return y

        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    def loss_fn(params, rng, x):
        h = forward_fn.apply(params, rng, x)
        return jnp.sum(h)

    value = loss_fn(params, rng, jnp.ones((1, 2, 3)))
    value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))
    print(grad)
