from functools import partial

import jax
from jax import value_and_grad

from src.modules.rootfind import rootfind
import jax.numpy as jnp
import haiku as hk
def test_rootfind_equality():
    pass


def test_rootfind_1D():
    @jax.jit
    def func(x):
        return x**2

    jit_rootfind = jax.jit(rootfind, static_argnums=[1, 2])

    def toy_model(x):
        y = func(x)
        return jnp.sum(rootfind(y, func, 30))

    root = jit_rootfind(jnp.ones((1,2,3)), func, 30)
    value, grad = value_and_grad(toy_model)(jnp.ones((1,2,3)))

    assert (6.0 == value or 0. == value)

def test_rootfind_multi_variable():
    def func(x, w):
        return w * x

    def toy_model(params, data):
        y = func(data, params)
        f = partial(func, w=params)
        return jnp.sum(rootfind(y, f, 30))

    params = jnp.zeros((1, 2, 3))
    data = jnp.ones((1,2,3))

    g = lambda x: func(x, params) - x
    y = func(data, params)
    root = rootfind(y, g, 30)
    _, f_vjp = jax.vjp(g, root)

    value, grad = value_and_grad(toy_model, argnums=1)(params, data)

    value, grad = value_and_grad(toy_model)(params, data)

    assert (12.0 == value or 0. == value)

def test_rootfind_with_haiku_func():
    def build_forward(output_size):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            return hk.Linear(output_size)(x)
        return forward_fn

    inputs = jnp.zeros((1,2,3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, inputs)

    def toy_model(params, rng, x):
        y = forward_fn.apply(params, rng, x)
        return jnp.sum(rootfind(y, lambda z: forward_fn.apply(params, rng, z), 30))

    root = rootfind(jnp.ones((1, 2, 3)),
                    lambda x: forward_fn.apply(params, rng, x),
                    30)

    value = toy_model(params, rng, jnp.ones((1, 2, 3)))

    value, grad = value_and_grad(toy_model)(params, rng, inputs)
    assert (jnp.sum(root) == value).any()


def test_rootfind_in_haiku():
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear = hk.Linear(output_size, name='l1')
            h = linear(x)
            func = lambda z: linear(z) - z
            y = rootfind(h, func, max_iter)
            return hk.Linear(output_size, name='l2')
        return forward_fn


    input = jnp.ones((1,2,3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    def loss_fn(params, rng, x):
        return jnp.sum(forward_fn.apply(params, rng, x))


    value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))


