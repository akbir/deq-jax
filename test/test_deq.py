from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad, grad

from src.modules.deq import deq


def test_deq_multi_variable():
    def layer(params, rng, x):
        return params['w'] * x + params['b']

    def toy_model(params, data):
        h = deq(params, None, data, layer, 30, True)
        return jnp.sum(h)

    params = {'w': -2 * jnp.ones((1, 2, 3)), 'b': jnp.ones((1, 2, 3))}
    data = jnp.ones((1, 2, 3))
    value, grad_params = value_and_grad(toy_model)(params, data)
    grad_data = grad(toy_model, argnums=1)(params, data)

    np.testing.assert_almost_equal(2, value, decimal=4)
    np.testing.assert_almost_equal(-2/3 * np.ones((1, 2, 3)), grad_data)


def test_deq_haiku():
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = hk.Linear(output_size,
                                 name='l1',
                                 w_init=hk.initializers.Constant(1),
                                 b_init=hk.initializers.Constant(1))
            transformed_linear = hk.transform(linear_1)
            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            z = deq(inner_params, hk.next_rng_key(), x, transformed_linear.apply, max_iter, True)
            return z

        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    @jax.jit
    def loss_fn(params, rng, x):
        return jnp.sum(forward_fn.apply(params, rng, x))

    value = loss_fn(params, rng, 2 * input)
    params_grad = grad(loss_fn)(params, rng, 2 * input)
    np.testing.assert_almost_equal(-3., value)

    expected_weights = 1/2 * np.ones((3,3))
    np.testing.assert_almost_equal(expected_weights, params_grad['lifted/l1']['w'])

def test_deq_with_hk_using_rng():
    class linear_with_dropout(hk.Module):
        def __init__(self,
                     output_size: int,
                     name: Optional[str] = None):
            super().__init__(name=name)
            self.output_size = output_size

        def __call__(self,
                     x: jnp.ndarray) -> jnp.ndarray:

            h = hk.Linear(self.output_size,
                          w_init=hk.initializers.Constant(1),
                          b_init=hk.initializers.Constant(1))(x)
            return hk.dropout(hk.next_rng_key(), 0.1, h)

    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = linear_with_dropout(3)
            transformed_linear = hk.transform(linear_1)

            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            def fun(_params, _rng, h): return transformed_linear.apply(_params, _rng, h)
            z = deq(inner_params, hk.next_rng_key(), x, fun, max_iter, True)
            return hk.Linear(output_size, name='l2', with_bias=False)(z)

        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    @jax.jit
    def loss_fn(params, rng, x):
        return jnp.sum(forward_fn.apply(params, rng, x))

    value = loss_fn(params, rng, 2 * input)
    params_grad = grad(loss_fn)(params, rng, 2 * input)
    np.testing.assert_almost_equal(-2.094507, value)

    expected_weights = np.asarray([[ 0.47394884, -0.25200003,  0.08285569],
                                    [ 0.47394884,  0.08946458,  1.0635897 ],
                                    [ 0.47394884,  0.08946458,  1.0635897 ]])

    np.testing.assert_almost_equal(expected_weights, params_grad['lifted/linear_with_dropout/linear']['w'])
