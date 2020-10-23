import haiku as hk
import jax
import jax.numpy as jnp
from jax import value_and_grad, grad
import numpy as np

from deq_jax.src.modules.rootfind import rootfind, rootfind


def test_rootfind_multi_variable():
    def layer(params, x):
        return params['w'] * x + params['b']

    def toy_model(params, data):
        def fun(params, z): return layer(params, z) - z
        h = rootfind(fun, 30, params, data)
        return jnp.sum(h)

    params = {'w': -2 * jnp.ones((1, 2, 3)), 'b': jnp.ones((1, 2, 3))}
    data = jnp.ones((1, 2, 3))
    value, grad_params = value_and_grad(toy_model)(params, data)
    grad_data = grad(toy_model, argnums=1)(params, data)

    np.testing.assert_almost_equal(2, value, decimal=4)
    np.testing.assert_almost_equal(-2* np.ones((1, 2, 3)), grad_data)



def test_rootfind_haiku():
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = hk.Linear(output_size,
                                 name='l1',
                                 w_init=hk.initializers.Constant(1),
                                 b_init=hk.initializers.Constant(1))

            transformed_linear = hk.without_apply_rng(
                hk.transform(linear_1)
            )
            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            def fun(params, z): return transformed_linear.apply(params, z) - z

            h = rootfind(fun, max_iter, inner_params, x)
            y = transformed_linear.apply(inner_params, h)
            return hk.Linear(output_size, name='l2', with_bias=False)(y)
        return forward_fn

    input = jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    @jax.jit
    def loss_fn(params, rng, x):
        return jnp.sum(forward_fn.apply(params, rng, x))

    value = loss_fn(params, rng, 2*input)
    params_grad = grad(loss_fn)(params, rng, 2*input)
    np.testing.assert_almost_equal(1.3233, value)
    np.testing.assert_almost_equal(np.ones((1,2,3)), params_grad)
