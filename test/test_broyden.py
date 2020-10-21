
from src.modules.broyden import broyden
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

def test_broyden_in_haiku():
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = hk.Linear(output_size, name='l1')
            transformed_linear = hk.without_apply_rng(hk.transform(linear_1))
            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            f = lambda z: transformed_linear.apply(inner_params, z)-z
            y = broyden(f, x, max_iter, eps=0.01)['result']
            return f(y)-y

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
    np.testing.assert_almost_equal(value, jnp.zeros(1))