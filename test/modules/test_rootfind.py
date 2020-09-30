import jax

from src.modules.rootfind import rootfind
import jax.numpy as jnp
import haiku as hk
def test_rootfind_equality():
    pass


def test_rootfind_jit():
    @jax.jit
    def func(x):
        return x**2

    jit_rootfind = jax.jit(rootfind, static_argnums=[0,2])
    jit_rootfind(func, jnp.ones((1,2,3)), 30)


def test_rootfind_with_haiku_func():
    def build_forward(output_size):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            return hk.Linear(output_size)(x)
        return forward_fn

    input = jnp.ones((1,2,3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    func = lambda x: forward_fn.apply(params, rng, input)
    root = rootfind(jax.jit(func), jnp.ones((1,2,3)), 30)



def test_rootfind_in_haiku():
    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear = hk.Linear(output_size, name='l1')
            h = linear(x)
            func = lambda x: linear(x) - x
            y = rootfind(func, h, max_iter)
            return hk.Linear(output_size, name='l2')(y)
        return forward_fn

    input = jnp.ones((1,2,3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 10)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    fwd = jax.jit(forward_fn.apply)
    return fwd(params, rng, input)