import haiku as hk
from src.modules.vjp import vjp as custom_vjp
import jax.numpy as jnp
import jax

class TestVJP:
    def test_simple(self):
        class MyModule(hk.Module):
            def __call__(self, x):
                return x ** 2
        def f(x):
            m = MyModule()
            _, m_vjp = jax.vjp(m, x)
            return m_vjp(x)

        f = hk.transform(f)
        x = jnp.array(2.)
        f.init(x)
        params, state = jax.jit(f.init)(x)
