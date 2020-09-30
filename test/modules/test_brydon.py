import jax.numpy as jnp
import jax
import numpy as np
import torch

from src.modules.th_brydon import th_broyden, th_rmatvec, th_matvec
from src.modules.broyden import broyden, rmatvec


def test_brydon():
    def quadratic(x):
        return x ** 2

    values = np.random.rand(2, 3, 4)
    threshold = 30
    eps = 1e-6
    th = torch.tensor(values, dtype=torch.float32)
    ja = jnp.asarray(values, dtype=jnp.float32)

    broyden_jit = jax.jit(broyden, static_argnums=(0, 2, 3))

    jax_ans = broyden_jit(jax.jit(quadratic), ja, threshold, eps)
    th_ans = th_broyden(quadratic, th, threshold, eps)

    np.testing.assert_array_almost_equal(np.array(jax_ans['result']), th_ans['result'].numpy(), decimal=6)

    assert jax_ans['n_step'] == th_ans['nstep']
    assert jax_ans['prot_break'] == th_ans['prot_break']
    assert jax_ans['eps'] == th_ans['eps']

    np.testing.assert_array_almost_equal(jax_ans['diff'], th_ans['diff'], decimal=6)
    np.testing.assert_array_almost_equal(np.array(jax_ans['diff_detail']), th_ans['diff_detail'].numpy(), decimal=6)
    np.testing.assert_array_almost_equal(jax_ans['trace'], th_ans['trace'], decimal=6)


def test_rmatvec():
    for _ in range(10):
        values = np.random.rand(2, 3, 4)
        th = torch.tensor(values, dtype=torch.float32)
        ja = jnp.asarray(values, dtype=jnp.float32)

        values_2 = np.random.rand(2, 3, 4, 5)
        values_3 = np.random.rand(2, 5, 3, 4)
        th_2, th_3 = torch.tensor(values_2, dtype=torch.float32), torch.tensor(values_3, dtype=torch.float32)
        ja_2, ja_3 = jnp.asarray(values_2, dtype=jnp.float32), jnp.asarray(values_3, dtype=jnp.float32)

        np.testing.assert_almost_equal(np.array(rmatvec(ja_2, ja_3, ja)),
                                       th_rmatvec(th_2, th_3, th).numpy(), decimal=5)


def test_matvec():
    for _ in range(10):
        values = np.random.rand(2, 3, 4)
        th = torch.tensor(values, dtype=torch.float32)
        ja = jnp.asarray(values, dtype=jnp.float32)

        values_2 = np.random.rand(2, 3, 4, 5)
        values_3 = np.random.rand(2, 5, 3, 4)
        th_2, th_3 = torch.tensor(values_2, dtype=torch.float32), torch.tensor(values_3, dtype=torch.float32)
        ja_2, ja_3 = jnp.asarray(values_2, dtype=jnp.float32), jnp.asarray(values_3, dtype=jnp.float32)

        np.testing.assert_almost_equal(np.array(matvec(ja_2, ja_3, ja)),
                                       th_matvec(th_2, th_3, th).numpy(), decimal=5)



