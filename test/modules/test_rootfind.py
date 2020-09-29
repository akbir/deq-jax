import jax
import numpy as np
import pytest
import jax.numpy as jnp
import torch

from DEQModel.modules.th_rootfind import RootFind as THROOTFIND
from DEQModel.modules.rootfind import rootfind as jax_rootfind
BATCH_SIZE, HIDDEN_SIZE, SEQ_LEN = 32, 100, 2

@pytest.fixture()
def z1ss():
    return np.random.rand(BATCH_SIZE, HIDDEN_SIZE, SEQ_LEN)

def test_rootfind_forward(z1ss):

    def f(x, *args):
        return x**2

    torch_rootfind = THROOTFIND()

    torch_min = torch_rootfind.apply(f, torch.Tensor(z1ss), 0, 0, 0, 10)

    jax_min = jax_rootfind(jax.jit(f), jnp.asarray(z1ss), 0, 0)

    np.testing.assert_almost_equal(np.array(jax_min), np.array(torch_min))