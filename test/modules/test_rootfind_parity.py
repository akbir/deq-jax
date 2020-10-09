import copy

import jax.numpy as jnp
import numpy as np
import torch
from jax import value_and_grad, grad
from torch import nn

from src.modules.rootfind import rootfind
from src.legacy.torch_rootfind import DEQModule, RootFind

def test_parity():
    def layer(x, w):
        return x * w
    def toy_model(params, data):
        z_star = rootfind(data, layer, 30, params)
        y = layer(z_star, params)
        return jnp.sum(y)

    params = -2 * jnp.ones((1,2,3))
    data = jnp.ones((1,2,3))
    value, grad_params = value_and_grad(toy_model)(params, data)
    grad_inputs = grad(toy_model, argnums=1)(params, data)

    assert (-6.0 == value or 0. == value)

    class DumbLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(-2 * torch.ones(1, 2, 3), requires_grad=True)
            self.register_parameter(name='W', param=self.weight)

        def forward(self, z1ss, *args):
            return z1ss * self.weight

        def copy(self, func):
            self.weight.data = func.weight.data.clone()

    class linearDEQ(DEQModule):
        def __init__(self, func, func_copy):
            super(linearDEQ, self).__init__(func, func_copy)

        def forward(self, z1s):
            zero = torch.zeros_like(z1s)
            threshold, train_step = 30, -1
            z1s_out = RootFind.apply(self.func, z1s, zero, zero, zero, threshold, train_step)
            if self.training:
                z1s_out = RootFind.f(self.func, z1s_out, zero, zero, zero, threshold, train_step)
                z1s_out = self.Backward.apply(self.func_copy, z1s_out, zero, zero, zero, threshold, train_step)
            return z1s_out


    class DEQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.func = DumbLinear()
            self.func_copy = copy.deepcopy(self.func)
            for params in self.func_copy.parameters():
                params.requires_grad_(False)

            self.deq = linearDEQ(self.func, self.func_copy)

        def forward(self, input):
            self.func_copy.copy(self.func)
            return self.deq(input)

    th_net = DEQ()
    input = torch.ones(1,2,3)
    input.requires_grad = True
    torch_value = th_net(input).sum()
    torch_value.backward()
    torch_gradient = th_net.func.W.grad

    np.testing.assert_almost_equal(np.asarray(value), torch_value.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_inputs), input.grad.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_params), torch_gradient.detach().numpy())



