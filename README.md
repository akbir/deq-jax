# Deep Equilibrium Models [NeurIPS'19]
Jax Implementation for the deep equilibrium (DEQ) model, an implicit-depth architecture proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Unlike many existing "deep" techniques, the DEQ model is a implicit-depth architecture that directly solves for and backpropagates through the equilibrium state of an (effectively) infinitely deep network. 

## Prerequisite
Python >= 3.5 and Haiku >= 0.0.2

## Major Components
This repo provides the following re-usable components:

1. Jax implementation of the Broyden's method, a quasi-Newton method for finding roots in k variables. This method is JIT-able.
2. Jax implementation of Rootfind forward and backward method with custom vector-Jacobi product (backwards method)
3. (WIP) Haiku implementation of the Transformer XL

## Usage
All DEQ instantiations share the same underlying framework, whose core functionalities are provided in src/modules.
In particular, `rootfind.py` provides the Jax functions that solves for the roots in forward and backward passes. `broyden.py` provides an implementation of the Broyden's method.

 ```python
import haiku as hk
import jax
import jax.numpy as jnp
from jax import value_and_grad

from deq_jax.src.modules.deq import deq

def build_forward(output_size, max_iter):
    def forward_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        # create original layers and transform them 
        network = hk.Linear(output_size, name='l1')
        transformed_net = hk.transform(network)

        # lift params
        inner_params = hk.experimental.lift(
            transformed_net.init)(hk.next_rng_key(), x)
        
        # apply deq to functions of form f(params, z)
        z = deq(inner_params, hk.next_rng_key(), x,
                 transformed_net.apply, max_iter, is_training)

        return hk.Linear(output_size)(z)
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


value, grad = value_and_grad(loss_fn)(params, rng, jnp.ones((1, 2, 3)))
```
## Credits
The repo takes direct inspiration from the [original implementation](https://github.com/locuslab/deq/tree/master) by Shaojie in Torch.
 The transformer module is modified from an [example](https://github.com/deepmind/dm-haiku/blob/master/examples/transformer/model.py) provided by Haiku.