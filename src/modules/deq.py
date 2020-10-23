from typing import Callable
import jax.numpy as jnp

from src.modules.rootfind import rootfind, rootfind_grad


def deq(fun: Callable, max_iter: int, params: dict, x: jnp.ndarray, with_gradients: bool) -> jnp.ndarray:
    """
    Apply Deep Equilibrium Network to haiku function.
    :param fun: func to apply in the deep equilibrium limit
    :param max_iter: maximum number of integers for the broyden method
    :param x: initial guess for broyden method
    :param params: Hk.Params
    :param with_gradients: Set True to calculate gradients for backward pass
    :return:
    """

    # define equilibrium eq (f(z)-z)
    def g(_params, _x): return fun(_params, _x) - _x

    # find equilibrium point
    z = rootfind(g, max_iter, params, x)

    if with_gradients:
        # if we need gradients, then we have to set up correct graph for chain rule
        z = fun(params, z)
        z = rootfind_grad(g, max_iter, params, z)
    return z