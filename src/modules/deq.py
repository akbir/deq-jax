from typing import Callable
import jax.numpy as jnp

from src.modules.rootfind import rootfind, rootfind_grad


def deq(params: dict, rng, x: jnp.ndarray, fun: Callable, max_iter: int, is_training: bool, *args) -> jnp.ndarray:
    """
    Apply Deep Equilibrium Network to haiku function.
    :param rng:
    :param fun: func to apply in the deep equilibrium limit
    :param max_iter: maximum number of integers for the broyden method
    :param x: initial guess for broyden method
    :param params: Hk.Params
    :param is_training: Set True to calculate gradients for backward pass
    :return:
    """

    # define equilibrium eq (f(z)-z)
    def g(_params, _rng, _x, *args): return fun(_params, _rng, _x, *args) - _x

    # find equilibrium point
    z = rootfind(g, max_iter, params, rng, x, *args)

    if is_training:
        # set up correct graph for chain rule (bk pass)
        z = fun(params, rng, z, *args)
        z = rootfind_grad(g, max_iter, params, rng, z, *args)
    return z