from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax

from src.modules.broyden import broyden


def rootfind_fwd(fun: Callable, max_iter: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    z_star = rootfind(fun, max_iter, params, rng, x, *args)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return z_star, (params, rng, z_star, *args)


def rootfind_bwd(fun, max_iter, res, grad):
    # returns dl/dz_star * J^(-1)_{g}
    (params, rng, z_star, *args) = res
    (_, vjp_fun) = jax.vjp(fun, params, rng, z_star, *args)

    def h_fun(x):
        #  J^(-1)_{g} x^T + (dl/dz_star)^T
        (JTp, JTr, JTx, *_) = vjp_fun(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_fun, dl_df_est, max_iter, eps)
    dl_df_est = result_info['result']

    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, dl_df_est, *arg_grads)
    return return_tuple


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind(fun: Callable, max_iter: int, params: dict, rng: jnp.ndarray, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(fun, params, rng)

    result_info = jax.lax.stop_gradient(
        broyden(fun, x, max_iter, eps, *args)
    )
    return result_info['result']


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind_grad(fun: Callable, max_iter: int, params: dict, rng, x: jnp.ndarray, *args):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(fun, params, rng)
    result_info = jax.lax.stop_gradient(
        broyden(fun, x, max_iter, eps, *args)
    )
    return result_info['result']


def dumb_fwd(fun: Callable, max_iter: int, params: dict, rng, x: jnp.ndarray, *args):
    return x, (params, rng, x, *args)

def dumb_bwd(fun, max_iter, res, grad):
    (params, rng, z_star, *args) = res
    # passed back gradient via d/dx and return nothing to other params
    arg_grads = tuple([None for _ in args])
    return_tuple = (None, None, grad, *arg_grads)
    return return_tuple

rootfind.defvjp(rootfind_fwd, dumb_bwd)
rootfind_grad.defvjp(dumb_fwd, rootfind_bwd)

