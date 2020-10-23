from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax

from src.modules.broyden import broyden


def rootfind_fwd(fun: Callable, max_iter: int, params: dict, x: jnp.ndarray):
    """Forward for applying rootfind to a function (fun)
    Requires fun to be form f(x), where x is the value to optimise, other values must be frozen
    """
    z_star = rootfind(fun, max_iter, params, x)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return z_star, (z_star, params)


def rootfind_bwd(fun, max_iter, res, grad):
    """Backward method for evaluating update for gradients for DEQ see

    :param fun:
    :param max_iter:
    :param res:
    :param grad:
    :return:
    """
    # returns dl/dz_star * J^(-1)_{g}
    (z_star, params) = res
    _, vjp_fun = jax.vjp(fun, params, z_star)

    def h_fun(x):
        #  J^(-1)_{g} x^T + (dl/dz_star)^T
        JTp, JTx = vjp_fun(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_fun, dl_df_est, max_iter, eps)
    dl_df_est = result_info['result']
    # passed back gradient via d/dx and return nothing to other params
    return None, dl_df_est


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind(fun: Callable, max_iter: int, params: dict, x: jnp.ndarray):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(fun, params)

    result_info = jax.lax.stop_gradient(
        broyden(fun, x, max_iter, eps)
    )
    return result_info['result']


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind_grad(fun: Callable, max_iter: int, params: dict, x: jnp.ndarray):
    eps = 1e-6 * jnp.sqrt(x.size)
    fun = partial(fun, params)
    result_info = jax.lax.stop_gradient(
        broyden(fun, x, max_iter, eps)
    )
    return result_info['result']


def dumb_fwd(fun: Callable, max_iter: int, params: dict, x: jnp.ndarray):
    return x, (x, params)

def dumb_bwd(fun, max_iter, res, grad):
    return None, grad

rootfind.defvjp(rootfind_fwd, dumb_bwd)
rootfind_grad.defvjp(dumb_fwd, rootfind_bwd)

