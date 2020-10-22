from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax

from src.modules.broyden import broyden


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def rootfind(fun: Callable, max_iter: int, x: jnp.ndarray):
    eps = 1e-6 * jnp.sqrt(x.size)
    result_info = jax.lax.stop_gradient(
        broyden(fun, x, max_iter, eps)
    )
    return result_info['result']


def rootfind_fwd(fun: Callable, max_iter: int, x: jnp.ndarray):
    """Layer for applying rootfind(g, z, x) to a function (fun)
    Requires fun to be form f(x), where x is the value to optimise, other values must be frozen
    """
    z_star = rootfind(fun, max_iter, x)

    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return z_star, (z_star)


def rootfind_bwd(fun, max_iter, res, grad):
    # returns dl/dz_star * J ^ (-1)_{g}
    (z_star,) = res
    _, g_vjp = jax.vjp(fun, z_star)

    def h_function(x):
        # ToDo: g_vjp is dependent on fun (e.g haiku func or jax func)
        JTx, _ = g_vjp(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_function, dl_df_est, max_iter, eps=eps)
    dl_df_est = result_info['result']
    # passed back gradient via d/dx and return nothing to other params
    return dl_df_est,


rootfind.defvjp(rootfind_fwd, rootfind_bwd)