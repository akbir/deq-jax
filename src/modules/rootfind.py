from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax

from src.modules.broyden import broyden


def g(func, x, *args):
    return f(func, x, *args) - x

def f(func, x, *args):
    return func(x, *args)

@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def rootfind(x: jnp.ndarray, func: Callable, max_iter: int):
    g_to_opt = partial(g, func)
    eps = 1e-6 * jnp.sqrt(x.size)
    # stop gradients for anything from root search
    result_info = jax.lax.stop_gradient(
        broyden(g_to_opt, x, max_iter, eps))
    return result_info['result']

def rootfind_fwd(x, func, max_iter):
    z_star = rootfind(x, func, max_iter)
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return z_star, (z_star,)

def rootfind_bwd(func, max_iter, res, grad):
    # returns dl/dz_star * J^(-1)_{g}
    (z_star,) = res
    g_to_opt = partial(g, func)
    _, f_vjp = jax.vjp(g_to_opt, z_star)

    def h_function(x):
        # returns tuple for each arg
        (JTx,) = f_vjp(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_function, dl_df_est, max_iter, eps=eps)
    dl_df_est = result_info['result']
    return dl_df_est,


rootfind.defvjp(rootfind_fwd, rootfind_bwd)
