from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax
import haiku as hk

from src.modules.broyden import broyden


def g(fun, x, *args):
    return f(fun, x, *args) - x


def f(fun, x, *args):
    return fun(x, *args)


@partial(jax.custom_vjp, nondiff_argnums=(0, 2))
def rootfind(fun: Callable, x: jnp.ndarray, max_iter: int, *args):
    g_to_opt = partial(g, fun)
    eps = 1e-6 * jnp.sqrt(x.size)
    result_info = jax.lax.stop_gradient(
        broyden(g_to_opt, x, max_iter, eps, *args)
    )
    return result_info['result']


def rootfind_fwd(fun: Callable, x: jnp.ndarray, max_iter: int, *args):
    """A JAX layer for applying rootfind(g, z, *args) to a function (fun)
    Requires fun to be form f(x, *args), where x is the value to optimise
    """
    z_star = rootfind(fun, x, max_iter, *args)

    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return z_star, (z_star, *args)


def rootfind_bwd(fun, max_iter, res, grad):
    # returns dl/dz_star * J ^ (-1)_{g}
    (z_star, *args) = res
    g_to_opt = partial(g, fun)
    _, g_vjp = jax.vjp(g_to_opt, z_star, *args)

    def h_function(x):
        # ToDo: g_vjp is dependent on fun (e.g haiku module, haiku transform or jax func)
        (JTx, *JT_args) = g_vjp(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_function, dl_df_est, max_iter, eps=eps)
    dl_df_est = result_info['result']
    # passed back gradient via d/dx and return nothing to other params
    empty_grads = tuple(jnp.zeros_like(arg) for arg in args)
    return dl_df_est, *empty_grads


rootfind.defvjp(rootfind_fwd, rootfind_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def h(fun: Callable, x: jnp.ndarray):
    return jax.lax.stop_gradient(fun(x))

def h_fwd(fun: Callable, x: jnp.ndarray):
    return h(fun, x), h(fun, x)

def h_bwd(fun, res, grad):
    output, = res
    x_vjp = jax.vjp(fun, output)
    return grad*x_vjp(output)

h.defvjp(h_fwd, h_bwd)
