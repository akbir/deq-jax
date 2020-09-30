from functools import partial
import jax.numpy as jnp
import jax

from src.modules.broyden import broyden


@partial(jax.custom_vjp, nondiff_argnums=(0, 2))
def rootfind(func, x: jnp.ndarray, max_iter):
    eps = 1e-6 * jnp.sqrt(x.size)
    result_info = broyden(func, x, max_iter, eps)
    return result_info['result']


def rootfind_fwd(func, z1ss, uss, z0, threshold):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return rootfind(func, z1ss, uss, z0, threshold), (func, z1ss, uss, z0, threshold)


def rootfind_bwd(res, grad):
    grad = grad.copy()
    func, z1ss, uss, z0, _ = res

    y = lambda x: g(func, x, uss, z0)

    def h_function(x):
        primal, vjp = jax.vjp(y, z1ss)
        JTx = vjp(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = broyden(h_function, dl_df_est, max_iter=30, eps=eps)
    dl_df_est = result_info['result']
    return dl_df_est


rootfind.defvjp(rootfind_fwd, rootfind_bwd)


