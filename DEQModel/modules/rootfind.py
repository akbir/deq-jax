from functools import partial
import jax.numpy as jnp
import jax

from jax.scipy.optimize import minimize


def f(func, z1ss, uss, z0, *args):
    return func(z1ss, uss, z0, *args)


def g(func, z1ss, uss, z0, *args):
    return jnp.linalg.norm(f(func, z1ss, uss, z0, *args) - z1ss)


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def rootfind(func, z1ss: jnp.ndarray, uss: jnp.ndarray, z0: jnp.ndarray):
    z1ss_est = z1ss.copy()
    def g_to_optimise(x):
        return g(func, x, uss, z0)

    result_info = minimize(g_to_optimise, z1ss_est, method='BFGS')
    z1ss_est = result_info['result']
    return z1ss_est


def rootfind_fwd(func, z1ss, uss, z0, threshold):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return rootfind(func, z1ss, uss, z0, threshold), (func, z1ss, uss, z0, threshold)


def rootfind_bwd(res, grad):
    grad = grad.copy()
    func, z1ss, uss, z0, threshold = res

    z1ss = z1ss.copy()
    uss = uss.copy()
    z0 = z0.copy()

    y = lambda x: g(func, x, uss, z0, threshold)

    def h_function(x):
        primal, vjp = jax.vjp(y, z1ss)
        JTx = vjp(x)
        return JTx + grad

    eps = 2e-10 * jnp.sqrt(grad.size)
    dl_df_est = jnp.zeros_like(grad)

    result_info = minimize(h_function, dl_df_est, method='BFGS')
    dl_df_est = result_info['result']
    return dl_df_est


# rootfind.defvjp(rootfind_fwd, rootfind_bwd)
