from typing import Callable, NamedTuple, Union

import jax.numpy as jnp
import haiku as hk
import jax
from jax import partial, lax

class _BroydenResults(NamedTuple):
    """Results from Broyden optimization.
    Parameters:
        converged: True if minimization converged.
        n_steps: integer the number of iterations of the BFGS update.
        min_x: array containing the minimum argument value found during the search. If
          the search converged, then this value is the argmin of the objective
          function.
        min_gx: array containing the value of the objective function at `min_x`. If the
          search converged, then this is the (local) minimum of the objective
          function.
        min_objective: array containing lowest 2 norm of the objective function
        x: array containing the prev argument value found during the search.
        gx: array containing the prev value of the objective function at `x`
        objective: array containing prev lowest 2 norm of the objective function
        trace: array of previous objectives
        Us: array containing the fraction component of the Jacobian approximation (N, 2d, L', n_step)
        VTs: array containing the \delta x_n^T J_{n-1}^{-1} of the estimated Jacobian (N, n_step, 2d, L')
        prot_break: True if protection threshold broken (no convergence)
        prog_break: True if progression threshold broken (no convergence)
    """
    converged: Union[bool, jnp.ndarray]
    n_step: Union[int, jnp.ndarray]
    min_x: jnp.ndarray
    min_gx: jnp.ndarray
    min_objective: jnp.ndarray
    x: jnp.ndarray
    gx: jnp.ndarray
    objective: jnp.ndarray
    trace: list
    Us: jnp.ndarray
    VTs: jnp.ndarray
    prot_break: Union[bool, jnp.ndarray]
    prog_break: Union[bool, jnp.ndarray]


_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)

def rmatvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if Us.size == 0:
        return -x
    xTU = _einsum('bij, bijd -> bd', x, Us)  # (N, threshold)
    return -x + _einsum('bd, bdij -> bij', xTU, VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))

def matvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # Us: (N, 2d, L', threshold)
    # VTs: (N, threshold, 2d, L')
    if Us.size == 0:
        return -x
    VTx = _einsum('bdij, bij -> bd', VTs, x)  # (N, threshold)
    return -x + _einsum('bijd, bd -> bij', Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def update(delta_x, delta_gx, Us, VTs, n_step):
    # Add column/row to Us/VTs with updated approximation
    # Calculate J_i
    vT = rmatvec(Us, VTs, delta_x)
    u = (delta_x - matvec(Us, VTs, delta_gx)) / _einsum('bij, bij -> b', vT, delta_gx)[:, None, None]

    vT = jnp.nan_to_num(vT)
    u = jnp.nan_to_num(u)

    # Store in UTs and VTs for calculating J
    VTs = jax.ops.index_update(VTs, jax.ops.index[:, n_step - 1], vT)
    Us = jax.ops.index_update(Us, jax.ops.index[:, :, :, n_step - 1], u)

    return Us, VTs


def line_search(g: Callable, update, x0: jnp.ndarray, g0: jnp.ndarray, *args):
    """
    `update` is the proposed direction of update.
    """
    s = 1.0
    x_est = x0 + s * update
    g0_new = g(x_est, *args)
    return x_est - x0, g0_new - g0

def broyden(g: Callable, x0: jnp.ndarray, max_iter: int, eps: float, *args) -> dict:
    """
    :param g: Function to find root of (e.g g(x) = f(x)-x)
    :param x0: Initial guess  (batch_size, hidden, seq_length)
    :param max_iter: maximum number of iterations.
    :param eps: terminates minimization when |J^_1|_norm < eps

    ``broyden`` supports ``jit`` compilation. It does not yet support
    differentiation or arguments in the form of multi-dimensional arrays
    :return:
    """

    # Update rule is J_n = J_n-1 + delta_J, so iteratively write J = J_0 + J_1 + J_2 + ...
    # For memory constraints J = U * V^T
    # So J = U_0 * V^T_0 + U_1 * V^T_1 + ..
    # For fast calculation of inv_jacobian (approximately) we store as Us and VTs

    bsz, total_hsize, seq_len = x0.shape
    gx = g(x0, *args)  # (bsz, 2d, L')
    init_objective = jnp.linalg.norm(gx)

    # To be used in protective breaks
    trace = jnp.zeros(max_iter)
    trace = jax.ops.index_update(trace, jax.ops.index[0], init_objective)
    protect_thres = 1e5 * seq_len

    state = _BroydenResults(
        converged=False,
        n_step=0,
        min_x=x0,
        min_gx=gx,
        min_objective=init_objective,
        x=x0,
        gx=gx,
        objective=init_objective,
        trace=trace,
        Us=jnp.zeros((bsz, total_hsize, seq_len, max_iter)),
        VTs=jnp.zeros((bsz, max_iter, total_hsize, seq_len)),
        prot_break=False,
        prog_break=False,
    )

    def cond_fun(state: _BroydenResults):
        return (jnp.logical_not(state.converged) &
                jnp.logical_not(state.prot_break) &
                jnp.logical_not(state.prog_break) &
                (state.n_step < max_iter))

    def body_fun(_, state: _BroydenResults):
        inv_jacobian = -matvec(state.Us, state.VTs, state.gx)
        dx, delta_gx = line_search(g, inv_jacobian, state.x, state.gx, *args)

        state = state._replace(
            x=state.x + dx,
            gx=state.gx + delta_gx,
            n_step=state.n_step + 1,
        )

        new_objective = jnp.linalg.norm(state.gx)
        trace = jax.ops.index_update(state.trace, jax.ops.index[state.n_step], new_objective)

        min_found = new_objective < state.min_objective
        state = state._replace(
            # if a new minimum is found
            min_x=jnp.where(min_found, state.x, state.min_x),
            min_gx=jnp.where(min_found, state.gx, state.min_gx),
            min_objective=jnp.where(min_found, new_objective, state.min_objective),
            trace=trace,
            # check convergence
            converged=(new_objective < eps),
            prot_break=(new_objective > init_objective * protect_thres),
            prog_break=(new_objective < 3. * eps) & (state.n_step > 30) & (jnp.max(state.trace[-30:]) / jnp.min(state.trace[-30:]) < 1.3)
        )

        # update for next jacobian
        Us, VTs = update(dx, delta_gx, state.Us, state.VTs, state.n_step)
        state = state._replace(Us=Us, VTs=VTs)

        return state


    # state = body_fun(state)
    # state = jax.lax.while_loop(cond_fun, body_fun, state)
    state = jax.lax.fori_loop(0, max_iter, body_fun, state)
    # state = hk.fori_loop(0, max_iter, body_fun, state)
    return {"result": state.min_x,
            "n_step": state.n_step,
            "diff": jnp.linalg.norm(state.min_gx),
            "diff_detail": jnp.linalg.norm(state.min_gx, axis=1),
            "prot_break": state.prot_break,
            "trace": state.trace,
            "eps": eps,
            "maxiter": max_iter}
