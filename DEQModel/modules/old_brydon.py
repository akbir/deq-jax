from typing import Callable

import jax.numpy as jnp
import jax
from jax import partial, lax

_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)

@jax.jit
def _safe_norm(v: jnp.ndarray) -> jnp.ndarray:
    if not jnp.isfinite(v).all():
        return jnp.array(jnp.inf)
    return jnp.linalg.norm(v)


@jax.jit
def rmatvec(part_Us: jnp.ndarray, part_VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.size == 0:
        return -x
    xTU = _einsum('bij, bijd -> bd', x, part_Us)  # (N, threshold)
    return -x + _einsum('bd, bdij -> bij', xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


@jax.jit
def matvec(part_Us: jnp.ndarray, part_VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.size == 0:
        return -x
    VTx = _einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + _einsum('bijd, bd -> bij', part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(g: Callable, x0: jnp.ndarray, threshold: int, eps: float) -> dict:
    # When doing low-rank updates at a (sub)sequence level, we still only store the low-rank updates,
    # instead of the huge matrices
    def line_search(g: Callable, update, x0: jnp.ndarray, g0: jnp.ndarray):
        """
        `update` is the proposed direction of update.
        """
        s = 1.0
        x_est = x0 + s * update
        g0_new = g(x_est)
        return x_est - x0, g0_new - g0

    bsz, total_hsize, seq_len = x0.shape

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = jnp.zeros((bsz, total_hsize, seq_len, threshold))  # One can also use an L-BFGS scheme to further reduce memory
    VTs = jnp.zeros((bsz, threshold, total_hsize, seq_len))
    update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    new_objective = init_objective = jnp.linalg.norm(gx).item()
    prot_break = False
    trace = [init_objective]

    # To be used in protective breaks
    protect_thres = 1e5 * seq_len
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep

    line_search = jax.jit(line_search, static_argnums=0)

    # ToDO: replace with while_loop function
    while new_objective >= eps and nstep < threshold:
        dx, delta_gx = line_search(g, update, x_est, gx)
        x_est += dx
        gx += delta_gx
        nstep += 1
        tnstep += 1
        new_objective = jnp.linalg.norm(gx).item()
        trace.append(new_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est, gx
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3 * eps and nstep > 30 and jnp.max(trace[-30:]) / jnp.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :nstep - 1]
        vT = rmatvec(part_Us, part_VTs, dx)
        u = (dx - matvec(part_Us, part_VTs, delta_gx)) / _einsum('bij, bij -> b', vT, delta_gx)[:, None, None]
        vT = jnp.nan_to_num(vT)
        u = jnp.nan_to_num(u)

        VTs = jax.ops.index_update(VTs, jax.ops.index[:, nstep - 1], vT)
        Us = jax.ops.index_update(Us, jax.ops.index[:, :, :, nstep - 1], u)
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": jnp.linalg.norm(lowest_gx).item(),
            "diff_detail": jnp.linalg.norm(lowest_gx, axis=1),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold}
