import numpy as np
import torch


def th_safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def th_scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1 * alpha2 * derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def th_line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = th_safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = th_scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est - x0, g0_new - g0, ite


def th_rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)  # (N, threshold)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def th_matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def th_broyden(g, x0, threshold, eps, ls=False, name="unknown"):
    # When doing low-rank updates at a (sub)sequence level, we still only store the low-rank updates,
    # instead of the huge matrices
    bsz, total_hsize, seq_len = x0.size()

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold)  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len)
    update = -th_matvec(Us[:, :, :, :nstep], VTs[:, :nstep],
                        gx)  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]

    # To be used in protective breaks
    protect_thres = 1e5 * seq_len
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    while new_objective >= eps and nstep < threshold:
        delta_x, delta_gx, ite = th_line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        x_est += delta_x
        gx += delta_gx
        nstep += 1
        tnstep += (ite + 1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3 * eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :nstep - 1]
        vT = th_rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - th_matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:, None,
                                                                 None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -th_matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold}


def th_find_update(Us, VTs, nstep, delta_x, delta_gx, gx):
    part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :nstep - 1]
    vT = th_rmatvec(part_Us, part_VTs, delta_x)
    u = (delta_x - th_matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:, None, None]
    vT[vT != vT] = 0
    u[u != u] = 0
    VTs[:, nstep - 1] = vT
    Us[:, :, :, nstep - 1] = u
    update = -th_matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)
    return update, VTs, Us
