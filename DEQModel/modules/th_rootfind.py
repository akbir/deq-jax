import numpy as np
import torch
from torch.autograd import Function

from DEQModel.modules.th_brydon import th_broyden


class RootFind(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """
    @staticmethod
    def f(func, z1ss, uss, z0, *args):
        return func(z1ss, uss, z0, *args)

    @staticmethod
    def g(func, z1ss, uss, z0, *args):
        return RootFind.f(func, z1ss, uss, z0, *args) - z1ss

    @staticmethod
    def broyden_find_root(func, z1ss, uss, z0, eps, *args):
        z1ss_est = z1ss.clone().detach()
        threshold = args[-2]  # Can also set this to be different, based on training/inference

        g = lambda x: RootFind.g(func, x, uss, z0, *args)
        result_info = th_broyden(g, z1ss_est, threshold=threshold, eps=eps, name="forward")
        z1ss_est = result_info['result']

        if threshold > 100:
            torch.cuda.empty_cache()
        return z1ss_est.clone().detach()

    @staticmethod
    def forward(ctx, func, z1ss, uss, z0, *args):
        bsz, d_model, seq_len = z1ss.size()
        eps = 1e-6 * np.sqrt(bsz * seq_len * d_model)
        root_find = RootFind.broyden_find_root
        ctx.args_len = len(args)
        with torch.no_grad():
            z1ss_est = root_find(func, z1ss, uss, z0, eps, *args)  # args include pos_emb, threshold, train_step

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return z1ss_est

    @staticmethod
    def backward(ctx, grad_z1):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, None, *grad_args)
