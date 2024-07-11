from typing import Optional

import numpy as np

import torch
from torch.autograd import Function
from torch import Tensor

from pyscf.dft import libxc


# def stack_inputs(x, spin=0):

#     if spin:

#         x_ = rearrange(x, "... x s -> s (... x)")

#         batch_shape = x.shape[:-2]
#         b = prod(batch_shape)
#         unstack = lambda x_: rearrange(x_, "s (b x) -> b x s", b=b).view(*batch_shape, -1, 2)

#     else:
#         shape = x.shape
#         x_ = rearrange(x, "... x -> (... x)")
#         unstack = lambda x_: x_.view(*shape)

#     return x_, unstack


def to_numpy(x, dtype=np.float64):
    return x.cpu().numpy().astype(dtype)


def _fake_grad_rho(gamma, spin=0):
    grad_x, grad_pad = np.sqrt(gamma), np.zeros_like(gamma)
    return np.stack([grad_x, grad_pad, grad_pad], axis=spin)


class LDAFunctional(Function):
    @staticmethod
    def forward(ctx, xc, rho):

        shape, dtype, device = rho.shape, rho.dtype, rho.device

        rho_data = to_numpy(rho).reshape(-1)
        exc, (vrho, *_), *__ = libxc.eval_xc(xc, rho_data)

        exc = torch.tensor(exc, dtype=dtype, device=device).view(shape)
        vrho = torch.tensor(vrho, dtype=dtype, device=device).view(shape)

        ctx.save_for_backward(vrho)

        return rho * exc

    @staticmethod
    def backward(ctx, d_exc):
        (vrho,) = ctx.saved_tensors
        return None, d_exc * vrho


class GGAFunctional(Function):
    @staticmethod
    def forward(ctx, xc, rho, gamma):

        shape, dtype, device = rho.shape, rho.dtype, rho.device

        rho_ = to_numpy(rho).reshape(1, -1)
        grad_rho_ = _fake_grad_rho(to_numpy(gamma).reshape(-1))

        rho_data = np.concatenate([rho_, grad_rho_], axis=0)
        exc, (vrho, vgamma, *_), *__ = libxc.eval_xc(xc, rho_data, spin=0)

        exc = torch.tensor(exc, dtype=dtype, device=device).view(shape)
        vrho = torch.tensor(vrho, dtype=dtype, device=device).view(shape)
        vgamma = torch.tensor(vgamma, dtype=dtype, device=device).view(shape)

        ctx.save_for_backward(vrho, vgamma)

        return rho * exc

    @staticmethod
    def backward(ctx, d_exc):
        vrho, vgamma = ctx.saved_tensors
        return None, d_exc * vrho, d_exc * vgamma


class MGGAFunctional(Function):
    @staticmethod
    def forward(ctx, xc, rho, gamma, tau):

        shape, dtype, device = rho.shape, rho.dtype, rho.device

        rho_ = to_numpy(rho).reshape(1, -1)
        grad_rho_ = _fake_grad_rho(to_numpy(gamma).reshape(-1))
        tau_ = to_numpy(tau).reshape(1, -1)

        rho_data = np.concatenate([rho_, grad_rho_, tau_, tau_], axis=0)
        exc, (vrho, vgamma, _, vtau), *__ = libxc.eval_xc(xc, rho_data, spin=0)

        exc = torch.tensor(exc, dtype=dtype, device=device).view(shape)
        vrho = torch.tensor(vrho, dtype=dtype, device=device).view(shape)
        vgamma = torch.tensor(vgamma, dtype=dtype, device=device).view(shape)
        vtau = torch.tensor(vtau, dtype=dtype, device=device).view(shape)

        ctx.save_for_backward(vrho, vgamma, vtau)

        return rho * exc

    @staticmethod
    def backward(ctx, d_exc):
        vrho, vgamma, vtau = ctx.saved_tensors
        return None, d_exc * vrho, d_exc * vgamma, d_exc * vtau


def eval_xc(
    xc: str, rho: Tensor, gamma: Optional[Tensor] = None, tau: Optional[Tensor] = None
) -> Tensor:

    xc_type = libxc.xc_type(xc)

    if xc_type == "LDA":
        return LDAFunctional.apply(xc, rho)

    elif xc_type == "GGA":
        assert gamma is not None, "GGA XC requires gradients but `gamma=None`."
        return GGAFunctional.apply(xc, rho, gamma)

    elif xc_type == "MGGA":
        assert gamma is not None, "MGGA XC requires gradients but `gamma=None`."
        assert tau is not None, "MGGA XC requires tau but `tau=None`."
        return MGGAFunctional.apply(xc, rho, gamma, tau)

    else:
        raise ValueError(f"Unsupported XC functional type: {xc_type}")
