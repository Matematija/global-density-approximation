from typing import Optional

import numpy as np

import torch
from torch.autograd import Function
from torch import Tensor

from einops import rearrange

from pyscf.dft import libxc


def to_numpy(x, dtype=np.float64):
    return x.cpu().numpy().astype(dtype)


def _fake_grad_rho(gamma, dim=0):
    grad_x, grad_pad = torch.sqrt(gamma), torch.zeros_like(gamma)
    return torch.stack([grad_x, grad_pad, grad_pad], dim=dim)


def extract_rho(rho_data, xc_type):

    has_spin = rho_data.ndim >= 2 and rho_data.shape[0] == 2

    if not has_spin:

        if xc_type == "LDA" or xc_type == "HF":
            return rho_data
        elif xc_type == "GGA" or xc_type == "MGGA":
            return rho_data[0, :]
        else:
            raise KeyError(f"Unsupported XC type: {xc_type}")

    else:

        if xc_type == "LDA" or xc_type == "HF":
            return rho_data.sum(0)
        elif xc_type == "GGA" or xc_type == "MGGA":
            return rho_data[:, 0, :].sum(0)
        else:
            raise KeyError(f"Unsupported XC type: {xc_type}")


class LibXCEnergy(Function):
    @staticmethod
    def forward(ctx, xc, rho_data):

        dtype, device = rho_data.dtype, rho_data.device

        ctx.save_for_backward(rho_data)
        ctx.xc = xc

        rho_data_ = to_numpy(rho_data)
        exc = libxc.eval_xc_eff(xc, rho_data_, deriv=0)
        rho = extract_rho(rho_data_, libxc.xc_type(xc))

        return torch.tensor(rho * exc, dtype=dtype, device=device)

    @staticmethod
    def vjp(ctx, d_exc):
        (rho_data,) = ctx.saved_tensors
        vxc = LibXCPotential.apply(ctx.xc, rho_data)
        return None, d_exc * vxc


class LibXCPotential(Function):
    @staticmethod
    def forward(ctx, xc, rho_data):

        dtype, device = rho_data.dtype, rho_data.device

        ctx.save_for_backward(rho_data)
        ctx.xc = xc

        vxc = libxc.eval_xc_eff(xc, to_numpy(rho_data), deriv=1)
        return torch.tensor(vxc, dtype=dtype, device=device)

    @staticmethod
    def vjp(ctx, d_vxc):

        dtype, device = d_vxc.dtype, d_vxc.device
        (rho_data,) = ctx.saved_tensors

        fxc = libxc.eval_xc_eff(ctx.xc, to_numpy(rho_data), deriv=2)
        fxc = torch.tensor(fxc, dtype=dtype, device=device)

        has_spin = rho_data.ndim >= 2 and rho_data.shape[0] == 2

        if not has_spin:
            out_grad = torch.einsum("ix,ijx->jx", d_vxc, fxc)
        else:
            out_grad = torch.einsum("aix,aibjx->bjx", d_vxc, fxc)

        return None, out_grad


def stack_rho_data_rks(
    rho: Tensor, gamma: Optional[Tensor] = None, tau: Optional[Tensor] = None
) -> tuple[Tensor, tuple[int]]:

    batch_shape = rho.shape
    rho_data = rho.ravel()

    if gamma is not None:
        grad_rho_ = _fake_grad_rho(gamma.ravel(), dim=0)
        rho_data = torch.cat([rho_data.unsqueeze(0), grad_rho_], dim=0)

    if tau is not None:
        assert gamma is not None, "Gamma must be provided for MGGAs."
        rho_data = torch.cat([rho_data, tau.ravel().unsqueeze(0)], dim=0)

    return rho_data, batch_shape


def stack_rho_data_uks(
    rho: Tensor, gamma: Optional[Tensor] = None, tau: Optional[Tensor] = None
) -> tuple[Tensor, tuple[int]]:

    batch_shape = rho.shape[:-1]
    rho_data = rearrange(rho, "... s -> s (...)")

    if gamma is not None:
        gamma_ = rearrange(gamma, "... s -> s (...)")
        grad_rho_ = _fake_grad_rho(gamma_, dim=1)
        rho_data = torch.cat([rho_data.unsqueeze(1), grad_rho_], dim=1)

    if tau is not None:
        assert gamma is not None, "Gamma must be provided for MGGAs."
        tau_ = rearrange(tau, "... s -> s 1 (...)")
        rho_data = torch.cat([rho_data, tau_], dim=1)

    return rho_data, batch_shape


def eval_xc(
    xc: str,
    rho: Tensor,
    gamma: Optional[Tensor] = None,
    tau: Optional[Tensor] = None,
    *,
    spin: int = 0,
) -> Tensor:

    if not spin:
        rho_data, batch_shape = stack_rho_data_rks(rho, gamma, tau)
    else:
        rho_data, batch_shape = stack_rho_data_uks(rho, gamma, tau)

    return LibXCEnergy.apply(xc, rho_data).view(batch_shape)
