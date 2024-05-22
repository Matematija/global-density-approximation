from typing import Optional
from math import sqrt, pi

import numpy as np

import torch
from torch import nn
from torch import Tensor

from einops import rearrange

import pykeops
from pykeops.torch import Genred, LazyTensor

pykeops.set_verbose(False)


class GaussianPool(nn.Module):
    def __init__(self, n_basis: int, max_std: float, ndim: int = 3):

        super().__init__()

        self.n_basis = n_basis
        self.max_std = max_std

        sigmas = torch.linspace(0, max_std, n_basis + 1)[1:]
        self.register_buffer("gammas", 1 / (2 * sigmas**2))

        norms = (2 * torch.pi * sigmas**2) ** (ndim / 2)
        self.register_buffer("norms", norms)

        formula = "Exp(-B * SqDist(X,Y)) * F"
        variables = [f"B = Pm({n_basis})", f"X = Vi({ndim})", f"Y = Vj({ndim})", "F = Vj(1)"]

        self.conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def forward(self, f: Tensor, coords: Tensor, anchor_coords: Tensor, *args, **kwargs) -> Tensor:

        if f.ndim == 1 and coords.ndim == 2 and anchor_coords.ndim == 2:
            fconv = self.conv_fn(self.gammas, anchor_coords, coords, f, *args, **kwargs)
        elif f.ndim == 2 and coords.ndim == 3 and anchor_coords.ndim == 3:
            fconv = self.conv_fn(self.gammas[None, ...], anchor_coords, coords, f, *args, **kwargs)
        else:
            raise ValueError(
                f"Incompatible shapes: f {f.shape}, coords {coords.shape}, anchor_coords {anchor_coords.shape}"
            )

        return fconv / self.norms


class ExponentialPool(nn.Module):
    def __init__(self, n_basis: int, max_scale: float):

        super().__init__()

        self.n_basis = n_basis
        self.max_scale = max_scale

        scales = torch.linspace(0, max_scale, n_basis + 1)[1:]
        self.register_buffer("mu", 1 / scales)

        norms = 8 * torch.pi * scales**3
        self.register_buffer("norms", norms)

        formula = "Exp(-B * Norm2(X-Y)) * F"
        variables = [f"B = Pm({n_basis})", f"X = Vi(3)", f"Y = Vj(3)", "F = Vj(1)"]

        self.conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def forward(self, f: Tensor, coords: Tensor, anchor_coords: Tensor, *args, **kwargs) -> Tensor:

        if f.ndim == 1 and coords.ndim == 2 and anchor_coords.ndim == 2:
            fconv = self.conv_fn(self.mu, anchor_coords, coords, f, *args, **kwargs)
        elif f.ndim == 2 and coords.ndim == 3 and anchor_coords.ndim == 3:
            fconv = self.conv_fn(self.mu[None, ...], anchor_coords, coords, f, *args, **kwargs)
        else:
            raise ValueError(
                f"Incompatible shapes: f {f.shape}, coords {coords.shape}, anchor_coords {anchor_coords.shape}"
            )

        return fconv / self.norms


def _real_ylm_norm(l: int) -> np.ndarray:

    if l == 0:
        x = np.array([1])
    elif l == 1:
        x = np.array([3, 3, 3])
    elif l == 2:
        x = np.array([15, 15, 5 / 4, 15, 15 / 4])
    else:
        raise NotImplementedError(f"l={l} is not implemented.")

    return np.sqrt(x / (4 * np.pi)).astype(np.float32)


class SphericalVectorPool(nn.Module):
    def __init__(self, n_radial_basis: int, max_scale: float):

        super().__init__()

        self.n_radial_basis = n_radial_basis
        self.max_scale = max_scale
        self.n_basis = n_radial_basis * 4

        scales = torch.linspace(0, max_scale, n_radial_basis + 1)[1:]
        self.register_buffer("mu", 1 / scales)

        r_norms = torch.rsqrt(2 * scales**3)
        a_norm_0 = torch.tensor(_real_ylm_norm(0)[0])
        a_norm_1 = torch.tensor(_real_ylm_norm(1)[0])

        self.register_buffer("r_norms", r_norms)
        self.register_buffer("a_norm_0", a_norm_0)
        self.register_buffer("a_norm_1", a_norm_1)

        variables = [f"B = Pm({n_radial_basis})", f"R = Vj(3)", f"r = Vi(3)", "F = Vi(1)"]

        formula_0 = "Exp(-B * Norm2(r - R)) * F"
        self.conv_fn_0 = Genred(formula_0, aliases=variables, reduction_op="Sum", axis=0)

        formula_11 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 0) * F"
        formula_12 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 1) * F"
        formula_13 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 2) * F"

        self.conv_fn_11 = Genred(formula_11, aliases=variables, reduction_op="Sum", axis=0)
        self.conv_fn_12 = Genred(formula_12, aliases=variables, reduction_op="Sum", axis=0)
        self.conv_fn_13 = Genred(formula_13, aliases=variables, reduction_op="Sum", axis=0)

    def forward(self, f: Tensor, coords: Tensor, out_coords: Optional[Tensor] = None) -> Tensor:

        if out_coords is None:
            out_coords = coords

        if f.ndim == 1 and coords.ndim == 2 and out_coords.ndim == 2:
            mu = self.mu
        elif f.ndim == 2 and coords.ndim == 3 and out_coords.ndim == 3:
            mu = self.mu[None, ...]
        else:
            raise ValueError(
                f"Incompatible shapes: f {f.shape}, coords {coords.shape}, anchor_coords {out_coords.shape}"
            )

        return torch.cat(
            [
                self.conv_fn_0(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_0),
                self.conv_fn_11(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
                self.conv_fn_12(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
                self.conv_fn_13(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
            ],
            dim=-1,
        )
