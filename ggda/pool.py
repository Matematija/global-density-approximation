from typing import Optional

import torch
from torch import nn
from torch import Tensor

import pykeops
from pykeops.torch import Genred

from .utils import log_cosh

pykeops.set_verbose(False)


class GaussianPool(nn.Module):
    def __init__(self, n_gaussians: int, max_std: float, ndim: int = 3):

        super().__init__()

        self.n_gaussians = n_gaussians
        self.max_std = max_std

        sigmas = torch.linspace(0, max_std, n_gaussians + 1)[1:]
        self.register_buffer("gammas", 1 / (2 * sigmas**2))

        norms = (2 * torch.pi * sigmas**2) ** (ndim / 2)
        self.register_buffer("norms", norms)

        formula = "Exp(-B * SqDist(X,Y)) * F"
        variables = [f"B = Pm({n_gaussians})", f"X = Vi({ndim})", f"Y = Vj({ndim})", "F = Vj(1)"]

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


class WeightedGaussianPool(nn.Module):
    def __init__(self, n_basis: int, cutoff: float, scale: float = 1.0, ndim: int = 3):

        super().__init__()

        self.n_basis = n_basis
        self.cutoff = cutoff

        means = torch.linspace(0, cutoff, n_basis) / cutoff
        self.register_buffer("means", means)

        self.register_buffer("beta", torch.tensor((scale * self.n_basis) ** 2))
        self.register_buffer("pi_half", torch.tensor(torch.pi / 2))

        formula = (
            "Cos(C * Norm2(R - r)) * Step(1 - Norm2(R - r)) * Exp(-B * Square(Norm2(R -r) - M)) * F"
        )

        variables = [
            "C = Pm(1)",
            "B = Pm(1)",
            f"M = Pm({n_basis})",
            f"R = Vi({ndim})",
            f"r = Vj({ndim})",
            "F = Vj(1)",
        ]

        self._conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def conv_fn(self, f: Tensor, coords: Tensor, out_coords: Tensor, *args, **kwargs) -> Tensor:

        batch_dims = f.shape[:-1]
        pi_half = torch.broadcast_to(self.pi_half, batch_dims + (1,))
        beta = torch.broadcast_to(self.beta, batch_dims + (1,))
        means = torch.broadcast_to(self.means, batch_dims + (self.n_basis,))

        return self._conv_fn(pi_half, beta, means, out_coords, coords, f, *args, **kwargs)

    def forward(
        self, f: Tensor, coords: Tensor, out_coords: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:

        if out_coords is None:
            out_coords = coords

        coords = coords / self.cutoff
        out_coords = out_coords / self.cutoff

        return self.conv_fn(f, coords, out_coords, *args, **kwargs)


class RangedCoulombPool(nn.Module):
    def __init__(self, n_basis: int, length_scale: float, ndim: int = 3):

        super().__init__()

        self.n_basis = n_basis

        kmax = 2 * torch.pi / length_scale
        self.k = nn.Parameter(torch.linspace(0, kmax, n_basis))

        formula = "S * SinXDivX(S * Norm2(R - r)) * F"
        variables = [f"S = Pm({n_basis})", f"R = Vi({ndim})", f"r = Vj({ndim})", "F = Vj(1)"]
        self.conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def forward(
        self, f: Tensor, coords: Tensor, out_coords: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:

        if out_coords is None:
            out_coords = coords

        k = torch.broadcast_to(log_cosh(self.k) + 1e-5, f.shape[:-1] + (self.n_basis,))
        return self.conv_fn(k, out_coords, coords, f, *args, **kwargs)
