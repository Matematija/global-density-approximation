from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .features import ttf

import pykeops
from pykeops.torch import Genred

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


class CiderFeatures(nn.Module):
    def __init__(self, ndim: int = 3):

        super().__init__()

        self.ndim = ndim
        self.params = nn.Parameter(2 * torch.ones(2))

        formula = "Exp(- (a+b) * SqDist(R,r) ) * F"
        variables = ["a = Vj(1)", "b = Vj(3)", f"R = Vi({ndim})", f"r = Vj({ndim})", "F = Vj(1)"]
        self.conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def forward(
        self,
        rho: Tensor,
        gamma: Tensor,
        coords: Tensor,
        weights: Tensor,
        out_coords: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # convention: gamma = ( \nabla n ) ^2

        if out_coords is None:
            out_coords = coords

        A, D = self.params

        B2, C2 = A, (6 * torch.pi**2) ** (2 / 3) * (6 * A / (160 * torch.pi))
        B1, C1 = B2 / 2, C2 / 2
        B3, C3 = 2 * B2, 2 * C2
        B0, C0 = (D / A) * B2, (D / A) * C2

        Bs = torch.stack([B0, B1, B2, B3], dim=-1)
        Cs = torch.stack([C0, C1, C2, C3], dim=-1)

        rho_, gamma_ = rho.unsqueeze(-1), gamma.unsqueeze(-1)
        ab = torch.pi * (rho_ / 2) ** (2 / 3) * (Bs + Cs * gamma_ / (8 * rho_ * ttf(rho_)))
        ab = torch.broadcast_to(ab, coords.shape[:-1] + (4,))
        a, b = torch.split(ab, [1, 3], dim=-1)

        norms = ((Bs[..., 0] + Bs[..., 1:]) / 2) ** (self.ndim / 2)
        G = self.conv_fn(a, b, out_coords, coords, weights * rho, *args, **kwargs) * norms

        return (G / (2 + G)) - 0.5
