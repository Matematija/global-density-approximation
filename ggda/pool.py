import torch
from torch import nn
from torch import Tensor

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
