from typing import Optional

import torch
from torch import nn
from torch import Tensor

import pykeops
from pykeops.torch import Genred

from .utils import log_cosh
from .features import rescaled_grad

pykeops.set_verbose(False)


class CoarseGraining(nn.Module):
    def __init__(self, n_basis: int, enhancement: float = 2.0, eps: float = 1e-4):

        super().__init__()

        self.n_basis = n_basis
        self.eps = eps
        width = int(n_basis * enhancement)

        self.field_embed = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, n_basis))

        self.register_buffer("x0", torch.tensor([0.0]))

        formula = "Exp(-B * SqDist(R,r)) * F"
        variables = [f"B = Vj({n_basis})", f"R = Vi(3)", f"r = Vj(3)", "F = Vj(1)"]
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

        x = rescaled_grad(rho, gamma)
        x = torch.log(x + self.eps).unsqueeze(-1)

        exponent = log_cosh(self.field_embed(x))
        heg_scale = log_cosh(self.field_embed(self.x0)) ** (3 / 2)

        beta = torch.pi * (rho.unsqueeze(-1) / 2) ** (2 / 3) * exponent
        beta = torch.broadcast_to(beta, rho.shape + (self.n_basis,))

        coords, out_coords = coords.contiguous(), out_coords.contiguous()
        wrho = (weights * rho).contiguous()

        y = self.conv_fn(beta, out_coords, coords, wrho, *args, **kwargs)

        return y * heg_scale
