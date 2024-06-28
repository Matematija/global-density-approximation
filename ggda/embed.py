from math import pi

import torch

from torch import nn
from torch import Tensor

from pykeops.torch import Genred

from .features import t_weisacker, t_thomas_fermi


class CiderFeatures(nn.Module):
    def __init__(self, A: float = 2.0, D: float = 2.0):

        super().__init__()

        B2, C2 = A, (6 * pi**2) ** (2 / 3) * (6 * A / (160 * pi))
        B1, C1 = B2 / 2, C2 / 2
        B3, C3 = 2 * B2, 2 * C2
        B0, C0 = (D / A) * B2, (D / A) * C2

        Bs = torch.tensor([B0, B1, B2, B3], dtype=torch.float32)
        Cs = torch.tensor([C0, C1, C2, C3], dtype=torch.float32)
        norms = ((Bs[..., 0] + Bs[..., 1:]) / 2) ** (3 / 2)

        self.register_buffer("B", Bs)
        self.register_buffer("C", Cs)
        self.register_buffer("norms", norms)

        formula = "Exp(- (a+b) * SqDist(x,y) ) * f"
        variables = ["a = Vj(1)", "b = Vi(3)", f"x = Vi(3)", f"y = Vj(3)", "f = Vj(1)"]
        self.conv_fn = Genred(formula, aliases=variables, reduction_op="Sum", axis=1)

    def forward(
        self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:

        # convention: gamma = ( \nabla n ) ^2

        rho_ = (rho + 1e-8).unsqueeze(-1)
        gamma_ = gamma.unsqueeze(-1)
        x = t_weisacker(rho_, gamma_) / t_thomas_fermi(rho_)

        ab = torch.pi * (rho_ / 2) ** (2 / 3) * (self.B + self.C * x)
        ab = torch.broadcast_to(ab, coords.shape[:-1] + (4,))
        a, b = torch.split(ab, [1, 3], dim=-1)

        a, b = a.contiguous(), b.contiguous()
        coords = coords.contiguous()
        wrho = (weights * rho).contiguous()

        device_id = wrho.device.index or -1
        y = self.conv_fn(a, b, coords, coords, wrho, device_id=device_id, *args, **kwargs)

        return y * self.norms
