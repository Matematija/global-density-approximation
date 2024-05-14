import torch
from torch import nn
from torch import Tensor
from pykeops.torch import Genred


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


class DensityPooling(nn.Module):
    def __init__(self, embed_dim: int, n_basis: int, max_std: float = 4, eps: float = 1e-4):

        super().__init__()

        self.eps = eps
        self.gaussian_pool = GaussianPool(n_basis, max_std)
        self.lift = nn.Linear(n_basis, embed_dim)

    def forward(self, wrho: Tensor, coords: Tensor, anchor_coords: Tensor) -> Tensor:
        pooled_rho = self.gaussian_pool(wrho, coords, anchor_coords)
        phi = torch.log(pooled_rho + self.eps)
        return self.lift(phi)
