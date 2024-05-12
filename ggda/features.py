from typing import Optional

import torch
from torch import Tensor

from einops import rearrange


def ttf(rho: Tensor) -> Tensor:
    return (3 / 10) * (3 * torch.pi**2 * rho) ** (2 / 3)


def lda_x(rho: Tensor) -> Tensor:
    return -(3 / 4) * (3 * rho / torch.pi) ** (1 / 3)


def lda_c_pz(rho: Tensor) -> Tensor:

    # Perdew-Zunger LDA correlation
    # J. P. Perdew and A. Zunger., Phys. Rev. B 23, 5048 (1981) (doi: 10.1103/PhysRevB.23.5048)

    # Ceperley-Alder
    gamma, beta_1, beta_2 = -0.1423, 1.0529, 0.3334
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116

    rs = (3 / (4 * torch.pi * rho)) ** (1 / 3)
    lnrs = torch.log(rs)

    low_density = gamma / (1 + beta_1 * torch.sqrt(rs) + beta_2 * rs)
    high_density = A * lnrs + B + C * rs * lnrs + D * rs

    return torch.where(rs < 1, high_density, low_density)


def lda_c(rho):

    # Vosko-Wilk-Nusair LDA correlation
    # S. H. Vosko, L. Wilk, and M. Nusair., Can. J. Phys. 58, 1200 (1980) (doi: 10.1139/p80-159)

    A, x0, b, c = 0.0310907, -0.10498, 3.72744, 12.9352

    x = (3 / (4 * torch.pi * rho)) ** (1 / 6)  # sqrt(rs)
    X = x**2 + b * x + c
    X0 = x0**2 + b * x0 + c
    Q = (4 * c - b**2) ** 0.5

    atan = torch.atan(Q / (2 * x + b))

    term1 = torch.log(x**2 / X) + (2 * b / Q) * atan
    term2 = -(b * x0 / X0) * (torch.log((x - x0) ** 2 / X) + (2 * (b + 2 * x0) / Q) * atan)

    return A * (term1 + term2)


def lda_xc(rho: Tensor) -> Tensor:
    return torch.stack([lda_x(rho), lda_c(rho)], dim=-1)


def lda(rho: Tensor) -> Tensor:
    return lda_x(rho) + lda_c(rho)


def dipole_moment(wrho: Tensor, coords: Tensor, norm: Optional[Tensor] = None) -> Tensor:

    if norm is not None:
        wrho = wrho / norm.unsqueeze(-1)

    return torch.einsum("...n,...ni->...i", wrho, coords)


def quadrupole_moment(
    wrho: Tensor, coords: Tensor, traceless: bool = True, norm: Optional[Tensor] = None
) -> Tensor:

    if norm is not None:
        wrho = wrho / norm.unsqueeze(-1)

    ndim = coords.shape[-1]
    Q = ndim * torch.einsum("...n,...ni,...nj->...ij", wrho, coords, coords)

    if traceless:
        trace = torch.einsum("...ii->...", Q / ndim)
        Q = Q - rearrange(trace, "... -> ... 1 1")

    return Q
