from typing import Optional
from math import log, pi

import torch
from torch import Tensor

from einops import rearrange


def rescaled_grad(rho: Tensor, gamma: Tensor) -> Tensor:
    return gamma / (4 * (3 * torch.pi**2) ** (2 / 3) * rho ** (8 / 3))


def t_weiszacker(rho: Tensor, gamma: Tensor) -> Tensor:
    return gamma / (8 * rho)


def log_t_weiszacker(rho: Tensor, gamma: Tensor, eps: float = 0.0) -> Tensor:
    return torch.log(gamma.clip(min=eps)) - torch.log(8 * rho.clip(min=eps))


def t_thomas_fermi(rho: Tensor) -> Tensor:
    return (3 / 10) * (3 * torch.pi**2) ** (2 / 3) * rho ** (5 / 3)


def log_t_thomas_fermi(rho: Tensor, eps: float = 0.0) -> Tensor:
    log_const = log((3 / 10) * (3 * pi**2) ** (2 / 3))
    return log_const + (5 / 3) * torch.log(rho.clip(min=eps))


def fermi_momentum(rho: Tensor) -> Tensor:
    return (3 * torch.pi**2 * rho) ** (1 / 3)


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


def mean_displacement(wrho: Tensor, coords: Tensor, norm: Optional[Tensor] = None) -> Tensor:

    if norm is None:
        norm = torch.sum(wrho, dim=-1)

    return dipole_moment(wrho, coords, norm=norm)


def covariance_matrix(
    wrho: Tensor, coords: Tensor, mean: Optional[Tensor] = None, norm: Optional[Tensor] = None
) -> Tensor:

    if norm is None:
        norm = torch.sum(wrho, dim=-1)

    if mean is None:
        mean = mean_displacement(wrho, coords, norm=norm)

    r_bar = coords - mean.unsqueeze(-2)

    return torch.einsum("...n,...ni,...nj->...ij", wrho / norm, r_bar, r_bar)


def mean_and_covariance(wrho: Tensor, coords: Tensor) -> Tensor:

    p = wrho / torch.sum(wrho, dim=-1, keepdim=True)

    mean = torch.einsum("...n,...ni->...i", p, coords)
    r_bar = coords - mean.unsqueeze(-2)

    cov = torch.einsum("...n,...ni,...nj->...ij", p, r_bar, r_bar)

    return mean, cov
