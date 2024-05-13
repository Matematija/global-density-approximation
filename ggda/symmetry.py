from typing import NamedTuple

import torch
from torch import Tensor, device as Device, dtype as Dtype

from .features import dipole_moment, quadrupole_moment


class Symmetry(NamedTuple):

    translation: Tensor
    rotation: Tensor

    def __call__(self, vector: Tensor) -> Tensor:
        return (vector - self.translation) @ self.rotation


# def diagonalizing_rotation(Q: Tensor) -> Tensor:
#     m, R = torch.linalg.eigh(Q)
#     R, s = torch.linalg.qr(R)
#     return R, m * s.diagonal(dim1=-2, dim2=-1)


# def principal_axes_rotation(wrho: Tensor, coords: Tensor, normalize: bool = True) -> Tensor:

#     norm = torch.sum(wrho, dim=-1) if normalize else None

#     p = dipole_moment(wrho, coords, norm=norm)
#     shifted_coords = coords - p.unsqueeze(-2)

#     Q = quadrupole_moment(wrho, shifted_coords, norm=norm)
#     R, m = diagonalizing_rotation(Q)

#     return Symmetry(p, R), m


def principal_axes_rotation(wrho: Tensor, coords: Tensor) -> Tensor:

    p = wrho / torch.sum(wrho, dim=-1, keepdim=True)

    mean = torch.einsum("...n,...ni->...i", p, coords)
    r_bar = coords - mean.unsqueeze(-2)

    cov = torch.einsum("...n,...ni,...nj->...ij", p, r_bar, r_bar)
    s2, R = torch.linalg.eigh(cov)

    return Symmetry(mean, R), s2
