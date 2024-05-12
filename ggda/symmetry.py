from typing import NamedTuple

import torch
from torch import Tensor

from .features import dipole_moment, quadrupole_moment


class Symmetry(NamedTuple):

    translation: Tensor
    rotation: Tensor

    def __call__(self, vector: Tensor) -> Tensor:
        return (vector - self.translation) @ self.rotation


def diagonalizing_rotation(Q: Tensor) -> Tensor:
    _, R = torch.linalg.eigh(Q)
    R, _ = torch.linalg.qr(R)
    return R


def principal_axes_rotation(
    rho: Tensor, coords: Tensor, weights: Tensor, normalize: bool = True
) -> Tensor:

    norm = torch.einsum("...n,...n->...", weights, rho) if normalize else None

    p = dipole_moment(rho, coords, weights, norm)
    shifted_coords = coords - p.unsqueeze(-2)

    Q = quadrupole_moment(rho, shifted_coords, weights, norm=norm)
    R = diagonalizing_rotation(Q)

    return Symmetry(p, R)
