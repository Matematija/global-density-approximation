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


def principal_axes_rotation(wrho: Tensor, coords: Tensor, normalize: bool = True) -> Tensor:

    norm = torch.sum(wrho, dim=-1) if normalize else None

    p = dipole_moment(wrho, coords, norm=norm)
    shifted_coords = coords - p.unsqueeze(-2)

    Q = quadrupole_moment(wrho, shifted_coords, norm=norm)
    R = diagonalizing_rotation(Q)

    return Symmetry(p, R)
