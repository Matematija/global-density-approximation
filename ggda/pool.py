from typing import Optional
from math import sqrt, pi

import numpy as np

import torch
from torch import nn
from torch import Tensor

from einops import rearrange

import pykeops
from pykeops.torch import Genred, LazyTensor

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


def _real_ylm_norm(l: int) -> np.ndarray:

    if l == 0:
        x = np.array([1])
    elif l == 1:
        x = np.array([3, 3, 3])
    elif l == 2:
        x = np.array([15, 15, 5 / 4, 15, 15 / 4])
    else:
        raise NotImplementedError(f"l={l} is not implemented.")

    return np.sqrt(x / (4 * np.pi)).astype(np.float32)


class SphericalVectorPool(nn.Module):
    def __init__(self, n_radial_basis: int, max_scale: float):

        super().__init__()

        self.n_radial_basis = n_radial_basis
        self.max_scale = max_scale
        self.n_basis = n_radial_basis * 4

        scales = torch.linspace(0, max_scale, n_radial_basis + 1)[1:]
        self.register_buffer("mu", 1 / scales)

        r_norms = torch.rsqrt(2 * scales**3)
        a_norm_0 = torch.tensor(_real_ylm_norm(0)[0])
        a_norm_1 = torch.tensor(_real_ylm_norm(1)[0])

        self.register_buffer("r_norms", r_norms)
        self.register_buffer("a_norm_0", a_norm_0)
        self.register_buffer("a_norm_1", a_norm_1)

        variables = [f"B = Pm({n_radial_basis})", f"R = Vj(3)", f"r = Vi(3)", "F = Vi(1)"]

        formula_0 = "Exp(-B * Norm2(r - R)) * F"
        self.conv_fn_0 = Genred(formula_0, aliases=variables, reduction_op="Sum", axis=0)

        formula_11 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 0) * F"
        formula_12 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 1) * F"
        formula_13 = "Exp(-B * Norm2(r - R)) * Elem(Normalize(r - R), 2) * F"

        self.conv_fn_11 = Genred(formula_11, aliases=variables, reduction_op="Sum", axis=0)
        self.conv_fn_12 = Genred(formula_12, aliases=variables, reduction_op="Sum", axis=0)
        self.conv_fn_13 = Genred(formula_13, aliases=variables, reduction_op="Sum", axis=0)

    def forward(self, f: Tensor, coords: Tensor, out_coords: Optional[Tensor] = None) -> Tensor:

        if out_coords is None:
            out_coords = coords

        if f.ndim == 1 and coords.ndim == 2 and out_coords.ndim == 2:
            mu = self.mu
        elif f.ndim == 2 and coords.ndim == 3 and out_coords.ndim == 3:
            mu = self.mu[None, ...]
        else:
            raise ValueError(
                f"Incompatible shapes: f {f.shape}, coords {coords.shape}, anchor_coords {out_coords.shape}"
            )

        return torch.cat(
            [
                self.conv_fn_0(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_0),
                self.conv_fn_11(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
                self.conv_fn_12(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
                self.conv_fn_13(mu, out_coords, coords, f) * (self.r_norms * self.a_norm_1),
            ],
            dim=-1,
        )


class HydrogenPool(nn.Module):
    def __init__(self, n_max: float, max_scale: float):

        super().__init__()

        self.n_max = n_max
        self.n_basis = (n_max * (1 + n_max) * (1 + 2 * n_max)) // 6
        self.max_scale = max_scale
        self.bohr = max_scale / n_max

        # fmt: off
        norms_1 = torch.tensor([1])
        norms_2 = 32 * torch.tensor([1, 1, 1, 1])
        norms_3 = 81**2 * torch.tensor([3, 2, 2, 2, 2, 2, 6, 2, 2])
        norms_4 = 512**2 * torch.tensor([9, 5, 5, 5, 3, 3, 36, 3, 12, 72, 3, 120, 180, 120, 12, 72])
        # fmt: on

        self.register_buffer("scales_1", torch.rsqrt(torch.pi * self.bohr**3 * norms_1))
        self.register_buffer("scales_2", torch.rsqrt(torch.pi * self.bohr**3 * norms_2))
        self.register_buffer("scales_3", torch.rsqrt(torch.pi * self.bohr**3 * norms_3))
        self.register_buffer("scales_4", torch.rsqrt(torch.pi * self.bohr**3 * norms_4))

    def _eval_1(self, d: LazyTensor, F: LazyTensor, reduce_dim: int) -> LazyTensor:
        r = d.norm2()
        kernel = (-r).exp()
        return (kernel * F).sum_reduction(dim=reduce_dim) * self.scales_1[0]

    def _eval_2(self, d: LazyTensor, F: LazyTensor, reduce_dim: int) -> LazyTensor:

        x, y, z, r = d[0], d[1], d[2], d.norm2()
        kernel = (-r / 2).exp()

        fconv = [
            (-kernel * (2 - r) * F).sum_reduction(dim=reduce_dim),
            -(kernel * x * F).sum_reduction(dim=reduce_dim),
            (kernel * z * F).sum_reduction(dim=reduce_dim),
            -(kernel * y * F).sum_reduction(dim=reduce_dim),
        ]

        return torch.cat(fconv, dim=-1) * self.scales_2

    def _eval_3(self, d: LazyTensor, F: LazyTensor, reduce_dim: int) -> LazyTensor:

        x, y, z, r = d[0], d[1], d[2], d.norm2()
        kernel = (-r / 3).exp()

        fconv = [
            (kernel * (27 + 2 * (-9 + r) * r) * F).sum_reduction(dim=reduce_dim),
            (kernel * (-6 + r) * x * F).sum_reduction(dim=reduce_dim),
            -(kernel * (-6 + r) * z * F).sum_reduction(dim=reduce_dim),
            (kernel * (-6 + r) * y * F).sum_reduction(dim=reduce_dim),
            (kernel * x * y * F).sum_reduction(dim=reduce_dim),
            -(kernel * x * z * F).sum_reduction(dim=reduce_dim),
            (kernel * (2 * r**2 - 3 * (x**2 + y**2)) * F).sum_reduction(dim=reduce_dim),
            -(kernel * y * z * F).sum_reduction(dim=reduce_dim),
            (kernel * (-(x**2) + y**2) * F).sum_reduction(dim=reduce_dim),
        ]

        return torch.cat(fconv, dim=-1) * self.scales_3

    def _eval_4(self, d: LazyTensor, F: LazyTensor, reduce_dim: int) -> LazyTensor:

        x, y, z, r = d[0], d[1], d[2], d.norm2()
        kernel = (-r / 4).exp()

        # fmt: off
        fconv = [
            (kernel * (192 - (-12 + r) ** 2 * r) * F).sum_reduction(dim=reduce_dim),
            -(kernel * (80 + (-20 + r) * r) * x * F).sum_reduction(dim=reduce_dim),
            (kernel * (80 + (-20 + r) * r) * z * F).sum_reduction(dim=reduce_dim),
            -(kernel * (80 + (-20 + r) * r) * y * F).sum_reduction(dim=reduce_dim),
            -(kernel * (-12 + r) * x * y * F).sum_reduction(dim=reduce_dim),
            (kernel * (-12 + r) * x * z * F).sum_reduction(dim=reduce_dim),
            -(kernel * (-12 + r) * (2 * r**2 - 3 * (x**2 + y**2)) * F).sum_reduction(dim=reduce_dim),
            (kernel * (-12 + r) * y * z * F).sum_reduction(dim=reduce_dim),
            (kernel * (-12 + r) * (x - y) * (x + y) * F).sum_reduction(dim=reduce_dim),
            (kernel * x * (x**2 - 3 * y**2) * F).sum_reduction(dim=reduce_dim),
            (kernel * x * y * z * F).sum_reduction(dim=reduce_dim),
            (kernel * x * (-4 * r**2 + 5 * (x**2 + y**2)) * F).sum_reduction(dim=reduce_dim),
            (kernel * (-3 * (x**2 + y**2) * z + 2 * z**3) * F).sum_reduction(dim=reduce_dim),
            (kernel * y * (-4 * r**2 + 5 * (x**2 + y**2)) * F).sum_reduction(dim=reduce_dim),
            (kernel * (-(x**2) + y**2) * z * F).sum_reduction(dim=reduce_dim),
            -(kernel * y * (-3 * x**2 + y**2) * F).sum_reduction(dim=reduce_dim),
        ]
        # fmt: on

        return torch.cat(fconv, dim=-1) * self.scales_4

    def forward(self, f: Tensor, coords: Tensor, out_coords: Optional[Tensor] = None) -> Tensor:

        if out_coords is None:
            out_coords = coords

        reduce_dim = (f.ndim - 1) + 1

        # fmt: off
        F = LazyTensor(rearrange(                     f, "... i   -> ... 1 i 1").contiguous())
        r = LazyTensor(rearrange(    coords / self.bohr, "... i c -> ... 1 i c").contiguous())
        R = LazyTensor(rearrange(out_coords / self.bohr, "... a c -> ... a 1 c").contiguous())
        # fmt: on

        d = R - r

        fconv = [self._eval_1(d, F, reduce_dim)]

        if self.n_max >= 2:
            fconv.append(self._eval_2(d, F, reduce_dim))

        if self.n_max >= 3:
            fconv.append(self._eval_3(d, F, reduce_dim))

        if self.n_max >= 4:
            fconv.append(self._eval_4(d, F, reduce_dim))

        if self.n_max >= 5:
            raise NotImplementedError("n_max >= 5 is not implemented.")

        return torch.cat(fconv, dim=-1)
