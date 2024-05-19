from typing import Union, Callable, Optional, Sequence
from math import log

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, BoolTensor
from torch import device as Device, dtype as Dtype

Activation = Union[str, Callable[[Tensor], Tensor]]


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shifted_softplus(x: Tensor) -> Tensor:
    return F.softplus(x) - log(2)


def log_cosh(x: Tensor) -> Tensor:
    return -log(2) - x + F.softplus(2 * x)


def activation_func(activation: Activation) -> Callable[[Tensor], Tensor]:

    if callable(activation):
        return activation

    elif isinstance(activation, str):

        name = activation.strip().lower()

        if name in ["shifted_softplus", "ssp"]:
            return shifted_softplus
        elif name == "log_cosh":
            return log_cosh
        else:
            return getattr(F, activation)

    else:
        raise ValueError("Activation must be a callable or a string!")


def dist(x: Tensor, y: Tensor) -> Tensor:
    return torch.norm(x[..., :, None, :] - y[..., None, :, :], dim=-1)


def sqdist(x: Tensor, y: Tensor) -> Tensor:
    return torch.sum((x[..., :, None, :] - y[..., None, :, :]) ** 2, dim=-1)


def std_scale(
    x: Tensor,
    dims: Union[int, Sequence[int]] = -1,
    mask: Optional[BoolTensor] = None,
    eps: float = 0.0,
) -> Tensor:

    if mask is None:
        var = torch.var(x, dim=dims, keepdim=True)
    else:
        denom = torch.sum(mask, dim=dims, keepdim=True)
        mean = torch.sum(x.masked_fill(~mask, 0), dim=dims, keepdim=True) / denom
        var = torch.sum(((x - mean) ** 2).masked_fill(~mask, 0), dim=dims, keepdim=True) / denom

    return x / torch.sqrt(var + eps)


_ShapeOrInt = Union[int, Sequence[int]]


def cubic_grid(
    npts: _ShapeOrInt,
    stack_dim: int = -1,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:

    nx, ny, nz = (npts,) * 3 if isinstance(npts, int) else npts

    xs = torch.linspace(-1, 1, nx, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, ny, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, nz, device=device, dtype=dtype)

    x, y, z = torch.meshgrid(xs, ys, zs, indexing="ij")

    return torch.stack([x, y, z], dim=stack_dim)


def spherical_grid(
    npts: _ShapeOrInt, device: Optional[Device] = None, dtype: Optional[Dtype] = None
) -> Tensor:

    nr, nt, np = (npts,) * 3 if isinstance(npts, int) else npts

    rs = torch.linspace(0, 1, nr + 1, device=device, dtype=dtype)[1:]
    thetas = torch.linspace(0, torch.pi, nt, device=device, dtype=dtype)
    phis = torch.linspace(0, 2 * torch.pi, np, device=device, dtype=dtype)

    r, t, p = torch.meshgrid(rs, thetas, phis, indexing="ij")

    x = r * torch.sin(t) * torch.cos(p)
    y = r * torch.sin(t) * torch.sin(p)
    z = r * torch.cos(t)

    return torch.stack([x, y, z], dim=-1).view(-1, 3)
