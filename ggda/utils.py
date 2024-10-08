from typing import Union, Callable, Optional, Sequence, Any
from math import log

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, BoolTensor
from torch import device as Device, dtype as Dtype

Activation = Union[str, Callable[[Tensor], Tensor]]


def to_numpy(x: Tensor, dtype: Any = np.float64):
    return x.detach().cpu().numpy().astype(dtype)


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
