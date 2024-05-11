from typing import Union, Callable
from math import log

from torch import nn
from torch.nn import functional as F
from torch import Tensor

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
