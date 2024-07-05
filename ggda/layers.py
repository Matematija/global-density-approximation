from typing import Optional
from math import sqrt

import torch
from torch import nn
from torch import Tensor

from .utils import Activation, activation_func


class MLP(nn.Module):
    def __init__(
        self,
        features: int,
        enhancement: float,
        activation: Activation = "gelu",
        out_features: Optional[int] = None,
        bias: bool = True,
    ):

        super().__init__()

        self.activation = activation_func(activation)

        in_features = features
        out_features = out_features or features
        width = int(features * enhancement)

        self.in_linear = nn.Linear(in_features, width, bias=bias)
        self.out_linear = nn.Linear(width, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_linear(self.activation(self.in_linear(x)))


class GatedMLP(nn.Module):
    def __init__(
        self,
        features: int,
        enhancement: float,
        activation: Activation = "gelu",
        out_features: Optional[int] = None,
        bias: bool = True,
    ):

        super().__init__()

        self.activation = activation_func(activation)

        in_features = features
        out_features = out_features or features
        width = int(features * enhancement)

        self.in_linear = nn.Linear(in_features, 2 * width, bias=bias)
        self.out_linear = nn.Linear(width, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.in_linear(x).chunk(2, dim=-1)
        return self.out_linear(a * self.activation(b))


class FourierPositionalEncoding(nn.Module):
    def __init__(
        self, embed_dim: int, kernel_scale: float, n_modes: Optional[int] = None, ndim: int = 3
    ):

        super().__init__()

        self.n_modes = n_modes or embed_dim
        assert self.n_modes % 2 == 0, "Embedding dimension must be even"

        self.mode_lift = nn.Linear(ndim, self.n_modes // 2, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=1 / kernel_scale)

        self.linear = nn.Linear(self.n_modes, embed_dim)

    def forward(self, coords: Tensor) -> Tensor:
        phases = self.mode_lift(coords)
        x = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
        return self.linear(x / sqrt(self.n_modes))
