from typing import Optional
from math import sqrt

import torch
from torch import nn
from torch import Tensor

from .features import mean_and_covariance
from .utils import Activation, activation_func


class MLP(nn.Module):
    def __init__(
        self,
        features: int,
        enhancement: float,
        activation: Activation = "silu",
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
        activation: Activation = "silu",
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


def grid_norm(f: Tensor, wrho: Tensor, eps: float = 1e-5) -> Tensor:

    N = wrho.sum(dim=-1, keepdim=True)
    p = torch.unsqueeze(wrho / N, dim=-1)

    mean = torch.sum(p * f, dim=-2, keepdim=True)
    var = torch.sum(p * (f - mean) ** 2, dim=-2, keepdim=True)

    return (f - mean) / torch.sqrt(var + eps)


def orbital_norm(psi: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:
    norm_squared = torch.einsum("...x,...xi->...i", weights, torch.abs(psi) ** 2)
    return psi / torch.sqrt(norm_squared + eps).unsqueeze(-2)


def coordinate_norm(coords: Tensor, weights: Tensor) -> Tensor:

    means, covs = mean_and_covariance(weights.detach(), coords)
    s2, R = torch.linalg.eigh(covs)

    coords_ = (coords - means.unsqueeze(-2)) @ R.mT.detach()
    return coords_ / torch.sqrt(s2 + 1e-5).unsqueeze(-2)


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


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, kernel_scale: float, ndim: int = 3):

        super().__init__()

        assert embed_dim % 2 == 0, "Embedding dimension must be even"

        self.mode_lift = nn.Linear(ndim, embed_dim // 2, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=1 / kernel_scale)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:

        phases = self.mode_lift(coords)
        c, s = torch.cos(phases), torch.sin(phases)
        a, b = x.chunk(2, dim=-1)

        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, kernel_scale: float, bias: bool = True):

        super().__init__()

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.positional_encoding = RotaryPositionalEncoding(embed_dim, kernel_scale)

    @staticmethod
    def _initialize_proj(proj: nn.Linear):

        embed_dim = proj.weight.shape[-1]
        delta, sigma = 1 / embed_dim, 1 / embed_dim

        with torch.no_grad():

            A, B = torch.zeros(2, embed_dim, embed_dim)
            nn.init.orthogonal_(A, gain=sigma)
            nn.init.eye_(B)

            proj.weight.copy_(A + delta * B)

            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        query, key, value = self.query_proj(phi), self.key_proj(phi), self.value_proj(phi)
        query, key, value = self._normalize(query, key, value, weights)

        query = self.positional_encoding(query, coords)
        key = self.positional_encoding(key, coords)

        return query @ (key.mT @ value)


class GalerkinAttention(LinearSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_proj(self.query_proj)

    def _normalize(self, query, key, value, weights):

        key = orbital_norm(key, weights)
        value = orbital_norm(value, weights) * weights.unsqueeze(-1)

        return query, key, value


class FourierAttention(LinearSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_proj(self.value_proj)

    def _normalize(self, query, key, value, weights):

        query = orbital_norm(query, weights)
        key = orbital_norm(key, weights)
        value = value * weights.unsqueeze(-1)

        return query, key, value
