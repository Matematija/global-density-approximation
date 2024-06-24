from typing import Optional
from math import sqrt

import torch
from torch import nn
from torch import Tensor, BoolTensor

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


# def basis_norm(psi: Tensor, weights: Optional[Tensor], eps: float = 1e-5) -> Tensor:

#     psi2 = torch.abs(psi) ** 2

#     if weights is None:
#         norm_squared = torch.sum(psi2, dim=-2, keepdim=True)

#     elif torch.is_floating_point(weights):
#         w = weights.unsqueeze(-1)
#         norm_squared = torch.sum(w * psi2, dim=-2, keepdim=True)

#     else:
#         psi2 = psi2.masked_fill(~weights.unsqueeze(-1), 0.0)
#         norm_squared = torch.sum(psi2, dim=-2, keepdim=True)

#     return psi / torch.sqrt(norm_squared + eps)


def orbital_norm(psi: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:
    norm_squared = torch.einsum("...x,...xi->...i", weights, torch.abs(psi) ** 2)
    return psi / torch.sqrt(norm_squared + eps).unsqueeze(-2)


# def gaussian_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
#     norm = (2 * torch.pi * sigma**2) ** (x.shape[-1] / 2)
#     d2 = torch.sum((x[..., :, None, :] - y[..., None, :, :]) ** 2, dim=-1)
#     return torch.exp(-d2 / (2 * sigma**2)) / norm


# def interpolated_gaussian_kernel(
#     x: Tensor,
#     sigma: float,
#     rank: int,
#     weights: Optional[Tensor] = None,
#     generator: Optional[Generator] = None,
#     **lstsq_kwargs,
# ) -> tuple[Tensor, Tensor]:

#     if weights is None:
#         weights = torch.ones_like(x[..., 0])

#     idx = torch.multinomial(weights, rank, replacement=False, generator=generator)
#     y = torch.take_along_dim(x, idx.unsqueeze(-1), dim=-2)

#     left = gaussian_kernel(y, x, sigma)
#     middle = gaussian_kernel(y, y, sigma)
#     right, *_ = torch.linalg.lstsq(middle, left, **lstsq_kwargs)

#     return left, right


# class FourierPositionalEncoding(nn.Module):
#     def __init__(self, n_modes: int, kernel_scale: float, ndim: int = 3):

#         super().__init__()

#         self.n_modes = n_modes

#         self.mode_lift = nn.Linear(ndim, n_modes)
#         nn.init.normal_(self.mode_lift.weight, std=1 / kernel_scale)
#         nn.init.uniform_(self.mode_lift.bias, a=-torch.pi, b=torch.pi)

#     def forward(self, coords: Tensor) -> Tensor:
#         return sqrt(2 / self.n_modes) * torch.cos(self.mode_lift(coords)).mT


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

        # query, key = self.query_proj(phi), self.key_proj(phi)
        # value = self.value_proj(torch.tanh(phi))

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


# class LinearSelfAttention(nn.Module):
#     def __init__(self, embed_dim: int, kernel_rank: int, kernel_scale: float, bias: bool = True):

#         super().__init__()

#         self.rank, self.scale = kernel_rank, kernel_scale

#         self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#     @staticmethod
#     def _initialize_proj(proj: nn.Linear):

#         embed_dim = proj.weight.shape[-1]
#         delta, sigma = 1 / embed_dim, 1 / embed_dim

#         with torch.no_grad():

#             A, B = torch.empty(2, embed_dim, embed_dim)
#             nn.init.orthogonal_(A, gain=1)
#             nn.init.eye_(B)

#             proj.weight.copy_(sigma * A + delta * B)

#             if proj.bias is not None:
#                 nn.init.zeros_(proj.bias)

#     def forward(
#         self, phi: Tensor, coords: Tensor, weights: Tensor, mask: Optional[BoolTensor] = None
#     ) -> Tensor:

#         if mask is None:
#             mask = weights > 0

#         point_weights = (weights > 0).to(weights.dtype)

#         query, key, value = self.query_proj(phi), self.key_proj(phi), self.value_proj(phi)
#         query, key, value = self._normalize(query, key, value, weights)

#         query_pe, key_pe = interpolated_gaussian_kernel(
#             coords, self.scale, self.rank, weights=point_weights
#         )

#         mask = ~mask.unsqueeze(-1)
#         query = query.masked_fill(mask, 0.0)
#         key = key.masked_fill(mask, 0.0)
#         value = value.masked_fill(mask, 0.0)

#         signature = "...xj, ...mx, ...my, ...yj, ...yi -> ...xi"
#         return torch.einsum(signature, query, query_pe, key_pe, key, value)

#         # query, key = query_pe @ query, key_pe @ key
#         # return query @ (key.mT @ value)
#         # Maybe add activation here in Galerkin case? GeLU(K.T @ V)?
