from typing import Optional

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
        n_hidden: int = 0,
        out_features: Optional[int] = None,
        bias: bool = True,
    ):

        super().__init__()

        self.activation = activation_func(activation)

        in_features = features
        out_features = out_features or features
        width = int(features * enhancement)

        self.in_linear = nn.Linear(in_features, width, bias=bias)
        self.hidden = nn.ModuleList([nn.Linear(width, width, bias=bias) for _ in range(n_hidden)])
        self.out_linear = nn.Linear(width, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:

        h = self.activation(self.in_linear(x))

        for linear in self.hidden:
            h = self.activation(linear(h))

        return self.out_linear(h)


class CoordinateEncoding(nn.Module):
    def __init__(self, embed_dim: int, init_std: float, ndim: int = 3):

        super().__init__()

        self.mode_lift = nn.Linear(ndim, embed_dim // 2, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=2 * torch.pi * init_std)

    def forward(self, coords: Tensor) -> Tensor:
        emb = self.mode_lift(coords)
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class InstanceNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5):

        super().__init__()

        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, f: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        if weights is None:
            mean = f.mean(dim=-2, keepdim=True)
            var = f.var(dim=-2, keepdim=True)
        else:
            mean = torch.einsum("...x,...xi->...i", weights, f).unsqueeze(-2)
            var = torch.einsum("...x,...xi->...i", weights, (f - mean) ** 2).unsqueeze(-2)

        f_bar = (f - mean) / torch.sqrt(var + self.eps)
        return f_bar * self.scale + self.bias


class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, enhancement: float = 1.0, bias: bool = False):

        super().__init__()

        self.inner_dim = int(embed_dim * enhancement)

        self.query_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=bias)

        self.key_norm = InstanceNorm(self.inner_dim)
        self.value_norm = InstanceNorm(self.inner_dim)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, *, weights: Optional[Tensor] = None
    ) -> Tensor:

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        key = self.key_norm(key, weights)
        value = self.value_norm(value, weights)

        norm = query.size(-2) ** 0.5

        if weights is not None:

            if weights.dtype == torch.bool:
                mask = ~weights.unsqueeze(-1)
                key = key.masked_fill(mask, 0.0) / norm
                value = value.masked_fill(mask, 0.0) / norm
            else:
                key = key * weights.unsqueeze(-1)

        else:
            key = key / norm
            value = value / norm

        return self.out_proj(query @ (key.mT @ value))
