from typing import Optional

import torch
from torch import nn
from torch import Tensor, BoolTensor

from einops import rearrange

from .utils import Activation, activation_func, log_cosh


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
    def __init__(self, embed_dim: int, init_std: float, n_modes: int, ndim: int = 3):

        super().__init__()

        assert embed_dim >= 2, "Embedding dimension must be at least 2."
        assert embed_dim % 2 == 0, "Embedding dimension must be even."

        self.n_modes = n_modes

        self.mode_lift = nn.Linear(ndim, n_modes, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=2 * torch.pi * init_std)

        self.proj = nn.Linear(2 * n_modes, embed_dim, bias=False)

    def forward(self, coords: Tensor) -> Tensor:
        emb = self.mode_lift(coords)
        x = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return self.proj(x)


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


class ProximalAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: Optional[int] = None, bias: bool = False):

        super().__init__()

        self.n_heads = n_heads or max(embed_dim // 64, 1)

        self.attention = nn.MultiheadAttention(
            embed_dim, self.n_heads, bias=bias, dropout=False, batch_first=True
        )

        self.mass_proj = nn.Linear(embed_dim, self.n_heads, bias=bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        distances: Tensor,
        *,
        kv_mask: Optional[BoolTensor] = None
    ) -> Tensor:

        mass = log_cosh(self.mass_proj(key))
        bias = torch.einsum("...ah,...ia->...hia", -mass, distances)

        if kv_mask is not None:
            kv_mask = rearrange(kv_mask, "... a -> ... 1 1 a")
            bias = bias.masked_fill(~kv_mask, float("-inf"))

        bias = rearrange(bias, "... h i a -> (... h) i a")
        out, _ = self.attention(query, key, value, attn_mask=bias, need_weights=False)

        return out


class Attention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: Optional[int] = None, bias: bool = False):

        super().__init__()

        self.n_heads = n_heads or max(embed_dim // 64, 1)

        self.attention = nn.MultiheadAttention(
            embed_dim, self.n_heads, bias=bias, dropout=False, batch_first=True
        )

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, *, kv_mask: Optional[BoolTensor] = None
    ) -> Tensor:

        out, _ = self.attention(query, key, value, key_padding_mask=~kv_mask, need_weights=False)
        return out
