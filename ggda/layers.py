from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
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
    def __init__(self, n_modes: int, init_std: float, ndim: int = 3):

        super().__init__()

        assert n_modes % 2 == 0, "Number of modes must be even."
        self.n_modes = n_modes

        self.mode_lift = nn.Linear(ndim, n_modes // 2, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=1 / init_std)

    def forward(self, coords: Tensor) -> Tensor:
        emb = self.mode_lift(coords)
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


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
        kv_mask: Optional[BoolTensor] = None,
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
        mask = ~kv_mask if kv_mask is not None else None
        out, _ = self.attention(query, key, value, key_padding_mask=mask, need_weights=False)
        return out


class AxialAttention(nn.Module):
    def __init__(self, embed_dim: int, heads_per_dim: Optional[int] = None, bias: bool = True):

        self.heads_per_dim = heads_per_dim or max(embed_dim // 32, 1)
        self.head_size = embed_dim // self.heads_per_dim

        self.q_proj = nn.Linear(embed_dim, self.heads_per_dim * self.head_size, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.heads_per_dim * self.head_size, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.heads_per_dim * self.head_size, bias=bias)

        self.out_proj = nn.Linear(3 * self.heads_per_dim * self.head_size, embed_dim)

    def forward(self, x: Tensor):

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        qx = rearrange(q, "... x y z (h d) -> ... h y z x d", h=self.heads_per_dim)
        kx = rearrange(k, "... x y z (h d) -> ... h y z x d", h=self.heads_per_dim)
        vx = rearrange(v, "... x y z (h d) -> ... h y z x d", h=self.heads_per_dim)

        qy = rearrange(q, "... x y z (h d) -> ... h z x y d", h=self.heads_per_dim)
        ky = rearrange(k, "... x y z (h d) -> ... h z x y d", h=self.heads_per_dim)
        vy = rearrange(v, "... x y z (h d) -> ... h z x y d", h=self.heads_per_dim)

        qz = rearrange(q, "... x y z (h d) -> ... h x y z d", h=self.heads_per_dim)
        kz = rearrange(k, "... x y z (h d) -> ... h x y z d", h=self.heads_per_dim)
        vz = rearrange(v, "... x y z (h d) -> ... h x y z d", h=self.heads_per_dim)

        attn_x = F.scaled_dot_product_attention(qx, kx, vx)
        attn_y = F.scaled_dot_product_attention(qy, ky, vy)
        attn_z = F.scaled_dot_product_attention(qz, kz, vz)

        attn_x = rearrange(attn_x, "... h y z x d -> ... x y z (h d)")
        attn_y = rearrange(attn_y, "... h z x y d -> ... x y z (h d)")
        attn_z = rearrange(attn_z, "... h x y z d -> ... x y z (h d)")

        attn = torch.cat([attn_x, attn_y, attn_z], dim=-1)

        return self.out_proj(attn)
