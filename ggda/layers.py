from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .utils import Activation, activation_func, random_unit_vec


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


def basis_norm(psi: Tensor, weights: Optional[Tensor], eps: float = 1e-5) -> Tensor:

    psi2 = torch.abs(psi) ** 2

    if weights is None:
        norm_squared = torch.sum(psi2, dim=-2, keepdim=True)

    elif torch.is_floating_point(weights):
        w = weights.unsqueeze(-1)
        norm_squared = torch.sum(w * psi2, dim=-2, keepdim=True)

    else:
        psi2 = psi2.masked_fill(~weights.unsqueeze(-1), 0.0)
        norm_squared = torch.sum(psi2, dim=-2, keepdim=True)

    return psi / torch.sqrt(norm_squared + eps)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, min_scale: float, base: float = 100.0, ndim: int = 3):

        super().__init__()

        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.lift = nn.Linear(ndim, embed_dim // 2, bias=False)

        with torch.no_grad():
            scales = base ** (-2 * torch.arange(0, embed_dim // 2) / embed_dim)
            k_norms = (2 * torch.pi / min_scale) * scales
            k_vals = k_norms.unsqueeze(-1) * random_unit_vec(embed_dim // 2)
            self.lift.weight.copy_(k_vals)

    def forward(self, f: Tensor, coords: Tensor) -> Tensor:

        phase = self.lift(coords)  # (..., n_grid, embed_dim // 2)
        c, s = torch.cos(phase), torch.sin(phase)
        f1, f2 = torch.chunk(f, 2, dim=-1)

        return torch.cat([f1 * c - f2 * s, f1 * s + f2 * c], dim=-1)


class LinearSelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int, min_scale: float, n_heads: Optional[int] = None, bias: bool = True
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads or max(embed_dim // 64, 1)
        self.head_dim = embed_dim // self.n_heads
        self.inner_dim = self.head_dim * self.n_heads

        self.query_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=bias)

        self.pos_enc = RotaryPositionalEncoding(embed_dim, min_scale)

        self._initialize_query_proj()

    def _initialize_query_proj(self):

        delta, sigma = 1 / self.head_dim, 1 / self.head_dim

        with torch.no_grad():

            A, B = torch.empty(2, self.inner_dim, self.embed_dim)
            nn.init.orthogonal_(A, gain=1)
            nn.init.eye_(B)

            self.query_proj.weight.copy_(sigma * A + delta * B)

            if self.query_proj.bias is not None:
                nn.init.zeros_(self.query_proj.bias)

    def forward(self, phi: Tensor, coords: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        query = self.query_proj(phi)
        key = basis_norm(self.key_proj(phi), weights)
        value = basis_norm(self.value_proj(phi), weights)

        query = self.pos_enc(query, coords)
        key = self.pos_enc(key, coords)

        if weights is not None:

            if weights.dtype == torch.bool:
                norm = query.size(-2) ** 0.5
                mask = ~weights.unsqueeze(-1)
                key = key.masked_fill(mask, 0.0) / norm
                value = value.masked_fill(mask, 0.0) / norm
            else:
                key = key * weights.unsqueeze(-1)

        else:
            norm = query.size(-2) ** 0.5
            key = key / norm
            value = value / norm

        return self.out_proj(query @ (key.mT @ value))
