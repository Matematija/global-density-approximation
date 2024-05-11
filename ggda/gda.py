from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, LongTensor

from .layers import MLP, LinearAttention, CoordinateEncoding
from .features import lda_x
from .utils import Activation, log_cosh, activation_func


class FieldEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_std: float = 0.1,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        assert embed_dim % 2 == 0, "Embedding dimension must be even"

        self.field_embed = nn.Sequential(nn.Linear(1, embed_dim // 2, bias=False), nn.Tanh())
        self.coord_embed = CoordinateEncoding(embed_dim // 2, init_std)

        self.mlp = MLP(embed_dim, enhancement, activation)

    def forward(self, rho: LongTensor, coords: Tensor) -> Tensor:

        x1 = torch.log(rho + 1e-4).unsqueeze(-1)
        x1 = self.field_embed(x1)
        x2 = self.coord_embed(coords)
        x = torch.cat([x1, x2], dim=-1)

        return self.mlp(x)


class InteractionBlock(nn.Module):
    def __init__(self, embed_dim: int, enhancement: float, activation: Activation = "gelu"):

        super().__init__()

        self.attn = LinearAttention(embed_dim)
        self.mlp = MLP(embed_dim, enhancement, activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, f: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        f = self.attn_norm(f + self.attn(f, f, f, weights=weights))
        return self.mlp_norm(f + self.mlp(f))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        self.mlp = MLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_features)

    def forward(self, x: Tensor) -> Tensor:
        h = self.activation(self.norm(self.mlp(x)))
        return self.proj(h)


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        init_std: float = 0.1,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.embedding = FieldEmbedding(embed_dim, init_std, enhancement, activation)

        make_block = lambda: InteractionBlock(embed_dim, enhancement, activation)
        self.blocks = nn.ModuleList(make_block() for _ in range(n_blocks))

        self.projection = FieldProjection(embed_dim, 2, enhancement, activation)

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        phi = self.embedding(rho, coords)

        for block in self.blocks:
            phi = block(phi, weights)

        y = self.projection(phi)

        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale * lda_x(rho.clip(min=1e-7)) + bias
