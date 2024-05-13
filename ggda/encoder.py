from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .layers import CoordinateEncoding, MLP, ProximalAttention
from .utils import Activation, std_scale


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.attention = ProximalAttention(embed_dim, n_heads)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, distances: Tensor) -> Tensor:
        attn = self.attention(x, x, x, distances)
        x = self.attn_norm(x + attn)
        return self.mlp_norm(x + self.mlp(x))


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        make_block = lambda: EncoderBlock(embed_dim, n_heads, enhancement, activation)

        self.coord_embed = CoordinateEncoding(embed_dim, init_std=0.05, n_modes=256)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

    def forward(self, phi: Tensor, coords: Tensor) -> Tensor:

        x = phi + self.coord_embed(coords)

        distances = torch.cdist(coords, coords)
        distances = std_scale(distances, eps=1e-5)

        for block in self.blocks:
            x = block(x, distances)

        return x
