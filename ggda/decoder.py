from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .layers import MLP, Attention, CoordinateEncoding
from .utils import Activation, activation_func


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_std: float,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        assert embed_dim % 2 == 0, "Embedding dimension must be even."

        self.field_embed = nn.Sequential(nn.Linear(2, embed_dim // 2), nn.Tanh())
        self.coord_embed = CoordinateEncoding(embed_dim // 2, init_std)

        self.mlp = MLP(embed_dim, enhancement, activation)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor) -> Tensor:

        f = torch.log(torch.stack([rho, gamma], dim=-1) + 1e-4)

        field_emb = self.field_embed(f)
        coord_emb = self.coord_embed(coords)
        x = torch.cat([field_emb, coord_emb], dim=-1)

        return self.mlp(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = Attention(embed_dim, n_heads)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, context: Tensor) -> Tensor:
        phi = phi + self.attention(self.attn_norm(phi), context, context)
        return phi + self.mlp(self.mlp_norm(phi))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_components: int,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)
        self.proj = nn.Linear(embed_dim, out_components)

    def __call__(self, phi: Tensor) -> Tensor:
        h = self.mlp(self.activation(self.norm(phi)))
        return self.proj(self.activation(h))


class Decoder(nn.Module):
    def __init__(
        self,
        out_components: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        coord_std: float = 0.5,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        make_block = lambda: DecoderBlock(embed_dim, n_heads, enhancement, activation=activation)

        self.embed = DensityEmbedding(embed_dim, coord_std, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.project = FieldProjection(embed_dim, out_components, activation=activation)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, context: Tensor) -> Tensor:

        phi = self.embed(rho, gamma, coords)

        for block in self.blocks:
            phi = block(phi, context)

        return self.project(phi)
