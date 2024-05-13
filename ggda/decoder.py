from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .layers import CoordinateEncoding, MLP, ProximalAttention
from .utils import Activation, activation_func, std_scale


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_modes: int = 256,
        init_std: float = 0.1,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
        eps: float = 1e-4,
    ):

        super().__init__()

        assert embed_dim % 2 == 0, "Embedding dimension must be even."

        self.eps = eps

        self.field_embed = nn.Linear(1, embed_dim // 2, bias=False)
        self.coord_embed = CoordinateEncoding(embed_dim // 2, init_std, n_modes=n_modes)

        self.mlp = MLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rho: Tensor, coords: Tensor) -> Tensor:

        log_rho = torch.log(rho + self.eps).unsqueeze(-1)
        field_emb = self.field_embed(log_rho)
        coord_emb = self.coord_embed(coords)

        x = torch.cat([field_emb, coord_emb], dim=-1)
        return self.norm(self.mlp(x))


class DecoderBlock(nn.Module):
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

    # def forward(self, phi: Tensor, context: Tensor, distances: Tensor) -> Tensor:
    #     attn = self.attention(phi, context, context, distances)
    #     phi = self.attn_norm(phi + attn)
    #     return self.mlp_norm(phi + self.mlp(phi))

    def forward(self, phi: Tensor, context: Tensor, distances: Tensor) -> Tensor:
        phi = phi + self.attention(self.attn_norm(phi), context, context, distances)
        return phi + self.mlp(self.mlp_norm(phi))


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

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)
        self.proj = nn.Linear(embed_dim, out_features)

    def __call__(self, phi: Tensor) -> Tensor:
        h = self.mlp(self.norm(phi))
        return self.proj(self.activation(h))


class Decoder(nn.Module):
    def __init__(
        self,
        out_features: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        make_block = lambda: DecoderBlock(embed_dim, n_heads, enhancement, activation=activation)

        self.embedding = DensityEmbedding(embed_dim, enhancement=enhancement, activation=activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.project = FieldProjection(embed_dim, out_features, activation=activation)

    def forward(self, rho: Tensor, coords: Tensor, context: Tensor, distances: Tensor) -> Tensor:

        phi = self.embedding(rho, coords)
        distances = std_scale(distances, eps=1e-5)

        for block in self.blocks:
            phi = block(phi, context, distances)

        return self.project(phi)
