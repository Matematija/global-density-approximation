from typing import Optional

from torch import nn
from torch import Tensor

from .layers import MLP, FieldEmbedding, ProximalAttention
from .utils import Activation, activation_func, std_scale


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = ProximalAttention(embed_dim, n_heads)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, context: Tensor, distances: Tensor) -> Tensor:
        attn = self.attention(phi, context, context, distances)
        phi = self.attn_norm(phi + attn)
        return self.mlp_norm(phi + self.mlp(phi))

    # def forward(self, phi: Tensor, context: Tensor, distances: Tensor) -> Tensor:
    #     phi = phi + self.attention(self.attn_norm(phi), context, context, distances)
    #     return phi + self.mlp(self.mlp_norm(phi))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        # self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)
        self.proj = nn.Linear(embed_dim, out_features)

    def __call__(self, phi: Tensor) -> Tensor:
        return self.proj(self.activation(self.mlp(phi)))

    # def __call__(self, phi: Tensor) -> Tensor:
    #     h = self.mlp(self.activation(self.norm(phi)))
    #     return self.proj(self.activation(h))


class Decoder(nn.Module):
    def __init__(
        self,
        out_features: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        coord_std: float = 1.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        make_block = lambda: DecoderBlock(embed_dim, n_heads, enhancement, activation=activation)

        self.embed = FieldEmbedding(1, embed_dim, coord_std, enhancement, activation)

        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.project = FieldProjection(embed_dim, out_features, activation=activation)

    def forward(self, rho: Tensor, coords: Tensor, context: Tensor, distances: Tensor) -> Tensor:

        phi = self.embed(rho.unsqueeze(-1), coords)
        distances = std_scale(distances, eps=1e-5)

        for block in self.blocks:
            phi = block(phi, context, distances)

        return self.project(phi)
