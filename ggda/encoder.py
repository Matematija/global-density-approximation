from typing import Optional

from torch import nn
from torch import Tensor

from .layers import MLP, ProximalAttention, CoordinateEncoding
from .utils import Activation, std_scale, dist, activation_func


class FeatureEmbedding(nn.Module):
    def __init__(
        self,
        in_components: int,
        embed_dim: int,
        coord_std: float = 8.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.activation = activation_func(activation)
        width = int(in_components * enhancement)

        self.feature_embed = nn.Linear(in_components, width)

        self.coord_embed = nn.Sequential(
            CoordinateEncoding(width, coord_std), nn.Linear(width, width)
        )

        self.proj = nn.Linear(width, embed_dim)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:

        feature_emb = self.feature_embed(x)
        coord_emb = self.coord_embed(coords)
        x = feature_emb * self.activation(coord_emb)

        return self.proj(x)


class EncoderBlock(nn.Module):
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

    def forward(self, x: Tensor, distances: Tensor) -> Tensor:
        y = self.attn_norm(x)
        x = x + self.attention(y, y, y, distances)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(
        self,
        in_components: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        coord_std: float = 3.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        make_block = lambda: EncoderBlock(embed_dim, n_heads, enhancement, activation)

        self.embed = FeatureEmbedding(in_components, embed_dim, coord_std, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:

        x = self.embed(x, coords)
        distances = std_scale(dist(coords, coords + 1e-5))

        for block in self.blocks:
            x = block(x, distances)

        return self.final_norm(x)
