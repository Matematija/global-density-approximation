from typing import Optional

from torch import nn
from torch import Tensor

from .layers import MLP, ProximalAttention, CoordinateEncoding
from .utils import Activation, std_scale, dist


class FeatureEmbeding(nn.Module):
    def __init__(
        self,
        in_components: int,
        embed_dim: int,
        coord_std: float,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        width = int(enhancement * embed_dim)

        self.feature_embed = nn.Sequential(
            nn.Linear(in_components, width, bias=False), nn.Softsign(), nn.Linear(width, embed_dim)
        )

        self.coord_embed = nn.Sequential(
            CoordinateEncoding(embed_dim, coord_std),
            MLP(embed_dim, enhancement, activation=activation),
        )

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        return self.feature_embed(x) + self.coord_embed(coords)


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
        n_basis: int,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        coord_std: float = 3.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        make_block = lambda: EncoderBlock(embed_dim, n_heads, enhancement, activation)

        self.embed = FeatureEmbeding(n_basis, embed_dim, coord_std, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:

        x = self.embed(x, coords)
        distances = std_scale(dist(coords, coords + 1e-5))

        for block in self.blocks:
            x = block(x, distances)

        return self.final_norm(x)
