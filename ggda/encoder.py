from typing import Optional

from torch import nn
from torch import Tensor

from .layers import MLP, ProximalAttention, FieldEmbedding
from .utils import Activation, std_scale, dist


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

    # def forward(self, x: Tensor, distances: Tensor) -> Tensor:
    #     attn = self.attention(x, x, x, distances)
    #     x = self.attn_norm(x + attn)
    #     return self.mlp_norm(x + self.mlp(x))

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

        self.embed = FieldEmbedding(n_basis, embed_dim, coord_std, enhancement, activation)

        make_block = lambda: EncoderBlock(embed_dim, n_heads, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, coords: Tensor) -> Tensor:

        distances = std_scale(dist(coords, coords + 1e-5))
        x = self.embed(phi, coords)

        for block in self.blocks:
            x = block(x, distances)

        # return x
        return self.final_norm(x)
