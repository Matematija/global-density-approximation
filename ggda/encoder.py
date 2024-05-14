from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .layers import CoordinateEncoding
from .utils import Activation, activation_func


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        coord_std: float = 0.05,
        coord_modes: int = 1024,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        n_heads = n_heads or max(embed_dim // 64, 1)
        activation = activation_func(activation)
        mlp_width = int(embed_dim * enhancement)

        self.coord_embed = CoordinateEncoding(embed_dim, init_std=coord_std, n_modes=coord_modes)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_width,
            activation=activation,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_blocks,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )

    def forward(self, phi: Tensor, coords: Tensor) -> Tensor:
        x = phi + self.coord_embed(coords)
        return self.transformer(x)
