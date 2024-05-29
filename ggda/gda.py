from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .pool import CiderFeatures
from .encoder import Encoder
from .decoder import Decoder
from .features import lda_x, mean_and_covariance
from .utils import Activation, log_cosh, cubic_grid


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        grid_size: int = 8,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size).view(-1, 3))

        self.pooling = CiderFeatures()

        self.encoder = Encoder(
            in_components=3,
            embed_dim=embed_dim,
            n_blocks=n_encoder_blocks,
            n_heads=n_heads,
            enhancement=enhancement,
            activation=activation,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_components=2,
            n_blocks=n_decoder_blocks,
            n_heads=n_heads,
            enhancement=enhancement,
            activation=activation,
        )

    def encode(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        means, covs = mean_and_covariance(weights * rho, coords)
        s2, R = torch.linalg.eigh(covs)
        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        anchor_coords = 3 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid

        phi = self.pooling(rho, gamma, coords, weights, anchor_coords)

        return self.encoder(phi, anchor_coords)

    def decode(self, rho: Tensor, gamma: Tensor, coords: Tensor, context: Tensor) -> Tensor:

        y = self.decoder(rho, gamma, coords, context)

        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale * lda_x(rho.clip(min=1e-7)) + bias

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        context = self.encode(rho, gamma, coords, weights)
        return self.decode(rho, gamma, coords, context)
