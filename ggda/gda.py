from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .pool import RangedCoulombPool
from .encoder import Encoder
from .decoder import Decoder
from .features import lda, mean_and_covariance
from .utils import Activation, log_cosh, cubic_grid


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        n_basis: Optional[int] = None,
        basis_cutoff: float = 4.0,
        grid_size: int = 8,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size).view(-1, 3))

        self.pooling = RangedCoulombPool(n_basis or embed_dim, basis_cutoff)

        self.encoder = Encoder(
            n_basis=self.pooling.n_basis,
            embed_dim=embed_dim,
            n_blocks=n_encoder_blocks,
            n_heads=n_heads,
            enhancement=enhancement,
            activation=activation,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_features=2,
            n_blocks=n_decoder_blocks,
            n_heads=n_heads,
            enhancement=enhancement,
            activation=activation,
        )

    def setup_grid(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)

        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        anchor_coords = 3 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid

        return coords, anchor_coords

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        wrho = weights * rho

        coords, anchor_coords = self.setup_grid(wrho, coords)
        phi = self.pooling(wrho, coords, anchor_coords)

        context = self.encoder(phi, anchor_coords)
        y = self.decoder(rho, coords, context)

        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)
        lda_val = lda(rho.clip(min=1e-7))

        return scale * lda_val + bias
