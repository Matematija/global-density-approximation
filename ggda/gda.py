from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .pool import CoarseGraining
from .encoder import Encoder
from .decoder import Decoder
from .layers import MLP
from .features import lda, mean_and_covariance
from .utils import Activation, cubic_grid


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        n_basis: Optional[int] = None,
        n_decoder_blocks: int = 1,
        coord_std: float = 2.0,
        grid_size: int = 8,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.n_basis = n_basis or embed_dim

        self.register_buffer("grid", cubic_grid(grid_size).view(-1, 3))
        self.pooling = CoarseGraining(self.n_basis, enhancement)

        self.encoder = Encoder(
            n_basis=self.n_basis,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            coord_std=coord_std,
            enhancement=enhancement,
            activation=activation,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_features=2,
            n_blocks=n_decoder_blocks,
            n_heads=n_heads,
            coord_std=coord_std,
            enhancement=enhancement,
            activation=activation,
        )

        self.nonlocal_head = MLP(embed_dim, enhancement, activation, out_features=1)

    def setup_grid(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)

        coords = (coords - means.unsqueeze(-2)) @ R.mT
        anchor_coords = 3 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid

        return coords, anchor_coords

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        wrho = weights * rho
        coords, anchor_coords = self.setup_grid(wrho, coords)

        phi = self.pooling(rho, gamma, coords, weights, anchor_coords)
        context = self.encoder(phi, anchor_coords)

        scale, bias = self.decoder(rho, gamma, coords, context).unbind(dim=-1)

        lda_val = lda(rho.clip(min=1e-7))

        E_local = torch.sum(wrho * (scale * lda_val + bias), dim=-1)
        E_nonlocal = self.nonlocal_head(context).squeeze(-1).mean(-1)

        return E_local + E_nonlocal
