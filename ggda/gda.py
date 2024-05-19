import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .pool import GaussianPool
from .encoder import Encoder
from .decoder import Decoder
from .features import lda_x, mean_and_covariance
from .utils import Activation, log_cosh, cubic_grid, dist


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        n_basis: int,
        max_std: float = 2.0,
        grid_size: int = 8,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size).view(-1, 3))
        self.pooling = GaussianPool(n_basis, max_std)

        self.encoder = Encoder(
            n_basis=n_basis,
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

    def encode(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)
        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        anchor_coords = 2 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid
        distances = dist(coords, anchor_coords)

        phi = self.pooling(wrho, coords, anchor_coords)
        phi = torch.log(phi + 1e-4)
        context = self.encoder(phi, anchor_coords)

        return context, distances

    def decode(self, rho: Tensor, coords: Tensor, context: Tensor, distances: Tensor) -> Tensor:

        y = self.decoder(rho, coords, context, distances)
        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale * lda_x(rho.clip(min=1e-7)) + bias

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        context, distances = self.encode(weights * rho, coords)
        return self.decode(rho, coords, context, distances)
