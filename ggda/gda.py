import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .pool import DensityPooling
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
        n_basis: int,
        max_std: float = 4.0,
        grid_size: int = 8,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size))
        self.pooling = DensityPooling(embed_dim, n_basis, max_std)

        self.encoder = Encoder(
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
        anchor_coords = 3 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid

        phi = self.pooling(wrho, coords, anchor_coords)
        return self.encoder(phi, anchor_coords)

    def decode(self, rho: Tensor, coords: Tensor, context: Tensor) -> Tensor:

        y = self.decoder(rho, coords, context)
        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale * lda_x(rho.clip(min=1e-7)) + bias

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        context = self.encode(weights * rho, coords)
        return self.decode(rho, coords, context)
