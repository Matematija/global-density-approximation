import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .encoder import Encoder
from .decoder import Decoder
from .features import lda_x, mean_and_covariance
from .utils import Activation, log_cosh, cubic_grid, dist


class DensityPooling(nn.Module):
    def __init__(self, n_basis: int, embed_dim: int, max_std: float = 4, eps: float = 1e-4):

        super().__init__()

        self.eps = eps

        stds = torch.linspace(0.0, max_std, n_basis + 1)[1:]
        self.register_buffer("gammas", 1 / (2 * stds**2))

        self.lift = nn.Linear(n_basis, embed_dim, bias=False)

    def forward(self, wrho: Tensor, distances: Tensor) -> Tensor:

        norms = (torch.pi / self.gammas) ** (3 / 2)
        basis_vals = norms * torch.exp(-self.gammas * distances.unsqueeze(-1) ** 2)

        pooled_rho = torch.einsum("...xas,...x->...as", basis_vals, wrho)
        phi = torch.log(pooled_rho + self.eps)

        return self.lift(phi)


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        n_basis: int,
        max_std: float = 4.0,
        grid_size: int = 8,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size))
        self.pooling = DensityPooling(n_basis, embed_dim, max_std)

        self.encoder = Encoder(
            embed_dim=embed_dim,
            n_blocks=n_encoder_blocks,
            enhancement=enhancement,
            activation=activation,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_features=2,
            n_blocks=n_decoder_blocks,
            enhancement=enhancement,
            activation=activation,
        )

    def encode(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)

        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        anchor_coords = 3 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid
        distances = dist(coords, anchor_coords)

        phi = self.pooling(wrho, distances)
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
