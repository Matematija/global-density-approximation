from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .pool import RangedCoulombPool, WeightedGaussianPool
from .layers import CoordinateEncoding, ProximalAttention, MLP
from .features import lda, mean_and_covariance
from .utils import Activation, ShapeOrInt, log_cosh, cubic_grid, activation_func, dist, std_scale


class FieldEmbedding(nn.Module):
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
        self.activation = activation_func(activation)

        self.feature_embed = nn.Linear(in_components, width, bias=False)

        self.coord_embed = nn.Sequential(
            CoordinateEncoding(embed_dim, coord_std),
            nn.Linear(embed_dim, width),
        )

        self.proj = nn.Linear(width, embed_dim)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:

        x = torch.log(x + 1e-4)
        field_emb = self.feature_embed(x)
        coord_emb = self.coord_embed(coords)

        y = field_emb * self.activation(coord_emb)

        return self.proj(y)


class Block(nn.Module):
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


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        self.mlp = MLP(embed_dim, enhancement, activation=activation)
        self.proj = nn.Linear(embed_dim, out_features)

        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, phi: Tensor) -> Tensor:
        h = self.mlp(self.norm(phi)).mean(dim=-2)
        return self.proj(self.activation(h))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        embed_dim: int,
        n_basis: Optional[int] = None,
        coord_std: float = 5.0,
        grid_size: ShapeOrInt = 8,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        n_basis = n_basis or embed_dim

        self.register_buffer("grid", cubic_grid(grid_size).view(-1, 3))
        self.pooling = WeightedGaussianPool(n_basis, coord_std)
        # self.pooling = RangedCoulombPool(n_basis, coord_std)

        self.field_embed = FieldEmbedding(
            in_components=n_basis,
            embed_dim=embed_dim,
            coord_std=coord_std,
            enhancement=enhancement,
            activation=activation,
        )

        make_block = lambda: Block(embed_dim, n_heads, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.field_proj = FieldProjection(
            embed_dim=embed_dim,
            out_features=1,
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
        phi = self.field_embed(phi, anchor_coords)

        distances = dist(anchor_coords, anchor_coords + 1e-5)
        distances = std_scale(distances)

        for block in self.blocks:
            phi = block(phi, distances)

        E_corr = -log_cosh(self.field_proj(phi)).squeeze(dim=-1)
        E_lda = torch.sum(wrho * lda(rho.clip(min=1e-7)), dim=-1)

        return E_lda + E_corr
