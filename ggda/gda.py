from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from einops import rearrange

from .pool import GaussianPool
from .layers import FieldEmbedding, CoordinateEmbedding, ConvBlock, AttentionBlock, FieldProjection
from .features import lda, mean_and_covariance
from .utils import Activation, ShapeOrInt, log_cosh, cubic_grid


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_size: ShapeOrInt = 3,
        n_groups: int = 4,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        attention: bool = False,
        n_heads: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.conv_block = ConvBlock(embed_dim, kernel_size, n_groups, enhancement, activation, bias)

        if attention:
            self.attn_block = AttentionBlock(embed_dim, n_heads, n_groups)
        else:
            self.attn_block = nn.Identity()

    def forward(self, f: Tensor, x: Tensor) -> Tensor:
        f = self.conv_block(f, x)
        return self.attn_block(f)


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        embed_dim: int,
        n_basis: int,
        max_pool_std: float = 6.0,
        grid_size: ShapeOrInt = 8,
        kernel_size: ShapeOrInt = 3,
        coord_std: float = 10.0,
        n_groups: int = 4,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        attention: bool = False,
        n_heads: Optional[int] = None,
        bias: bool = True,
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(grid_size))
        self.pooling = GaussianPool(n_basis, max_pool_std)
        self.grid_size = dict(zip("xyz", self.grid.shape[:3]))

        self.field_embed = FieldEmbedding(
            in_components=n_basis,
            embed_dim=embed_dim,
            n_groups=n_groups,
            enhancement=enhancement,
            activation=activation,
            bias=bias,
        )

        self.coord_embed = CoordinateEmbedding(
            embed_dim=embed_dim,
            init_std=coord_std,
            enhancement=enhancement,
            activation=activation,
            bias=bias,
        )

        make_block = lambda: Block(
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            enhancement=enhancement,
            activation=activation,
            attention=attention,
            n_heads=n_heads,
            bias=bias,
        )

        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.field_proj = FieldProjection(
            embed_dim=embed_dim,
            out_components=2,
            n_groups=n_groups,
            activation=activation,
            bias=bias,
        )

    def setup_grid(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)
        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()

        stds = rearrange(torch.sqrt(s2 + 1e-5), "... c -> ... 1 1 1 c")
        anchor_coords = 3 * stds * self.grid

        return anchor_coords

    def scale_and_bias(self, wrho: Tensor, coords: Tensor) -> Tensor:

        anchor_coords = self.setup_grid(wrho, coords)
        x = self.coord_embed(anchor_coords)

        anchor_coords = rearrange(anchor_coords, "... x y z c -> ... (x y z) c")
        phi = self.pooling(wrho, coords, anchor_coords)
        phi = rearrange(phi, "... (x y z) c -> (...) c x y z", **self.grid_size)
        phi = self.field_embed(torch.log(phi + 1e-4))

        for block in self.blocks:
            phi = block(phi, x)

        scale, bias = self.field_proj(phi).squeeze().unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale, bias

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        scale, bias = self.scale_and_bias(weights * rho, coords)
        lda_energy = torch.sum(weights * rho * lda(rho.clip(min=1e-7)), dim=-1)

        return scale * lda_energy + bias
