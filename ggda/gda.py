from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .pool import GaussianPool
from .layers import MLP, ProximalAttention, CoordinateEncoding
from .features import lda, mean_and_covariance
from .utils import Activation, log_cosh, dist, activation_func, std_scale


class FieldEmbedding(nn.Module):
    def __init__(
        self,
        in_components: int,
        embed_dim: int,
        init_std: float,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        eps: float = 1e-4,
    ):

        super().__init__()

        self.activation = activation_func(activation)
        self.width = int(embed_dim * enhancement)
        self.eps = eps

        assert self.width % 2 == 0, f"`embed_dim * enhancement` must be even, got {self.width}."

        self.field_embed = nn.Linear(in_components, self.width, bias=False)
        self.coord_embed = CoordinateEncoding(self.width // 2, init_std)
        self.proj = nn.Linear(self.width, embed_dim)

    def forward(self, field: Tensor, coords: Tensor) -> Tensor:
        field_emb = self.field_embed(field)
        coord_emb = self.coord_embed(coords)
        return self.proj(self.activation(field_emb) * coord_emb)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: Optional[int] = None,
        enhancement: int = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = ProximalAttention(embed_dim, n_heads, enhancement)
        self.mlp = MLP(embed_dim, enhancement, activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, distances: Tensor) -> Tensor:
        phi = self.attn_norm(phi + self.attention(phi, phi, phi, distances))
        return self.mlp_norm(phi + self.mlp(phi))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_components: int,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.mlp = MLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_components)

    def forward(self, phi: Tensor) -> Tensor:
        x = self.mlp(phi).sum(dim=-2)
        return self.proj(self.norm(x))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        n_basis: int = 32,
        max_std: float = 1.0,
        n_pts: int = 512,
        n_heads: int = None,
        coord_std: float = 1.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.n_pts = n_pts
        self.register_buffer("grid", torch.randn(n_pts, 3))
        self.pooling = GaussianPool(n_basis, max_std)

        self.field_embed = FieldEmbedding(n_basis, embed_dim, coord_std, enhancement, activation)

        make_block = lambda: TransformerBlock(embed_dim, n_heads, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.field_proj = FieldProjection(embed_dim, 2, enhancement, activation)

    def pool_density(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)
        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()

        grid = torch.randn(self.n_pts, 3) if self.training else self.grid
        anchor_coords = torch.sqrt(s2 + 1e-5).unsqueeze(-2) * grid

        phi = self.pooling(wrho, coords, anchor_coords)
        phi = torch.log(phi + 1e-4)

        return phi, anchor_coords

    def scale_and_bias(self, wrho: Tensor, coords: Tensor) -> Tensor:

        phi, anchor_coords = self.pool_density(wrho, coords)
        phi = self.field_embed(phi, anchor_coords)

        distances = dist(anchor_coords, anchor_coords + 1e-5)
        distances = std_scale(distances)

        for block in self.blocks:
            phi = block(phi, distances)

        y = self.field_proj(phi)
        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale, bias

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        wrho = weights * rho

        E_lda = torch.sum(wrho * lda(rho.clip(min=1e-7)), dim=-1)
        scale, bias = self.scale_and_bias(wrho, coords)

        return scale * E_lda + bias
