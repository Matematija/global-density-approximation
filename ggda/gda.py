from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from einops import repeat

from .layers import MLP, Attention, ProximalAttention, CoordinateEncoding
from .features import lda, mean_and_covariance
from .utils import Activation, log_cosh, dist, activation_func, std_scale, cubic_grid


class CoordinateEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_std: float,
        enhancement: float = 1.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.encode = CoordinateEncoding(embed_dim, init_std)
        self.mlp = MLP(embed_dim, enhancement, activation)

    def forward(self, coords: Tensor) -> Tensor:
        return self.mlp(self.encode(coords))


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
        self.coord_embed = CoordinateEmbedding(self.width, init_std, activation=activation)
        self.proj = nn.Linear(self.width, embed_dim)

    def forward(self, field: Tensor, coords: Tensor) -> Tensor:

        field_emb = self.field_embed(field)
        coord_emb = self.coord_embed(coords)
        x = self.activation(field_emb) * coord_emb

        return self.proj(x)


class InterpolationBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: Optional[int] = None):

        super().__init__()

        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.value_norm = nn.Identity()

        self.attention = Attention(embed_dim, n_heads)

    def forward(self, src: Tensor, tgt: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        query = self.query_norm(tgt)
        key = self.key_norm(src)
        value = self.value_norm(src)

        if weights is not None:
            n_heads, n_tgt = self.attention.n_heads, tgt.size(-2)
            attn_bias = repeat(torch.log(weights), "... r -> (... h) R r", h=n_heads, R=n_tgt)

        else:
            attn_bias = None

        return self.attention(query, key, value, attn_mask=attn_bias)


class InteractionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: Optional[int] = None,
        enhancement: int = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = ProximalAttention(embed_dim, n_heads)
        self.mlp = MLP(embed_dim, enhancement, activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, distances: Tensor) -> Tensor:
        y = self.attn_norm(phi)
        phi = phi + self.attention(y, y, y, distances)
        return phi + self.mlp_norm(self.mlp(phi))


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
        return self.proj(self.mlp(self.norm(phi)))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int = 1,
        n_pts: int = 8,
        n_heads: int = None,
        coord_std: float = 1.0,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.register_buffer("grid", cubic_grid(n_pts).view(-1, 3))

        self.grid_coord_embed = CoordinateEmbedding(embed_dim, coord_std, activation=activation)
        self.field_embed = FieldEmbedding(1, embed_dim, coord_std, enhancement, activation)

        self.input_interp_block = InterpolationBlock(embed_dim, n_heads)

        make_block = lambda: InteractionBlock(embed_dim, n_heads, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.output_interp_block = InterpolationBlock(embed_dim, n_heads)
        self.field_proj = FieldProjection(embed_dim, 2, enhancement, activation)

    def setup_grid(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho, coords)
        s2, R = torch.linalg.eigh(covs)
        coords = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        anchor_coords = 2 * torch.sqrt(s2 + 1e-5).unsqueeze(-2) * self.grid

        return anchor_coords

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        phi = torch.log(rho + 1e-4).unsqueeze(-1)
        phi = self.field_embed(phi, coords)

        anchor_coords = self.setup_grid(weights * rho, coords)
        grid_emb = self.grid_coord_embed(anchor_coords)

        phi_grid = self.input_interp_block(phi, grid_emb, weights=weights)

        distances = dist(anchor_coords, anchor_coords + 1e-5)
        distances = std_scale(distances)

        for block in self.blocks:
            phi_grid = block(phi_grid, distances)

        phi_points = phi + self.output_interp_block(phi_grid, phi)

        scale, bias = self.field_proj(phi_points).unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return scale * lda(rho.clip(min=1e-7)) + bias
