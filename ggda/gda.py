from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .layers import MLP, LinearSelfAttention
from .features import lda_x, lda_c, mean_and_covariance
from .utils import Activation, activation_func, log_cosh


class DensityEmbedding(nn.Module):
    def __init__(self, embed_dim: int, enhancement: float = 4.0, eps: float = 1e-4):

        super().__init__()

        self.eps = eps

        width = int(enhancement * embed_dim)
        self.lift = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, embed_dim))

    def forward(self, rho: Tensor, gamma: Tensor) -> Tensor:
        x = torch.stack([rho, gamma], dim=-1)
        return self.lift(torch.log(x + self.eps))


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        min_scale: float,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = LinearSelfAttention(embed_dim, min_scale, n_heads)
        self.mlp = MLP(embed_dim, enhancement, activation=activation)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, phi: Tensor, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor
    ) -> Tensor:

        attn = self.attention(phi, rho, gamma, coords, weights)
        phi = self.attn_norm(phi + attn)

        return self.mlp_norm(phi + self.mlp(phi))


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
        self.mlp = MLP(embed_dim, enhancement, activation)
        self.proj = nn.Linear(embed_dim, out_features)

    def __call__(self, phi: Tensor) -> Tensor:
        return self.proj(self.activation(self.mlp(phi)))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        min_scale: float,
        n_heads: int = None,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.embedding = DensityEmbedding(embed_dim, enhancement)

        make_block = lambda: Block(embed_dim, min_scale, n_heads, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 3, enhancement, activation)

    def rotate_coords(self, wrho: Tensor, coords: Tensor) -> Tensor:
        means, covs = mean_and_covariance(wrho, coords)
        _, R = torch.linalg.eigh(covs)
        return (coords - means.unsqueeze(-2)) @ R.mT.detach()

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        coords = self.rotate_coords(weights * rho, coords)
        phi = self.embedding(rho, gamma)

        for block in self.blocks:
            phi = block(phi, rho, gamma, coords, weights)

        scale_x, scale_c, bias = self.proj(phi).unbind(dim=-1)
        scale_x, scale_c, bias = F.softplus(scale_x), F.softplus(scale_c), -log_cosh(bias)

        return scale_x * lda_x(rho + 1e-7) + scale_c * lda_c(rho + 1e-7) + bias
