from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .layers import GatedMLP, FourierAttention, FourierPositionalEncoding
from .layers import coordinate_norm, grid_norm
from .features import t_thomas_fermi
from .utils import Activation, activation_func


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "gelu",
        eps: float = 1e-4,
    ):

        super().__init__()
        self.eps = eps

        self.activation = activation_func(activation)
        width = int(embed_dim * enhancement)

        self.in_linear = nn.Linear(2, width)
        self.out_linear = nn.Linear(width, embed_dim)
        self.coord_emb = FourierPositionalEncoding(width, kernel_scale)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        x = torch.stack([rho, gamma], dim=-1)
        x = grid_norm(torch.log(x + self.eps), weights)
        gate = self.activation(self.coord_emb(coords))

        return self.out_linear(self.in_linear(x) * gate)


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.attention = FourierAttention(embed_dim, kernel_scale)
        self.mlp = GatedMLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        phi = phi + self.attention(phi, coords, weights)
        return phi + self.mlp(self.norm(phi))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int = 1,
        enhancement: float = 2.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        self.mlp = GatedMLP(embed_dim, enhancement, activation)
        self.proj = nn.Linear(embed_dim, out_features)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor) -> Tensor:
        phi = self.mlp(self.norm(phi))
        return self.proj(torch.tanh(phi))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.embedding = DensityEmbedding(embed_dim, kernel_scale, enhancement, activation)

        make_block = lambda: Block(embed_dim, kernel_scale, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 1, enhancement, activation)

    def tau_pauli(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        phi = self(rho, gamma, coords, weights)
        return torch.exp(phi) * t_thomas_fermi(rho)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        wrho = weights * rho
        coords = coordinate_norm(coords, wrho)
        phi = self.embedding(rho, gamma, coords, wrho)

        for block in self.blocks:
            phi = block(phi, coords, wrho)

        return self.proj(phi).squeeze(dim=-1)
