from typing import Optional

import torch
from torch import nn
from torch import Tensor

from einops import rearrange

from .utils import Activation, ShapeOrInt, activation_func


class CoordinateEncoding(nn.Module):
    def __init__(self, embed_dim: int, init_std: float, ndim: int = 3):

        super().__init__()

        assert embed_dim >= 2, "Embedding dimension must be at least 2."
        assert embed_dim % 2 == 0, "Embedding dimension must be even."

        self.mode_lift = nn.Linear(ndim, embed_dim // 2, bias=False)
        nn.init.normal_(self.mode_lift.weight, std=1 / init_std)

    def forward(self, coords: Tensor) -> Tensor:
        emb = self.mode_lift(coords)
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class CoordinateEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_std: float,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        bias: bool = True,
    ):

        super().__init__()

        self.width = int(enhancement * embed_dim)

        self.encode = CoordinateEncoding(self.width, init_std)
        self.conv1 = nn.Conv3d(self.width, self.width, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv3d(self.width, embed_dim, kernel_size=1, bias=bias)
        self.activation = activation_func(activation)

    def forward(self, coords: Tensor) -> Tensor:
        emb = rearrange(self.encode(coords), "... x y z c -> ... c x y z")
        return self.conv2(self.activation(self.conv1(emb)))


class FieldEmbedding(nn.Module):
    def __init__(
        self,
        in_components: int,
        embed_dim: int,
        n_groups: int = 4,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        bias: bool = True,
    ):

        super().__init__()

        self.activation = activation_func(activation)
        self.width = int(embed_dim * enhancement)

        self.conv1 = nn.Conv3d(in_components, self.width, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv3d(self.width, embed_dim, kernel_size=1, bias=bias)
        self.norm = nn.GroupNorm(n_groups, self.width)

    def forward(self, field: Tensor) -> Tensor:
        emb = self.norm(self.conv1(field))
        return self.conv2(self.activation(emb))


class ConvBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_size: ShapeOrInt = 3,
        n_groups: int = 4,
        enhancement: float = 4.0,
        activation: Activation = "silu",
        bias: bool = True,
    ):

        super().__init__()

        self.width = int(enhancement * embed_dim)

        self.lift = nn.Conv3d(embed_dim, self.width, kernel_size, padding="same", bias=bias)
        self.proj = nn.Conv3d(self.width, embed_dim, kernel_size, padding="same", bias=bias)
        self.coord_lift = nn.Conv3d(embed_dim, self.width, kernel_size=1, bias=bias)

        self.norm1 = nn.GroupNorm(n_groups, embed_dim)
        self.norm2 = nn.GroupNorm(n_groups, self.width)

        self.activation = activation_func(activation)

    def forward(self, f: Tensor, x: Tensor) -> Tensor:

        g = self.lift(self.activation(self.norm1(f)))
        g = g + self.coord_lift(x)
        g = self.proj(self.activation(self.norm2(g)))

        return f + g


class AttentionBlock(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: Optional[int] = None, n_groups: int = 4, bias: bool = False
    ):

        super().__init__()

        self.n_heads = n_heads or max(embed_dim // 64, 1)

        self.norm = nn.GroupNorm(n_groups, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim, self.n_heads, dropout=0.0, bias=bias, batch_first=True
        )

    def forward(self, f: Tensor) -> Tensor:

        *_, dx, dy, dz = f.shape

        g = rearrange(self.norm(f), "... c x y z -> ... (xyz) c")
        g = self.attention(g, g, g)
        g = rearrange(g, "... (x y z) c -> ... c x y z", x=dx, y=dy, z=dz)

        return f + g


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_components: int,
        n_groups: int = 4,
        activation: Activation = "silu",
        bias: bool = True,
    ):

        super().__init__()

        self.activation = activation_func(activation)
        self.norm = nn.GroupNorm(n_groups, embed_dim)
        self.proj = nn.Conv3d(embed_dim, out_components, kernel_size=1, bias=bias)

    def forward(self, f: Tensor) -> Tensor:
        h = self.proj(self.activation(self.norm(f)))
        return torch.mean(h, dim=(-3, -2, -1))
