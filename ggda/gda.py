import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .embed import CiderFeatures
from .layers import GatedMLP, FourierAttention
from .features import rescaled_grad, mean_and_covariance
from .utils import Activation, activation_func


class DensityEmbedding(nn.Module):
    def __init__(
        self, embed_dim: int, embed_params: tuple[float, float] = (4.0, 4.0), eps: float = 1e-4
    ):

        super().__init__()
        self.eps = eps

        self.nonlocal_features = CiderFeatures(*embed_params)
        self.lift = nn.Linear(4, embed_dim)

    def forward(
        self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:

        x0 = rescaled_grad(rho + 1e-12, gamma).unsqueeze(-1)
        x1 = self.nonlocal_features(rho, gamma, coords, weights, *args, **kwargs)
        x = torch.cat([x0, x1], dim=-1)

        return self.lift(torch.log(x + self.eps))


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        enhancement: float = 4.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = FourierAttention(embed_dim, kernel_scale)
        self.mlp = GatedMLP(embed_dim, enhancement, activation=activation)

    def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        phi = phi + self.attention(phi, coords, weights)
        return phi + self.mlp(F.tanh(phi))


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int = 1,
        enhancement: float = 2.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.activation = activation_func(activation)
        self.mlp = GatedMLP(embed_dim, enhancement, activation)
        self.proj = nn.Linear(embed_dim, out_features)

    def __call__(self, phi: Tensor) -> Tensor:
        phi = self.mlp(F.tanh(phi))
        return self.proj(self.activation(phi))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        kernel_scale: float,
        embed_params: tuple[float, float] = (4.0, 4.0),
        enhancement: float = 2.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.embedding = DensityEmbedding(embed_dim, embed_params)

        make_block = lambda: Block(embed_dim, kernel_scale, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 1, enhancement, activation)

    def rotate_coords(self, wrho: Tensor, coords: Tensor) -> Tensor:
        means, covs = mean_and_covariance(wrho, coords)
        _, R = torch.linalg.eigh(covs)
        return (coords - means.unsqueeze(-2)) @ R.mT.detach()

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        coords = self.rotate_coords(weights * rho, coords)
        phi = self.embedding(rho, gamma, coords, weights)

        for block in self.blocks:
            phi = block(phi, coords, weights)

        return self.proj(phi).squeeze(dim=-1)
