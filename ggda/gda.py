import torch
from torch import nn
from torch import Tensor

from .embed import CiderFeatures
from .layers import GatedMLP, FourierAttention
from .features import rescaled_grad, mean_and_covariance, t_weisacker, t_thomas_fermi
from .utils import Activation


# class DensityEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         embed_params: tuple[float, float] = (4.0, 4.0),
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#         eps: float = 1e-4,
#     ):

#         super().__init__()
#         self.eps = eps

#         width = int(embed_dim * enhancement)

#         self.nonlocal_features = CiderFeatures(*embed_params)
#         self.field_embed = nn.Sequential(
#             nn.Linear(4, width), nn.Tanh(), nn.Linear(width, embed_dim)
#         )

#         self.coord_embed = FourierPositionalEncoding(width, kernel_scale, 1.0, activation)

#     def forward(
#         self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor, *args, **kwargs
#     ) -> Tensor:

#         x0 = rescaled_grad(rho + 1e-12, gamma).unsqueeze(-1)
#         x1 = self.nonlocal_features(rho, gamma, coords, weights, *args, **kwargs)
#         x = torch.log(torch.cat([x0, x1], dim=-1) + self.eps)

#         return self.field_embed(x) + self.coord_embed(coords)


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        embed_params: tuple[float, float] = (4.0, 4.0),
        enhancement: float = 2.0,
        eps: float = 1e-4,
    ):

        super().__init__()
        self.eps = eps

        width = int(embed_dim * enhancement)

        self.nonlocal_features = CiderFeatures(*embed_params)
        self.lift = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, embed_dim))

    def forward(
        self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:

        x0 = rescaled_grad(rho + 1e-8, gamma).unsqueeze(-1)
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
        activation: Activation = "silu",
    ):

        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = GatedMLP(embed_dim, enhancement, activation)
        self.proj = nn.Linear(embed_dim, out_features)

    def __call__(self, phi: Tensor) -> Tensor:
        phi = self.mlp(self.norm(phi))
        return self.proj(torch.tanh(phi))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        kernel_scale: float,
        embed_params: tuple[float, float] = (4.0, 4.0),
        enhancement: float = 2.0,
        activation: Activation = "silu",
        eta: float = 1e-3,
    ):

        super().__init__()

        self.eta = eta

        self.embedding = DensityEmbedding(embed_dim, embed_params, enhancement)

        make_block = lambda: Block(embed_dim, kernel_scale, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 1, enhancement, activation)

    def eval_tau(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        phi = self(rho, gamma, coords, weights)
        t0, tw = t_thomas_fermi(rho + 1e-12), t_weisacker(rho + 1e-12, gamma)

        return tw + torch.exp(phi) * (t0 + self.eta * tw)

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
