import torch
from torch import nn
from torch import Tensor

from .embed import CiderFeatures
from .layers import GatedMLP, FourierAttention, FourierPositionalEncoding
from .features import rescaled_grad, mean_and_covariance, t_weisacker, t_thomas_fermi
from .utils import Activation, activation_func


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        embed_params: tuple[float, float] = (4.0, 4.0),
        enhancement: float = 2.0,
        activation: Activation = "silu",
        eps: float = 1e-4,
    ):

        super().__init__()
        self.eps = eps

        width = int(embed_dim * enhancement)

        self.nonlocal_features = CiderFeatures(*embed_params)
        self.lift = nn.Linear(4, width)
        self.proj = nn.Linear(width, embed_dim)

        self.pos_enc = FourierPositionalEncoding(width, kernel_scale, 1.0, activation)

    def forward(
        self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor, *args, **kwargs
    ) -> Tensor:

        x0 = rescaled_grad(rho + 1e-12, gamma).unsqueeze(-1)
        x1 = self.nonlocal_features(rho, gamma, coords, weights, *args, **kwargs)
        x = torch.cat([x0, x1], dim=-1)

        phi = torch.log(x + self.eps)
        phi = self.lift(phi) + self.pos_enc(coords)
        return self.proj(torch.tanh(phi))


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

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        attn = self.attention(phi, coords, weights)
        phi = self.attn_norm(phi + attn)
        return self.mlp_norm(phi + self.mlp(phi))


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
        return self.proj(self.activation(self.mlp(phi)))


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

        self.embedding = DensityEmbedding(
            embed_dim, kernel_scale, embed_params, enhancement, activation
        )

        make_block = lambda: Block(embed_dim, kernel_scale, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 1, enhancement, activation)

    def eval_tau(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        phi = self(rho, gamma, coords, weights)
        t0, tw = t_thomas_fermi(rho), t_weisacker(rho + 1e-12, gamma)

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
