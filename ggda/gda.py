from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .layers import MLP
from .features import lda
from .symmetry import principal_axes_rotation
from .utils import Activation, log_cosh, activation_func


class WaveletEmbedding(nn.Module):
    def __init__(self, embed_dim: int, n_gaussians: int, modes_per_gaussian: int, max_std: float):

        super().__init__()

        std_vals = torch.linspace(0, max_std, n_gaussians + 1)[1:]
        self.register_buffer("stds", std_vals)

        k_vals = torch.randn(n_gaussians, modes_per_gaussian, 3)
        k_vals = (2 * torch.pi / std_vals)[..., None, None] * k_vals
        self.k = nn.Parameter(k_vals)

        self.lift = nn.Linear(modes_per_gaussian, embed_dim)

    def eval_basis(self, coords: Tensor) -> Tensor:

        z = coords.unsqueeze(-1) / self.stds
        envelope = torch.exp(-(z**2) / 2)
        # envelope.shape = (n_coords, n_gaussians)

        phases = torch.einsum("...xc,skc->...skx", coords, self.k)
        waves = torch.cos(phases)
        # waves.shape = (... n_gaussians, n_modes, n_coords)

        k2 = torch.sum(self.k**2, dim=-1)
        s2 = self.stds.unsqueeze(-1) ** 2
        norms = torch.exp(-0.5 * k2 * s2) / (2 * torch.pi * s2) ** (3 / 2)
        # norms.shape = (n_modes, n_gaussians)

        return torch.einsum("ks,...xs,...skx->...skx", norms, envelope, waves)

    def forward(self, wrho: Tensor, coords: Tensor) -> Tensor:
        basis = self.eval_basis(coords)
        phi = torch.einsum("...x,...skx->...sk", wrho, basis)
        return self.lift(phi)


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        n_heads: Optional[int] = None,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        n_heads = n_heads or max(embed_dim // 64, 1)
        mlp_width = int(embed_dim * enhancement)
        activation = activation_func(activation)

        block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_width,
            activation=activation,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            block, num_layers=n_blocks, norm=nn.LayerNorm(embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class FieldProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_features: int,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.activation = activation_func(activation)

        self.mlp = MLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.activation(self.norm(self.mlp(x))))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_gaussians: int,
        modes_per_gaussian: int,
        max_std: float,
        n_blocks: int,
        enhancement: float = 4.0,
        activation: Activation = "gelu",
    ):

        super().__init__()

        self.embedding = WaveletEmbedding(embed_dim, n_gaussians, modes_per_gaussian, max_std)
        self.transformer = Transformer(embed_dim, n_blocks, enhancement, activation)
        self.projection = FieldProjection(embed_dim, 2, enhancement, activation)

    def scale_and_bias(self, wrho: Tensor, coords: Tensor) -> Tensor:

        symmetry = principal_axes_rotation(wrho, coords)
        coords = symmetry(coords)

        phi = self.embedding(wrho, coords)
        phi = self.transformer(phi)
        y = self.projection(phi)

        scale, bias = y.unbind(dim=-1)
        scale, bias = F.softplus(scale), -log_cosh(bias)

        return torch.stack([scale, bias], dim=-1)

    def forward(self, rho: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        rho.requires_grad_(True)
        wrho = weights * rho

        exc_lda = rho * lda(rho.clip(min=1e-7))
        E_lda = torch.einsum("...n,...n->...", weights, exc_lda)

        vxc_lda = torch.autograd.grad(
            E_lda, rho, grad_outputs=torch.ones_like(E_lda), create_graph=True
        )

        sb = self.scale_and_bias(wrho, coords)

        (grad_sb,) = torch.autograd.grad(
            sb, wrho, grad_outputs=torch.ones_like(sb), create_graph=True
        )

        grad_scale, grad_bias = grad_sb.unbind(dim=-1)

        return
