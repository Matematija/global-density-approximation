import torch
from torch import nn
from torch import Tensor

# from .embed import CiderFeatures
from .layers import GatedMLP, MLP, FourierAttention, FourierPositionalEncoding

from .features import mean_and_covariance, t_weisacker, t_thomas_fermi, fermi_momentum

from .utils import Activation, activation_func, log_cosh


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


# class DensityEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#         eps: float = 1e-4,
#     ):

#         super().__init__()
#         self.eps = eps

#         width = int(embed_dim * enhancement)

#         self.field_emb = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, embed_dim))
#         self.coord_emb = FourierPositionalEncoding(embed_dim, kernel_scale, enhancement, activation)

#     def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor) -> Tensor:
#         x = torch.stack([rho, gamma], dim=-1)
#         x = torch.log(x + self.eps)
#         return self.field_emb(x) + self.coord_emb(coords)


# class DensityEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#         eps: float = 1e-4,
#     ):

#         super().__init__()
#         self.eps = eps

#         width = int(embed_dim * enhancement)
#         self.activation = activation_func(activation)

#         self.in_linear = nn.Linear(1, width)
#         self.out_linear = nn.Linear(width, embed_dim)
#         self.coord_emb = FourierPositionalEncoding(width, kernel_scale)

#     def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor) -> Tensor:

#         s2 = rescaled_grad(rho + 1e-12, gamma)

#         x = torch.log(s2 + self.eps).unsqueeze(-1)
#         x = torch.tanh(self.in_linear(x)) * self.activation(self.coord_emb(coords))

#         return self.out_linear(x)
#         # return self.out_linear(x)


# class DensityEmbedding(nn.Module):
#     def __init__(self, embed_dim: int, enhancement: float = 2.0, eps: float = 1e-4):

#         super().__init__()
#         self.eps = eps

#         width = int(embed_dim * enhancement)
#         self.lift = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, embed_dim))

#     def forward(self, rho: Tensor, gamma: Tensor) -> Tensor:

#         # x = rescaled_grad(rho + 1e-12, gamma)
#         # x = torch.log(x + self.eps).unsqueeze(-1)

#         x = torch.stack([rho, gamma], dim=-1)
#         x = torch.log(x + self.eps)

#         return self.lift(x)


# class DensityEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         eps: float = 1e-4,
#     ):

#         super().__init__()
#         self.eps = eps

#         width = int(embed_dim * enhancement)

#         self.in_linear = nn.Linear(1, width)
#         self.out_linear = nn.Linear(width, embed_dim)
#         self.coord_emb = FourierPositionalEncoding(width, kernel_scale)

#     def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor) -> Tensor:

#         s2 = rescaled_grad(rho + 1e-12, gamma)
#         kF = fermi_momentum(rho + 1e-12).unsqueeze(-1)

#         x = torch.log(s2 + self.eps).unsqueeze(-1)
#         x = self.in_linear(x) + self.coord_emb(kF * coords)

#         return self.out_linear(torch.tanh(x))
# return self.out_linear(x)


class DensityEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "silu",
        eps: float = 1e-4,
    ):

        super().__init__()
        self.eps = eps

        self.activation = activation_func(activation)
        width = int(embed_dim * enhancement)

        self.in_linear = nn.Linear(2, width)
        self.out_linear = nn.Linear(width, embed_dim)
        self.coord_emb = FourierPositionalEncoding(width, kernel_scale)

    @staticmethod
    def normalize(f: Tensor, rho: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:

        wrho = weights * rho
        N = wrho.sum(dim=-1, keepdim=True)
        p = torch.unsqueeze(wrho / N, dim=-1)

        mean = torch.sum(p * f, dim=-2, keepdim=True)
        var = torch.sum(p * (f - mean) ** 2, dim=-2, keepdim=True)

        return (f - mean) / torch.sqrt(var + eps)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        x = torch.stack([rho, gamma], dim=-1)
        x = self.normalize(torch.log(x + self.eps), rho, weights)
        x = self.activation(self.in_linear(x))

        return self.out_linear(x * self.coord_emb(coords))


# class DensityEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#         eps: float = 1e-4,
#     ):

#         super().__init__()
#         self.eps = eps

#         self.activation = activation_func(activation)
#         width = int(embed_dim * enhancement)

#         self.in_linear = nn.Linear(2, width)
#         self.out_linear = nn.Linear(width, embed_dim)
#         self.coord_emb = FourierPositionalEncoding(width, kernel_scale)

#     @staticmethod
#     def normalize(f: Tensor, wrho: Tensor, eps: float = 1e-5) -> Tensor:

#         N = wrho.sum(dim=-1, keepdim=True)
#         p = torch.unsqueeze(wrho / N, dim=-1)

#         mean = torch.sum(p * f, dim=-2, keepdim=True)
#         var = torch.sum(p * (f - mean) ** 2, dim=-2, keepdim=True)

#         return (f - mean) / torch.sqrt(var + eps)

#     def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

#         x = torch.stack([rho, gamma], dim=-1)
#         x = self.normalize(torch.log(x + self.eps), weights * rho)
#         gate = self.activation(self.coord_emb(coords))

#         return self.out_linear(self.in_linear(x) * gate)


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "silu",
    ):

        super().__init__()

        self.attention = FourierAttention(embed_dim, kernel_scale)
        self.mlp = GatedMLP(embed_dim, enhancement, activation)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        phi = phi + self.attention(phi, coords, weights)
        return phi + self.mlp(self.norm(phi))


# class Block(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#     ):

#         super().__init__()

#         self.attention = FourierAttention(embed_dim, kernel_scale)
#         self.mlp = GatedMLP(embed_dim, enhancement, activation)

#     def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
#         phi = phi + self.attention(phi, coords, weights)
#         return phi + self.mlp(torch.tanh(phi))


# class Block(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         kernel_scale: float,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#     ):

#         super().__init__()

#         self.attention = FourierAttention(embed_dim, kernel_scale)
#         self.mlp = GatedMLP(embed_dim, enhancement, activation)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, phi: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
#         phi = phi + self.attention(phi, coords, weights)
#         return self.norm(phi + self.mlp(phi))


# class FieldProjection(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         out_features: int = 1,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#     ):

#         super().__init__()

#         self.activation = activation_func(activation)

#         self.norm = nn.LayerNorm(embed_dim)
#         self.mlp = GatedMLP(embed_dim, enhancement, activation)
#         self.proj = nn.Linear(embed_dim, out_features)

#     def forward(self, phi: Tensor) -> Tensor:
#         phi = self.mlp(self.norm(phi))
#         return self.proj(torch.tanh(phi))


# class FieldProjection(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         out_features: int = 1,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#     ):

#         super().__init__()

#         self.activation = activation_func(activation)

#         self.mlp = MLP(embed_dim, enhancement, activation)
#         self.proj = nn.Linear(embed_dim, out_features)

#     def forward(self, phi: Tensor) -> Tensor:
#         phi = self.mlp(torch.tanh(phi))
#         return self.proj(self.activation(phi))


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

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, phi: Tensor) -> Tensor:
        phi = self.mlp(self.norm1(phi))
        return self.proj(self.activation(self.norm2(phi)))


# class FieldProjection(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         out_features: int = 1,
#         enhancement: float = 2.0,
#         activation: Activation = "silu",
#     ):

#         super().__init__()

#         self.activation = activation_func(activation)

#         self.mlp = GatedMLP(embed_dim, enhancement, activation)
#         self.proj = nn.Linear(embed_dim, out_features)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, phi: Tensor) -> Tensor:
#         phi = self.mlp(self.norm(phi))
#         return self.proj(self.activation(phi))


class GlobalDensityApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        kernel_scale: float,
        enhancement: float = 2.0,
        activation: Activation = "silu",
        eta: float = 1e-3,
    ):

        super().__init__()

        self.eta = eta

        # self.embedding = DensityEmbedding(embed_dim, kernel_scale, enhancement, activation)
        self.embedding = DensityEmbedding(embed_dim, kernel_scale, enhancement)

        make_block = lambda: Block(embed_dim, kernel_scale, enhancement, activation)
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])

        self.proj = FieldProjection(embed_dim, 1, enhancement, activation)

    # def eval_tau(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

    #     phi = self(rho, gamma, coords, weights)
    #     t0, tw = t_thomas_fermi(rho + 1e-12), t_weisacker(rho + 1e-12, gamma)

    #     # return tw + torch.exp(phi) * (t0 + self.eta * tw)
    #     return tw + torch.expm1(phi) * (t0 + self.eta * tw)

    def eval_tau(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:
        phi = self(rho, gamma, coords, weights)
        t0, tw = t_thomas_fermi(rho + 1e-12), t_weisacker(rho + 1e-12, gamma)
        return tw + phi * (t0 + tw)

    def rotate_coords(self, wrho: Tensor, coords: Tensor) -> Tensor:

        means, covs = mean_and_covariance(wrho.detach(), coords)
        s2, R = torch.linalg.eigh(covs)

        coords_ = (coords - means.unsqueeze(-2)) @ R.mT.detach()
        return coords_ / torch.sqrt(s2 + 1e-5).unsqueeze(-2)

    def forward(self, rho: Tensor, gamma: Tensor, coords: Tensor, weights: Tensor) -> Tensor:

        coords = self.rotate_coords(weights * rho, coords)
        phi = self.embedding(rho, gamma, coords, weights)
        # phi = self.embedding(rho, gamma)

        for block in self.blocks:
            phi = block(phi, coords, weights)

        phi = self.proj(phi).squeeze(dim=-1)

        # return torch.tanh(log_cosh(phi))
        return log_cosh(self.proj(phi))
