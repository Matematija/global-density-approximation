from typing import Any, NamedTuple

from warnings import warn

import numpy as np

import torch
from torch import autograd
from torch import Tensor

from pyscf import gto, dft
from pyscf.dft.numint import NumInt
from pyscf.gto.eval_gto import BLKSIZE

from .gda import GlobalDensityApprox

Molecule = gto.mole.Mole
KohnShamDFT = dft.RKS
Grid = dft.gen_grid.Grids


class CachedGrid(NamedTuple):

    coords: Tensor
    weights: Tensor

    def __len__(self):
        return len(self.weights)


class GDANumInt(NumInt):
    def __init__(self, gda: GlobalDensityApprox):

        super().__init__()

        if torch.cuda.is_available():
            self.gda = gda.cuda().eval()
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.gda = gda.cpu().eval()
            self.device = torch.device("cpu")

        self.gda.zero_grad()

        self._grid = None

    def _xc_type(self, *_, **__):
        return "LDA"

    def cache_grid(self, grids: Grid):

        coords, weights = grids.coords, grids.weights
        coords_ = torch.tensor(coords, device=self.device, dtype=torch.float32)
        weights_ = torch.tensor(weights, device=self.device, dtype=torch.float32)

        self._grid = CachedGrid(coords_, weights_)

    def block_loop(
        self,
        mol: Molecule,
        grids: Grid,
        nao=None,
        deriv=0,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        buf=None,
    ):

        npts = len(grids.weights)
        blksize = (npts // BLKSIZE + 1) * BLKSIZE

        if self._grid is None or len(self._grid) != npts:
            self.cache_grid(grids)

        return super().block_loop(
            mol=mol,
            grids=grids,
            nao=nao,
            deriv=deriv,
            max_memory=max_memory,
            non0tab=non0tab,
            blksize=blksize,
            buf=buf,
        )

    def eval_exc(self, rho: Tensor, spin: int = 0) -> Tensor:

        coords, weights = self._grid

        if spin == 0:
            exc = self.gda.predict(rho, coords, weights)

        else:
            rho_a, rho_b = rho.unbind(dim=0)
            exc_a = self.gda.predict(2 * rho_a, coords, weights)
            exc_b = self.gda.predict(2 * rho_b, coords, weights)
            exc = (exc_a + exc_b) / 2

        return exc

    def eval_vxc(
        self, rho: Tensor, exc: Tensor, spin: int = 0, *, create_graph: bool = False
    ) -> Tensor:

        if spin == 0:
            (vrho,) = autograd.grad(rho @ exc, rho, create_graph=create_graph)

        else:
            rho_a, rho_b = torch.unbind(rho, dim=0)

            (vrho_a,) = autograd.grad(rho_a @ exc, rho_a, create_graph=create_graph)
            (vrho_b,) = autograd.grad(rho_b @ exc, rho_b, create_graph=create_graph)

            vrho = torch.stack([vrho_a, vrho_b], dim=0)

        return vrho

    @staticmethod
    def to_numpy(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    def eval_xc(
        self,
        xc_code: str,
        rho: np.ndarray,
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        omega: Any = None,
        verbose: int = None,
    ):

        del xc_code, relativity, omega, verbose

        assert deriv < 4, f"Invalid derivative order: {deriv}!"

        rho = torch.tensor(rho, requires_grad=deriv >= 1, device=self.device, dtype=torch.float32)

        exc_ = self.eval_exc(rho, spin=spin)
        exc = self.to_numpy(exc_)

        vxc, fxc, kxc = None, None, None

        if deriv >= 1:
            vrho = self.eval_vxc(rho, exc_, spin=spin, create_graph=deriv >= 2)
            vxc = (self.to_numpy(vrho),)

        if deriv >= 2:
            v2rho = self.eval_fxc(rho, vrho, spin=spin, create_graph=deriv >= 3)
            fxc = (self.to_numpy(v2rho),)

        if deriv >= 3:
            v3rho = self.eval_kxc(rho, v2rho, spin=spin, create_graph=deriv >= 4)
            kxc = (self.to_numpy(v3rho),)

        return exc, vxc, fxc, kxc
