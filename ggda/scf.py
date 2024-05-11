from typing import Any, NamedTuple, Optional

from warnings import warn

import numpy as np

import torch
from torch import autograd
from torch import Tensor

from pyscf import gto, dft
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox

Molecule = gto.mole.Mole
KohnShamDFT = dft.RKS
Grid = dft.gen_grid.Grids


class GridChunk(NamedTuple):
    coords: Tensor
    weights: Tensor


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

        self._mol_data = None
        self._grid_chunk = None

    def _xc_type(self, *_, **__):
        return "LDA"

    @property
    def is_encoded(self):
        return self._mol_data is not None

    def encode_molecule(self, mol: Molecule):

        atom_coords, atom_charges = mol.atom_coords(), mol.atom_charges()
        atom_coords = torch.from_numpy(atom_coords).to(device=self.device, dtype=torch.float32)
        atom_charges = torch.from_numpy(atom_charges).to(device=self.device, dtype=torch.long)

        with torch.no_grad():
            return self.gda.encode(atom_coords, atom_charges)

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

        if not self.is_encoded:
            self._mol_data = self.encode_molecule(mol)

        for ao, mask, weights, coords in super().block_loop(
            mol=mol,
            grids=grids,
            nao=nao,
            deriv=deriv,
            max_memory=max_memory,
            non0tab=non0tab,
            blksize=blksize,
            buf=buf,
        ):

            coords_ = torch.tensor(coords, device=self.device, dtype=torch.float32)
            weights_ = torch.tensor(weights, device=self.device, dtype=torch.float32)
            self._grid_chunk = GridChunk(coords_, weights_)

            yield ao, mask, weights, coords

    def eval_exc(self, rho: Tensor, spin: int = 0) -> Tensor:

        if spin == 0:
            exc = self.gda.predict(rho, self._grid_chunk.coords, self._mol_data)

        else:
            rho_a, rho_b = torch.unbind(rho, dim=0)
            exc_a = self.gda.predict(2 * rho_a, self._grid_chunk.coords, self._mol_data)
            exc_b = self.gda.predict(2 * rho_b, self._grid_chunk.coords, self._mol_data)
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

    def eval_fxc(
        self, rho: Tensor, vxc: Tensor, spin: int = 0, *, create_graph: bool = False
    ) -> Tensor:

        if spin == 0:
            grad_outputs = torch.ones_like(rho)
            (v2rho,) = autograd.grad(vxc, rho, grad_outputs=grad_outputs, create_graph=create_graph)

        else:
            rho_a, rho_b = torch.unbind(rho, dim=0)
            vxc_a, vxc_b = torch.unbind(vxc, dim=0)
            grad_outputs = torch.ones_like(rho_a)

            # fmt: off
            (v2rho_aa,) = autograd.grad(vxc_a, rho_a, grad_outputs=grad_outputs, create_graph=create_graph)
            (v2rho_ab,) = autograd.grad(vxc_a, rho_b, grad_outputs=grad_outputs, create_graph=create_graph)
            (v2rho_bb,) = autograd.grad(vxc_b, rho_b, grad_outputs=grad_outputs, create_graph=create_graph)
            # fmt: on

            v2rho = torch.stack([v2rho_aa, v2rho_ab, v2rho_bb], dim=0)

        return v2rho

    def eval_kxc(self, *_, **__) -> Tensor:
        raise NotImplementedError("Third density derivatives of functionals are not supported yet!")

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
