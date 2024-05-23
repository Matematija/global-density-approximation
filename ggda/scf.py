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

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del xc_code, relativity, hermi, max_memory, verbose

        if grids.coords is None:
            grids.build(with_non0tab=True)

        ao_vals = self.eval_ao(mol, grids.coords, deriv=0, cutoff=grids.cutoff)

        coords = torch.tensor(grids.coords, device=self.device, dtype=torch.float32)
        weights = torch.tensor(grids.weights, device=self.device, dtype=torch.float32)
        ao = torch.tensor(ao_vals, device=self.device, dtype=torch.float32)
        dm = torch.tensor(dms, device=self.device, dtype=torch.float32, requires_grad=True)

        rho = torch.einsum("mn,xm,xn->x", dm, ao, ao)
        wrho = weights * rho

        Exc = wrho @ self.gda(rho, coords, weights)
        (X,) = autograd.grad(Exc, dm, grad_outputs=torch.ones_like(Exc))

        with torch.no_grad():
            N = torch.sum(wrho)

        nelec = N.detach().item()
        excsum = Exc.detach().item()
        vmat = X.detach().cpu().numpy()

        return nelec, excsum, vmat

    def nr_uks(self, *args, **kwargs):
        raise NotImplementedError("Unrestricted DFT is not implemented yet for GDA functionals.")
