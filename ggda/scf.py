from warnings import warn

import numpy as np

import torch
from torch import autograd

from pyscf import gto, dft
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox
from .libxc import eval_xc

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

    def to_tensor(self, arr, requires_grad=False):
        return torch.tensor(
            arr, device=self.device, dtype=torch.float32, requires_grad=requires_grad
        )

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, hermi, max_memory, verbose

        if grids.coords is None:
            grids.build(with_non0tab=True)

        xc_type = libxc.xc_type(xc_code)

        if xc_type not in ["LDA", "GGA", "MGGA"]:
            raise NotImplementedError("Only LDA, GGA, and MGGA XC functionals are supported.")

        if xc_type == "LDA":
            ao_vals = self.eval_ao(mol, grids.coords, deriv=0, cutoff=grids.cutoff)
            ao = torch.tensor(ao_vals, device=self.device, dtype=torch.float32)
        else:
            ao_vals = self.eval_ao(mol, grids.coords, deriv=1, cutoff=grids.cutoff)
            ao, grad_ao = self.to_tensor(ao_vals[0]), self.to_tensor(ao_vals[1:4])

        coords = self.to_tensor(grids.coords)
        weights = self.to_tensor(grids.weights)

        dm = self.to_tensor(dms, requires_grad=True)
        rho = torch.einsum("mn,xm,xn->x", dm, ao, ao)
        gamma, tau = None, None

        if xc_type in ["GGA", "MGGA"]:
            grad_rho = 2 * torch.einsum("mn,xm,cxn->xc", dm, ao, grad_ao)
            gamma = torch.sum(grad_rho**2, dim=-1)

        if xc_type == "MGGA":
            tau = self.gda.eval_tau(rho, gamma, coords, weights)

        Exc = weights @ eval_xc(xc_code, rho, gamma, tau)
        (X,) = autograd.grad(Exc, dm, grad_outputs=torch.ones_like(Exc))

        X = (X + X.mT) / 2

        with torch.no_grad():
            N = weights @ rho

        nelec = N.detach().item()
        excsum = Exc.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def nr_uks(self, *args, **kwargs):
        raise NotImplementedError("Unrestricted DFT is not implemented yet for GDA functionals.")
