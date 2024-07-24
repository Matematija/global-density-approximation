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


def _density_inputs(dm, ao, grad_ao, xc_type="LDA"):

    rho = torch.einsum("mn,xm,xn->x", dm, ao, ao)

    if xc_type in ["GGA", "MGGA"]:
        grad_rho = 2 * torch.einsum("mn,xm,cxn->xc", dm, ao, grad_ao)
        gamma = torch.sum(grad_rho**2, dim=-1)
    else:
        gamma = None

    return rho, gamma


class GDANumInt(NumInt):
    def __init__(self, gda: GlobalDensityApprox, kinetic: bool = True, xc: bool = True):

        super().__init__()

        if torch.cuda.is_available():
            self.gda = gda.cuda().eval()
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.gda = gda.cpu().eval()
            self.device = torch.device("cpu")

        self.gda.zero_grad()

        if not kinetic and not xc:
            raise ValueError("At least one of kinetic or XC energy must be included.")

        self.kinetic = kinetic
        self.xc = xc

    def _process_mol(self, mol, grids, xc_type="LDA", device=None, dtype=torch.float32):

        if grids.coords is None:
            grids.build(with_non0tab=True)

        if xc_type not in ["LDA", "GGA", "MGGA"]:
            raise NotImplementedError("Only LDA, GGA, and MGGA XC functionals are supported.")

        if xc_type == "LDA":
            ao_vals = self.eval_ao(mol, grids.coords, deriv=0, cutoff=grids.cutoff)
            ao = torch.tensor(ao_vals, device=device, dtype=dtype)
            grad_ao = None
        else:
            ao_vals = self.eval_ao(mol, grids.coords, deriv=1, cutoff=grids.cutoff)
            ao = torch.tensor(ao_vals[0], device=device, dtype=dtype)
            grad_ao = torch.tensor(ao_vals[1:4], device=device, dtype=dtype)

        coords = torch.tensor(grids.coords, device=device, dtype=dtype)
        weights = torch.tensor(grids.weights, device=device, dtype=dtype)

        return ao, grad_ao, coords, weights

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, hermi, max_memory, verbose

        xc_type = libxc.xc_type(xc_code)

        ao, grad_ao, coords, weights = self._process_mol(
            mol, grids, xc_type=xc_type, device=self.device
        )

        dm = torch.tensor(dms, requires_grad=True, device=self.device, dtype=torch.float32)
        rho, gamma = _density_inputs(dm, ao, grad_ao, xc_type=xc_type)

        if self.kinetic or xc_type == "MGGA":
            tau = self.gda.eval_tau(rho, gamma, coords, weights)
        else:
            tau = None

        E = 0.0

        if self.kinetic:
            E = E + weights @ tau

        if self.xc:
            E = E + weights @ eval_xc(xc_code, rho, gamma, tau)

        (X,) = autograd.grad(E, dm)
        X = (X + X.mT) / 2

        with torch.no_grad():
            N = weights @ rho

        nelec = N.detach().item()
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def nr_uks(self, *args, **kwargs):
        raise NotImplementedError("Unrestricted DFT is not implemented yet for GDA functionals.")


class RKS(dft.rks.RKS):
    def __init__(self, *args, gda=None, **kwargs):
        super().__init__(*args, **kwargs)

        if gda is not None:
            self.gda = gda

    @property
    def gda(self):
        if isinstance(self._numint, GDANumInt):
            return self._numint.gda
        else:
            return None

    @gda.setter
    def gda(self, value):
        self._numint = GDANumInt(value, kinetic=False, xc=True)
