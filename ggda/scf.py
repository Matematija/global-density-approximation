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
    def __init__(self, gda: GlobalDensityApprox, kinetic: bool = False, eps: float = 1e-12):

        super().__init__()

        if torch.cuda.is_available():
            self.gda = gda.cuda().eval()
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.gda = gda.cpu().eval()
            self.device = torch.device("cpu")

        self.eps = eps
        self.kinetic = kinetic
        self.gda.zero_grad()

    def process_mol(self, mol, grids, xc_type, device=None, dtype=torch.float32):

        if grids.coords is None:
            grids.build(with_non0tab=True)

        if xc_type not in ["LDA", "GGA", "MGGA"]:
            raise NotImplementedError("Only LDA, GGA, and MGGA XC functionals are supported.")

        if xc_type == "LDA" and not self.kinetic:
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

    def density_inputs(self, dm, ao, grad_ao, xc_type):

        rho = torch.einsum("...mn,xm,xn->x...", dm, ao, ao)

        if xc_type in ["GGA", "MGGA"] or self.kinetic:
            grad_rho = 2 * torch.einsum("...mn,xm,cxn->x...c", dm, ao, grad_ao)
            gamma = torch.sum(grad_rho**2, dim=-1)
        else:
            gamma = None

        return rho, gamma

    def total_energy(self, xc_code, dm, ao, grad_ao, coords, weights, spin=0):

        xc_type = libxc.xc_type(xc_code)

        rho, gamma = self.density_inputs(dm, ao, grad_ao, xc_type=xc_type)
        tau = None

        if self.kinetic or xc_type == "MGGA":

            if not spin:
                tau = self.gda.eval_tau(rho, gamma, coords, weights, eps=self.eps)
            else:

                rho_a, rho_b = rho.unbind(-1)
                gamma_a, gamma_b = gamma.unbind(-1)

                tau_a = self.gda.eval_tau(rho_a, gamma_a, coords, weights, eps=self.eps)
                tau_b = self.gda.eval_tau(rho_b, gamma_b, coords, weights, eps=self.eps)

                tau = torch.stack([tau_a, tau_b], dim=-1)

        E = weights @ eval_xc(xc_code, rho, gamma, tau, spin=spin)

        if self.kinetic:

            if spin:
                tau = tau.sum(dim=-1)

            E = E + weights @ tau

        with torch.no_grad():
            N = weights @ rho

        return E, N

    def xc_mat(self, E, dm):
        (X,) = autograd.grad(E, dm)
        return (X + X.mT) / 2

    def nr_aux(self, mol, grids, xc_code, dms, spin):

        ao, grad_ao, coords, weights = self.process_mol(
            mol, grids, xc_type=libxc.xc_type(xc_code), device=self.device
        )

        dm = torch.tensor(dms, requires_grad=True, device=self.device, dtype=torch.float32)
        E, N = self.total_energy(xc_code, dm, ao, grad_ao, coords, weights, spin=spin)
        X = self.xc_mat(E, dm)

        return E, N, X

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, hermi, max_memory, verbose

        E, N, X = self.nr_aux(mol, grids, xc_code, dms, spin=0)

        nelec = N.detach().item()
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def nr_uks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):
        del relativity, hermi, max_memory, verbose

        E, N, X = self.nr_aux(mol, grids, xc_code, dms, spin=1)

        nelec = N.detach().cpu().numpy().astype(np.float64)
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat


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
        self._numint = GDANumInt(value, kinetic=False)


class UKS(dft.uks.UKS):
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
        self._numint = GDANumInt(value, kinetic=False)
