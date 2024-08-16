from typing import Any
from warnings import warn
from copy import deepcopy

import numpy as np

import torch
from torch import autograd
from torch.autograd.functional import hvp

from pyscf import dft
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox
from .features import log_t_weiszacker
from .libxc import eval_xc


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


####################################################################################################


class GDANumInt(NumInt):
    def __init__(
        self,
        gda: GlobalDensityApprox,
        kinetic: bool = False,
        eps: float = 0.0,
        dtype: Any = torch.float64,
    ):

        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.device = torch.device("cpu")

        self.gda = deepcopy(gda).eval().to(device=self.device, dtype=dtype)

        self.kinetic = kinetic
        self.eps = eps
        self.dtype = dtype

        self.gda.zero_grad()

    def process_mol(self, mol, grids, xc_type):

        if grids.coords is None:
            grids.build(with_non0tab=True)

        if xc_type not in ["LDA", "GGA", "MGGA"]:
            raise NotImplementedError("Only LDA, GGA, and MGGA XC functionals are supported.")

        if xc_type == "LDA" and not self.kinetic:
            ao_vals = self.eval_ao(mol, grids.coords, deriv=0, cutoff=grids.cutoff)
            ao = torch.tensor(ao_vals, device=self.device, dtype=self.dtype)
            grad_ao = None
        else:
            ao_vals = self.eval_ao(mol, grids.coords, deriv=1, cutoff=grids.cutoff)
            ao = torch.tensor(ao_vals[0], device=self.device, dtype=self.dtype)
            grad_ao = torch.tensor(ao_vals[1:4], device=self.device, dtype=self.dtype)

        coords = torch.tensor(grids.coords, device=self.device, dtype=self.dtype)
        weights = torch.tensor(grids.weights, device=self.device, dtype=self.dtype)

        return ao, grad_ao, coords, weights

    def density_inputs(self, dm, ao, grad_ao=None, xc_type="LDA"):

        rho = torch.einsum("...mn,xm,xn->x...", dm, ao, ao)

        if xc_type in ["GGA", "MGGA"] or self.kinetic:
            grad_rho = 2 * torch.einsum("...mn,xm,cxn->x...c", dm, ao, grad_ao)
        else:
            grad_rho = None

        return rho, grad_rho

    def xc_energy(self, xc_code, rho, grad_rho, coords, weights, spin=0, xc_type=None):

        if xc_type is None:
            xc_type = libxc.xc_type(xc_code)

        if self.kinetic or xc_type == "MGGA":

            if not spin:
                log_tau_p = self.gda.log_tau(
                    rho, grad_rho, coords, weights, pauli=True, eps=self.eps
                )
            else:
                raise NotImplementedError("UKS not working yet.")

        if xc_type == "MGGA":

            if not spin:
                gamma = torch.sum(grad_rho**2, dim=-1)
                log_tw = log_t_weiszacker(rho, gamma, eps=self.eps)
                tau = torch.logaddexp(log_tw, log_tau_p).exp()
            else:
                raise NotImplementedError("Meta-GGA is not implemented for UKS.")

        else:
            tau = None

        E = weights @ eval_xc(xc_code, rho, grad_rho, tau, spin=spin)

        if self.kinetic:

            if not spin:
                Tp = torch.logsumexp(torch.log(weights) + log_tau_p, dim=-1).exp()
            else:
                raise NotImplementedError("UKS not working yet.")

            E = E + Tp

        return E

    def nr_vxc_aux(self, mol, grids, xc_code, dms, spin):

        xc_type = libxc.xc_type(xc_code)

        ao, grad_ao, coords, weights = self.process_mol(mol, grids, xc_type=xc_type)
        dm = torch.tensor(dms, requires_grad=True, device=self.device, dtype=self.dtype)

        rho, grad_rho = self.density_inputs(dm, ao, grad_ao, xc_type=xc_type)
        E = self.xc_energy(xc_code, rho, grad_rho, coords, weights, spin=spin, xc_type=xc_type)

        (X,) = autograd.grad(E, dm)
        X = (X + X.mT) / 2

        with torch.no_grad():
            N = weights @ rho

        return E, N, X

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, hermi, max_memory, verbose

        E, N, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=0)

        nelec = N.detach().item()
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def nr_uks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):
        del relativity, hermi, max_memory, verbose

        E, N, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=1)

        nelec = N.detach().cpu().numpy().astype(np.float64)
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0, max_memory=2000):

        self.dm0 = np.einsum("a,ma,na->mn", mo_occ, mo_coeff, mo_coeff)

        return super().cache_xc_kernel(
            mol, grids, xc_code, mo_coeff, mo_occ, spin=spin, max_memory=max_memory
        )

    def nr_rks_fxc(
        self,
        mol,
        grids,
        xc_code,
        dm0,
        dms,
        relativity=0,
        hermi=0,
        rho0=None,
        vxc=None,
        fxc=None,
        max_memory=2000,
        verbose=None,
    ):

        if dm0 is None:
            dm0 = self.dm0

        del rho0, vxc, fxc, relativity, hermi, max_memory, verbose

        xc_type = libxc.xc_type(xc_code)

        ao, grad_ao, coords, weights = self.process_mol(mol, grids, xc_type=xc_type)

        dm0 = torch.tensor(dm0, device=self.device, dtype=self.dtype)
        dms = torch.tensor(dms, device=self.device, dtype=self.dtype)

        def eval_energy(dm):
            rho, gamma = self.density_inputs(dm, ao, grad_ao, xc_type=xc_type)
            return self.xc_energy(xc_code, rho, gamma, coords, weights, spin=0, xc_type=xc_type)

        hvps = [hvp(eval_energy, dm0, dm)[1] for dm in dms]
        return np.stack([h.cpu().numpy() for h in hvps], axis=0)

    def nr_uks_fxc(
        self,
        mol,
        grids,
        xc_code,
        dm0,
        dms,
        relativity=0,
        hermi=0,
        rho0=None,
        vxc=None,
        fxc=None,
        max_memory=2000,
        verbose=None,
    ):

        del rho0, vxc, fxc, relativity, hermi, max_memory, verbose

        raise NotImplementedError("Second derivatives is not implemented for UKS.")
