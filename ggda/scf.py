from typing import Any
from warnings import warn
from copy import deepcopy

import numpy as np

import torch
from torch import autograd

from pyscf import dft
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox
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
        self._numint = GDANumInt(value)


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
        self._numint = GDANumInt(value)


####################################################################################################


class GDANumInt(NumInt):
    def __init__(self, gda: GlobalDensityApprox, eps: float = 0.0, dtype: Any = torch.float64):

        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.device = torch.device("cpu")

        self.gda = deepcopy(gda).eval().to(device=self.device, dtype=dtype)

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

        if xc_type == "MGGA":

            if not spin:
                tau = self.gda.log_tau(rho, grad_rho, coords, weights, eps=self.eps).exp()

            else:
                rho_a, rho_b = rho.unbind(dim=-1)
                grad_rho_a, grad_rho_b = grad_rho.unbind(dim=-1)

                # fmt: off
                tau_a = self.gda.log_tau(2*rho_a, 2*grad_rho_a, coords, weights, eps=self.eps).exp()
                tau_b = self.gda.log_tau(2*rho_b, 2*grad_rho_b, coords, weights, eps=self.eps).exp()
                # fmt: on

                tau = 0.5 * torch.stack([tau_a, tau_b], dim=-1)

        else:
            tau = None

        return weights @ eval_xc(xc_code, rho, grad_rho, tau, spin=spin)

    def nr_vxc_aux(self, mol, grids, xc_code, dm, spin, hermi, *, create_graph=False):

        xc_type = libxc.xc_type(xc_code)

        ao, grad_ao, coords, weights = self.process_mol(mol, grids, xc_type=xc_type)
        dm = torch.as_tensor(dm, device=self.device, dtype=self.dtype).requires_grad_(True)
        rho, grad_rho = self.density_inputs(dm, ao, grad_ao, xc_type=xc_type)
        E = self.xc_energy(xc_code, rho, grad_rho, coords, weights, spin=spin, xc_type=xc_type)
        (X,) = autograd.grad(E, dm, create_graph=create_graph)

        if hermi:
            X = (X + X.mT) / 2

        with torch.no_grad():
            N = weights @ rho

        return E, N, X

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, max_memory, verbose

        E, N, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=0, hermi=hermi)

        nelec = N.detach().item()
        excsum = E.detach().item()
        vmat = X.detach().cpu().numpy().astype(np.float64)

        return nelec, excsum, vmat

    def nr_uks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):
        del relativity, max_memory, verbose

        E, N, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=1, hermi=hermi)

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

        del rho0, vxc, fxc, relativity, max_memory, verbose

        dm0 = torch.tensor(dm0, requires_grad=True, device=self.device, dtype=self.dtype)
        _, _, X = self.nr_vxc_aux(mol, grids, xc_code, dm0, spin=0, hermi=hermi, create_graph=True)

        if isinstance(dms, np.ndarray) and dms.ndim == 2:
            dms = [dms]

        hvps = []

        for dm in dms:

            dm = torch.tensor(dm, device=self.device, dtype=self.dtype)
            (hvp,) = autograd.grad(X, dm0, grad_outputs=dm, retain_graph=True)

            if hermi:
                hvp = (hvp + hvp.mT) / 2

            hvps.append(hvp.detach().cpu().numpy().astype(np.float64))

        if len(hvps) == 1:
            return hvps[0]
        else:
            return np.stack(hvps, axis=0)

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
