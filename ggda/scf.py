from typing import Any, Optional
from warnings import warn
from copy import deepcopy
from functools import lru_cache

import numpy as np

import torch
from torch import autograd
from torch import Tensor

from pyscf import dft
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox
from .libxc import eval_xc
from .utils import to_numpy


class KohnShamGDA:
    def __init__(self, *args, **kwargs):

        gda = kwargs.pop("gda", None)

        if gda is not None:
            self.gda = gda

        super().__init__(*args, **kwargs)

    @property
    def is_gda(self):
        return isinstance(self._numint, GDANumInt)

    @property
    def gda(self):

        if self.is_gda:
            return self._numint.gda
        else:
            return None

    @gda.setter
    def gda(self, value):

        if isinstance(value, GlobalDensityApprox):
            self._numint = GDANumInt(value)
        else:
            raise ValueError(f"Invalid GDA model, got type {type(value)}")

    @property
    def gda_chunk_size(self):

        if self.is_gda:
            return self._numint.chunk_size
        else:
            raise AttributeError("GDA is not set.")

    @gda_chunk_size.setter
    def gda_chunk_size(self, value):

        if self.is_gda:
            self._numint.chunk_size = value
        else:
            raise AttributeError("GDA is not set.")


class RKS(KohnShamGDA, dft.rks.RKS):
    pass


class UKS(KohnShamGDA, dft.uks.UKS):
    pass


####################################################################################################


class GDANumInt(NumInt):
    def __init__(
        self,
        gda: GlobalDensityApprox,
        eps: float = 0.0,
        chunk_size: Optional[int] = None,
        dtype: Any = torch.float64,
    ):

        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warn("CUDA is not available. Using CPU.")
            self.device = torch.device("cpu")

        self.gda = deepcopy(gda).eval().to(device=self.device, dtype=dtype)
        self.gda.zero_grad()

        self.eps = eps
        self.chunk_size = chunk_size
        self.dtype = dtype

    def _to_tensor(self, arr: np.ndarray, requires_grad: bool = False) -> Tensor:
        return torch.tensor(arr, requires_grad=requires_grad, device=self.device, dtype=self.dtype)

    @lru_cache(maxsize=4)
    def process_mol(self, mol, grids, xc_type):

        if grids.coords is None:
            grids.build(with_non0tab=True)

        if xc_type == "LDA":
            ao_vals = self.eval_ao(mol, grids.coords, deriv=0, cutoff=grids.cutoff)
            ao, grad_ao = self._to_tensor(ao_vals), None

        elif xc_type in ["GGA", "MGGA"]:
            ao_vals = self.eval_ao(mol, grids.coords, deriv=1, cutoff=grids.cutoff)
            ao, grad_ao = self._to_tensor(ao_vals[0]), self._to_tensor(ao_vals[1:4])

        else:
            raise NotImplementedError("Only LDA, GGA, and MGGA XC functionals are supported.")

        coords, weights = self._to_tensor(grids.coords), self._to_tensor(grids.weights)

        return ao, grad_ao, coords, weights

    def density_inputs(self, dm, ao, grad_ao=None, xc_type="LDA"):

        rho = torch.einsum("...mn,xm,xn->x...", dm, ao, ao)

        if xc_type in ["GGA", "MGGA"]:
            grad_rho = 2 * torch.einsum("...mn,xm,cxn->xc...", dm, ao, grad_ao)
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

    def nr_vxc_aux(self, mol, grids, xc_code, dms, spin, hermi, *, create_graph=False):

        xc_type = libxc.xc_type(xc_code)

        ao, grad_ao, coords, weights = self.process_mol(mol, grids, xc_type=xc_type)

        def vxc_fn(dm):

            rho, grad_rho = self.density_inputs(dm, ao, grad_ao, xc_type=xc_type)
            E = self.xc_energy(xc_code, rho, grad_rho, coords, weights, spin=spin, xc_type=xc_type)
            (X,) = autograd.grad(E, dm, create_graph=create_graph)

            with torch.no_grad():
                N = weights @ rho

            return N, E, X

        if (spin == 0 and dms.ndim == 2) or (spin == 1 and dms.ndim == 3):
            N, E, X = vxc_fn(dms)
        else:
            N, E, X = torch.vmap(vxc_fn, out_dims=(0, 0, 0), chunk_size=self.chunk_size)(dms)

        if hermi:
            X = (X + X.mT) / 2

        return N, E, X

    def nr_rks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):

        del relativity, max_memory, verbose

        dms = self._to_tensor(dms, requires_grad=True)
        N, E, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=0, hermi=hermi)

        return N.detach().item(), E.detach().item(), to_numpy(X)

    def nr_uks(
        self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
    ):
        del relativity, max_memory, verbose

        dms = self._to_tensor(dms, requires_grad=True)
        E, N, X = self.nr_vxc_aux(mol, grids, xc_code, dms, spin=1, hermi=hermi)

        return to_numpy(N), E.detach().item(), to_numpy(X)

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0, max_memory=2000):
        self._cached_dm = np.einsum("a,ma,na->mn", mo_occ, mo_coeff, mo_coeff)
        return super().cache_xc_kernel(mol, grids, xc_code, mo_coeff, mo_occ, spin, max_memory)

    def cache_xc_kernel1(self, mol, grids, xc_code, dm, spin=0, max_memory=2000):
        self._cached_dm = dm
        return super().cache_xc_kernel1(mol, grids, xc_code, dm, spin, max_memory)

    def nr_rks_fxc_aux(self, mol, grids, xc_code, dm0, dms, hermi, spin):

        _, _, X = self.nr_vxc_aux(
            mol, grids, xc_code, dm0, spin=spin, hermi=hermi, create_graph=True
        )

        Xa = X[0] if spin else X

        def hvp_fn(dm):
            (hvp,) = autograd.grad(Xa, dm0, grad_outputs=dm, retain_graph=True)
            return hvp

        if dms.ndim == 2:
            hvps = hvp_fn(dms)
        else:
            hvps = torch.vmap(hvp_fn, chunk_size=self.chunk_size)(dms)

        if hermi:
            hvps = (hvps + hvps.mT) / 2

        return hvps

    def nr_rks_fxc_st(
        self,
        mol,
        grids,
        xc_code,
        dm0,
        dms_alpha,
        relativity=0,
        singlet=True,
        rho0=None,
        vxc=None,
        fxc=None,
        max_memory=2000,
        verbose=None,
    ):

        assert singlet, "Only singlet hessians are supported."

        del rho0, vxc, fxc, relativity, max_memory, verbose

        dm0 = dm0 if dm0 is not None else self._cached_dm
        dm_ab = np.repeat(0.5 * dm0[None], axis=0, repeats=2)
        dm_ab = self._to_tensor(dm_ab, requires_grad=True)
        dms_alpha = self._to_tensor(dms_alpha)

        kdm = self.nr_rks_fxc_aux(mol, grids, xc_code, dm_ab, dms_alpha, hermi=1, spin=1)

        return to_numpy(kdm).sum(axis=1)

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

        del rho0, vxc, fxc, relativity, max_memory, verbose

        dm0 = self._to_tensor(dm0 if dm0 is not None else self._cached_dm, requires_grad=True)
        dms = self._to_tensor(dms)

        hvps = self.nr_rks_fxc_aux(mol, grids, xc_code, dm0, dms, hermi=hermi, spin=0)

        return to_numpy(hvps)

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
