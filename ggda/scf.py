from warnings import warn

import torch
from torch import autograd

from pyscf import gto, dft
from pyscf.dft.numint import NumInt

from .gda import GlobalDensityApprox
from .libxc import eval_xc

Molecule = gto.mole.Mole
KohnShamDFT = dft.RKS
Grid = dft.gen_grid.Grids


class GDANumInt(NumInt):
    def __init__(self, xc: str, gda: GlobalDensityApprox):

        super().__init__()
        self.xc = xc

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

        ao_vals = self.eval_ao(mol, grids.coords, deriv=1, cutoff=grids.cutoff)

        coords = torch.tensor(grids.coords, device=self.device, dtype=torch.float32)
        weights = torch.tensor(grids.weights, device=self.device, dtype=torch.float32)
        ao = torch.tensor(ao_vals[0], device=self.device, dtype=torch.float32)
        grad_ao = torch.tensor(ao_vals[1:], device=self.device, dtype=torch.float32)
        dm = torch.tensor(dms, device=self.device, dtype=torch.float32, requires_grad=True)

        rho = torch.einsum("mn,xm,xn->x", dm, ao, ao)
        grad_rho = 2 * torch.einsum("mn,xm,cxn->xc", dm, ao, grad_ao)
        gamma = torch.sum(grad_rho**2, dim=-1)

        tau = self.gda.eval_tau(rho, gamma, coords, weights)
        exc = eval_xc(self.xc, rho, gamma, tau)
        Exc = weights @ exc

        (X,) = autograd.grad(Exc, dm, grad_outputs=torch.ones_like(Exc))

        with torch.no_grad():
            N = weights @ rho

        nelec = N.detach().item()
        excsum = Exc.detach().item()
        vmat = X.detach().cpu().numpy()

        return nelec, excsum, vmat

    def nr_uks(self, *args, **kwargs):
        raise NotImplementedError("Unrestricted DFT is not implemented yet for GDA functionals.")
