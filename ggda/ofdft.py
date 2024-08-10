import numpy as np

from scipy.linalg import solve
from scipy.optimize import minimize

from pyscf.dft.rks import RKS

from .gda import GlobalDensityApprox
from .scf import GDANumInt


def _get_iprint(verbose):

    if verbose == 0:
        return -1
    elif verbose == 1 and verbose < 2:
        return 0
    elif verbose >= 2 and verbose < 4:
        return 10
    elif verbose >= 4 and verbose < 6:
        return 1
    elif verbose >= 6 and verbose < 8:
        return 99
    elif verbose >= 8:
        return 100
    else:
        return 1


class OrbitalFree(RKS):
    def __init__(self, *args, kef=None, gda=None, eps=1e-12, **kwargs):

        super().__init__(*args, **kwargs)

        if kef is None or isinstance(kef, str):
            self.kef = kef
        else:
            raise ValueError(
                f"Invalid value for `kef`. Must be a string or `None`, got {type(kef)}."
            )

        if gda is not None:
            self.gda = gda

        self.eps = eps
        self.grad_hook = None
        self.of_reset()

    @property
    def gda(self):
        if isinstance(self._numint, GDANumInt):
            return self._numint.gda
        else:
            return None

    @gda.setter
    def gda(self, value):

        if not isinstance(value, GlobalDensityApprox):
            raise TypeError(f"Expected a `GlobalDensityApprox` object, got {type(value)}.")

        self._numint = GDANumInt(value, kinetic=True)

    @property
    def chemical_potential(self):

        if self.mo_energy is None:
            return None
        else:
            return self.mo_energy[0]

    @property
    def s1e(self):

        if self._s1e is None:
            self._s1e = self.get_ovlp()

        return self._s1e

    @property
    def h1e(self):

        if self._h1e is None:
            self._h1e = self.get_hcore()

        return self._h1e

    def get_init_guess(self, *args, **kwargs):

        dm0 = super().get_init_guess(*args, **kwargs)

        self.grids.build()
        coords, weights = self.grids.coords, self.grids.weights

        ao = self._numint.eval_ao(self.mol, coords, deriv=0, cutoff=self.grids.cutoff)
        rho0 = self._numint.eval_rho(self.mol, ao, dm0, xctype="LDA")
        p0 = rho0 / self.mol.nelectron

        s1e = self.mol.intor_symmetric("int1e_ovlp")
        b = (weights * np.sqrt(p0)) @ ao

        return solve(s1e, b, assume_a="sym")

    def get_veff(self, *args, **kwargs):

        if self.kef is None:
            return super().get_veff(*args, **kwargs)

        else:
            xc = self.xc
            self.xc += " + " + self.kef
            veff = super().get_veff(*args, **kwargs)
            self.xc = xc

            return veff

    def energy(self, a):

        sa = self.s1e @ a
        asa = a @ sa

        N = self.mol.nelectron
        dm = N * np.outer(a, a) / asa

        veff = self.get_veff(self.mol, dm)
        E = self.energy_tot(dm, self.h1e, veff)
        heff = self.get_fock(self.h1e, self.s1e, veff, dm)

        ha = heff @ a
        aha = a @ ha
        mu = aha / asa
        grad = 2 * N * mu * (ha / aha - sa / asa)

        if self.grad_hook is not None:
            grad = self.grad_hook(grad)

        return E, grad, mu

    def scf(self, c0=None):

        self.dump_flags()
        self.build(self.mol)

        if c0 is None:
            c0 = self.get_init_guess()

        options = {
            "maxiter": self.max_cycle,
            "iprint": _get_iprint(self.verbose),
            "ftol": self.conv_tol * 1e-2,
        }

        if self.verbose >= 4:
            E0, _, _ = self.energy(c0)
            print(f"Initial Energy: {E0:.6f}")

        def energy_value_and_grad(a):
            E, grad, _ = self.energy(a)
            return E, grad

        res = minimize(
            energy_value_and_grad,
            x0=c0,
            method="L-BFGS-B",
            jac=True,
            tol=self.conv_tol,
            options=options,
        )

        c = res.x / np.sqrt(res.x @ self.s1e @ res.x)
        _, _, mu = self.energy(res.x)

        self.converged = res.success
        self.e_tot = res.fun
        self.mo_energy = np.array([mu])
        self.mo_coeff = np.expand_dims(c, axis=-1)
        self.mo_occ = np.array([self.mol.nelectron])

        return self.e_tot

    def of_reset(self):
        self._s1e = None
        self._h1e = None

    def reset(self):
        super().reset()
        self.of_reset()
