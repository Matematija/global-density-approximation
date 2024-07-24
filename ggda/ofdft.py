from typing import Optional

import numpy as np

from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize

from .scf import RKS, GDANumInt


class OrbitalFree(RKS):
    def __init__(self, *args, kef=None, kind="of", **kwargs):

        super().__init__(*args, **kwargs)

        if kef is None or isinstance(kef, str):
            self.kef = kef
        else:
            raise ValueError(
                f"Invalid value for `kef`. Must be a string or callable, got {type(kef)}."
            )

        self.kind = kind
        self.of_reset()

    @classmethod
    def from_rks(cls, ks: RKS, kef=None, kind="of"):
        return cls(
            ks.mol,
            ks.xc,
            kef=kef,
            kind=kind,
            max_cycle=ks.max_cycle,
            conv_tol=ks.conv_tol,
            verbose=ks.verbose,
        )

    @property
    def gda(self):
        return self._numint.gda

    @gda.setter
    def gda(self, value):
        self._numint = GDANumInt(value, kinetic=True, xc=True)

    def of_init(self):
        # self._h1e = self.get_hcore()
        self._h1e = self.mol.intor("int1e_nuc")
        self._ovlp = self.get_ovlp()

    def of_init_guess(self):
        z = np.ones(self.mol.nao) / np.sqrt(self.mol.nao)
        h = fractional_matrix_power(self._ovlp, -0.5)
        return h @ z

    def get_veff(self, *args, **kwargs):

        if self.kef is None:
            return super().get_veff(*args, **kwargs)

        else:
            xc = self.xc
            self.xc += " + " + self.kef
            veff = super().get_veff(*args, **kwargs)
            self.xc = xc

            return veff

    def of_energy_value_and_grad(self, c):

        N = self.mol.nelectron
        dm = np.outer(c, N * c)

        veff = self.get_veff(self.mol, dm)
        heff = self._h1e + veff
        v = 2 * N * heff @ c

        E = self.energy_tot(dm, self._h1e, veff)

        self._last_energy = E
        self._last_chem_pot = self.chemical_potential(c, heff)

        return E, v

    def of_constraint(self, c):
        return c @ self._ovlp @ c - 1.0

    def of_constraint_jac(self, c):
        return 2 * self._ovlp @ c

    def of_callback(self, _):

        self._iteration += 1

        if self.verbose >= 4:
            i = self._iteration
            E = self._last_energy
            mu = self._last_chem_pot

            print(f"Iteration {i} | Energy: {E:.6f} | Chemical Potential: {mu:.6f}")

    def chemical_potential(self, c=None, heff=None):

        if c is None:
            c = self.mo_coeff[:, 0]

        if heff is None:

            N = self.mol.nelectron
            dm = np.outer(c, N * c)

            h1e = self.mol.intor("int1e_nuc")
            veff = self.get_veff(self.mol, dm)
            heff = h1e + veff

        return c @ heff @ c

    def of_optimize(self, c0=None):

        self.dump_flags()
        self.build(self.mol)
        self.of_init()

        c0 = c0 or self.of_init_guess()
        constraints = [{"type": "eq", "fun": self.of_constraint, "jac": self.of_constraint_jac}]
        options = {"maxiter": self.max_cycle, "disp": self.verbose >= 1}

        res = minimize(
            self.of_energy_value_and_grad,
            x0=c0,
            method="SLSQP",
            jac=True,
            constraints=constraints,
            callback=self.of_callback,
            tol=self.conv_tol,
            options=options,
        )

        mu = self.chemical_potential(res.x)

        self.converged = res.success
        self.e_tot = res.fun
        self.mo_energy = np.array([mu])
        self.mo_coeff = res.x[:, None]
        self.mo_occ = np.array([self.mol.nelectron])

        self.of_reset()

        return self.e_tot

    def of_reset(self):
        self._iteration = 0
        self._last_energy = None
        self._last_chem_pot = None
        self._ovlp = None
        self._h1e = None

    def scf(self, *args, kind=None, **kwargs):

        if kind is None:
            kind = self.kind

        kind = kind.lower().strip()

        if kind == "ks":
            return super().scf(*args, **kwargs)
        elif kind == "of":
            return self.of_optimize(*args, **kwargs)
        else:
            raise ValueError(f"Invalid kind: {kind}")


def orbital_free(ks: RKS, kef: Optional[str] = None) -> OrbitalFree:
    of = OrbitalFree.from_rks(ks, kef=kef)
    of.kernel()
    return of
