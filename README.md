# <h1 align='center'>Global Density Approximation</h1>

Research code for neural-network approximations of electronic density functionals. This code is attached to the following publication:

**Neural network distillation of orbital dependent density functional theory** ([arXiv:2410.16408](https://arxiv.org/abs/2410.16408)).

Code author: Matija Medvidović

<center>
    <img src="./images/workflow.png" alt="workflow" class="center" width="800"/>
</center>

## Installation

Global density approximations (GDA) is not available through package managers yet. However, you can install it by pointing `pip` to this repository:

```bash
pip3 install 'gda@git+https://github.com/Matematija/global-density-approximation.git'
```

## Basic usage

### GDA as a [PyTorch](https://pytorch.org/) module

<center>
    <img src="./images/diagram.png" alt="diagram" class="center" width="250"/>
</center>

The `gda` package exports only one PyTorch module you can construct in the following way:

```python
import torch
from gda import GlobalDensityApprox

gda = GlobalDensityApprox(embed_dim=128, n_blocks=2)
```

and call to evaluate the function $\phi(\mathbf{r})$:

```python
# Dummy data
n = torch.randn(20000) # shape = (..., grid_size,)
grad_n = torch.randn(20000, 3) # shape = (..., grid_size, 3)
coords = torch.randn(20000, 3) # shape = (..., grid_size, 3)
weights = torch.randn(20000) # shape = (..., grid_size,)

phi = gda(n, grad_n, coords, weights)
log_tau = gda.log_tau(n, grad_n, coords, weights)
# Log-tau implemented for numerical stability
```

defined by

$$
\tau ( \mathbf{r} ) = \tau _W ( \mathbf{r} ) + e^{ \phi ( \mathbf{r} ) } \left( \tau _U ( \mathbf{r} ) + \eta \tau _W ( \mathbf{r} ) \right)
$$

where $\tau_U = \frac{3}{10} (3 \pi ^2 )^{2/3} n ^{5/3}$ is the uniform electron gas kinetic energy, $\tau_W = \frac{| \nabla n |^2}{8 n}$ is the von Weizsäcker kinetic functional and $\eta = 10^{-3}$.

### The [PySCF](https://pyscf.org/) interface

We also provide a custom `RKS` (Restricted Kohn-Sham) DFT class that can be used to run DFT loops using trained GDA models. Example interface: 

```python
from ggda.scf import RKS

ks = RKS(mol)
ks.xc = 'tpss'
ks.gda = gda
ks.grids.level = 1
ks.conv_tol = 1e-5
ks.verbose = 4
ks.kernel()
```

If `RKS.gda` field is not set, then the `RKS.kernel()` method will just run "normal" DFT using PySCF defaults. Furthermore, since the GDA approximation only models the kinetic energy density $\tau (\mathbf{r})$, if the `RKS.xc` field is set to a functional that does not require $\tau$ as an input, unmodified LibXC functionals are called as well.

## A differentiable [LibXC](https://libxc.gitlab.io/) wrapper

This library includes a differentiable PyTorch wrapper around LibXC as a convenience. In short - we make the [`eval_xc`](https://github.com/pyscf/pyscf/blob/f2c2d3f963916fb64ae77241f1b44f24fa484d96/pyscf/dft/libxc.py#L684) function available in PySCF transparent to PyTorch [Autograd](https://pytorch.org/docs/stable/autograd.html). This presents a convenient unified API for calculating XC potentials and higher-order derivatives for experimentation with general parametrized functionals.

An example calculation yielding the PBE potential matrix in the `ccpvdz` basis set $\chi _\mu (\mathbf{r})$:

$$
V ^{XC} _{\mu \nu} = \int d^3 \mathbf{r} \; \frac{\delta E _{XC}}{\delta n (\mathbf{r})} \chi _\mu (\mathbf{r}) \chi _\nu (\mathbf{r}) = \frac{\partial E_{XC}}{\partial {\Gamma _{\mu \nu}}}
$$

when the density is represented as $n (\mathbf{r}) = \sum _{\mu \nu} \Gamma _{\mu \nu} \chi _\mu (\mathbf{r}) \chi _\nu (\mathbf{r})$.

```python
from pyscf import gto, dft
from torch import autograd
from gda.libxc import eval_xc

ao = torch.tensor(dft.numint.eval_ao(ks.mol, ks.grids.coords, deriv=1))
dm = torch.tensor(ks.make_rdm1()).requires_grad_(True)

def eval_energy(dm, ao):

    ao_, grad_ao_ = ao[0], ao[1:]

    density = torch.einsum('mn,im,in->i', dm, ao_, ao_)
    grad_density = torch.einsum('mn,im,cin->ic', dm, ao_, grad_ao_)

    exc = eval_xc('pbe', density, grad_density)

    return weights @ exc

E = eval_energy(dm, ao)
V, = autograd.grad(E, dm)

V.shape # (n_ao, n_ao)
```