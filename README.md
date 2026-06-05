# Disordered_Chern

Numerical tools for studying the interplay between topology and localisation in two-dimensional Chern band models subject to quasiperiodic (and, prospectively, random) disorder potentials.

## Physics background

Topological phases of matter are characterised by global invariants — such as the Chern number — that enforce the existence of extended edge states robust to perturbations. Disorder, on the other hand, generically drives Anderson localisation, exponentially confining eigenstates in real space. The competition between these two tendencies is particularly rich in two dimensions, where all single-particle states in a disordered system with broken time-reversal symmetry are generically localised, yet topological protection can stabilise extended states even at finite disorder strength.

Quasiperiodic potentials — spatially ordered but lacking translational periodicity — offer a distinct setting for this competition. Unlike random disorder, which is fully uncorrelated, quasiperiodic potentials have long-range correlations set by an underlying incommensurate structure. In one dimension this is well-studied via the Aubry–André model, which exhibits an exact localisation transition. In two dimensions the picture is less settled, and the effect of quasiperiodic modulations on topological Chern bands remains an active area of investigation.

This repository implements two paradigmatic models of Chern bands — the **Haldane model** on the honeycomb lattice and the **Hofstadter model** on the square lattice — and provides tools for adding quasiperiodic on-site potentials, diagonalising the resulting Hamiltonians, and characterising the spectral and localisation properties of eigenstates via quantities such as the inverse participation ratio (IPR). The framework is also intended to accommodate random disorder in future work, enabling direct comparison between the two classes of potential.

## Repository structure

### `Haldane.py`

Implements the Haldane model on a finite honeycomb lattice with open boundary conditions.

**`Haldane(L, t, m, t2, a, exclude_endsites, V, V_args)`**  
Constructs an $L \times L$ unit-cell honeycomb lattice with nearest-neighbour hopping $t$, sublattice mass $m$, and complex next-nearest-neighbour hopping $t_2$ (which breaks time-reversal symmetry and opens a topological gap). An optional on-site potential $V(x, y, \texttt{**V\_args})$ can be supplied. The `exclude_endsites` flag removes two isolated corner sites that would otherwise be disconnected from the bulk.

Key methods:
- `build_lattice()` — constructs the `Site` graph including nearest and next-nearest-neighbour connectivity.
- `calc_H()` — assembles the real-space Hamiltonian matrix, including the Peierls-like phase structure of $t_2$ that encodes the net flux through each plaquette.
- `plot_lattice()` — visualises the lattice geometry, optionally overlaying the disorder potential.
- `remove_site()` — removes a site and all associated bonds from the graph.

**`Site`**  
A lightweight data class storing a site's real-space position, unit-cell index, sublattice label, and adjacency lists for nearest and next-nearest neighbours.

**Potential functions** (all accept array-valued `x`, `y`):
- `V_square` — separable square lattice potential $V(\cos G x + \cos G y)$.
- `V_pw` — plane-wave potential $V \cos G y$.
- `V_hex` — potential with hexagonal symmetry.
- `V_sep` — two-wave separable potential with hexagonal reciprocal-lattice vectors.

---

### `Hofstadter.py`

Implements the Hofstadter model on a finite square lattice, describing a charged particle on a 2D lattice in a perpendicular magnetic field.

**`Hofstadter(L, t, phi, a, V, V_args, show_progress)`**  
Constructs an $L \times L$ square lattice. The magnetic flux per plaquette is $\phi$ (in units where $\phi_0 = 2\pi$). Hopping along the $x$-direction acquires Peierls phases $e^{\pm i \phi j}$, where $j$ is the $y$-coordinate of the site (Landau gauge). An optional on-site potential can be supplied.

Key methods:
- `build_lattice()` — constructs the site graph with nearest-neighbour bonds.
- `calc_H()` — assembles the Hamiltonian with Peierls phases.
- `plot_lattice()` — visualises the lattice and optional disorder potential.

**Potential functions:**
- `V_nonsep` — non-separable quasiperiodic potential $V(\cos \beta G (x+y) + \cos \beta G (x-y))$; choosing $\beta = 1/\sqrt{2}$ (or another irrational) makes the modulation incommensurate with the lattice.

---

### `Calculations.py`

Numerical routines for spectral calculations.

- `calc_spectrum(H, return_evects)` — diagonalises a Hermitian Hamiltonian using `numpy.linalg.eigh`, returning eigenvalues and optionally eigenvectors.
- `calc_butterfly(phi_vals, L, calc_ipr, save, save_filename, **kwargs)` — sweeps over a range of flux values `phi_vals`, diagonalises the Hofstadter Hamiltonian at each, and optionally computes the IPR of every eigenstate. Results are saved as a compressed `.npz` archive. The IPR of eigenstate $|\psi\rangle$, defined as $\sum_i |\psi_i|^4$, distinguishes extended states (small IPR, scaling as $1/N$) from localised states (IPR of order unity).

---

### `Plots.py`

Plotting utilities for visualising results.

- `plot_butterfly(filename, ms, title_params, color_ipr, cmap, ipr_scale)` — loads a saved `.npz` butterfly dataset and plots the spectrum as a function of $\phi / \phi_0$. When `color_ipr=True`, each eigenvalue is coloured by the IPR of the corresponding eigenstate, providing a visual map of localisation across the butterfly spectrum. Supports linear and logarithmic IPR colour scales.
- `make_title_str` — helper for constructing plot titles from saved parameter dictionaries.

## Dependencies

- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

## Usage

A typical workflow for the Hofstadter model with quasiperiodic disorder:

```python
from Hofstadter import Hofstadter, V_nonsep
from Calculations import calc_butterfly
import numpy as np

phi_vals = np.linspace(0, 2 * np.pi, 101)

calc_butterfly(
    phi_vals=phi_vals,
    L=20,
    calc_ipr=True,
    save=True,
    save_filename='Data/Butterfly_qp.npz',
    V=V_nonsep,
    V_args={'beta': 1 / np.sqrt(2), 'V': 1.0}
)
```

Then visualise the result:

```python
from Plots import plot_butterfly

plot_butterfly(
    'Data/Butterfly_qp.npz',
    ms=0.5,
    color_ipr=True,
    ipr_scale='log',
    cmap='plasma'
)
```
