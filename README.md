# Topological Phases Induced by Quasiperiodic Disorder

This repository contains the Python codebase supporting the numerical simulations, lattice constructions, potential implementations, and visualization utilities for the study of the Chern insulators in the presence of different random or quasiperiodic disorder potentials. 

## Repository Structure

The core codebase is organized into four modular Python scripts:

| File | Description |
| :--- | :--- |
| **`Lattice.py`** | Object-oriented framework for real-space tight-binding lattices. Defines base `Lattice` classes and specific implementations for `Haldane` (honeycomb), `Ladder` (1D two-leg ladders), `Hofstadter` (square lattice in magnetic field), and `Ladder_Network` (2D network of coupled ladders). |
| **`Potentials.py`** | On-site potential generators, including the structured quasiperiodic potential supporting weak potential lines (WPLs) (`V_hex_wpl`), alternative 1D/2D quasiperiodic potentials, superlattices, and uncorrelated random disorder (`V_random`). |
| **`Calculations.py`** | Real-space topological and transport diagnostics. Computes eigenspectra, Inverse Participation Ratio (IPR), density of states (DOS), Bianco-Resta real-space Chern markers, bulk-averaged Chern invariants, real-space equilibrium bond currents, and Fermi-sea time evolution. |
| **`Plots.py`** | Visualization routines for plotting energy spectra, spatial wavefunctions/eigenstates, real-space Chern marker maps, 2D phase diagrams, real-space bond current distributions, and scaling analyses. |

---

## Key Features

- **Real-Space Chern Invariant Diagnostics:** Efficient computation of the local Bianco-Resta Chern marker $C(\mathbf{r})$ and bulk-averaged topological invariants $\overline{C}$ for arbitrary finite geometries and disorder realizations.
- **Weak Potential Line (WPL) Network Physics:** Tools for constructing full honeycomb lattices under structured quasiperiodic disorder as well as low-energy effective 2D coupled ladder networks (`Ladder_Network`).
- **Real-Space Transport & Dynamics:** Exact computation of bond current matrices $I_{j \leftarrow i}$ for non-interacting filled Fermi seas, chiral edge currents, and non-equilibrium Fermi-sea time evolution under quenches.

---

## Requirements & Installation

The code relies on standard Scientific Python libraries. 

### Dependencies
- Python $\ge$ 3.8
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

To install the required dependencies:
```bash
pip install numpy scipy matplotlib tqdm
```

---

## Quickstart Examples

### 1. Constructing a Haldane Lattice with Quasiperiodic WPL Disorder
```python
import numpy as np
import Lattice
import Potentials
import Calculations

# Parameters
L = 20  # Lattice dimension (L x L unit cells, N_sites = 2 * L^2)
V_mag = 4.0  # Disorder strength
beta = 1.0 / np.sqrt(2)  # Irrational quasiperiodic frequency ratio

# Instantiate Haldane model with WPL quasiperiodic potential
system = Lattice.Haldane(
    L=L,
    t=1.0,
    t2=0.2j,
    m=0.0,
    V=Potentials.V_hex_wpl,
    V_args={'beta': beta, 'V': V_mag}
)

# Compute Hamiltonian and full eigensystem
H = system.calc_H()
evals, evects = Calculations.calc_spectrum(H, return_evects=True)

# Calculate Bianco-Resta real-space Chern marker at half-filling
N_occ = L**2
chern_marker = Calculations.calc_chern_marker(evects, system, N_occ=N_occ)
C_avg = Calculations.calc_avg_chern(chern_marker, system, N_max=3)

print(f"Bulk-averaged Chern marker C_bar = {C_avg:.4f}")
```

### 2. Plotting the Real-Space Chern Marker
```python
import Plots

# Plot spatial distribution of local Chern marker
Plots.plot_chern_marker(
    chern=chern_marker,
    system=system,
    calc_avg=True,
    N_max=3,
    title_str=r'Haldane + WPL Potential ($V/t = 4$)'
)
```

### 3. Evaluating an Effective Coupled Ladder Network
```python
# Construct an effective network of coupled ladders with spacing d = 5 unit cells
L_net = 17
d_net = 5
i0 = np.arange(1, L_net, d_net)
j0 = np.copy(i0)

network = Lattice.Ladder_Network(
    i0=i0,
    j0=j0,
    L=L_net,
    V1=4.0,
    V2=4.0,
    calc_effective=True
)

H_net = network.calc_H()
evals_net, evects_net = Calculations.calc_spectrum(H_net, return_evects=True)
```

---

## Data Availability & Reproduction

Data generated during numerical runs (e.g., phase diagrams, spectrum sweeps, IPR scaling data) are saved as NumPy compressed archives (`.npz`). The plotting functions in `Plots.py` directly accept these `.npz` files for automated figure generation.

---