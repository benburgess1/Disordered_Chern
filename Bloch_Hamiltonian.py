"""
Bloch Hamiltonian construction for periodic (supercell) potentials
on the Haldane model, via real-space bond folding.

Convention
----------
- Primitive lattice vectors:
    a1 = (1, 0)
    a2 = (1/2, sqrt(3)/2)
  (lattice constant = 1; NN bond length = 1/sqrt(3))

- Two-site basis per primitive cell: sublattice A (s=0), sublattice B (s=1).

- A supercell is an N x N block of primitive cells (N along a1, N along a2).
  Supercell lattice vectors: A1 = N*a1, A2 = N*a2.
  Total sites in supercell Bloch Hamiltonian: n_orb * N^2 = 2*N^2.

- Site indexing within the supercell:
    idx(i, j, s) = n_orb*(i*N + j) + s
  where i, j in {0,...,N-1} label the primitive cell's position inside the
  supercell (i along a1, j along a2), and s in {0,...,n_orb-1} is the
  sublattice/orbital index.

- Gauge: "periodic" (superlattice) gauge. A bond that crosses the supercell
  boundary picks up a phase exp(i*(theta1*W1 + theta2*W2)), where (W1,W2) is
  the integer number of supercell lattice vectors (A1, A2) crossed, and
  (theta1, theta2) in [0, 2*pi) x [0, 2*pi) is the crystal momentum
  *reduced to the supercell Brillouin zone*, i.e. theta_i = k . A_i.
  This does NOT include the intracell atomic offset (bA, bB). If you need
  the full position gauge to match an existing bond-current convention,
  see the note at the bottom of this file.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict


def primitive_lattice_vectors():
    """Real-space primitive vectors a1, a2 of the (unfolded) honeycomb lattice."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    return a1, a2


def primitive_reciprocal_vectors():
    """
    Reciprocal vectors b1, b2 of the *original* (unfolded) lattice, satisfying
    a_i . b_j = 2*pi*delta_ij. These define the true (hexagonal) Brillouin
    zone of the Haldane model, independent of any supercell.
    """
    a1, a2 = primitive_lattice_vectors()
    A = np.array([a1, a2])
    Bmat = 2 * np.pi * np.linalg.inv(A).T
    return Bmat[0, :], Bmat[1, :]


def supercell_reciprocal_vectors(N: int):
    """
    Reciprocal vectors B1, B2 of the N x N supercell, satisfying
    A_i . B_j = 2*pi*delta_ij with A_i = N*a_i. Related to the primitive
    reciprocal vectors simply by B_i = b_i / N.
    """
    b1, b2 = primitive_reciprocal_vectors()
    return b1 / N, b2 / N


def high_symmetry_points() -> Dict[str, np.ndarray]:
    """
    High-symmetry points of the *original* hexagonal Brillouin zone
    (Gamma, K, K', M), in Cartesian (kx, ky). Useful for locating where the
    original Dirac points fold to inside the supercell BZ, or for plotting
    band structure along a Gamma-K-M-Gamma path on the unfolded lattice.
    """
    b1, b2 = primitive_reciprocal_vectors()
    return {
        "Gamma": np.array([0.0, 0.0]),
        "K": (2 * b2 + b1) / 3,
        "K'": (b2 - b1) / 3,
        "M": (b2) / 2,
    }


def cartesian_to_theta(kx: float, ky: float, N: int):
    """
    Convert a Cartesian crystal momentum (kx, ky) to the reduced supercell
    coordinates (theta1, theta2) = (k . A1, k . A2), A_i = N*a_i, which is
    what H0(theta1, theta2) expects. Any (kx, ky) is accepted -- H0 is
    automatically 2*pi-periodic in each theta, so you don't need to first
    restrict k to a particular BZ choice.
    """
    a1, a2 = primitive_lattice_vectors()
    A1, A2 = N * a1, N * a2
    theta1 = kx * A1[0] + ky * A1[1]
    theta2 = kx * A2[0] + ky * A2[1]
    return theta1, theta2


def make_H0_k(H0: Callable[[float, float], np.ndarray], N: int):
    """
    Wrap a theta-based Bloch Hamiltonian H0(theta1, theta2) (as returned by
    build_supercell_bloch_hamiltonian) into a Cartesian-k version
    H0_k(kx, ky), via the linear map theta_i = k . A_i.

    Use H0 directly (on a uniform theta grid) for BZ-integrated quantities
    (Chern numbers via Fukui-Hatsugai-Suzuki, DOS, etc.), where uniform
    sampling of the true BZ matters. Use H0_k for anything where thinking
    in physical, Cartesian k is more natural: band structure along a
    Gamma-K-M-Gamma path, checking where original Dirac points fold to,
    comparing against unfolded-lattice velocities, etc.
    """
    def H0_k(kx: float, ky: float) -> np.ndarray:
        theta1, theta2 = cartesian_to_theta(kx, ky, N)
        return H0(theta1, theta2)
    return H0_k


@dataclass(frozen=True)
class PrimitiveBond:
    """
    A single directed hopping bond within the primitive-cell description
    of the lattice, to be tiled over the supercell.

    Represents a hop from (cell (0,0), sublattice s1) to
    (cell (n1,n2), sublattice s2) with amplitude `amp`.

    List each *physical* bond only ONCE (as one directed representative);
    the Hermitian conjugate is added automatically during folding. Do not
    separately add the reverse-direction bond, or it will be double counted.
    """
    s1: int
    s2: int
    n1: int
    n2: int
    amp: complex


def haldane_primitive_bonds(t: float, t2_mag: float, phi: float) -> List[PrimitiveBond]:
    """
    Directed primitive-cell bonds for the Haldane model.

    Sublattices: 0 = A, 1 = B.

    NN bonds (A -> B), real amplitude t:
        A(0,0)  -> B(0,0)
        A(0,0)  -> B(-1,0)
        A(0,0)  -> B(0,-1)

    NNN bonds, complex amplitude t2_mag*exp(+/- i*phi):
        A -> A along +a1, -a2, +(a2-a1), amplitude t2_mag*exp(+i*phi)
        B -> B along +a1, -a2, +(a2-a1), amplitude t2_mag*exp(-i*phi)

    This is one specific (common) choice of chirality/sign convention for
    the Haldane flux. If it disagrees in sign with your existing Haldane.py
    real-space bond-current convention, flip the sign of phi here (or swap
    the +i*phi / -i*phi assignment between A and B) -- do NOT silently mix
    conventions between this file and your existing code.
    """
    A, B = 0, 1
    bonds = [
        PrimitiveBond(A, B, 0, 0, t),
        PrimitiveBond(A, B, -1, 0, t),
        PrimitiveBond(A, B, 0, -1, t),
    ]
    nnn_dirs = [(1, 0), (0, -1), (-1, 1)]  # +a1, -a2, +(a2-a1)
    for (n1, n2) in nnn_dirs:
        bonds.append(PrimitiveBond(A, A, n1, n2, t2_mag * np.exp(1j * phi)))
        bonds.append(PrimitiveBond(B, B, n1, n2, t2_mag * np.exp(-1j * phi)))
    return bonds


def build_supercell_bloch_hamiltonian(
    N: int,
    primitive_bonds: List[PrimitiveBond],
    n_orb: int = 2,
    m: float = 0.,
) -> Callable[[float, float], np.ndarray]:
    """
    Master function: folds a primitive-cell bond list onto an N x N
    supercell and returns a function H0(theta1, theta2) -> (n_orb*N^2,
    n_orb*N^2) complex Hermitian ndarray, the zero-potential Bloch
    Hamiltonian.

    theta1, theta2 are each in [0, 2*pi) -- crystal momentum reduced to the
    supercell Brillouin zone (theta_i = k . A_i, A_i = N * a_i).

    Call build_supercell_bloch_hamiltonian once per (N, model params); the
    returned H0 is cheap to evaluate on a k-grid. Potentials are added
    afterwards as a diagonal array (see example_stripe_potential below).
    """
    dim = n_orb * N * N

    def idx(i, j, s):
        return n_orb * (i * N + j) + s

    # Precompute the (site_a, site_b, amp[, W1, W2]) tuples once, so that
    # H0(theta1, theta2) evaluation is just a phase multiply + scatter-add,
    # not a full re-derivation of the bond geometry on every call.
    intracell_terms = []   # (a, b, amp)                -- no phase
    intercell_terms = []   # (a, b, amp, W1, W2)         -- phase exp(i(W1 th1 + W2 th2))

    for i in range(N):
        for j in range(N):
            for bond in primitive_bonds:
                i2, j2 = i + bond.n1, j + bond.n2
                I2, W1 = i2 % N, i2 // N
                J2, W2 = j2 % N, j2 // N
                a = idx(i, j, bond.s1)
                b = idx(I2, J2, bond.s2)
                if W1 == 0 and W2 == 0:
                    intracell_terms.append((a, b, bond.amp))
                else:
                    intercell_terms.append((a, b, bond.amp, W1, W2))

    def H0(theta1: float, theta2: float) -> np.ndarray:
        H = np.zeros((dim, dim), dtype=complex)
        for a, b, amp in intracell_terms:
            H[a, b] += amp
            H[b, a] += np.conj(amp)
        for a, b, amp, W1, W2 in intercell_terms:
            phase = np.exp(1j * (W1 * theta1 + W2 * theta2))
            H[a, b] += amp * phase
            H[b, a] += np.conj(amp * phase)
        for i in range(dim):        # Semenoff mass
            H[i, i] += m * (-1)**i
        return H

    return H0


def site_grid_indices(N: int, n_orb: int = 2):
    """
    Convenience: returns arrays i_of[a], j_of[a], s_of[a] giving the
    (primitive-cell i, primitive-cell j, sublattice) label of each
    supercell site index a. Use this to build potentials V(i,j,s).
    """
    dim = n_orb * N * N
    i_of = np.zeros(dim, dtype=int)
    j_of = np.zeros(dim, dtype=int)
    s_of = np.zeros(dim, dtype=int)
    for i in range(N):
        for j in range(N):
            for s in range(n_orb):
                a = n_orb * (i * N + j) + s
                i_of[a], j_of[a], s_of[a] = i, j, s
    return i_of, j_of, s_of


def example_stripe_potential(N: int, V0: float, n_orb: int = 2) -> np.ndarray:
    """
    Example potential: V depends only on i (period N along a1), same on
    both sublattices. This is just a template for the pattern -- swap in
    whatever V(i,j,s) you actually need and add it as np.diag(V) to H0.
    """
    i_of, j_of, s_of = site_grid_indices(N, n_orb)
    return V0 * np.cos(2 * np.pi * i_of / N)


def beta_potential(beta, V, N, phi_1=0, phi_2=0, n_orb=2):
    a1, a2 = primitive_lattice_vectors()
    d1 = np.array([0.5, 0.5/np.sqrt(3)])
    i_of, j_of, s_of = site_grid_indices(N, n_orb)
    r = i_of[:, None] * a1[None, :] + j_of[:, None] * a2[None, :] + s_of[:, None] * d1[None, :]
    b1, b2 = primitive_reciprocal_vectors()
    # return 2*V * np.cos(beta*(r @ b1) + phi_1) * np.cos(beta*(r @ b2) + phi_2)
    return V * (np.cos(beta*(r @ (b1+b2)) + phi_1) + np.cos(beta*(r @ (b1-b2)) + phi_2))

    


if __name__ == "__main__":
    V = beta_potential(beta=0.6, V=1, N=5)
    print(V.shape)
    print(np.diag(V).shape)