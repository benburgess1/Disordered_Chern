"""
Bloch_Calculations.py

Computational routines operating on a theta-based Bloch Hamiltonian
H0(theta1, theta2), as constructed by
haldane_supercell.build_supercell_bloch_hamiltonian: diagonalization at a
single point, along a path, and over a mesh.

Berry curvature / Chern number routines will be added to this file later,
reusing the same H0-centric design (and the eigvecs=True machinery already
built into diagonalize_bloch_h) so nothing here needs to change.
"""

import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Dict

from Bloch_Hamiltonian import cartesian_to_theta, high_symmetry_points
from tqdm import tqdm
from contextlib import nullcontext

def _progress(total, desc, show):
    return tqdm(total=total, desc=desc) if show else nullcontext()

def diagonalize_bloch_h(
    H0: Callable[[float, float], np.ndarray],
    k: Optional[Tuple[float, float]] = None,
    theta: Optional[Tuple[float, float]] = None,
    N: Optional[int] = None,
    potential: Optional[np.ndarray] = None,
    eigvecs: bool = False,
):
    """
    Diagonalize a Bloch Hamiltonian at a single momentum.

    Specify EXACTLY ONE of:
      - theta = (theta1, theta2)   reduced supercell coordinates, or
      - k = (kx, ky), with N       Cartesian crystal momentum (converted
                                     internally via cartesian_to_theta)

    Parameters
    ----------
    H0 : callable(theta1, theta2) -> (dim, dim) complex ndarray
        Zero-potential (or potential-free) Bloch Hamiltonian, e.g. from
        build_supercell_bloch_hamiltonian.
    potential : optional (dim,) real ndarray
        Added as np.diag(potential) before diagonalizing.
    eigvecs : bool
        If True, returns (energies, states) with states[:, n] the
        eigenvector belonging to energies[n] (ascending order, as returned
        by np.linalg.eigh). Eigenvector phases are gauge-arbitrary -- fine
        for band energies, not safe to use directly for finite-difference
        Berry curvature. If False (default), returns only energies.

    Returns
    -------
    energies : (dim,) ndarray, ascending
    states : (dim, dim) ndarray, only if eigvecs=True
    """
    if (k is None) == (theta is None):
        raise ValueError("Specify exactly one of `k` or `theta`, not both/neither.")

    if k is not None:
        if N is None:
            raise ValueError("`N` is required to convert Cartesian `k` to theta.")
        theta1, theta2 = cartesian_to_theta(k[0], k[1], N)
    else:
        theta1, theta2 = theta

    H = H0(theta1, theta2)
    if potential is not None:
        H = H + np.diag(potential)

    if eigvecs:
        return np.linalg.eigh(H)
    return np.linalg.eigvalsh(H)


def _points_to_theta(points, N, in_theta):
    """Internal: convert a list of Cartesian (kx,ky) or theta points to theta pairs."""
    if in_theta:
        return [tuple(p) for p in points]
    if N is None:
        raise ValueError("`N` is required when points are given as Cartesian k (in_theta=False).")
    return [cartesian_to_theta(p[0], p[1], N) for p in points]


def default_path_Gamma_M_K_Gamma():
    """Convenience: the standard Gamma-M-K-Gamma path, in Cartesian k."""
    hs = high_symmetry_points()
    return np.array([hs["Gamma"], hs["M"], hs["K"], hs["Gamma"]])


def band_structure_along_path(
    H0: Callable[[float, float], np.ndarray],
    points: Sequence[Tuple[float, float]],
    n_per_segment: int = 100,
    N: Optional[int] = None,
    in_theta: bool = False,
    potential: Optional[np.ndarray] = None,
    eigvecs: bool = False,
) -> Dict:
    """
    Evaluate bands along a piecewise-linear path through momentum space, by
    repeated calls to diagonalize_bloch_h.

    Parameters
    ----------
    points : sequence of (x, y) tuples
        Path nodes. By default (in_theta=False) these are Cartesian (kx, ky)
        and N must be supplied; set in_theta=True to give (theta1, theta2)
        nodes directly instead. See default_path_Gamma_K_M_Gamma() for a
        ready-made example.
    n_per_segment : int
        Number of points sampled along each segment between consecutive
        nodes (shared nodes are not duplicated).
    eigvecs : bool
        If True, also return the eigenvectors at each point.

    Returns
    -------
    dict with keys:
        'distance'   : (n_k,) cumulative distance along the path (same units
                       as the input points: Cartesian k, or theta if
                       in_theta=True)
        'energies'   : (n_k, dim) ascending-sorted eigenvalues at each point
        'node_dist'  : (n_nodes,) distance values at the supplied path
                       nodes, for placing tick labels (e.g. Gamma, K, M)
        'theta_path' : (n_k, 2) the theta values actually evaluated
        'states'     : (n_k, dim, dim), only if eigvecs=True
    """
    theta_nodes = np.array(_points_to_theta(points, N, in_theta))
    n_segs = len(theta_nodes) - 1
    if n_segs < 1:
        raise ValueError("Need at least two points to define a path.")

    theta_path = []
    distance = []
    node_dist = [0.0]
    cum = 0.0

    for seg in range(n_segs):
        start, end = theta_nodes[seg], theta_nodes[seg + 1]
        seg_len = np.linalg.norm(end - start)
        is_last = (seg == n_segs - 1)
        ts = np.linspace(0.0, 1.0, n_per_segment if not is_last else n_per_segment+1, endpoint=is_last)
        for t in ts:
            theta_path.append(start + t * (end - start))
            distance.append(cum + t * seg_len)
        cum += seg_len
        node_dist.append(cum)

    theta_path = np.array(theta_path)
    distance = np.array(distance)
    n_k = len(theta_path)

    energies = None
    states = None
    for i, (th1, th2) in enumerate(theta_path):
        if eigvecs:
            e, v = diagonalize_bloch_h(H0, theta=(th1, th2), potential=potential, eigvecs=True)
            if energies is None:
                dim = e.shape[0]
                energies = np.zeros((n_k, dim))
                states = np.zeros((n_k, dim, dim), dtype=complex)
            energies[i], states[i] = e, v
        else:
            e = diagonalize_bloch_h(H0, theta=(th1, th2), potential=potential, eigvecs=False)
            if energies is None:
                energies = np.zeros((n_k, e.shape[0]))
            energies[i] = e

    result = {
        "distance": distance,
        "energies": energies,
        "node_dist": np.array(node_dist),
        "theta_path": theta_path,
    }
    if eigvecs:
        result["states"] = states
    return result


def band_structure_mesh(
    H0: Callable[[float, float], np.ndarray],
    n1: int,
    n2: int,
    N: Optional[int] = None,
    k_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    potential: Optional[np.ndarray] = None,
    eigvecs: bool = False,
    show_progress: bool = True,
) -> Dict:
    """
    Evaluate bands over a uniform mesh, by repeated calls to
    diagonalize_bloch_h.

    Default (k_range=None): a uniform mesh in theta over [0,2pi) x [0,2pi),
    which tiles the supercell Brillouin zone exactly once -- use this for
    anything BZ-integrated later (Chern numbers, DOS, k-averages).

    k_range=((kx_min,kx_max),(ky_min,ky_max)): a uniform mesh in Cartesian
    (kx,ky) instead (requires N). Convenient for visualization (e.g.
    plotting folded Dirac cones over an extended zone) but does NOT, in
    general, tile the BZ exactly once -- don't use this for BZ sums.

    Returns
    -------
    dict with keys:
        'axis1', 'axis2' : (n1,), (n2,) coordinate arrays (theta, or
                            Cartesian k if k_range is given)
        'energies'       : (n1, n2, dim) ascending-sorted eigenvalues
        'states'         : (n1, n2, dim, dim), only if eigvecs=True
    """
    if k_range is None:
        ax1 = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
        ax2 = np.linspace(0.0, 2 * np.pi, n2, endpoint=False)
        as_theta = lambda x1, x2: (x1, x2)
    else:
        if N is None:
            raise ValueError("`N` is required when specifying a Cartesian k_range.")
        (kx_min, kx_max), (ky_min, ky_max) = k_range
        ax1 = np.linspace(kx_min, kx_max, n1)
        ax2 = np.linspace(ky_min, ky_max, n2)
        as_theta = lambda kx, ky: cartesian_to_theta(kx, ky, N)

    energies = None
    states = None
    with _progress(total=len(ax1)*len(ax2), desc='Diagonalising over mesh', show=show_progress) as pbar:
        for i, x1 in enumerate(ax1):
            for j, x2 in enumerate(ax2):
                th1, th2 = as_theta(x1, x2)
                if eigvecs:
                    e, v = diagonalize_bloch_h(H0, theta=(th1, th2), potential=potential, eigvecs=True)
                    if energies is None:
                        dim = e.shape[0]
                        energies = np.zeros((n1, n2, dim))
                        states = np.zeros((n1, n2, dim, dim), dtype=complex)
                    energies[i, j], states[i, j] = e, v
                else:
                    e = diagonalize_bloch_h(H0, theta=(th1, th2), potential=potential, eigvecs=False)
                    if energies is None:
                        energies = np.zeros((n1, n2, e.shape[0]))
                    energies[i, j] = e
                if pbar: pbar.update(1)

    result = {"axis1": ax1, "axis2": ax2, "energies": energies}
    if eigvecs:
        result["states"] = states
    return result


def calc_dos(E_vals=None, sigma=0.1, E_grid=None, n_sigma=5, N_grid=1000,
             **kwargs):
    """
    Calculate the density of states by convolving eigenvalues with a Gaussian kernel.

    Parameters
    ----------
    E_vals : ndarray, optional
        Energy eigenvalues. If not supplied, filename must be given.
    sigma : float
        Width of the Gaussian smoothing kernel.
    E_grid : ndarray, optional
        Energy grid on which to evaluate the DOS. If None, auto-determined
        from the range of E_vals padded by n_sigma * sigma on each side.
    n_sigma : float
        Number of sigma by which to pad the auto-determined E_grid.
    N_grid : int
        Number of points in the auto-determined E_grid.
    Returns
    -------
    E_grid : ndarray, shape (N_grid,)
        Energy values at which DOS is evaluated.
    DOS : ndarray, shape (N_grid,)
        Density of states.
    """
    E_vals = np.asarray(E_vals).ravel()

    if E_grid is None:
        E_min = E_vals.min() - n_sigma * sigma
        E_max = E_vals.max() + n_sigma * sigma
        E_grid = np.linspace(E_min, E_max, N_grid)

    dE = E_grid[:, np.newaxis] - E_vals[np.newaxis, :]   # shape (N_grid, N_evals)
    DOS = np.sum(np.exp(-0.5 * (dE / sigma)**2), axis=1) / (sigma * np.sqrt(2 * np.pi))

    return E_grid, DOS




def berry_curvature_fhs(
    eigvecs: np.ndarray,
    bands: Optional[Sequence[int]] = None,
    N_occ: Optional[int] = None,
    sum_bands: bool = True,
    non_abelian: bool = False,
    reverse_orientation: bool = False,
    periodic: bool = True,
) -> np.ndarray:
    """
    Berry curvature via the Fukui-Hatsugai-Suzuki (FHS) lattice method.
 
    Parameters
    ----------
    eigvecs : (N_k1, N_k2, dim, dim) complex ndarray
        eigvecs[i, j][:, n] = eigenvector of band n at mesh point (i, j),
        e.g. the 'states' array from band_structure_mesh(..., eigvecs=True).
        Must come from a uniform THETA mesh (band_structure_mesh with
        k_range=None), so neighbouring mesh indices (including the wrap
        from index N_k-1 back to index 0, if periodic=True) are physically
        adjacent points in the Brillouin zone.
    bands : sequence of int, optional
        Explicit band indices to compute curvature for. Mutually exclusive
        with N_occ.
    N_occ : int, optional
        If given, use bands 0, ..., N_occ-1 (the N_occ lowest bands).
        Mutually exclusive with `bands`.
    sum_bands : bool
        Abelian case only (ignored if non_abelian=True, which always
        returns a single combined quantity for the multiplet). If True,
        sum curvature over the selected bands. If False, keep bands
        separate (adds a leading band axis to the output).
    non_abelian : bool
        If True, compute the non-Abelian curvature of the multiplet spanned
        by the selected bands as a single quantity, via the determinant of
        the (N_bands x N_bands) overlap matrix at each link. This is exact
        even if bands within the selected set are degenerate or mix, and
        always returns a single array -- it inherently sums over the
        group, so sum_bands is ignored.
    reverse_orientation : bool
        Flips the sign of the result by reversing the plaquette traversal
        direction. Use this if the sign disagrees with your existing
        chirality/current convention elsewhere in the codebase.
    periodic : bool
        If True (default), wrap the last mesh index back to the first on
        each axis, so every plaquette on the torus is included -- this is
        what makes the FHS sum over all plaquettes an EXACTLY quantized
        integer (2*pi*Chern number) for any N_k, not just a large-N_k
        approximation. Requires eigvecs to come from the full [0,2*pi)
        theta mesh (band_structure_mesh with k_range=None) so the wrap is
        physically meaningful. Set periodic=False to instead compute only
        the (N_k1-1) x (N_k2-1) interior plaquettes with no wraparound
        (e.g. if you deliberately want an open, non-periodic patch).
 
    Returns
    -------
    ndarray
        Berry flux through each plaquette (radians, principal value in
        (-pi, pi]) -- NOT divided by plaquette area, so summing all entries
        and dividing by 2*pi gives the Chern number of the selected
        band(s) (exactly, if periodic=True). Shape (N_k1, N_k2) if
        periodic, else (N_k1-1, N_k2-1); with a leading band axis
        prepended if sum_bands=False.
    """
    if (bands is None) == (N_occ is None):
        raise ValueError("Specify exactly one of `bands` or `N_occ`.")
    bands = np.arange(N_occ) if bands is None else np.asarray(bands)
 
    n1, n2, dim, _ = eigvecs.shape
    m1, m2 = (n1, n2) if periodic else (n1 - 1, n2 - 1)
 
    # (di1,dj1) / (di2,dj2): the two link directions. Swapping them reverses
    # the plaquette traversal sense and hence the sign of the curvature.
    di1, dj1 = (0, 1) if reverse_orientation else (1, 0)
    di2, dj2 = (1, 0) if reverse_orientation else (0, 1)
 
    def wrap(i, j):
        return (i % n1, j % n2) if periodic else (i, j)
 
    def link_overlap(i, j, di, dj, b):
        i2, j2 = wrap(i + di, j + dj)
        return np.vdot(eigvecs[i, j][:, b], eigvecs[i2, j2][:, b])
 
    def link_det(i, j, di, dj):
        i2, j2 = wrap(i + di, j + dj)
        vA = eigvecs[i, j][:, bands]
        vB = eigvecs[i2, j2][:, bands]
        return np.linalg.det(vA.conj().T @ vB)
 
    def plaquette_phase(u1_ij, u2_ij, u1_next, u2_next):
        u1_ij, u2_ij = u1_ij / np.abs(u1_ij), u2_ij / np.abs(u2_ij)
        u1_next, u2_next = u1_next / np.abs(u1_next), u2_next / np.abs(u2_next)
        return np.angle((u1_ij * u2_next) / (u1_next * u2_ij))
 
    if non_abelian:
        curvature = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                u1_ij = link_det(i, j, di1, dj1)
                u2_ij = link_det(i, j, di2, dj2)
                i_n, j_n = wrap(i + di2, j + dj2)
                u1_next = link_det(i_n, j_n, di1, dj1)
                i_n, j_n = wrap(i + di1, j + dj1)
                u2_next = link_det(i_n, j_n, di2, dj2)
                curvature[i, j] = plaquette_phase(u1_ij, u2_ij, u1_next, u2_next)
        return curvature
 
    n_bands = len(bands)
    curvature = np.zeros((n_bands, m1, m2))
    for bi, b in enumerate(bands):
        for i in range(m1):
            for j in range(m2):
                u1_ij = link_overlap(i, j, di1, dj1, b)
                u2_ij = link_overlap(i, j, di2, dj2, b)
                i_n, j_n = wrap(i + di2, j + dj2)
                u1_next = link_overlap(i_n, j_n, di1, dj1, b)
                i_n, j_n = wrap(i + di1, j + dj1)
                u2_next = link_overlap(i_n, j_n, di2, dj2, b)
                curvature[bi, i, j] = plaquette_phase(u1_ij, u2_ij, u1_next, u2_next)
 
    if sum_bands:
        return curvature.sum(axis=0)
    return curvature


if __name__ == "__main__":
    from Bloch_Hamiltonian import haldane_primitive_bonds, build_supercell_bloch_hamiltonian

    N = 4
    t1, t2, phi = 1.0, 0.2, np.pi / 2
    bonds = haldane_primitive_bonds(t1, t2, phi)
    H0 = build_supercell_bloch_hamiltonian(N, bonds, n_orb=2)
    dim = 2 * N * N

    # 1) single-point diagonalization, both interfaces
    e_theta = diagonalize_bloch_h(H0, theta=(0.3, 1.1))
    e_k = diagonalize_bloch_h(H0, k=(0.1, -0.2), N=N)
    assert e_theta.shape == (dim,) and e_k.shape == (dim,)
    print("single-point OK, dim =", dim)

    e, v = diagonalize_bloch_h(H0, theta=(0.3, 1.1), eigvecs=True)
    assert np.allclose(v.conj().T @ v, np.eye(dim), atol=1e-8), "eigenvectors not orthonormal"
    print("eigenvector orthonormality OK")

    # 2) path
    path = default_path_Gamma_M_K_Gamma()
    res_path = band_structure_along_path(H0, path, n_per_segment=50, N=N)
    print("path: distance shape", res_path["distance"].shape,
          "energies shape", res_path["energies"].shape,
          "node_dist", np.round(res_path["node_dist"], 3))

    # 3) mesh (theta, BZ-tiling)
    res_mesh = band_structure_mesh(H0, n1=6, n2=6)
    print("mesh: energies shape", res_mesh["energies"].shape)
    assert res_mesh["energies"].shape == (6, 6, dim)

    # 4) mesh with potential, sanity: still Hermitian spectrum (real energies)
    V = 0.3 * np.cos(2 * np.pi * np.arange(dim) / dim)
    res_mesh_V = band_structure_mesh(H0, n1=4, n2=4, potential=V)
    assert np.all(np.isfinite(res_mesh_V["energies"]))
    print("mesh with potential OK")