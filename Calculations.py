import numpy as np
from tqdm import tqdm
from Lattice import Hofstadter, Haldane


def calc_spectrum(H, return_evects=False):
    if not return_evects:
        evals = np.linalg.eigvalsh(H)
        return evals
    else:
        evals, evects = np.linalg.eigh(H)
        return evals, evects
    

def calc_butterfly(phi_vals, L, calc_ipr=True, save=True, save_filename='Data/Butterly.npz', **kwargs):
    E_vals = np.zeros((L**2, phi_vals.size))
    if calc_ipr:
        ipr_vals = np.copy(E_vals)
    for i, phi in tqdm(enumerate(phi_vals), desc='Calculating spectrum over phi values'):
        system = Hofstadter(phi=phi, L=L, V=None, show_progress=False, **kwargs)
        H = system.calc_H()
        if calc_ipr:
            E_vals[:, i], evects = calc_spectrum(H, return_evects=True)
            ipr_vals[:, i] = np.sum(np.abs(evects)**4, axis=0)
        else:
            E_vals[:, i] = calc_spectrum(H, return_evects=False)
    if save:
        if calc_ipr:
            np.savez(save_filename, phi_vals=phi_vals, E_vals=E_vals, ipr_vals=ipr_vals, L=L, **kwargs)
        else:
            np.savez(save_filename, phi_vals=phi_vals, E_vals=E_vals, L=L, **kwargs)


def calc_chern_marker(evects, system, N_occ=None, E_max=None, E_vals=None, **kwargs):
    """
    Compute the Bianco-Resta real-space Chern marker for each site.

    Parameters
    ----------
    evects : ndarray, shape (N, N)
        Eigenvectors as columns, ordered by ascending eigenvalue.
    sites : list of Site
        System site list (after reindexing, sites[n].site_idx == n).
    N_occ : int, optional
        Number of occupied bands (lowest N_bands eigenstates).
    E_max : float, optional
        Fermi energy; occupies all states with E_vals <= E. Overrides N_bands.
    E_vals : ndarray, shape (N,), optional
        Eigenvalues corresponding to evects columns. Required if E is supplied.

    Returns
    -------
    chern_marker : ndarray, shape (N,)
        Real-space Chern marker C(r) at each site, indexed by site_idx.
    """
    if E_max is not None:
        if E_vals is None:
            raise ValueError("E_vals must be supplied when E is specified.")
        occ = evects[:, E_vals <= E_max]
    elif N_occ is not None:
        occ = evects[:, :N_occ]
    else:
        raise ValueError("Either N_bands or E (with E_vals) must be supplied.")
    
    sites = system.sites

    # Position operators (diagonal matrices)
    x = np.array([site.r[0] for site in sites])
    y = np.array([site.r[1] for site in sites])

    # Projector onto occupied subspace: P = occ @ occ†
    P = occ @ occ.conj().T

    # Bianco-Resta real-space Chern marker.
    # The canonical form (Bianco & Resta 2011) is:
    #   C(r) = -4π Im[ P x Q y P ]_rr,  where Q = I - P
    # Inserting Q = I - P and noting that Im[ P x y P ]_rr = 0
    # (since P x y P is Hermitian on the diagonal), this is equivalent to:
    #   C(r) = +4π Im[ P x P y P ]_rr
    # which is the form computed here.
    PxP = P * x[np.newaxis, :]   # P @ diag(x) @ P = (P * x[newaxis,:]) @ P
    PxP = PxP @ P
    PyP = P * y[np.newaxis, :]
    PyP = PyP @ P

    # Diagonal of PxP @ PyP
    diag_PxPyP = np.einsum('ij,ji->i', PxP, PyP)
    
    chern_marker = 4 * np.pi * np.imag(diag_PxPyP)
    return chern_marker

def calc_avg_chern(chern_marker, system, N_max, return_centre=False):
    """
    Compute the bulk-averaged real-space Chern number.

    The sum of the Chern marker over all bulk sites is divided by the unit
    cell area, giving the average Chern number per unit cell. The bulk region
    is defined as sites within +/- N_max lattice vectors of the lattice centre.

    Parameters
    ----------
    chern_marker : ndarray, shape (N,)
        Real-space Chern marker at each site, indexed by site_idx.
    system : Haldane or Hofstadter
        System object, from which sites, a1, and a2 are taken.
    N_max : float
        Half-width of bulk region in units of lattice vectors.

    Returns
    -------
    avg_chern : float
        Bulk-averaged Chern number.
    """
    # Unit cell area: magnitude of cross product a1 x a2
    uc_area = np.abs(system.a1[0] * system.a2[1] - system.a1[1] * system.a2[0])

    # Infer lattice centre from bounding box of all site positions
    r_all = np.array([site.r for site in system.sites])
    r_centre = 0.5 * np.array([r_all[:, 0].min() + r_all[:, 0].max(),
                                r_all[:, 1].min() + r_all[:, 1].max()])

    # Express displacement from centre in lattice vector coordinates,
    # i.e. find coefficients (c1, c2) such that dr = c1*a1 + c2*a2
    # Solve the 2x2 system [a1 | a2] @ [c1, c2]^T = dr for each site
    A = np.column_stack([system.a1, system.a2])  # shape (2, 2)
    dr = r_all - r_centre                         # shape (N, 2)
    coeffs = np.linalg.solve(A, dr.T).T           # shape (N, 2)

    # Select bulk sites: both lattice-vector coordinates within [-N_max, N_max]
    in_bulk = np.all(np.abs(coeffs) <= N_max, axis=1)

    avg_chern = np.mean(chern_marker[in_bulk]) * system.N_sublattice / uc_area
    if return_centre:
        return avg_chern, r_centre
    else:
        return avg_chern
    

def calc_chern_vs_E(system, E_F_vals=None, N_E_F=100, N_max=3, save=True, save_filename='Data/Haldane_C_vs_E.npz',
                    calc_ipr=False, save_spectrum=False):
    H = system.calc_H()
    evals, evects = calc_spectrum(H, return_evects=True)
    if E_F_vals is None:
        E_F_vals = np.linspace(np.min(evals), np.max(evals), N_E_F)
    if calc_ipr:
        ipr_vals = np.sum(np.abs(evects)**4, axis=0)
    C_vals = np.zeros_like(E_F_vals)
    for i, E in enumerate(tqdm(E_F_vals, desc='Calculating over Fermi energies')):
        chern_marker = calc_chern_marker(evects, system, E_max=E, E_vals=evals)
        C_vals[i] = calc_avg_chern(chern_marker, system, N_max=N_max)
    if save:
        save_dict = system.save_dict
        save_dict.update({
            'E_F_vals': E_F_vals,
            'C_vals':   C_vals,
        })
        if save_spectrum:
            save_dict['E_vals'] = evals
        if calc_ipr:
            save_dict['ipr_vals'] = ipr_vals
        np.savez(save_filename, **save_dict)


def calc_chern_vs_N(system, N_occ_vals=None, N_max=3, save=True, save_filename='Data/Haldane_C_vs_N.npz',
                    calc_ipr=False, save_spectrum=False):
    H = system.calc_H()
    evals, evects = calc_spectrum(H, return_evects=True)
    if N_occ_vals is None:
        N_occ_vals = np.arange(1, 2*system.L**2+1)
    if calc_ipr:
        ipr_vals = np.sum(np.abs(evects)**4, axis=0)
    C_vals = np.zeros(np.size(N_occ_vals))
    for i, N_occ in enumerate(tqdm(N_occ_vals, desc='Calculating over occupancy values')):
        chern_marker = calc_chern_marker(evects, system, N_occ=N_occ)
        C_vals[i] = calc_avg_chern(chern_marker, system, N_max=N_max)
    if save:
        save_dict = system.save_dict
        save_dict.update({
            'N_occ_vals': N_occ_vals,
            'C_vals':   C_vals,
        })
        if save_spectrum:
            save_dict['E_vals'] = evals
        if calc_ipr:
            save_dict['ipr_vals'] = ipr_vals
        np.savez(save_filename, **save_dict)


def calc_dos(E_vals=None, filename=None, sigma=0.1, E_grid=None, n_sigma=5, N_grid=1000,
             bulk=False, **kwargs):
    """
    Calculate the density of states by convolving eigenvalues with a Gaussian kernel.

    Parameters
    ----------
    E_vals : ndarray, optional
        Energy eigenvalues. If not supplied, filename must be given.
    filename : str, optional
        Path to .npz file containing 'E_vals'. Used only if E_vals is None.
    sigma : float
        Width of the Gaussian smoothing kernel.
    E_grid : ndarray, optional
        Energy grid on which to evaluate the DOS. If None, auto-determined
        from the range of E_vals padded by n_sigma * sigma on each side.
    n_sigma : float
        Number of sigma by which to pad the auto-determined E_grid.
    N_grid : int
        Number of points in the auto-determined E_grid.
    bulk : bool
        Project out states not hosted in the bulk of the system. Can only be passed if 
        a filename with stored evects is passed.
    **kwargs : 
        Passed to bulk_eigenstate_mask(). E.g. N_edge, c.
    Returns
    -------
    E_grid : ndarray, shape (N_grid,)
        Energy values at which DOS is evaluated.
    DOS : ndarray, shape (N_grid,)
        Density of states.
    """
    if E_vals is None:
        if filename is None:
            raise ValueError("Either E_vals or filename must be supplied.")
        data = np.load(filename)
        E_vals = data['E_vals'].ravel()
        if bulk:
            mask = bulk_eigenstate_mask(data['evects'], L=data['L'], **kwargs)
            E_vals = E_vals[mask]
    else:
        E_vals = np.asarray(E_vals).ravel()

    if E_grid is None:
        E_min = E_vals.min() - n_sigma * sigma
        E_max = E_vals.max() + n_sigma * sigma
        E_grid = np.linspace(E_min, E_max, N_grid)

    # DOS(E) = sum_n G(E - E_n), where G is a normalised Gaussian.
    # Computed via broadcasting: (N_grid, N_evals) -> sum over N_evals axis.
    # For very large systems, use the explicit loop commented below instead.
    dE = E_grid[:, np.newaxis] - E_vals[np.newaxis, :]   # shape (N_grid, N_evals)
    DOS = np.sum(np.exp(-0.5 * (dE / sigma)**2), axis=1) / (sigma * np.sqrt(2 * np.pi))

    return E_grid, DOS

def calc_sublattice_polarization(evects):
    """
    Calculate the sublattice polarization for each eigenstate.

    Assumes basis states are ordered ABAB..., so sublattice A (B) corresponds
    to even (odd) indices.

    Parameters
    ----------
    evects : ndarray, shape (N, N)
        Eigenvectors as columns.

    Returns
    -------
    polarization : ndarray, shape (N,)
        Sublattice polarization p = <psi|P_A|psi> - <psi|P_B|psi>
        for each eigenstate.
    """
    prob = np.abs(evects)**2          # shape (N, N)
    weight_A = np.sum(prob[0::2], axis=0)
    weight_B = np.sum(prob[1::2], axis=0)
    polarization = weight_A - weight_B
    return polarization


def calc_bloch_state(k, u, L, model='haldane'):
    """
    Construct a Bloch state on a finite lattice in the real-space site basis.

    The Bloch state is:
        psi_r = (1/sqrt(N_uc)) * exp(i k.r_uc) * u[sublattice_idx]
    where r_uc is the unit cell position of site r and sublattice_idx maps
    each site to its component in the spinor u.

    Parameters
    ----------
    k : array-like, shape (2,)
        Crystal momentum vector.
    u : array-like, shape (N_sublattice,)
        Spinor (internal degree of freedom) at this k-point, e.g. from
        diagonalising the 2x2 Bloch Hamiltonian H(k).
    L : int
        Linear dimension of the lattice (L^2 unit cells).
    model : {'haldane', 'hofstadter'}
        Determines which dummy system to construct for site positions/labels.

    Returns
    -------
    psi : ndarray, shape (N_sites,), dtype complex128
        Normalised Bloch state in the real-space site basis.
    """
    k = np.asarray(k)
    u = np.asarray(u, dtype=np.complex128)

    if model == 'haldane':
        system = Haldane(L=L, show_progress=False, exclude_endsites=False)
        sublattice_map = {'A': 0, 'B': 1}
        sublattice_idx = np.array([sublattice_map[site.sublattice] for site in system.sites])
    elif model == 'hofstadter':
        system = Hofstadter(L=L, show_progress=False)
        sublattice_idx = np.zeros(len(system.sites), dtype=int)
    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'haldane' or 'hofstadter'.")
    
    r = np.array([site.r for site in system.sites])  # full site positions
    phase = np.exp(1j * (k @ r.T))                    # shape (N_sites,)
    spinor_weights = u[sublattice_idx]
    psi = phase * spinor_weights
    psi /= np.linalg.norm(psi)

    return psi


def bulk_eigenstate_mask(evects, L=None, N_edge=2, c=0.8, model=Haldane, **kwargs):
    if L is None:
        L = int(np.sqrt(evects.shape[0]/2))
    system = model(L=L, show_progress=False, **kwargs)
    mask = system.get_bulk_mask(N_edge=N_edge)
    bulk_weights = np.sum(np.abs(evects[mask, :])**2, axis=0)
    return bulk_weights > c


def bond_currents(H, eigenvectors, N_occ, check_kirchhoff=True, atol=1e-9):
    """
    Compute all equilibrium bond currents in a finite-size real-space lattice
    model, from a filled Fermi sea of single-particle eigenstates.

    Convention (matches the Hamiltonian convention used throughout Lattice/
    Haldane/Hofstadter): H[j, i] is the hopping amplitude describing a
    particle moving from site i to site j, i.e. the term in H is
        H[j, i] * c_j^dagger c_i   (+ h.c. via H[i, j] = conj(H[j, i]))
    The returned current matrix uses the SAME convention:
        I[j, i] = equilibrium current flowing from site i to site j.
    So I[j, i] > 0 means net particle flow i -> j, and I is antisymmetric,
    I[j, i] = -I[i, j], as required for a single physical current per bond.

    Derivation: for site-occupation continuity dn_i/dt = i[H, n_i], the bond
    current operator between i and j is Hermitian only if it includes both
    the i->j and j->i hopping terms. This gives the bond current operator along
    i->j: 
    
        I_i->j = i(H_ij c_i^dagger c_j - H.c.)

    For a single occupied eigenstate psi_n, the current contribution is:

        I_i->j(n) = -2 * Im[ H_ij * conj(psi_n(i)) * psi_n(j)]
    
    Summing over occupied eigenstates, the currents along all bonds can be
    efficiently computed in one shot via an elementwise (Hadamard) product with
    the one-body density matrix of the occupied manifold:

        I[i, j] = -2 * Im[ H[i, j] * conj(P[i, j]) ].T

    where P is the one-body density matrix of the occupied manifold,
        P[i, j] = sum_{n in occ} psi_n(i) * conj(psi_n(j))
                = Psi_occ @ Psi_occ^dagger .

    The transpose is required so that the element I[j,i] is the current FROM i TO j,
    consistent with the convention of H_ji being the hopping matrix element FROM i TO j.

    This gives every bond current in the lattice
    simultaneously, in one O(N^2) pass, with zero entries automatically at
    any (i, j) pair that isn't an actual bond (since H is zero there).

    This definition is consistent with I_i->j = - del H / del phi_ji, where 
    phi_ji = arg(H_ji) is the complex phase of hopping from i->j.

    Parameters
    ----------
    H : (N, N) complex ndarray
        Real-space Hamiltonian, Hermitian, using the H[j, i] = hop(i->j)
        convention described above.
    eigenvectors : (N, N) complex ndarray
        Full eigenvector matrix from diagonalizing H (e.g. from
        np.linalg.eigh(H)), columns are eigenstates, assumed already sorted
        by ascending energy (as eigh returns them).
    n_occ : int
        Number of occupied single-particle states (e.g. N // 2 at half
        filling). The first n_occ columns of `eigenvectors` are taken as
        occupied.
    check_kirchhoff : bool, default True
        If True, assert that net current into every site vanishes
        (row sums of I are ~0), as a correctness check on H and I. This is
        cheap (O(N^2)) and catches Hamiltonian-construction or convention
        bugs, so it's left on by default; set False only if you're calling
        this in a tight loop (e.g. inside a phase-diagram sweep) and have
        already validated the Hamiltonian construction elsewhere.
    atol : float
        Absolute tolerance for the Kirchhoff-law check.

    Returns
    -------
    I : (N, N) complex ndarray (purely real up to floating point noise)
        Full antisymmetric bond-current matrix. I[j, i] = current i -> j.
        Zero at any (i, j) with no bond (H[j, i] == 0).
        Returned as real via .real for convenience (imaginary part is
        zero to floating-point precision by construction; not discarded
        silently -- see note below).

    Notes
    -----
    - Diagonal elements I[i, i] are exactly zero by construction (H has
      zero diagonal for a hopping-only model; if you ever add on-site
      terms, they don't contribute to bond currents anyway since the
      derivation only uses off-diagonal H).
    - To get the current along a specific bond (i, j), just index the
      result: I[j, i]. No separate single-bond routine is needed -- this
      always computes the full matrix in one shot, since the cost of doing
      so is the same order as computing the density matrix P itself.
    - At a given k-independent filling this is the direct real-space
      analogue of the Bloch-space Hellmann-Feynman bond current
      -dE/dphi discussed earlier: P plays the role the occupied Bloch
      eigenvector played there.
    """
    Psi_occ = eigenvectors[:, :N_occ]
    P = Psi_occ @ Psi_occ.conj().T  # P[i, j] = sum_n psi_n(i) psi_n*(j)

    I = -2.0 * np.imag(H * P.conj()).T  # elementwise (Hadamard) product, NOT matmul

    if check_kirchhoff:
        row_sums = I.sum(axis=1)
        max_violation = np.max(np.abs(row_sums))
        if max_violation > atol:
            raise ValueError(
                f"Kirchhoff's law violated: max |sum_i I[j,i]| = "
                f"{max_violation:.3e} > atol={atol:.1e}. This usually means "
                f"H is not Hermitian, or eigenvectors/H use inconsistent "
                f"conventions."
            )
    return I


import numpy as np


def chiral_edge_current(I, system=None, positions=None, edge_mask=None, center=None, tol=1e-10, **kwargs):
    """
    Compute the total chiral edge current: the sum, over all bonds lying
    entirely within the edge region, of each bond's current weighted by
    its chirality (anticlockwise/clockwise) about the lattice centre --
    the same cross-product quantity used to colour bonds in
    plot_bond_currents().

    Parameters
    ----------
    I : (N, N) array
        Full bond current matrix, I[j, i] = current from i to j (as
        returned by bond_currents()).
    system : Lattice object
        System for which the edge current is to be calculated. Used to 
        extract site positions and edge_mask. If not provided, then both 
        positions and edge_mask must be provided separately.
    positions : (N, 2) array
        Real-space (x, y) coordinates of every site.
    edge_mask : (N,) bool array
        True for sites belonging to an edge unit cell, e.g.
        edge_mask = ~system.get_bulk_mask(N_edge=1). Both the A and B
        sites of an edge unit cell should be marked True.
    center : (2,) array or None
        Point to measure chirality about. Defaults to the centroid of all
        `positions` (not just the edge sites) -- pass this explicitly if
        that's not the right reference point for your geometry.
    tol : float
        Bonds with |I_ij| below this, or whose midpoint sits within `tol`
        of `center`, are skipped (same as plot_bond_currents).

    Returns
    -------
    total : float
        Sum of the signed chirality-weighted current c_ij over every
        unique edge-edge bond. Positive = net anticlockwise circulation,
        negative = net clockwise, matching the red/blue convention in
        plot_bond_currents.
    """
    if system is not None:
        positions = np.array([site.r for site in system.sites])
        edge_mask = ~system.get_bulk_mask(**kwargs)
    else:
        if positions is None:
            raise ValueError('positions must be supplied if system is not supplied')
        if edge_mask is None:
            raise ValueError('edge_mask must be supplied if system is not supplied')
    
    if center is None:
        center = positions.mean(axis=0)

    edge_idx = np.where(edge_mask)[0]
    total = 0.0

    for a, i in enumerate(edge_idx):
        for j in edge_idx[a + 1:]:
            Iij = I[j, i]  # current i -> j
            if abs(Iij) < tol:
                continue

            r_i, r_j = positions[i], positions[j]
            bond_vec = r_j - r_i
            bond_len = np.linalg.norm(bond_vec)
            if bond_len < tol:
                continue  # not actually a bond / duplicate position

            mid = 0.5 * (r_i + r_j)
            radial = mid - center
            radial_norm = np.linalg.norm(radial)
            if radial_norm < tol:
                continue

            bond_hat = bond_vec / bond_len
            current_vec = Iij * bond_hat
            cross_z = radial[0]*current_vec[1] - radial[1]*current_vec[0]
            c = cross_z / radial_norm

            total += c

    return total