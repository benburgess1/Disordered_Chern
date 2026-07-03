import numpy as np
from tqdm import tqdm
from Lattice import Hofstadter


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


def calc_chern_marker(evects, system, N_occ=None, E_max=None, E_vals=None):
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


def calc_dos(E_vals=None, filename=None, sigma=0.1, E_grid=None, n_sigma=5, N_grid=1000):
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