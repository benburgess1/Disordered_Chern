import numpy as np
from Hofstadter import Hofstadter
from Haldane import Haldane
from tqdm import tqdm


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


def calc_chern_marker(evects, sites, N_occ=None, E=None, E_vals=None):
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
    E : float, optional
        Fermi energy; occupies all states with E_vals <= E. Overrides N_bands.
    E_vals : ndarray, shape (N,), optional
        Eigenvalues corresponding to evects columns. Required if E is supplied.

    Returns
    -------
    chern_marker : ndarray, shape (N,)
        Real-space Chern marker C(r) at each site, indexed by site_idx.
    """
    if E is not None:
        if E_vals is None:
            raise ValueError("E_vals must be supplied when E is specified.")
        occ = evects[:, E_vals <= E]
    elif N_occ is not None:
        occ = evects[:, :N_occ]
    else:
        raise ValueError("Either N_bands or E (with E_vals) must be supplied.")

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