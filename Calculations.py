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