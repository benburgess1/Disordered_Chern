import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def calc_H0(k, t=1, t2=1, m=0, a=1, phi=0, **kwargs):
    if np.angle(t2) > 1e-6:
        print('Warning: complex t2 passed. Extracting magnitude and phase.')
        phi = np.angle(t2)
        t2 = np.abs(t2)
    a1 = a * np.array([1, 0])
    a2 = a * np.array([-0.5, -0.5*np.sqrt(3)])
    a3 = a * np.array([-0.5, 0.5*np.sqrt(3)])
    b = a * np.sqrt(3) / 3
    b1 = b * np.array([-0.5*np.sqrt(3), 0.5])
    b2 = b * np.array([0, -1])
    b3 = b * np.array([0.5*np.sqrt(3), 0.5])
    H_plus = t * (np.exp(1j * k.dot(b1)) + np.exp(1j * k.dot(b2)) + np.exp(1j * k.dot(b3)))
    H_z = m + 2 * t2 * np.sin(phi) * (np.sin(k.dot(a1)) + np.sin(k.dot(a2)) + np.sin(k.dot(a3)))
    H_id = 2 * t2 * np.cos(phi) * (np.cos(k.dot(a1)) + np.cos(k.dot(a2)) + np.cos(k.dot(a3)))
    H = np.array([[H_id + H_z, H_plus],                 # NB various sign conventions/gauge choices could change the form of this Bloch Hamiltonian
                  [np.conj(H_plus), H_id - H_z]])       # This choice is consistent with the real-space tight-binding model considered here
    return H

def H0_components(k, 
                  t=1, t2=1, m=0, a=1, phi=0, 
                  **kwargs):     ### Something is broken with this function! ###
    # H0 = calc_H0(k, **kwargs)
    # return extract_components(H0)
    a1 = a * np.array([1, 0])
    a2 = a * np.array([-0.5, -0.5*np.sqrt(3)])
    a3 = a * np.array([-0.5, 0.5*np.sqrt(3)])
    b = a * np.sqrt(3) / 3
    b1 = b * np.array([-0.5*np.sqrt(3), 0.5])
    b2 = b * np.array([0, -1])
    b3 = b * np.array([0.5*np.sqrt(3), 0.5])
    H_z = m + 2 * t2 * np.sin(phi) * (np.sin(k.dot(a1)) + np.sin(k.dot(a2)) + np.sin(k.dot(a3)))
    H_id = 2 * t2 * np.cos(phi) * (np.cos(k.dot(a1)) + np.cos(k.dot(a2)) + np.cos(k.dot(a3)))
    H_plus = t * (np.exp(1j * k.dot(b1)) + np.exp(1j * k.dot(b2)) + np.exp(1j * k.dot(b3)))
    H_x = np.real(H_plus)
    H_y = -np.imag(H_plus)
    return H_id, H_x, H_y, H_z

def k_K(a=1):
    G = 4 * np.pi / (a * np.sqrt(3))
    return 0.5 * G * np.array([1/np.sqrt(3), 1])


def k_K_prime(a=1):
    G = 4 * np.pi / (a * np.sqrt(3))
    return 0.5 * G * np.array([-1/np.sqrt(3), 1])


def V_pair(theta1, theta2, V=1, beta=1/np.sqrt(2), a=1, **kwargs):
    # Calculates pair of potentials for an A site located at the origin, and B site 
    # separated by a displacement b1
    # Effective position of the A site / the origin is via the phases theta1, theta2
    # Varying theta1, theta2 within [0, 2pi) is equivalent to sampling all
    # configurations of the quasicrystal
    # Returns diag(V_A, V_B)
    b = a * np.sqrt(3) / 3
    b1 = b * np.array([-0.5*np.sqrt(3), 0.5])
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    V_A = V * np.cos(theta1) * np.cos(theta2)
    # V_B = V * np.cos(theta1 + beta * G1 @ b1) * np.cos(theta2 + beta * G2 @ b1)
    V_B = V * np.cos(theta1 - beta*4*np.pi/3) * np.cos(theta2 + beta*2*np.pi/3)
    return np.diag([V_A, V_B])


def calc_sigma(k, N_theta, **kwargs):
    theta1 = np.linspace(0, 2*np.pi, N_theta+1)[:-1]
    theta2 = np.copy(theta1)
    H0 = calc_H0(k=k, **kwargs)
    H_inv_tot = np.zeros((2,2), dtype=np.complex128)
    with tqdm(total=N_theta**2, desc='Evaluating over theta values') as pbar:
        for i in range(N_theta):
            for j in range(N_theta):
                H_inv_tot += np.linalg.inv(H0 + V_pair(theta1[i], theta2[j], **kwargs))
                pbar.update(1)
    H_inv_avg = H_inv_tot / (N_theta**2)
    sigma = np.linalg.inv(H_inv_avg) - H0
    return sigma

def extract_x(M):
    return 0.5 * (M[...,0,1] + M[...,1,0])

def extract_y(M):
    return 0.5j * (M[...,0,1] - M[...,1,0])

def extract_z(M):
    return 0.5 * (M[...,0,0] - M[...,1,1])

def extract_I(M):
    return 0.5 * (M[...,0,0] + M[...,1,1])

def extract_components(M):
    return extract_I(M), extract_x(M), extract_y(M), extract_z(M)

def calc_G0(E, k, eps=1e-6, **kwargs):
    H0 = calc_H0(k=k, **kwargs)
    return np.linalg.inv((E + 1j*eps) * np.eye(2) - H0)


def calc_sigma_born(E, k, q_arr=[], phi_arr=None, V=1, return_components=False, **kwargs):
    if phi_arr is None:
        phi_arr = np.zeros(len(q_arr))
    sigma = np.zeros((2,2), dtype=np.complex128)
    for s in [1, -1]:
    # for s in [1]:
        for i, q in enumerate(q_arr):
            G0 = calc_G0(E, k+s*q, **kwargs)
            U = np.diag([1, np.exp(1j*s*phi_arr[i])])
            sigma += (V/2)**2 * U @ G0 @ U.conj() #calc_G0(E, k+s*q, **kwargs) #/ (2*np.pi)**2
    if return_components:
        S0 = extract_I(sigma)
        Sx = extract_x(sigma)
        Sy = extract_y(sigma)
        Sz = extract_z(sigma)
        return S0, Sx, Sy, Sz
    else:
        return sigma


def sigma_z_vs_V(E, k, V_vals, **kwargs):
    sigma_z = np.zeros_like(V_vals)
    for i, V in enumerate(V_vals):
        sigma = calc_sigma_born(E, k, V=V, **kwargs)
        sigma_z[i] = extract_z(sigma)
    return sigma_z


def plot_sigma_z_vs_V(V_vals, sigma_z):
    fig, ax = plt.subplots()
    ax.plot(V_vals, sigma_z, color='b', ls='-', marker=None)
    ax.set_xlabel(r'$V$ / $t_1$')
    ax.set_ylabel(r'$\Sigma_z$ / $t_1$')
    ax.set_title('Self-Energy Mass Renormalisation vs V')
    plt.show()


def sigma_z_vs_dm(E, k, dm_vals, m0=0, **kwargs):
    sigma_z = np.zeros_like(dm_vals)
    for i, dm in enumerate(dm_vals):
        sigma = calc_sigma_born(E, k, m=m0+dm, **kwargs)
        sigma_z[i] = extract_z(sigma)
    return sigma_z


def sigma_components_vs_dm(E, dm_vals, m0=0, sigma_func=calc_sigma_born, **kwargs):
    sigma_I = np.zeros_like(dm_vals)
    sigma_x = np.zeros_like(dm_vals)
    sigma_y = np.zeros_like(dm_vals)
    sigma_z = np.zeros_like(dm_vals)
    for i, dm in enumerate(dm_vals):
        sigma = sigma_func(E=E, m=m0+dm, **kwargs)
        sigma_I[i] = extract_I(sigma)
        sigma_x[i] = extract_x(sigma)
        sigma_y[i] = extract_y(sigma)
        sigma_z[i] = extract_z(sigma)
    return sigma_I, sigma_x, sigma_y, sigma_z


def plot_sigma_z_vs_dm(dm_vals, sigma_z):
    fig, ax = plt.subplots()
    ax.plot(dm_vals, sigma_z, color='b', ls='-', marker=None)
    ax.set_xlabel(r'$\delta m$ / $t_1$')
    ax.set_ylabel(r'$\Sigma_z$ / $t_1$')
    ax.set_title(r'Self-Energy Mass Renormalisation vs $\delta m$')
    plt.show()


def plot_sigma_components_vs_dm(dm_vals, sigma_I, sigma_x, sigma_y, sigma_z):
    fig, ax = plt.subplots()
    ax.plot(dm_vals, sigma_I, color='r', ls='-', marker=None, label=r'$\Sigma_0$')
    ax.plot(dm_vals, sigma_x, color='c', ls='-', marker=None, label=r'$\Sigma_x$')
    ax.plot(dm_vals, sigma_y, color='g', ls='-', marker=None, label=r'$\Sigma_y$')
    ax.plot(dm_vals, sigma_z, color='b', ls='-', marker=None, label=r'$\Sigma_z$')
    ax.set_xlabel(r'$\delta m$ / $t_1$')
    ax.set_ylabel(r'$\Sigma$ / $t_1$')
    ax.set_title(r'Self-energy components vs $\delta m$')
    ax.legend()
    plt.show()


def calc_E_surface(kx_vals, ky_vals, E_F=0, **kwargs):
    E_vals = np.zeros((kx_vals.size, ky_vals.size))
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            k = np.array([kx, ky])
            hI, hx, hy, hz = H0_components(k=k, **kwargs)
            SI, Sx, Sy, Sz = calc_sigma_born(E=E_F, k=k, return_components=True, **kwargs)
            E_vals[i,j] = np.sqrt(np.abs(hI+SI)**2 + np.abs(hx+Sx)**2 + np.abs(hy+Sy)**2 + np.abs(hz+Sz)**2)
            # H0 = calc_H0(k=k, **kwargs)
            # Sigma = calc_sigma_born(E=E_F, k=k, return_components=False, **kwargs)
            # # hI, hx, hy, hz = extract_components(H0)
            # SI, Sx, Sy, Sz = extract_components(Sigma)
            # E_vals[i,j] = np.sqrt(np.abs(hI+SI)**2 + np.abs(hx+Sx)**2 + np.abs(hy+Sy)**2 + np.abs(hz+Sz)**2)
            # H = H0 + Sigma
            # E_vals[i,j] = np.linalg.eigvalsh(H)[1]
    return E_vals


def plot_E_surface(kx_vals, ky_vals, E_vals, plot_BZ=True, cmap='viridis', vmax=None, vmin=None,
                   title_params={}, **kwargs):
    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    fig, ax = plt.subplots()
    if vmax is None:
        vmax = E_vals.max()
    if vmin is None:
        vmin = E_vals.min()
    levels = np.linspace(vmin, vmax, 200)
    contour = ax.contourf(kxx, kyy, E_vals, levels=levels, cmap=cmap)
    ax.set_aspect('equal')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    cbar = fig.colorbar(contour)
    cbar.set_label(r'$E$ / $t$')
    if plot_BZ:
        a = kwargs.get('a', 1)
        G = 4 * np.pi / (a * np.sqrt(3))
        k_K = 0.5 * G * np.array([1/np.sqrt(3), 1])
        k_K_mag = np.linalg.norm(k_K)
        BZ = RegularPolygon(xy=(0,0), numVertices=6, radius=k_K_mag, orientation=np.pi/6,
                        ec='k', fc=(0,0,0,0))
        ax.add_patch(BZ)
    title_str = ''
    for k, v in title_params.items():
        if k in kwargs:
            if len(title_str) > 0:
                title_str += ', '
            title_str += v + f'={kwargs[k]:.4g}'
    ax.set_title(title_str)
    ax.set_xlim(np.min(kx_vals), np.max(kx_vals))
    ax.set_ylim(np.min(ky_vals), np.max(ky_vals))
    plt.show()


def calc_E_min(kx_vals=np.linspace(-5, 5, 50), ky_vals=np.linspace(-5, 5, 50), **kwargs):
    E_vals = calc_E_surface(kx_vals, ky_vals, **kwargs)
    return np.min(E_vals)


def calc_Vc(V_vals, E_threshold=0.01, **kwargs):
    E_min = np.zeros_like(V_vals)
    for i, V in enumerate(V_vals):
        E_min[i] = calc_E_min(V=V, **kwargs)
    if np.min(E_min) < E_threshold:
        return V_vals[np.argmin(E_min)]
    else:
        print(f'No energy gap values below threshold {E_threshold:.4g} (min = {np.min(E_min):.4g})')
        return np.nan


def calc_mc(m_vals, E_threshold=0.01, **kwargs):
    E_min = np.zeros_like(m_vals)
    for i, m in enumerate(m_vals):
        E_min[i] = calc_E_min(m=m, **kwargs)
    if np.min(E_min) < E_threshold:
        return m_vals[np.argmin(E_min)]
    else:
        print(f'No energy gap values below threshold {E_threshold:.4g} (min = {np.min(E_min):.4g})')
        return np.nan


def calc_Vc_vs_m(m_vals, V_vals, **kwargs):
    Vc_vals = np.full_like(m_vals, np.nan)
    for i, m in enumerate(tqdm(m_vals, desc='Evaluating over m values')):
        Vc_vals[i] = calc_Vc(V_vals, m=m, **kwargs)
    return Vc_vals


def calc_mc_vs_V(m_vals, V_vals, **kwargs):
    mc_vals = np.full_like(V_vals, np.nan)
    for i, V in enumerate(tqdm(V_vals, desc='Evaluating over V values')):
        mc_vals[i] = calc_mc(m_vals, V=V, **kwargs)
    return mc_vals

def lattice_vectors(a=1, **kwargs):
    a1 = a * np.array([1, 0])
    a2 = a * np.array([1/2, np.sqrt(3)/2])
    return a1, a2

def theta_to_cartesian(theta, **kwargs):
    a1, a2 = lattice_vectors(**kwargs)
    A = np.array([a1, a2])
    k = np.linalg.inv(A) @ theta
    return k


def calc_sigma_random(V=1, N_theta=50, E=0, eps=1e-6, sigma_prev=np.zeros((2,2), dtype=np.complex128), 
                      show_progress=False, **kwargs):
    theta1 = np.linspace(0, 2*np.pi, N_theta+1)[:-1]
    theta2 = np.copy(theta1)
    dtheta = theta1[1] - theta1[0]
    Sigma = np.zeros((2, 2), dtype=np.complex128)
    P_A = np.array([[1,0],
                    [0,0]])
    P_B = np.array([[0,0],
                    [0,1]])
    if show_progress:
        pbar = tqdm(total=N_theta**2, desc='Summing over BZ')
    else: 
        pbar = None
    for i in range(N_theta):
        for j in range(N_theta):
            theta = np.array([theta1[i], theta2[j]])
            k = theta_to_cartesian(theta)
            G = np.linalg.inv((E + 1j*eps)*np.eye(2) - calc_H0(k, **kwargs) - sigma_prev)
            Sigma += P_A @ G @ P_A + P_B @ G @ P_B
            if pbar is not None:
                pbar.update(1)
    Sigma *= (2*V)**2 / 3
    Sigma *= (dtheta/(2*np.pi))**2      # Normalize by BZ area
    return Sigma


def calc_sigma_random_iterative(N_iter=5, return_iterations=False, **kwargs):
    sigma = np.zeros((N_iter+1, 2, 2), dtype=np.complex128)
    for i in range(1, N_iter+1):
        sigma[i,:,:] = calc_sigma_random(sigma_prev=sigma[i-1, :, :], **kwargs)
    if return_iterations:
        return sigma[1:, :, :]
    else:
        return sigma[-1, :, :]


def plot_sigma_components_vs_iter(N_iter, sigma_I, sigma_x, sigma_y, sigma_z):
    fig, ax = plt.subplots()
    x = np.arange(1, N_iter+1)
    ax.plot(x, sigma_I, color='r', ls='-', marker=None, label=r'$\Sigma_0$')
    ax.plot(x, sigma_x, color='c', ls='-', marker=None, label=r'$\Sigma_x$')
    ax.plot(x, sigma_y, color='g', ls='-', marker=None, label=r'$\Sigma_y$')
    ax.plot(x, sigma_z, color='b', ls='-', marker=None, label=r'$\Sigma_z$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\Sigma$ / $t_1$')
    ax.set_title(r'Iterated self-energy components')
    ax.legend()
    plt.show()


def calc_random_mc(V_vals, auto_m=True, m_vals=None, m0=None, N_iter=5, **kwargs):
    mc_vals = np.zeros(V_vals.size, dtype=np.float64)
    if m0 is None:
        m0 = 3*np.sqrt(3)*kwargs.get('t2', 1.5)
    for i, V in enumerate(V_vals):
        print(f'Evaluating V = {V:.4g} ({i+1}/{V_vals.size})')
        if np.isclose(V, 0):
            mc_vals[i] = m0
            continue
        if auto_m:
            dm_min = 0.1 * V**2
            dm_max = 0.3 * V**2
            m_vals = m0 + np.linspace(dm_min, dm_max, 20)
        dm = m_vals[1] - m_vals[0]
        Sz = np.zeros_like(m_vals)
        for j, m in enumerate(m_vals):
            sigma = calc_sigma_random_iterative(N_iter=N_iter, return_iterations=False, V=V, m=m, 
                                                show_progress=False, **kwargs)
            Sz[j] = np.real(extract_z(sigma))
        y = Sz + m_vals - m0
        # Estimate zero crossing point by linearly interpolating between the values either side
        try:
            idx = np.argmax(y[y<0])
            y1, y2 = y[idx:idx+2]
            mc_vals[i] = m_vals[idx] + dm * np.abs(y1) / (y2 - y1)
        except ValueError:
            print(f'Error at V = {V:.4g}')
            print(f'Min y = {np.min(y):.3g}; max y = {np.max(y):.3g}')
            mc_vals[i] = np.nan
    return mc_vals