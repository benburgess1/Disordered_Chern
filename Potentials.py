import numpy as np

def V_sq_sep(x, y, beta=1, a=1, V=1, phi_x=0., phi_y=0., **kwargs):
    G = 2 * np.pi * beta / a
    return V * (np.cos(x*G + phi_x) + np.cos(y*G + phi_y))

def V_sq_nonsep(x, y, beta=1, a=1, V=1, phi_1=0., phi_2=0., **kwargs):
    G = 2 * np.pi / a
    return V * (np.cos(beta * G * (x+y) + phi_1) + np.cos(beta * G * (x-y) + phi_2))

def V_pw(x, beta=1, a=1, V=1, phi_x=0., **kwargs):
    G = 2 * np.pi * beta / a
    return V * np.cos(x*G + phi_x)

def V_hex_wpl(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos((G1[0]+G2[0])*x + (G1[1]+G2[1])*y + phi_1) + np.cos((G1[0]-G2[0])*x + (G1[1]-G2[1])*y + phi_2))

def V_hex_sep(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos(G1[0]*x + G1[1]*y + phi_1) + np.cos(G2[0]*x + G2[1]*y + phi_2))

def V_hex_rot(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, theta=0., **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    G1 = R @ G1
    G2 = R @ G2
    return V * (np.cos((G1[0]+G2[0])*x + (G1[1]+G2[1])*y + phi_1) + np.cos((G1[0]-G2[0])*x + (G1[1]-G2[1])*y + phi_2))

def origin_to_phases(x0, y0, beta=1, a=1):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    r0 = np.array([x0, y0])
    phi_1 = -beta * (G1+G2) @ r0
    phi_2 = -beta * (G1-G2) @ r0
    return phi_1, phi_2


def lattice_origin_to_phases(i0, j0, beta=1):
    phi_1 = -beta * 2*np.pi * (i0 + j0)
    phi_2 = -beta * 2*np.pi * (i0 - j0)
    return phi_1, phi_2

def V_random(x, y, V=1, rng=np.random.default_rng(0), **kwargs):
    return V * (2 * rng.random() - 1)

def V_hex_superlattice(x, y, V=1, n_i=5, n_j=5, i0=3, j0=3, a=1, **kwargs):
    # Calculate site position in basis of lattice vectors (a1, a2)
    G = 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    idx = np.array([G1, G2]) @ np.array([x, y]) / (2*np.pi)
    # Rounding and tolerance used to eliminate small floating point errors.
    # NB don't round to nearest integer since B sublattice points do not lie
    # on integer lattice points.
    idx = np.round(idx)
    tol = 1e-3
    i, j = idx + tol
    # print(i, j)
    if (i-i0) % n_i < 2*tol or (j-j0) % n_j < 2*tol:
        return -V
    else:
        return 0
    

def V_hex_superlattice_alt(i, j, sublattice, V=1, n_i=5, n_j=5, i0=3, j0=3, i_vals=None,
                           j_vals=None, L=30, **kwargs):
    if i_vals is None:
            i_vals = i0 + np.arange(0, L+1, n_i)
    if j_vals is None:
        j_vals = j0 + np.arange(0, L+1, n_j)
    # tol = 1e-3
    # i += tol
    # j += tol
    if sublattice == 'A':
        if i in i_vals or j in j_vals:
            return -V
        else:
            return 0
    elif sublattice == 'B':
        if i+1 in i_vals or j+1 in j_vals:
            return -V
        else:
            return 0
        

def V_hex_superlattice_random(i, j, sublattice, V=1, n_i=5, n_j=5, i0=3, j0=3, rng=np.random.default_rng(0), 
                              i_vals=None, j_vals=None, L=30, **kwargs):
    if i_vals is None:
        i_vals = i0 + np.arange(0, L+1, n_i)
    if j_vals is None:
        j_vals = j0 + np.arange(0, L+1, n_j)
    # tol = 1e-3
    # i += tol
    # j += tol
    if sublattice == 'A':
        if i in i_vals or j in j_vals:
            return 0
        else:
            return V * (2 * rng.random() - 1)
    elif sublattice == 'B':
        if i+1 in i_vals or j+1 in j_vals:
            return 0
        else:
            return V * (2 * rng.random() - 1)


def V_hex_superlattice_quasiperiodic(i, j, sublattice, V0=100, V=4, beta=1/np.sqrt(2), phi_1=0, phi_2=0, n_i=5, n_j=5, i0=3, j0=3, **kwargs):
    tol = 1e-3
    i += tol
    j += tol
    if sublattice == 'A':
        if (i-i0) % n_i < 2*tol or (j-j0) % n_j < 2*tol:
            return 2*V * np.cos(2*np.pi*beta*i) * np.cos(2*np.pi*beta*j)
        else:
            return V0
    elif sublattice == 'B':
        if (i-i0+1) % n_i < 2*tol or (j-j0+1) % n_j < 2*tol:
            return 2*V * np.cos(2*np.pi*beta*(i+1/3)) * np.cos(2*np.pi*beta*(j+1/3))
        else:
            return V0