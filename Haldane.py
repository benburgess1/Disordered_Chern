import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext

def _progress(total, desc, show):
    return tqdm(total=total, desc=desc) if show else nullcontext()

class Haldane:
    def __init__(self, L, t=1, m=0, t2=0.5j, a=1, exclude_endsites=True,
                 V=None, V_args={}, show_progress=True):
        self.L = L
        self.t = t
        self.t2 = t2
        self.m = m
        self.V = V
        self.V_args = V_args
        self.a = a
        self.a1 = self.a * np.array([1,0])
        self.a2 = self.a * np.array([0.5, 0.5*np.sqrt(3)])
        self.b = self.a * np.sqrt(3) / 3
        self.b1 = self.b * np.array([0.5*np.sqrt(3), 0.5])
        self.b2 = self.b * np.array([0, 1])
        self.N_sublattice = 2
        self.show_progress = show_progress
        self.sites = []
        self.build_lattice(exclude_endsites=exclude_endsites)
        
    def build_lattice(self, exclude_endsites):
        with _progress(total=self.L**2, desc='Building lattice', show=self.show_progress) as pbar:
            for i in range(self.L):
                for j in range(self.L):
                    r_uc = i * self.a1 + j * self.a2
                    site_A = Site(r=r_uc, uc_idx=(i,j), sublattice='A', site_idx=i*2*self.L+2*j)
                    site_B = Site(r=r_uc+self.b1, uc_idx=(i,j), sublattice='B', site_idx=i*2*self.L+2*j+1)
                    # Add near neighbours
                    site_A.add_neighbour(site_B)
                    if j > 0:
                        site_A.add_neighbour(self.sites[i*2*self.L+2*j-1])
                    if i > 0:
                        site_A.add_neighbour(self.sites[2*(i-1)*self.L+2*j+1])
                    # Add next-near neighbours
                    if j > 0:
                        site_A.add_next_neighbour(self.sites[i*2*self.L+2*(j-1)])
                        site_B.add_next_neighbour(self.sites[i*2*self.L+2*(j-1)+1])
                    if i > 0:
                        site_A.add_next_neighbour(self.sites[(i-1)*2*self.L+2*j])
                        site_B.add_next_neighbour(self.sites[(i-1)*2*self.L+2*j+1])
                        if j < self.L-1:
                            site_A.add_next_neighbour(self.sites[(i-1)*2*self.L+2*(j+1)])
                            site_B.add_next_neighbour(self.sites[(i-1)*2*self.L+2*(j+1)+1])
                    self.sites.append(site_A)
                    self.sites.append(site_B)
                    if pbar: pbar.update(1)
        if exclude_endsites:
            self.remove_site(0)
            self.remove_site(2*self.L**2 - 2)       # -2 since already removed the first site


    def plot_lattice(self, ax=None, color='k', ms=5, plot_V=False, p=0.5, Nx=100, plot_fig=False):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for site in self.sites:
            for neighbour in site.neighbours:
                ax.plot([site.r[0], neighbour.r[0]], [site.r[1], neighbour.r[1]], 
                        marker='o', color=color, ms=ms, ls='-')
        if plot_V and self.V is not None:
            # p = 0.5
            x = np.linspace(-p, (self.L-1)*(self.a1[0]+self.a2[0]) + self.b1[0] + p, Nx)
            y = np.linspace(-p, (self.L-1)*self.a2[1] + self.b1[1] + p, Nx)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            V = self.V(xx, yy, **self.V_args)
            levels = np.linspace(np.min(V), np.max(V), 200)
            ticks = [np.min(V), 0, np.max(V)]
            plot = ax.contourf(xx, yy, V, cmap=plt.colormaps['viridis'], levels=levels)
            cbar = fig.colorbar(plot, ticks=ticks)
            cbar.ax.set_ylabel(r'$V$', rotation=0)
        if plot_fig:
            plt.show()


    def remove_site(self, site_idx):
        site = self.sites[site_idx]
        for neighbour in site.neighbours:
            neighbour.neighbours.remove(site)
        for next_neighbour in site.next_neighbours:
            next_neighbour.next_neighbours.remove(site)
        self.sites.remove(site)
        self.reindex_sites()

    def reindex_sites(self):
        for n, site in enumerate(self.sites):
            site.site_idx = n
    
    def calc_H(self):
        N = len(self.sites)
        H = np.zeros((N, N), dtype=np.complex128)
        with _progress(total=N, desc='Building Hamiltonian', show=self.show_progress) as pbar:
            for site in self.sites:
                # Near neighbour hopping
                for neighbour in site.neighbours:
                    H[neighbour.site_idx, site.site_idx] = self.t
                # Next near neighbour hopping
                for next_neighbour in site.next_neighbours:
                    dr = next_neighbour.r - site.r
                    if site.sublattice == 'A':
                        if np.all(np.isclose(dr, -self.a1)) or np.all(np.isclose(dr, self.a2)) or np.all(np.isclose(dr, self.a1-self.a2)):
                            H[next_neighbour.site_idx, site.site_idx] = self.t2
                        else:
                            H[next_neighbour.site_idx, site.site_idx] = np.conj(self.t2)
                    else:
                        if np.all(np.isclose(dr, -self.a1)) or np.all(np.isclose(dr, self.a2)) or np.all(np.isclose(dr, self.a1-self.a2)):
                            H[next_neighbour.site_idx, site.site_idx] = np.conj(self.t2)
                        else:
                            H[next_neighbour.site_idx, site.site_idx] = self.t2
                # On-site potential
                H[site.site_idx, site.site_idx] = self.m if site.sublattice=='A' else -self.m
                if self.V is not None:
                    H[site.site_idx, site.site_idx] += self.V(*site.r, **self.V_args)
                if pbar: pbar.update(1)
        return H


class Site:
    def __init__(self, r, uc_idx, site_idx, sublattice=None):
        self.r = r
        self.uc_idx = uc_idx
        self.sublattice = sublattice
        self.site_idx = site_idx
        self.neighbours = []
        self.next_neighbours = []

    def add_neighbour(self, other):
        if self not in other.neighbours:
            other.neighbours.append(self)
            self.neighbours.append(other)

    def add_next_neighbour(self, other):
        if self not in other.next_neighbours:
            other.next_neighbours.append(self)
            self.next_neighbours.append(other)

def V_square(x, y, G=1, V=1):
    return V * (np.cos(x*G) + np.cos(y*G))

def V_pw(x, y, G=1, V=1):
    return V * np.cos(y*G)

def V_wpl(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos((G1[0]+G2[0])*x + (G1[1]+G2[1])*y + phi_1) + np.cos((G1[0]-G2[0])*x + (G1[1]-G2[1])*y + phi_2))

def V_sep(x, y, beta=1, V=1, a=1):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos(G1[0]*x + G1[1]*y) + np.cos(G2[0]*x + G2[1]*y))
