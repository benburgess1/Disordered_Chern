import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp
from tqdm import tqdm
from Haldane import Site
from contextlib import nullcontext

def _progress(total, desc, show):
    return tqdm(total=total, desc=desc) if show else nullcontext()

class Hofstadter:
    def __init__(self, L, t=1, phi=0., a=1,
                 V=None, V_args={}, show_progress=True,
                 **kwargs):
        self.L = L
        self.t = t
        self.phi = phi
        self.V = V
        self.V_args = V_args
        self.a = a
        self.a1 = self.a * np.array([1, 0])
        self.a2 = self.a * np.array([0, 1])
        self.N_sublattice = 1
        self.show_progress = show_progress
        self.sites = []
        self.build_lattice()
        
    def build_lattice(self):
        with _progress(total=self.L**2, desc='Building lattice', show=self.show_progress) as pbar:
            for i in range(self.L):
                for j in range(self.L):
                    r = i * self.a1 + j * self.a2
                    site = Site(r=r, uc_idx=(i,j), site_idx=i*self.L+j)
                    # Add near neighbours
                    if j > 0:
                        site.add_neighbour(self.sites[i*self.L+j-1])
                    if i > 0:
                        site.add_neighbour(self.sites[(i-1)*self.L+j])
                    self.sites.append(site)
                    if pbar: pbar.update(1)

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
            x = np.linspace(-p, (self.L-1)*self.a + p, Nx)
            y = np.copy(x)
            # y = np.linspace(-p, (self.L-1)*self.a + p, Nx)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            V = self.V(xx, yy, **self.V_args)
            levels = np.linspace(np.min(V), np.max(V), 200)
            ticks = [np.min(V), 0, np.max(V)]
            plot = ax.contourf(xx, yy, V, cmap=plt.colormaps['viridis'], levels=levels)
            cbar = fig.colorbar(plot, ticks=ticks)
            cbar.ax.set_ylabel(r'$V$', rotation=0)
        if plot_fig:
            plt.show()
        return 

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
                    if neighbour.uc_idx[0] == site.uc_idx[0]:
                        H[neighbour.site_idx, site.site_idx] = self.t
                    elif neighbour.uc_idx[0] - site.uc_idx[0] == 1:
                        H[neighbour.site_idx, site.site_idx] = self.t * np.exp(-1j * self.phi * site.uc_idx[1])
                    elif neighbour.uc_idx[0] - site.uc_idx[0] == -1:
                        H[neighbour.site_idx, site.site_idx] = self.t * np.exp(1j * self.phi * site.uc_idx[1])
                # On-site potential
                if self.V is not None:
                    H[site.site_idx, site.site_idx] += self.V(*site.r, **self.V_args)
                if pbar: pbar.update(1)
        return H
    

def V_nonsep(x, y, beta=1, a=1, V=1):
    G = 2 * np.pi / a
    return V * (np.cos(beta * G * (x+y)) + np.cos(beta * G * (x-y)))


