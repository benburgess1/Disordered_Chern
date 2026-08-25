import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from contextlib import nullcontext
from abc import ABC, abstractmethod


def _progress(total, desc, show):
    return tqdm(total=total, desc=desc) if show else nullcontext()


class Lattice(ABC):
    def __init__(self, L=None, Lx=None, Ly=None, t=1, a=1, V=None, V_args={}, show_progress=True, **kwargs):
        if L is not None:
            self.L = L
            self.Lx = L
            self.Ly = L
        elif Lx is not None and Ly is not None:
            self.L = Lx         # For convenience to avoid rewriting lots of code
            self.Lx = Lx
            self.Ly = Ly
        else:
            raise ValueError('Either L, or both Lx and Ly, must be supplied')
        self.t = t
        self.a = a
        self.V = V
        self.V_args = V_args
        self.V_name = self.V.__name__ if self.V is not None else 'None'
        self.show_progress = show_progress
        self.sites = []

    @abstractmethod
    def build_lattice(self):
        pass

    @abstractmethod
    def calc_H(self):
        pass

    @abstractmethod
    def build_save_dict(self):
        pass

    def set_potentials(self):
        if self.V is not None:
            if self.V_name == 'V_random':
                # Set seed once when initialising the potentials, which are calculated by 
                # sequentially calling rng.random(). Thereafter, V on each site is stored 
                # as an attribute for each site and never needs to be recalculated.
                seed = self.V_args['seed'] if 'seed' in self.V_args.keys() else None
                rng = np.random.default_rng(seed=seed)
                self.V_args['rng'] = rng    
            for site in self.sites:
                if self.V_name == 'V_hex_superlattice_alt' or self.V_name == 'V_hex_superlattice_random' or self.V_name == 'V_hex_superlattice_quasiperiodic':     # This potential takes unit cell index and sublattice as arguments, not site position
                    site.V = self.V(*site.uc_idx, site.sublattice, **self.V_args)
                else:
                    site.V = self.V(*site.r, **self.V_args)

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

    def plot_lattice(self, ax=None, color='k', ms=5, plot_V=False, p=0.5, Nx=100,
                     plot_fig=False, plot_V_onsite=False, cmap_name='viridis',
                     suppress_ticks=True, vmax=None, zorder=3, lw=1, plot_next_neighbours=False, 
                     aspect='equal', pad=0.5, title_str='',
                     **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(9, 5)
        ax.set_aspect(aspect)
        positions = np.array([site.r for site in self.sites])
        ax.set_xlim(np.min(positions[:,0] - pad), np.max(positions[:,0] + pad))
        ax.set_ylim(np.min(positions[:,1] - pad), np.max(positions[:,1] + pad))
        if suppress_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        for site in self.sites:
            for neighbour in site.neighbours:
                ax.plot([site.r[0], neighbour.r[0]], [site.r[1], neighbour.r[1]],
                        marker='o', color=color, ms=ms, ls='-', lw=lw, zorder=zorder)
            if plot_next_neighbours:
                for neighbour in site.next_neighbours:
                    ax.plot([site.r[0], neighbour.r[0]], [site.r[1], neighbour.r[1]],
                            marker=None, color=color, ms=ms, ls=':', lw=lw, zorder=zorder)
        if (plot_V or plot_V_onsite) and self.V is not None:
            r_all = np.array([site.r for site in self.sites])
            x = np.linspace(r_all[:, 0].min() - p, r_all[:, 0].max() + p, Nx)
            y = np.linspace(r_all[:, 1].min() - p, r_all[:, 1].max() + p, Nx)
            if plot_V:
                xx, yy = np.meshgrid(x, y, indexing='ij')
                V = self.V(xx, yy, **self.V_args)
                levels = np.linspace(np.min(V), np.max(V), 200)
                ticks = [np.min(V), 0, np.max(V)]
                plot = ax.contourf(xx, yy, V, cmap=plt.colormaps[cmap_name], levels=levels, zorder=zorder-1)
                cbar = fig.colorbar(plot, ticks=ticks)
                cbar.ax.set_ylabel(r'$V$', rotation=0)
            if plot_V_onsite:
                if vmax is None:
                    vmax = np.ceil(np.max(np.abs([site.V for site in self.sites])))
                norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
                sm.set_array([])
                for site in self.sites:
                    c = plt.get_cmap(cmap_name)(norm(site.V))
                    ax.plot([site.r[0]], [site.r[1]], marker='o', ms=ms, color=c, zorder=zorder)
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label(r'$V(\mathbf{r})$', rotation=0)
        ax.set_title(title_str)
        if plot_fig:
            plt.show()
        return ax

    def get_bulk_mask(self, N_edge=1):
        return np.array([~site.near_edge(N_edge, self.L) for site in self.sites], dtype=bool)


class Ladder(Lattice):
    def __init__(self, L, t_AA_mag=1., phi_AA=0., t_BB_mag=1., phi_BB=0., 
                 t_AB_mag=0., phi_AB=0., t_BA_mag=0., phi_BA=0., 
                 t=1, a=1, 
                 V=None, V_args={}, show_progress=True, **kwargs):
        super().__init__(L=L, t=t, a=a, V=V, V_args=V_args,
                         show_progress=show_progress, **kwargs)
        self.t_AA_mag = t_AA_mag
        self.t_BB_mag = t_BB_mag
        self.t_AB_mag = t_AB_mag
        self.t_BA_mag = t_BA_mag
        self.phi_AA = phi_AA
        self.phi_AB = phi_AB
        self.phi_BA = phi_BA
        self.phi_BB = phi_BB
        self.t_AA = self.t_AA_mag * np.exp(1j * self.phi_AA)
        self.t_AB = self.t_AB_mag * np.exp(1j * self.phi_AB)
        self.t_BA = self.t_BA_mag * np.exp(1j * self.phi_BA)
        self.t_BB = self.t_BB_mag * np.exp(1j * self.phi_BB)
        self.a1 = self.a * np.array([1, 0])
        self.b1 = self.a * np.array([0, -1])
        self.N_sublattice = 2
        self.build_lattice()
        self.set_potentials()
        # self.save_dict = self.build_save_dict()

    def build_lattice(self):
        with _progress(total=self.L, desc='Building lattice', show=self.show_progress) as pbar:
            for i in range(self.L):
                r_uc = i * self.a1
                site_A = Site(r=r_uc, uc_idx=np.array([i]), sublattice='A', site_idx=2*i)
                site_B = Site(r=r_uc+self.b1, uc_idx=np.array([i]), sublattice='B', site_idx=2*i+1)
                # Add neighbours - not distinguishing between A/B sites and same/different unit cell
                # here; that will be handled in the building of the Hamiltonian
                site_A.add_neighbour(site_B)
                if i > 0:
                    site_A.add_neighbour(self.sites[2*(i-1)])
                    site_B.add_neighbour(self.sites[2*(i-1)+1])
                    site_A.add_next_neighbour(self.sites[2*(i-1)+1])
                    site_B.add_next_neighbour(self.sites[2*(i-1)])
                self.sites.append(site_A)
                self.sites.append(site_B)
                if pbar: pbar.update(1)

    def calc_H(self):
        N = len(self.sites)
        H = np.zeros((N, N), dtype=np.complex128)
        with _progress(total=N, desc='Building Hamiltonian', show=self.show_progress) as pbar:
            for site in self.sites:
                # Hopping to neighbours
                for neighbour in site.neighbours:
                    # Difference in unit cell index
                    d = int(neighbour.uc_idx[0] - site.uc_idx[0])
                    # Intra-cell hopping:
                    if d == 0:
                        H[neighbour.site_idx, site.site_idx] = self.t
                    # Inter-cell hopping:
                    elif d == 1:
                        if site.sublattice == 'A' and neighbour.sublattice == 'A':
                            H[neighbour.site_idx, site.site_idx] = self.t_AA
                        elif site.sublattice == 'B' and neighbour.sublattice == 'B':
                            H[neighbour.site_idx, site.site_idx] = self.t_BB
                    elif d == -1:
                        if site.sublattice == 'A' and neighbour.sublattice == 'A':
                            H[neighbour.site_idx, site.site_idx] = np.conj(self.t_AA)
                        # elif site.sublattice == 'A' and neighbour.sublattice == 'B':
                        #     H[neighbour.site_idx, site.site_idx] = np.conj(self.t_AB)
                        # elif site.sublattice == 'B' and neighbour.sublattice == 'A':
                        #     H[neighbour.site_idx, site.site_idx] = np.conj(self.t_BA)
                        elif site.sublattice == 'B' and neighbour.sublattice == 'B':
                            H[neighbour.site_idx, site.site_idx] = np.conj(self.t_BB)
                for neighbour in site.next_neighbours:
                    d = int(neighbour.uc_idx[0] - site.uc_idx[0])
                    if d == 1:
                        if site.sublattice == 'A' and neighbour.sublattice == 'B':
                            H[neighbour.site_idx, site.site_idx] = self.t_AB
                        elif site.sublattice == 'B' and neighbour.sublattice == 'A':
                            H[neighbour.site_idx, site.site_idx] = self.t_BA
                    elif d == -1:
                        if site.sublattice == 'A' and neighbour.sublattice == 'B':
                            H[neighbour.site_idx, site.site_idx] = np.conj(self.t_BA)
                        elif site.sublattice == 'B' and neighbour.sublattice == 'A':
                            H[neighbour.site_idx, site.site_idx] = np.conj(self.t_AB)
                # On-site potential
                if self.V is not None:
                    H[site.site_idx, site.site_idx] += site.V
                if pbar: pbar.update(1)
        return H

    def calc_Py(self, average=True):
        """
        Calculate and return the y-polarisation operator:
            Py = sum_jm (-1)^m n_jm
        where j indexes unit cells, m indexes legs ('spin' states), n_jm is 
        a density operator, and (-1)^m is positive (negative) for sites on the 
        upper (lower) leg. 
        If average=True, calculates the average polarisation per unit cell.
        """
        y0 = np.mean([site.r[1] for site in self.sites])
        N = len(self.sites)
        Py = np.zeros((N, N))
        for i, site in enumerate(self.sites):
            Py[i, i] = site.r[1] - y0
        if average:
            Py /= self.L
        return Py


    def calc_Jx(self, i=None):
        """
        Calculate and return the operator giving current in the +x direction,
        evaluated between the two unit cells in the centre of the system.
        Takes into account t_AA, t_BB, t_AB and t_BA hoppings, but not 
        intra-cell t hopping which is purely in the y direction.
        Could be generalised to give currents between all unit cells, but this
        will suffice for initial purposes.
        """
        if i is None:
            i = int(np.floor(self.L/2) - 1)
        j = i + 1
        N = len(self.sites)
        Jx = np.zeros((N, N), dtype=np.complex128)
        # AA hopping
        Jx[2*j, 2*i] = -1j * self.t_AA
        # AB hopping
        Jx[2*j+1, 2*i] = -1j * self.t_AB
        # BA hopping
        Jx[2*j, 2*i+1] = -1j * self.t_BA
        # BB hopping
        Jx[2*j+1, 2*i+1] = -1j * self.t_BB
        # Add Hermitian conjugate terms
        Jx += Jx.conj().T
        return Jx

    
    def calc_Jx_tot(self, average=True):
        """
        Calculate and return the operator giving total current in the +x direction,
        summed and optionally averaged over all unit cells in the system.
        Takes into account t_AA, t_BB, t_AB and t_BA hoppings, but not 
        intra-cell t hopping which is purely in the y direction.
        """
        N = len(self.sites)
        Jx = np.zeros((N, N), dtype=np.complex128)
        for i in range(self.L-1):
            j = i + 1
            # AA hopping
            Jx[2*j, 2*i] = -1j * self.t_AA
            # AB hopping
            Jx[2*j+1, 2*i] = -1j * self.t_AB
            # BA hopping
            Jx[2*j, 2*i+1] = -1j * self.t_BA
            # BB hopping
            Jx[2*j+1, 2*i+1] = -1j * self.t_BB
        # Add Hermitian conjugate terms
        Jx += Jx.conj().T
        if average:
            Jx /= self.L
        return Jx


    def build_save_dict(self):
        save_dict = {
            't':                self.t,
            't_AA':             self.t_AA,
            't_AB':             self.t_AB,
            't_BA':             self.t_BA,
            't_BB':             self.t_BB,
            't_AA_mag':         self.t_AA_mag,
            't_AB_mag':         self.t_AB_mag,
            't_BA_mag':         self.t_BA_mag,
            't_BB_mag':         self.t_BB_mag,
            't_AA_mag':         self.t_AA_mag,
            'phi_AA':           self.phi_AA,
            'phi_AB':           self.phi_AB,
            'phi_BA':           self.phi_BA,
            'phi_BB':           self.phi_BB,
            'L':                self.L,
            'a':                self.a,
            'V_name':           self.V_name,
        }
        for p in ['V', 'beta', 'phi_1', 'phi_2', 'theta']:
            if p in self.V_args.keys():
                save_dict[p] = self.V_args[p]
        return save_dict


class Asym_Ladder(Ladder):
    def __init__(self, L, t_AA_mag=1, t=1, alpha=np.pi/10, t_AB_mag=0.2, a=1, 
                 V=None, V_args={}, show_progress=True, **kwargs):
        super().__init__(L=L, t=t, a=a, 
                         t_AA_mag=t_AA_mag, phi_AA=np.pi/2-alpha, t_BB_mag=t_AA_mag,
                         phi_BB=-np.pi/2-alpha, t_AB_mag=t_AB_mag, t_BA_mag=t_AB_mag,
                         phi_AB=np.pi/2, phi_BA=np.pi/2,
                         V=V, V_args=V_args,
                         show_progress=show_progress, **kwargs)
        self.alpha = alpha
        self.save_dict = self.build_save_dict()

    def build_save_dict(self):
        save_dict = {
            't':                self.t,
            't_AA':             self.t_AA,
            't_BB':             self.t_BB,
            't_AA_mag':         self.t_AA_mag,
            't_AB':             self.t_AB,
            't_BA':             self.t_BA,
            't_AB_mag':         self.t_AB_mag,
            'alpha':            self.alpha,
            'alpha_red':        self.alpha/np.pi,
            'L':                self.L,
            'a':                self.a,
            'V_name':           self.V_name,
        }
        for p in ['V', 'beta', 'phi_1', 'phi_2', 'theta']:
            if p in self.V_args.keys():
                save_dict[p] = self.V_args[p]
        return save_dict


class Plain_Ladder(Ladder):
    def __init__(self, L, t2_mag=1, t=1, phi=np.pi/2, **kwargs):
        super().__init__(L=L, t=t, t_AA_mag=t2_mag, t_BB_mag=t2_mag, 
                         phi_AA=phi, phi_BB=-phi, **kwargs)
        self.t2_mag = t2_mag
        self.phi = phi
        self.save_dict = self.build_save_dict()
    
    def build_save_dict(self):
        save_dict = {
            't':                self.t,
            't2_mag':           self.t2_mag,
            'phi':              self.phi,
            'phi_red':          self.phi/np.pi,
            'L':                self.L,
            'a':                self.a,
            'V_name':           self.V_name,
        }
        for p in ['V', 'beta', 'phi_1', 'phi_2', 'theta']:
            if p in self.V_args.keys():
                save_dict[p] = self.V_args[p]
        return save_dict


class Haldane(Lattice):
    def __init__(self, L=None, Lx=None, Ly=None, t=1, m=0, t2=0.5j, a=1, exclude_endsites=False,
                 V=None, V_args={}, show_progress=True, **kwargs):
        super().__init__(L=L, Lx=Lx, Ly=Ly, t=t, a=a, V=V, V_args=V_args,
                         show_progress=show_progress, **kwargs)
        self.t2 = t2
        self.m = m
        self.a1 = self.a * np.array([1, 0])
        self.a2 = self.a * np.array([0.5, 0.5 * np.sqrt(3)])
        self.b = self.a * np.sqrt(3) / 3
        self.b1 = self.b * np.array([0.5 * np.sqrt(3), 0.5])
        self.b2 = self.b * np.array([0, 1])
        self.N_sublattice = 2
        self.exclude_endsites = exclude_endsites
        self.build_lattice()
        self.set_potentials()
        self.save_dict = self.build_save_dict()

    def build_lattice(self):
        with _progress(total=self.Lx*self.Ly, desc='Building lattice', show=self.show_progress) as pbar:
            for i in range(self.Lx):
                for j in range(self.Ly):
                    r_uc = i * self.a1 + j * self.a2
                    site_A = Site(r=r_uc, uc_idx=np.array([i, j]), sublattice='A', site_idx=i*2*self.Ly+2*j)
                    site_B = Site(r=r_uc+self.b1, uc_idx=np.array([i, j]), sublattice='B', site_idx=i*2*self.Ly+2*j+1)
                    # Add near neighbours
                    site_A.add_neighbour(site_B)
                    if j > 0:
                        site_A.add_neighbour(self.sites[i*2*self.Ly+2*j-1])
                    if i > 0:
                        site_A.add_neighbour(self.sites[2*(i-1)*self.Ly+2*j+1])
                    # Add next-near neighbours
                    if j > 0:
                        site_A.add_next_neighbour(self.sites[i*2*self.Ly+2*(j-1)])
                        site_B.add_next_neighbour(self.sites[i*2*self.Ly+2*(j-1)+1])
                    if i > 0:
                        site_A.add_next_neighbour(self.sites[(i-1)*2*self.Ly+2*j])
                        site_B.add_next_neighbour(self.sites[(i-1)*2*self.Ly+2*j+1])
                        if j < self.Ly-1:
                            site_A.add_next_neighbour(self.sites[(i-1)*2*self.Ly+2*(j+1)])
                            site_B.add_next_neighbour(self.sites[(i-1)*2*self.Ly+2*(j+1)+1])
                    self.sites.append(site_A)
                    self.sites.append(site_B)
                    if pbar: pbar.update(1)
        if self.exclude_endsites:
            self.remove_site(0)
            self.remove_site(2*self.Lx*self.Ly - 2)    # -2 since already removed the first site

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
                            H[next_neighbour.site_idx, site.site_idx] = np.conj(self.t2)
                        else:
                            H[next_neighbour.site_idx, site.site_idx] = self.t2
                    else:
                        if np.all(np.isclose(dr, -self.a1)) or np.all(np.isclose(dr, self.a2)) or np.all(np.isclose(dr, self.a1-self.a2)):
                            H[next_neighbour.site_idx, site.site_idx] = self.t2
                        else:
                            H[next_neighbour.site_idx, site.site_idx] = np.conj(self.t2)
                # On-site potential
                H[site.site_idx, site.site_idx] = self.m if site.sublattice == 'A' else -self.m
                if self.V is not None:
                    H[site.site_idx, site.site_idx] += site.V
                if pbar: pbar.update(1)
        return H

    def build_save_dict(self):
        save_dict = {
            't':                self.t,
            't2':               self.t2,
            't2_mag':           np.abs(self.t2),
            'phi':              np.angle(self.t2),
            'phi_red':          np.angle(self.t2) / np.pi,
            'm':                self.m,
            'L':                self.L,
            'Lx':               self.Lx,
            'Ly':               self.Ly,
            'a':                self.a,
            'exclude_endsites': self.exclude_endsites,
            'V_name':           self.V_name,
        }
        for p in ['V', 'beta', 'phi_1', 'phi_2', 'theta']:
            if p in self.V_args.keys():
                save_dict[p] = self.V_args[p]
        return save_dict

    def calc_Py(self, average=True):
        y0 = np.mean([site.r[1] for site in self.sites])
        N = len(self.sites)
        Py = np.zeros((N, N))
        for i, site in enumerate(self.sites):
            Py[i,i] = site.r[1] - y0
        if average:
            Py /= self.Lx
        return Py

    def calc_Jx_tot(self, average=True):
        """
        Calculate and return the operator giving total current in the +x direction,
        summed and optionally averaged over all unit cells in the system.
        Takes into account all hoppings with components in the x-direction, but not 
        intra-cell t hopping which is purely in the y direction.
        """
        N = len(self.sites)
        Jx = np.zeros((N, N), dtype=np.complex128)
        for i in range(self.L-1):
            for j in range(self.Ly):
                # AA hopping
                Jx[2*self.Ly*(i+1)+2*j, 2*self.Ly*i+2*j] = -1j * self.t2
                # AB hopping
                Jx[2*self.Ly*i+2*j+1, 2*self.Ly*i+2*j] = -1j * self.t
                # BA hopping
                Jx[2*self.Ly*(i+1)+2*j, 2*self.Ly*i+2*j+1] = -1j * self.t
                # BB hopping
                Jx[2*self.Ly*(i+1)+2*j+1, 2*self.Ly*i+2*j+1] = -1j * np.conj(self.t2)
        # Add Hermitian conjugate terms
        Jx += Jx.conj().T
        if average:
            Jx /= self.Lx
        return Jx


class Hofstadter(Lattice):
    def __init__(self, L, t=1, phi=0., a=1,
                 V=None, V_args={}, show_progress=True, **kwargs):
        super().__init__(L=L, t=t, a=a, V=V, V_args=V_args,
                         show_progress=show_progress, **kwargs)
        self.phi = phi
        self.a1 = self.a * np.array([1, 0])
        self.a2 = self.a * np.array([0, 1])
        self.N_sublattice = 1
        self.build_lattice()
        self.set_potentials()
        self.save_dict = self.build_save_dict()

    def build_lattice(self):
        with _progress(total=self.L**2, desc='Building lattice', show=self.show_progress) as pbar:
            for i in range(self.L):
                for j in range(self.L):
                    r = i * self.a1 + j * self.a2
                    site = Site(r=r, uc_idx=np.array([i, j]), site_idx=i*self.L+j)
                    # Add near neighbours
                    if j > 0:
                        site.add_neighbour(self.sites[i*self.L+j-1])
                    if i > 0:
                        site.add_neighbour(self.sites[(i-1)*self.L+j])
                    self.sites.append(site)
                    if pbar: pbar.update(1)

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
                    H[site.site_idx, site.site_idx] += site.V
                if pbar: pbar.update(1)
        return H

    def build_save_dict(self):
        save_dict = {
            't':                self.t,
            'phi':              np.angle(self.t2),
            'L':                self.L,
            'a':                self.a,
            'V_name':           self.V_name,
        }
        for p in ['V', 'beta', 'phi_1', 'phi_2', 'theta']:
            if p in self.V_args.keys():
                save_dict[p] = self.V_args[p]
        return save_dict


class Site:
    def __init__(self, r, uc_idx, site_idx, sublattice=None):
        self.r = r
        self.uc_idx = uc_idx
        self.sublattice = sublattice
        self.site_idx = site_idx
        self.neighbours = []
        self.next_neighbours = []
        self.V = 0

    def add_neighbour(self, other):
        if self not in other.neighbours:
            other.neighbours.append(self)
            self.neighbours.append(other)

    def add_next_neighbour(self, other):
        if self not in other.next_neighbours:
            other.next_neighbours.append(self)
            self.next_neighbours.append(other)

    def near_edge(self, N_edge, L):
        return np.logical_or(np.any(self.uc_idx<=N_edge-1), np.any(self.uc_idx>=L-N_edge))


class Haldane_Ladder(Haldane):
    def __init__(self, L, **kwargs):
        super().__init__(Lx=L, Ly=2, **kwargs)

    def calc_Py(self, average=True):
        y0 = np.mean([site.r[1] for site in self.sites])
        N = len(self.sites)
        Py = np.zeros((N, N))
        for i, site in enumerate(self.sites):
            Py[i,i] = site.r[1] - y0
        if average:
            Py /= self.Lx
        return Py

    def calc_Jx_tot(self, average=True):
        """
        Calculate and return the operator giving total current in the +x direction,
        summed and optionally averaged over all unit cells in the system.
        Takes into account all hoppings with components in the x-direction, but not 
        intra-cell t hopping which is purely in the y direction.
        """
        N = len(self.sites)
        Jx = np.zeros((N, N), dtype=np.complex128)
        for i in range(self.L-1):
            for j in range(2):
                # AA hopping
                Jx[4*(i+1)+2*j, 4*i+2*j] = -1j * self.t2
                # AB hopping
                Jx[4*i+2*j+1, 4*i+2*j] = -1j * self.t
                # BA hopping
                Jx[4*(i+1)+2*j, 4*i+2*j+1] = -1j * self.t
                # BB hopping
                Jx[4*(i+1)+2*j+1, 4*i+2*j+1] = -1j * np.conj(self.t2)
        # Add Hermitian conjugate terms
        Jx += Jx.conj().T
        if average:
            Jx /= self.Lx
        return Jx