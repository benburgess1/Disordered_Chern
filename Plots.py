import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import Hofstadter
import Haldane
import Calculations


def make_title_str(data, base_str='', params={}):
    for k, v in params.items():
        if k in data.files:
            if len(base_str) > 0:
                base_str += ', '
            base_str += v + r'$=$' + f'{data[k]:.4g}'
    return base_str

def plot_spectrum(filenames, x_param='V', x_label=None, title_params={},
                 color_ipr=False, cmap='plasma', ipr_scale='log',
                 normalise_E=False, space_x=False,
                 line_width=0.1, lw=1.0):
    fig, ax = plt.subplots()

    all_ipr = []
    datasets = []

    for i, filename in enumerate(filenames):
        data = np.load(filename)
        E_vals = data['E_vals']
        x_val = data[x_param]
        x_pos = i if space_x else x_val
        if normalise_E:
            E_vals = E_vals / np.max(np.abs(E_vals))
        ipr_vals = data['ipr_vals'] if color_ipr else None
        datasets.append((x_pos, x_val, E_vals, ipr_vals))
        if color_ipr:
            all_ipr.append(ipr_vals)

    if color_ipr:
        all_ipr_concat = np.concatenate(all_ipr)
        if ipr_scale == 'log':
            norm = mcolors.LogNorm(vmin=all_ipr_concat.min(), vmax=all_ipr_concat.max())
        else:
            norm = mcolors.Normalize(vmin=all_ipr_concat.min(), vmax=all_ipr_concat.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    for x_pos, x_val, E_vals, ipr_vals in datasets:
        if color_ipr:
            colors = plt.get_cmap(cmap)(norm(ipr_vals))
            ax.hlines(E_vals, x_pos - line_width/2, x_pos + line_width/2,
                    colors=colors, linewidths=lw)
        else:
            ax.hlines(E_vals, x_pos - line_width/2, x_pos + line_width/2,
                    colors='b', linewidths=lw)

    if space_x:
        x_positions = [d[0] for d in datasets]
        x_labels = [str(d[1]) for d in datasets]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)

    if color_ipr:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('IPR')

    if x_label is None:
        x_label = x_param
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$E$ / $t$' if not normalise_E else r'$E$ / $E_\mathrm{max}$')
    title_str = make_title_str(data, base_str='Spectrum', params=title_params)
    ax.set_title(title_str)
    plt.show()


def plot_butterfly(filename, ms=1, title_params={}, color_ipr=False, cmap='viridis', ipr_scale='log'):
    data = np.load(filename)
    phi_vals = data['phi_vals']
    E_vals = data['E_vals']

    if color_ipr:
        ipr_vals = data['ipr_vals']
        if ipr_scale == 'log':
            norm = mcolors.LogNorm(vmin=ipr_vals.min(), vmax=ipr_vals.max())
        else:
            norm = mcolors.Normalize(vmin=ipr_vals.min(), vmax=ipr_vals.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    fig, ax = plt.subplots()

    for i, phi in enumerate(phi_vals):
        x = phi * np.ones(E_vals.shape[0]) / (2 * np.pi)
        y = E_vals[:, i]
        if color_ipr:
            c = plt.get_cmap(cmap)(norm(ipr_vals[:, i]))
            ax.scatter(x, y, c=c, s=ms, marker='.', linewidths=0)
        else:
            ax.plot(x, y, marker='.', ms=ms, color='b', ls='')

    if color_ipr:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('IPR')

    ax.set_xlabel(r'$\phi$ / $\phi_0$')
    ax.set_ylabel(r'$E$ / $t$')
    title_str = make_title_str(data, base_str='Hofstadter Butterfly', params=title_params)
    ax.set_title(title_str)
    plt.show()


def plot_eigenstate(filename, index, model=Hofstadter.Hofstadter, cmap='plasma', ms=5, log=False):
    data = np.load(filename)
    psi = data['evects'][:, index]
    psi2 = np.abs(psi)**2
    E = data['E_vals'][index]
    ipr = data['ipr_vals'][index]
    L = data['L']
    system = model(L, show_progress=False)
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    system.plot_lattice(ax=ax, ms=0, color='k')
    vmax = np.max(psi2)
    if log:
        norm = mcolors.LogNorm(vmin=np.min(psi2[psi2 > 0]), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for site in system.sites:
        c = plt.get_cmap(cmap)(norm(psi2[site.site_idx]))
        ax.plot([site.r[0]], [site.r[1]], marker='o', ms=ms, color=c)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$|\Psi|^2$', rotation=0)
    title_str = f'Eigenstate {index}, ' + r'$E/t=$' + f'{E:.4g}' + r', $IPR=$' + f'{ipr:.4g}'
    ax.set_title(title_str)
    plt.show()


def plot_chern_marker(filename, model=Hofstadter.Hofstadter, cmap='bwr', ms=5, vmax=None, title_params={},
                      calc_new=False, calc_avg=False, N_max=5, **kwargs):
    data = np.load(filename)
    L = data['L']
    system = model(L, show_progress=False)
    if calc_new:
        evects = data['evects']
        E_vals = data['E_vals']
        chern = Calculations.calc_chern_marker(evects, system, E_vals=E_vals, **kwargs)
    else:
        chern = data['chern_marker']
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    system.plot_lattice(ax=ax, ms=0, color='k')
    if vmax is None:
        vmax = np.max(np.abs(chern))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for site in system.sites:
        c = plt.get_cmap(cmap)(norm(chern[site.site_idx]))
        ax.plot([site.r[0]], [site.r[1]], marker='o', ms=ms, color=c)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$C(\mathbf{r})$', rotation=0)
    title_str = 'Chern Marker'
    if calc_avg:
        c_avg, r_centre = Calculations.calc_avg_chern(chern, system, N_max=N_max, return_centre=True)
        boundary = N_max * np.array([system.a1+system.a2, -system.a1+system.a2, -system.a1-system.a2, system.a1-system.a2, system.a1+system.a2]) + r_centre
        ax.plot(boundary[:,0], boundary[:,1], ls='--', color='k', lw=1, marker=None)
        title_str += r', $\overline{C}=$' + f'{c_avg:.4g}'
    if calc_new:
        if 'N_occ' in kwargs:
            title_str += r', $N_{occ}=$' + f'{kwargs.get('N_occ'):.3g}'
        elif 'E_max' in kwargs:
            title_str += r', $E_{max}=$' + f'{kwargs.get('E_max'):.3g}'
    else:
        if 'N_occ' in data.files:
            title_str += r', $N_{occ}=$' + f'{data['N_occ']:.3g}'
        elif 'E_max' in data.files:
            title_str += r', $E_{max}=$' + f'{data['E_max']:.3g}'
    title_str = make_title_str(data, base_str=title_str, params=title_params)
    ax.set_title(title_str)
    plt.show()

def plot_C_avg(filename, plot_std=False, color='b', ms=4, title_params={}, ylim=None):
    data = np.load(filename)
    V_vals = data['V_vals']
    C_mean = data['C_mean']
    fig, ax = plt.subplots()
    if plot_std:
        C_std = data['C_std']
        ax.errorbar(V_vals, C_mean, yerr=C_std, marker='o', color=color, ms=ms, ls='-', lw=1)
    else:
        ax.plot(V_vals, C_mean, marker='o', color=color, ms=ms, ls='-', lw=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r'$V$ / $t$')
    ax.set_ylabel(r'$\overline{C}$')
    title_str = make_title_str(data, base_str='Phase-Averaged Chern Marker', params=title_params)
    ax.set_title(title_str)
    plt.show()