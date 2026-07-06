import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import Lattice
import Calculations


def make_title_str(data, base_str='', params={}):
    for k, v in params.items():
        if k in data.files:
            if len(base_str) > 0:
                base_str += ', '
            base_str += v + r'$=$' + f'{data[k]:.4g}'
    return base_str

def plot_spectrum(filenames, x_param='V', x_label=None, title_params={},
                 color_mode=None, cmap='plasma', ipr_scale='log',
                 normalise_E=False, space_x=False, line_width=0.1, lw=1.0):
    """
    Plot spectrum from a list of .npz files.

    Parameters
    ----------
    color_mode : {None, 'ipr', 'exponent', 'polarization'}
        Quantity used to colour each eigenstate. None plots all states in blue.
    """
    fig, ax = plt.subplots()

    # --- Load datasets ---
    datasets = []
    all_color_vals = []

    for i, filename in enumerate(filenames):
        data = np.load(filename)
        E_vals = data['E_vals']
        x_val = float(data[x_param])
        x_pos = i if space_x else x_val

        if normalise_E:
            E_vals = (E_vals - E_vals.min()) / (E_vals.max() - E_vals.min())

        if color_mode == 'ipr':
            color_vals = data['ipr_vals']
        elif color_mode == 'exponent':
            L = data['L']
            color_vals = -np.log(data['ipr_vals']) / np.log(2 * L**2)
        elif color_mode == 'polarization':
            color_vals = data['polarization']
        else:
            color_vals = None

        datasets.append((x_pos, x_val, E_vals, color_vals))
        if color_vals is not None:
            all_color_vals.append(color_vals)

    # --- Set up norm and colorbar ---
    fontsize = 11
    if color_mode == 'ipr':
        all_concat = np.concatenate(all_color_vals)
        norm = (mcolors.LogNorm(vmin=all_concat.min(), vmax=all_concat.max())
                if ipr_scale == 'log' else
                mcolors.Normalize(vmin=all_concat.min(), vmax=all_concat.max()))
        cbar_label = 'IPR'
    elif color_mode == 'exponent':
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cbar_label = r'$-\frac{\ln{(\mathrm{IPR})}}{\ln{(N_\mathrm{sites})}}$'
        fontsize = 15
    elif color_mode == 'polarization':
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        cbar_label = r'$p$'
    else:
        norm = None
        cbar_label = None

    # --- Plot ---
    cmap_fn = plt.get_cmap(cmap)
    for x_pos, x_val, E_vals, color_vals in datasets:
        colors = cmap_fn(norm(color_vals)) if norm is not None else 'b'
        ax.hlines(E_vals, x_pos - line_width/2, x_pos + line_width/2,
                  colors=colors, linewidths=lw)

    # --- Colorbar ---
    if norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, rotation=0, fontsize=fontsize, labelpad=10, y=0.53)

    # --- Axes labels and formatting ---
    if space_x:
        ax.set_xticks([d[0] for d in datasets])
        ax.set_xticklabels([f'{d[1]:.3g}' for d in datasets])

    ax.set_xlabel(x_label if x_label is not None else x_param)
    ax.set_ylabel(r'$E$ / $t$' if not normalise_E else
                  r'$(E - E_\mathrm{min})$ / $(E_\mathrm{max} - E_\mathrm{min})$')
    ax.set_title(make_title_str(data, base_str='Spectrum', params=title_params))
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


def plot_eigenstate(psi=None, filename=None, index=None, model=Lattice.Haldane, L=30, cmap='plasma', ms=5, log=False, max_orders=None,
                    title_str=''):
    if psi is None:
        if filename is None:
            raise ValueError('Either state psi or filename and index must be specified')
        data = np.load(filename)
        psi = data['evects'][:, index]
        L = data['L']
    psi2 = np.abs(psi)**2
    system = model(L, show_progress=False, exclude_endsites=data['exclude_endsites'] if filename is not None else False)
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    system.plot_lattice(ax=ax, ms=0, color='k')
    vmax = np.max(psi2)
    if log:
        if max_orders is not None:
            vmin = vmax * 10 ** (-max_orders)
        else:
            vmin = np.min(psi2[psi2 >0 ])
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for site in system.sites:
        c = plt.get_cmap(cmap)(norm(psi2[site.site_idx]))
        ax.plot([site.r[0]], [site.r[1]], marker='o', ms=ms, color=c)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$|\Psi|^2$', rotation=0)
    if filename is not None:
        E = data['E_vals'][index]
        title_str = f'Eigenstate {index}, ' + r'$E/t=$' + f'{E:.4g}'
        if 'ipr_vals' in data.files:
            title_str += r', $IPR=$' + f'{data['ipr_vals'][index]:.4g}'
        if 'polarization' in data.files:
            title_str += r', $p=$' + f'{data['polarization'][index]:.4g}'
    ax.set_title(title_str)
    plt.show()


def plot_chern_marker(filename, model=Lattice.Hofstadter, cmap='bwr', ms=5, vmax=None, title_params={},
                      calc_new=False, calc_avg=False, N_max=5, **kwargs):
    data = np.load(filename)
    L = data['L']
    system = model(L, show_progress=False, exclude_endsites=data['exclude_endsites'])
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

def plot_C_avg(filename, ax=None, plot_std=False, color='b', ms=4, title_params={}, ylim=None, plot_fig=True,
               label=None):
    data = np.load(filename)
    V_vals = data['V_vals']
    C_mean = data['C_mean']
    if ax is None:
        fig, ax = plt.subplots()
    if plot_std:
        C_std = data['C_std']
        ax.errorbar(V_vals, C_mean, yerr=C_std, marker='o', color=color, ms=ms, ls='-', lw=1, label=label)
    else:
        ax.plot(V_vals, C_mean, marker='o', color=color, ms=ms, ls='-', lw=1, label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r'$V$ / $t$')
    ax.set_ylabel(r'$\overline{C}$')
    title_str = make_title_str(data, base_str='Phase-Averaged Chern Marker', params=title_params)
    ax.set_title(title_str)
    if plot_fig:
        plt.show()

def plot_C_avg_multi(filenames, cmap='viridis', title_params={}, **kwargs):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i / max(len(filenames) - 1, 1)) for i in range(len(filenames))]
    for i, f in enumerate(filenames):
        data = np.load(f)
        L = data['L']
        plot_C_avg(f, ax=ax, color=colors[i], label=str(L), plot_fig=False, **kwargs)
    ax.legend(title=r'$L$')
    title_str = make_title_str(data, base_str='Phase-Averaged Chern Marker', params=title_params)
    ax.set_title(title_str)
    plt.show()
    
def plot_phase_diagram(filename, x_param='t2_mag_vals', y_param='V_vals', z_param='C_mean',
                       cmap='RdBu_r', x_label=None, y_label=None, z_label=None,
                       title_params={}, vmax=None, plot_power_law=False, A=1, n=0.5):
    data = np.load(filename)
    x_data = data[x_param]
    y_data = data[y_param]
    z_data = data[z_param]

    if z_data.shape == (y_data.size, x_data.size):
        z_data = z_data.T
    elif z_data.shape != (x_data.size, y_data.size):
        raise ValueError(f"z_data shape {z_data.shape} inconsistent with x ({x_data.size},) and y ({y_data.size},)")

    dx = (x_data[1] - x_data[0]) / 2
    dy = (y_data[1] - y_data[0]) / 2
    extent = [x_data[0] - dx, x_data[-1] + dx, y_data[0] - dy, y_data[-1] + dy]

    if vmax is None:
        vmax = np.max(np.abs(z_data))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots()
    im = ax.imshow(z_data.T, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, norm=norm)
    
    if plot_power_law:
        x_pl = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_pl = A * x_pl ** n
        label = r'$V_c=A|t_2|^n, A=$' + f'{A:.3g}' r'$, n=$' + f'{n:.3g}'
        ax.plot(x_pl, y_pl, color='c', ls='--', marker=None, label=label)
        ax.legend()


    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(z_label if z_label is not None else z_param, rotation=0)

    ax.set_xlabel(x_label if x_label is not None else x_param)
    ax.set_ylabel(y_label if y_label is not None else y_param)

    title_str = make_title_str(data, base_str='Phase Diagram', params=title_params)
    ax.set_title(title_str)
    plt.show()

def plot_chern_vs_param(filenames, x_param='E_F_vals', x_label=None, filling_fraction=False,
                     legend_param='V', legend_title=None,
                     cmap='viridis', ms=4, lw=1.5, title_params={}):
    if legend_title is None:
        legend_title = legend_param
    if x_label is None:
        x_label = r'$E_F$ / $t$' if x_param == 'E_F_vals' else x_param

    fig, ax = plt.subplots()
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(filenames)))

    for filename, color in zip(filenames, colors):
        data = np.load(filename)
        if x_param == 'N_occ_vals':
            x_vals = data['N_occ_vals'] / (2 * data['L']**2) if filling_fraction else data['N_occ_vals']
        else:
            x_vals = data[x_param]
        C_vals = data['C_vals']
        label = f'{data[legend_param]:.3g}'
        ax.plot(x_vals, C_vals, marker='o', ms=ms, lw=lw, color=color, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$\overline{C}$', rotation=0)
    ax.legend(title=legend_title)
    title_str = make_title_str(data, base_str='Chern marker vs ' + x_label, params=title_params)
    ax.set_title(title_str)
    plt.show()


def plot_dos(filenames, legend_param='V', legend_title=None, cmap='viridis',
             lw=1.5, ax=None, compute_dos=True, title_params={}, **kwargs):
    """
    Plot the density of states for a list of data files.

    Parameters
    ----------
    filenames : list of str
        Paths to .npz files.
    legend_param : str
        Key in each data file used to label each curve in the legend.
    legend_title : str, optional
        Title for the legend. Defaults to legend_param.
    cmap : str
        Colormap from which curve colours are sampled uniformly.
    lw : float
        Line width.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    compute_dos : bool
        If True, compute DOS from E_vals using calc_dos() passing **kwargs.
        If False, read DOS and E_grid directly from data['DOS'] and data['E_grid'].
    **kwargs
        Passed to calc_dos() when compute_dos=True (e.g. sigma, N_grid, n_sigma).
    """
    if legend_title is None:
        legend_title = legend_param

    if ax is None:
        fig, ax = plt.subplots()

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(filenames)))

    for filename, color in zip(filenames, colors):
        data = np.load(filename)
        if compute_dos:
            E_grid, DOS = Calculations.calc_dos(filename=filename, **kwargs)
        else:
            E_grid = data['E_grid']
            DOS = data['DOS']
        label = np.round(data[legend_param], 3)
        ax.plot(E_grid, DOS, lw=lw, color=color, label=label)

    ax.set_xlabel(r'$E$ / $t$')
    ax.set_ylabel(r'$\rho(E)$ (arb.)')
    ax.legend(title=legend_title)
    title_str = make_title_str(data, base_str='Density of States', params=title_params)
    ax.set_title(title_str)
    plt.show()


def plot_polarization_vs_E(filenames, legend_param='V', legend_title=None,
                            cmap='viridis', ms=2, ax=None, lw=0,
                            title_params={}):
    """
    Plot sublattice polarization vs energy for a list of .npz files.

    Parameters
    ----------
    filenames : list of str
        Paths to .npz files, each containing 'E_vals' and 'polarization'.
    legend_param : str
        Key in each data file used to label each curve in the legend.
    legend_title : str, optional
        Title for the legend. Defaults to legend_param.
    cmap : str
        Colormap from which curve colours are sampled uniformly.
    ms : float
        Marker size.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    lw : float
        Line width. Defaults to 0 (markers only).
    title_params : dict
        Parameters to include in the plot title via make_title_str.
    """
    if legend_title is None:
        legend_title = legend_param

    if ax is None:
        fig, ax = plt.subplots()

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(filenames)))

    for filename, color in zip(filenames, colors):
        data = np.load(filename)
        E_vals = data['E_vals']
        polarization = data['polarization']
        label = data[legend_param]
        ax.plot(E_vals, polarization, marker='o', ms=ms, lw=lw,
                color=color, label=label)

    ax.set_xlabel(r'$E$ / $t$')
    ax.set_ylabel(r'$p$', rotation=0)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.legend(title=legend_title)
    ax.set_title(make_title_str(data, base_str='Sublattice polarization', params=title_params))
    plt.show()