import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
import warnings
from matplotlib import colormaps


def plot_bs_line(evals, ticks=[], tick_labels=[], title_str='Bandstructure', **kwargs):
    fig, ax = plt.subplots()
    for i in range(evals.shape[1]):
        ax.plot(evals[:,i], color='b', ls='-', marker=None)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(r'$E$ / $t$')
    ax.set_title(title_str)
    plt.show()


def plot_bs_GMKG(evals, n, **kwargs):
    plot_bs_line(evals, ticks=[0, n, 2*n, 3*n], tick_labels=[r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'], **kwargs)


def plot_dos(E_grid, dos, title_str='Density of States'):
    fig, ax = plt.subplots()
    ax.plot(E_grid, dos, color='b', ls='-', marker=None)
    ax.set_xlabel(r'$E$ / $t$')
    ax.set_ylabel('DoS')
    ax.set_title(title_str)
    plt.show()


def plot_berry_curvature(
    curvature: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    coord_type: str = "theta",
    offset: bool = True,
    n_levels: int = 21,
    cmap: str = "RdBu_r",
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    plot_fig: bool = True,
    title_str: str = 'Curvature',
    plot_BZ: bool = True,
    N: int = 1,
    vmax: float = None,
):
    """
    Plot Berry curvature as a filled contour (contourf) plot.
 
    Parameters
    ----------
    curvature : (N1, N2) or (N_bands, N1, N2) ndarray
        Per-plaquette Berry curvature, e.g. from berry_curvature_fhs. If
        3D, summed over axis 0 (bands) before plotting.
    axis1, axis2 : 1D ndarray
        The MESH GRID coordinate arrays used to generate curvature -- e.g.
        'axis1'/'axis2' from band_structure_mesh -- NOT plaquette centers.
        Assumed uniformly spaced. Works whether curvature has one entry per
        axis1/axis2 point (berry_curvature_fhs periodic=True) or one fewer
        (periodic=False): the number of plaquettes is read off curvature's
        shape, and the corresponding leading entries of axis1, axis2 are
        used as each plaquette's lower-left corner.
    coord_type : {'theta', 'k'}
        Only affects axis labels: 'theta' -> theta1/theta2, 'k' -> kx/ky.
        Does not transform the coordinate values themselves.
    offset : bool
        If True (default), shift plotted coordinates by half the grid
        spacing (dk/2) so each curvature value is centered on its
        plaquette -- where the flux is physically defined -- rather than
        anchored to the plaquette's lower-left grid corner. If False, plot
        at the raw axis1/axis2 corner coordinates.
    n_levels : int
        Number of contour levels, spread symmetrically about zero.
    cmap : str or Colormap
        Colormap; default 'RdBu_r' so curvature = 0 maps to white (paired
        with the symmetric vmin/vmax chosen from max(|curvature|) below).
    ax : matplotlib Axes, optional
        Axes to draw into. A new figure/axes is created if not given.
    colorbar : bool
        Whether to add a colorbar.
 
    Returns
    -------
    fig, ax, cf
        Figure, axes, and the QuadContourSet, in case further
        customization (titles, saving, etc.) is wanted.
    """
    curvature = np.asarray(curvature)
    if curvature.ndim == 3:
        curvature = curvature.sum(axis=0)
    elif curvature.ndim != 2:
        raise ValueError(f"curvature must be 2D or 3D, got shape {curvature.shape}")
 
    n1, n2 = curvature.shape
    x = np.asarray(axis1)[:n1]
    y = np.asarray(axis2)[:n2]
 
    if offset:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        x = x + dx / 2
        y = y + dy / 2
 
    X, Y = np.meshgrid(x, y, indexing="ij")

    if vmax is None:
        vmax = np.max(np.abs(curvature))
        vmax = vmax if vmax > 0 else 1.0  # avoid degenerate levels for all-zero input
    levels = np.linspace(-vmax, vmax, n_levels)
 
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
 
    cf = ax.contourf(X, Y, curvature, levels=levels, cmap=cmap, vmin=-vmax, vmax=vmax)
 
    if coord_type == "theta":
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
    elif coord_type == "k":
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    else:
        raise ValueError("coord_type must be 'theta' or 'k'")
 
    ax.set_aspect("equal")
 
    if colorbar:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Berry curvature (rad)")

    ax.set_title(title_str)

    if plot_BZ and coord_type == 'k':
        bz_patch = RegularPolygon(xy=(0,0), numVertices=6, radius=4*np.pi/(3*N), ec='k', fc=(0,0,0,0), orientation=np.pi/6)
        ax.add_patch(bz_patch)

    if plot_fig:
        plt.show()
 
    return fig, ax, cf


def plot_C_vs_V(filename, title_str='Behaviour vs V', plot_dE=True):
    data = np.load(filename)
    V = data['V_vals']
    C = data['C_vals']
    fig, ax = plt.subplots()
    if plot_dE:
        dE = data['dE_vals']
        ax.plot(V, C, color='r', ls='-', marker=None, label=r'$\Delta E$')         # Dummy plot for legend - plot underneath C so that scale of ax doesn't change
        ax2 = ax.twinx()
        ax2.plot(V, dE, color='r', ls='-', marker=None, label=r'$\Delta E$')
        ax2.axhline(y=0, color='k', ls=':')
        ax2.set_ylabel(r'$\Delta E$ / $t$')
    ax.plot(V, C, color='b', ls='-', marker=None, label=r'$C$')
    ax.set_xlabel(r'$V$ / $t$')
    ax.set_ylabel(r'$C$')
    ax.legend()
    ax.set_title(title_str)
    plt.show()


def plot_dE_vs_V(filename, log=True, title_str='Energy Gap'):
    data = np.load(filename)
    V = data['V_vals']
    dE = data['dE_vals']
    y = np.abs(dE-dE[0])
    fig, ax = plt.subplots()
    ax.plot(V, y, color='r', ls='-', marker=None, label=r'$\Delta E$')
    ax.plot(V, V**2, ls='--', color='k', label=r'$\sim V^2$')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(r'$V$ / $t$')
    ax.set_ylabel(r'$|\Delta E - \Delta E_0|$ / $t$')
    ax.legend()
    ax.set_title(title_str)
    plt.show()


# def plot_Vbeta_phasediagram(filename, plot_quantity='C', mask_zerogap=True, z_label=r'$C$'):
#     data = np.load(filename)
#     x_data = data['beta_vals']
#     y_data = data['V_vals']
 
#     mask_nan = False  # True only when NaN-masking is actually in play
 
#     if plot_quantity == 'C':
#         z = data['C_vals'].astype(float)  # must be float to hold NaN
#         if mask_zerogap:
#             dE = data['dE_vals']
#             z = np.where(dE > 0, z, np.nan)
#             mask_nan = True
#         cmap = plt.get_cmap('RdBu_r').copy()
#         vmax = np.nanmax(np.abs(z))
#         vmin = -vmax
 
#     elif plot_quantity == 'dE':
#         z = data['dE_vals']
#         if mask_zerogap:
#             z = np.where(z > 0, z, 0)
#         cmap = plt.get_cmap('hot').copy()
#         vmax = np.max(z)
#         vmin = 0 if mask_zerogap else np.min(z)
 
#     else:
#         raise ValueError(f"Unknown plot_quantity: {plot_quantity!r} (expected 'C' or 'dE')")
 
#     if mask_nan:
#         cmap.set_bad(color='lightgrey')
 
#     dx = (x_data[1] - x_data[0]) / 2
#     dy = (y_data[1] - y_data[0]) / 2
#     extent = [x_data[0] - dx, x_data[-1] + dx, y_data[0] - dy, y_data[-1] + dy]
 
#     fig, ax = plt.subplots()
#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     im = ax.imshow(z.T, origin='lower', extent=extent, aspect='auto',
#                     cmap=cmap, norm=norm)
 
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(z_label, rotation=0)
 
#     if mask_nan:
#         # Small grey swatch directly beneath the real colorbar, acting as
#         # an explicit "Delta E = 0" legend entry -- matplotlib has no
#         # native concept of a NaN category on a colorbar, so this is a
#         # second, tiny axes positioned to look like part of the same bar.
#         pos = cbar.ax.get_position()
#         gap = 0.015
#         swatch_height = 0.04
#         swatch_ax = fig.add_axes(
#             [pos.x0, pos.y0 - gap - swatch_height, pos.width, swatch_height]
#         )
#         swatch_ax.set_facecolor('lightgrey')
#         swatch_ax.set_xticks([])
#         swatch_ax.set_yticks([0.5])
#         swatch_ax.set_yticklabels([r'$\Delta E=0$'])
#         swatch_ax.yaxis.tick_right()
#         for spine in swatch_ax.spines.values():
#             spine.set_visible(True)
 
#     ax.set_xlabel(r'$\beta$')
#     ax.set_ylabel(r'$V$ / $t$')
 
#     plt.show()
 
#     return fig, ax


def plot_Vbeta_phasediagram(filename, plot_quantity='C', mask_zerogap=True, Ec=0,
                            sentinel=-1000, title_str='Phase Diagram', vmax=None, include_overshoots=True):
    data = np.load(filename)
    x_data = data['beta_vals']
    y_data = data['V_vals']

    mask_applied = False
    overshoot_applied = False

    if plot_quantity == 'C':
        z_raw = np.asarray(data['C_vals'], dtype=float)
        z_int = np.round(z_raw)
        if np.any(np.abs(z_raw - z_int) > 1e-6):
            warnings.warn(
                "plot_Vbeta_phasediagram: C_vals contains entries not close "
                "to integers; rounding to the nearest integer for plotting."
            )
        z = z_int.astype(int)
        if vmax is None:
            vmax = np.max(np.abs(z))
        elif include_overshoots:
            vmax += 1
            overshoot_applied = True
        vmin = -vmax
        z[z > vmax] = vmax
        z[z < vmin] = vmin
        N_c = 2*vmax + 1
        cmap = colormaps['RdBu_r']
        colors = list(cmap(np.linspace(0, 1, N_c)))
        
        if mask_zerogap:
            if sentinel >= vmin - 1:
                raise ValueError(
                    f"sentinel={sentinel} is too close to the data range "
                    f"[{vmin}, {vmax}] to be distinguishable as a masked value; "
                    f"choose something more extreme."
                )
            dE = data['dE_vals']
            z = np.where(dE > Ec, z, sentinel)
            mapping = {sentinel:0}
            for i in range(N_c):
                mapping[vmin+i] = i+1
            C_index = np.full(z.shape, 0)
            for k, v in mapping.items():
                C_index[z==k] = v
            z = C_index.copy()
            colors = ['lightgrey'] + colors
            cmap = mcolors.ListedColormap(colors)
            bounds = np.arange(-0.5, N_c+1, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            mask_applied = True
        else:
            cmap = mcolors.ListedColormap(colors)
            bounds = np.arange(vmin-0.5, vmax+1, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

    elif plot_quantity == 'dE':
        z = data['dE_vals']
        if mask_zerogap:
            z = np.where(z > Ec, z, 0)
        cmap = plt.get_cmap('hot').copy()
        vmax = np.max(z)
        vmin = 0 if mask_zerogap else np.min(z)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    else:
        raise ValueError(f"Unknown plot_quantity: {plot_quantity!r} (expected 'C' or 'dE')")

    dx = (x_data[1] - x_data[0]) / 2
    dy = (y_data[1] - y_data[0]) / 2
    extent = [x_data[0] - dx, x_data[-1] + dx, y_data[0] - dy, y_data[-1] + dy]

    fig, ax = plt.subplots()
    im = ax.imshow(z.T, origin='lower', extent=extent, aspect='auto',
                    cmap=cmap, norm=norm)

    if plot_quantity == 'C':
        if mask_applied:
            ticks = np.arange(0, N_c+1)
            tick_labels = [r'$\Delta E \leq$' + f'{Ec:.2g}'] + list(np.arange(vmin, vmax+1))
            if overshoot_applied:
                tick_labels[1] = r'$\leq$' + str(vmin)
                tick_labels[-1] = r'$\geq$' + str(vmax)
        else:
            ticks = np.arange(vmin, vmax+1)
            tick_labels = ticks
            if overshoot_applied:
                tick_labels[0] = r'$\leq$' + str(vmin)
                tick_labels[-1] = r'$\geq$' + str(vmax)
        cbar = fig.colorbar(im, ax=ax, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        cbar.set_label(r'$C$', rotation=0)
    else:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\Delta E$ / $t$')
        if mask_zerogap:
            tick_labels = cbar.ax.get_yticklabels()
            tick_labels[0] = r'$\leq$' + f'{Ec:.2g}'
            cbar.ax.set_yticklabels(tick_labels)

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$V$ / $t$')

    ax.set_title(title_str)

    plt.show()
    return fig, ax
