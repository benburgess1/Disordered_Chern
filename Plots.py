import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def make_title_str(data, base_str='', params={}):
    for k, v in params.items():
        if k in data.files:
            if len(base_str) > 0:
                base_str += ', '
            base_str += v + r'$=$' + f'{data[k]:.4g}'
    return base_str

def plot_spectrum(filenames, x_param='V', x_label=None, ms=1, title_params={}, color_ipr=False, cmap='viridis', ipr_scale='log', 
                  normalise_E=False, space_x=False):
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
        xs = x_pos * np.ones(E_vals.shape[0])
        if color_ipr:
            c = plt.get_cmap(cmap)(norm(ipr_vals))
            ax.scatter(xs, E_vals, c=c, s=ms, marker='.', linewidths=0)
        else:
            ax.plot(xs, E_vals, marker='.', ms=ms, color='b', ls='')

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


if __name__ == '__main__':
    # system = Hofstadter(L=10, V=V_nonsep, V_args={'beta':1/np.sqrt(2), 'V':1})
    # # H = system.calc_H()
    # # print(sp.linalg.ishermitian(H))
    # system.plot_lattice(plot_V=True, Nx=100)
    phi_vals = np.linspace(0, 2*np.pi, 101)
    f = 'Data/Butterfly_Test_3.npz'
    # calc_butterfly(phi_vals=phi_vals, L=20, calc_ipr=True, save=True, save_filename=f)
    plot_butterfly(f, ms=0.5, title_params={'L':r'$L$'}, color_ipr=True, ipr_scale='linear', cmap='plasma')