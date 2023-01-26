import csv
import h5py
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

import reduced_atmospheres
import reduced_atmospheres.equilibrate_melt as eq_melt

# path to 'itcovitz_reduced_atmospheres'
dir_path = reduced_atmospheres.dir_path

# global constants
gC = reduced_atmospheres.constants.Constants()

# get dpi for current screen (ease and consistency of figure display)
local_dpi = reduced_atmospheres.figure_files.get_dpi()

# color palette
wong = gC.color_wong
cols = {'H2O': wong[2], 'H2': wong[-2], 'CO2': wong[0], 'N2': wong[3],
        'CO': wong[1], 'CH4': wong[-3], 'NH3': wong[-1], 'fO2_A': 'grey',
        'fO2_M': 'k'}

fac = 1e-4 / (4. * np.pi * gC.r_earth**2.)


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
def plot_test():
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    # SET UP FIGURE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    fig = plt.figure(figsize=(5.5, 3.5), dpi=local_dpi)
    plt.subplots_adjust(wspace=0.4, left=0.15, right=0.95, top=0.93,
                        bottom=0.125)
    plt.rcParams.update({'font.size': 10})
    plot_params = {
        'axes.facecolor': '1.',
        'axes.edgecolor': '0.',
        'axes.grid': False,
        'axes.axisbelow': True,
        'axes.labelcolor': '.15',
        'figure.facecolor': '0.',
        'grid.color': '0.92',
        'grid.linestyle': '-',
        'text.color': '0.15',
        'xtick.color': '0.15',
        'ytick.color': '0.15',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'image.cmap': 'rocket',
        'font.family': ['sans-serif'],
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans',
                            'Bitstream Vera Sans', 'sans-serif'],
        'patch.force_edgecolor': True,
        'xtick.bottom': True,
        'xtick.top': True,
        'ytick.left': True,
        'ytick.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': True,
        'axes.spines.top': True}

    with sns.axes_style(plot_params):
        ax0 = fig.add_subplot(121)
        ax0.minorticks_on()
        ax1 = fig.add_subplot(122)
        ax1.minorticks_on()

    xticks = [2e21, 5e21, 1e22, 2e22]
    xticks = [X * 1e-22 for X in xticks]

    for ax in [ax0, ax1]:
        # ax.set_xlabel('Pre-Impact pCO$_2$ /bar')
        # ax.set_xlabel('Pre-Impact Surface Oceans /EO')
        ax.set_xlabel('Temperature /K')
        ax.tick_params(axis='both', which='both', direction='in')

        ax.set_ylabel('H$_2$ Column Density /moles cm$^{-2}$')

        ax.text(x=0., y=1.01, s="$10^4$", transform=ax.transAxes, fontsize=10)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    # IMPORT RELEVANT DATA
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    # impactor mass
    m_imps = np.logspace(np.log10(2.00e21), np.log10(2.44e22), 30, base=10.,
                         endpoint=True)

    # peridotitic melt phase ferric-to-iron ratio
    fracs = np.arange(0.010, 0.11, 0.01)

    # basaltic melt phase oxygen fugacity
    fo2_vals = np.linspace(-2., 1., 20, endpoint=True)

    # target initial surface water inventory
    waters = np.arange(1.4, 2.5, 0.1)

    # target initial CO2 partial pressure
    co2_vals = np.arange(10, 210, 10)

    # target initial mantle H2O wt%
    mantle_H2O = np.arange(0., 0.2, 0.01)

    # system temperature
    temps = np.linspace(700, 1700, 20, endpoint=True)

    # equilibrium H2 column density [moles cm-2]
    h2 = []

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    for temp in temps:
        # file name
        var_str = "%.0d" % temp
        var_str = var_str.replace('.', '_')
        var_str = var_str.replace('+', '')
        file = dir_path + '/output/temps/peridotite_3A_' + var_str

        # read data
        with h5py.File(file + '.hdf5', 'r') as f:
            # unpack trackers
            h2.append(np.array(list(f['atmos/h2'])))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    h2_pi = [h2[i][0] * fac * 1e-4 for i in range(len(h2))]  # post-impact H2
    h2_eq = [h2[i][-1] * fac * 1e-4 for i in range(len(h2))]  # post-equilibration H2

    ax0.plot(temps, h2_pi, color=cols['H2'], marker='s', ms=3,
             label='post-impact')
    ax0.legend(fontsize=8)

    ax1.plot(temps, h2_eq, color=cols['H2'], marker='o', ms=3,
             label='post-equilibration')
    ax1.legend(fontsize=8)

    # plt.savefig(dir_path + '/figures/figure_algebra.pdf', dpi=200)
    plt.show()