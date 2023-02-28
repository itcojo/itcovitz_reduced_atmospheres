import sys

import matplotlib.gridspec as gs
import matplotlib.lines as lines
import matplotlib.pyplot as plt
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


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def plot_figure_5() -> None:
    """Plot distribution of impactor iron between the target interior, atmosphere, and escaping the system, as a function of impactor mass.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SET UP FIGURE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.figure(figsize=(6.5, 3.5), dpi=local_dpi)
    plt.subplots_adjust(left=0.1, right=0.94, top=0.92, bottom=0.15, wspace=0.02)
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
    grid = gs.GridSpec(1, 3, width_ratios=[1.25, 2, 1.25])

    with sns.axes_style(plot_params):
        ax0 = plt.subplot(grid[0])
        ax0.minorticks_on()

        ax1 = plt.subplot(grid[1])
        ax1.minorticks_on()

        ax2 = plt.subplot(grid[2])
        ax2.minorticks_on()

    for ax in [ax0, ax1, ax2]:
        ax.set_xlim([0.2, 4.2])

        ax.set_ylim([-0.05, 1.05])

        ax.axvline(x=2.0, color='grey', linestyle='--')
        ax.axvline(x=2.44, color='grey', linestyle='--')

    # label axes
    ax1.set_xlabel('Impactor Mass /10$^{22}$ kg')
    ax0.set_ylabel('Iron Distribution Fraction')

    # titles
    ax0.set_title("θ = 30$^\circ$")
    ax1.set_title("θ = 45$^\circ$")
    ax2.set_title("θ = 60$^\circ$")

    # labels on central subplot
    ax0.text(x=2.0, y=0.82, s='standard', color='grey',
            fontsize=8, va='top', ha='left', rotation=270)
    ax0.text(x=2.44, y=0.82, s='maximum HSE', color='grey',
            fontsize=8, va='top', ha='left', rotation=270)

    ax1.text(x=0.35, y=1.0, s='$v_i = 2.0\,v_\mathrm{esc}$',
             fontsize=8, va='center')

    ax1.tick_params(axis='y', which='both', left=True, right=True,
                    labelleft=False, labelright=False)
    ax2.tick_params(axis='y', which='both', left=True, right=True,
                    labelleft=False, labelright=True)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # DATA VALUES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    masses, vels, thetas = [], [], []
    X_int, X_surf, X_atm, X_disc, X_ejec = [], [], [], [], []

    impact_m_earth = 5.9127e+24

    atm_mark = 'o'
    int_mark = 's'
    ejec_mark = 'X'

    file = f"{dir_path}/reduced_atmospheres/data/iron_distributions.txt"
    with open(file, 'r') as f:
        count = -1
        for line in f:
            count += 1
            if count < 2:
                continue
            else:
                row = line.split(",")

                if len(row) == 11:
                    masses.append(float(row[0]) * impact_m_earth)
                    vels.append(float(row[1]))
                    thetas.append(float(row[2]))
                    X_int.append(float(row[5]) + float(row[6]))
                    # X_surf.append(float(row[6]))
                    X_atm.append(float(row[7]))
                    # X_disc.append(float(row[8]))
                    X_ejec.append(float(row[8]) + float(row[9]))

    interp_mass_30, interp_int_30, interp_atm_30, interp_ejec_30 = [], [], [], []
    interp_mass_45, interp_int_45, interp_atm_45, interp_ejec_45 = [], [], [], []
    interp_mass_60, interp_int_60, interp_atm_60, interp_ejec_60 = [], [], [], []
    for i in range(len(masses)):
        if vels[i] == 2.0:
            # do not plot head-on collisions
            if thetas[i] == 0:
                continue

            # simply plot data for 30 or 60 degree impacts
            elif thetas[i] == 30.:
                interp_mass_30.append(masses[i] * 1e-22)
                interp_int_30.append(X_int[i])
                interp_atm_30.append(X_atm[i])
                interp_ejec_30.append(X_ejec[i])

                ax0.plot(masses[i] * 1e-22, X_int[i], marker=int_mark,
                         markersize=4, color=cols['CO2'])
                ax0.plot(masses[i] * 1e-22, X_atm[i], marker=atm_mark,
                         markersize=4, color=cols['H2O'])
                ax0.plot(masses[i] * 1e-22, X_ejec[i], marker=ejec_mark,
                         markersize=4, color=cols['N2'])

            # carry out interpolation for 45 degree impacts
            if thetas[i] == 45.:
                interp_mass_45.append(masses[i] * 1e-22)
                interp_int_45.append(X_int[i])
                interp_atm_45.append(X_atm[i])
                interp_ejec_45.append(X_ejec[i])

                ax1.plot(masses[i] * 1e-22, X_int[i], marker=int_mark,
                         markersize=4, color=cols['CO2'])
                ax1.plot(masses[i] * 1e-22, X_atm[i], marker=atm_mark,
                         markersize=4, color=cols['H2O'])
                ax1.plot(masses[i] * 1e-22, X_ejec[i], marker=ejec_mark,
                         markersize=4, color=cols['N2'])

            elif thetas[i] == 60.:
                interp_mass_60.append(masses[i] * 1e-22)
                interp_int_60.append(X_int[i])
                interp_atm_60.append(X_atm[i])
                interp_ejec_60.append(X_ejec[i])

                ax2.plot(masses[i] * 1e-22, X_int[i], marker=int_mark,
                         markersize=4, color=cols['CO2'])
                ax2.plot(masses[i] * 1e-22, X_atm[i], marker=atm_mark,
                         markersize=4, color=cols['H2O'])
                ax2.plot(masses[i] * 1e-22, X_ejec[i], marker=ejec_mark,
                         markersize=4, color=cols['N2'])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # REGRESSION LINES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit_int_30 = np.polyfit(interp_mass_30, interp_int_30, 2)
    fit_atm_30 = np.polyfit(interp_mass_30, interp_atm_30, 2)
    fit_ejec_30 = np.polyfit(interp_mass_30, interp_ejec_30, 2)

    plot_masses_30 = np.linspace(4e21, 4e22, 50)
    plot_masses_30 = plot_masses_30 * 1e-22

    plot_int_30 = (fit_int_30[0] * plot_masses_30 ** 2) + \
                  (fit_int_30[1] * plot_masses_30) + fit_int_30[2]

    plot_atm_30 = (fit_atm_30[0] * plot_masses_30 ** 2) + \
                  (fit_atm_30[1] * plot_masses_30) + fit_atm_30[2]

    plot_ejec_30 = (fit_ejec_30[0] * plot_masses_30 ** 2) + \
                   (fit_ejec_30[1] * plot_masses_30) + fit_ejec_30[2]

    for i in range(len(plot_int_30)):
        for item in [plot_int_30, plot_atm_30, plot_ejec_30]:
            if item[i] < 0.:
                item[i] = 0.

    ax0.plot(plot_masses_30, plot_int_30, color=cols['CO2'],
             linestyle='-', linewidth=1.2, label='interior')
    ax0.plot(plot_masses_30, plot_atm_30, color=cols['H2O'],
             linestyle='-', linewidth=1.2, label='atmosphere')
    ax0.plot(plot_masses_30, plot_ejec_30, color=cols['N2'],
             linestyle='-', linewidth=1.2, label='not accreted')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit_int_45 = np.polyfit(interp_mass_45, interp_int_45, 2)
    fit_atm_45 = np.polyfit(interp_mass_45, interp_atm_45, 2)
    fit_ejec_45 = np.polyfit(interp_mass_45, interp_ejec_45, 2)

    plot_masses_45 = np.linspace(4e21, 4e22, 50)
    plot_masses_45 = plot_masses_45 * 1e-22

    plot_int_45 = (fit_int_45[0] * plot_masses_45 ** 2) + \
                  (fit_int_45[1] * plot_masses_45) + fit_int_45[2]

    plot_atm_45 = (fit_atm_45[0] * plot_masses_45 ** 2) + \
                  (fit_atm_45[1] * plot_masses_45) + fit_atm_45[2]

    plot_ejec_45 = (fit_ejec_45[0] * plot_masses_45 ** 2) + \
                   (fit_ejec_45[1] * plot_masses_45) + fit_ejec_45[2]

    for i in range(len(plot_int_45)):
        for item in [plot_int_45, plot_atm_45, plot_ejec_45]:
            if item[i] < 0.:
                item[i] = 0.

    ax1.plot(plot_masses_45, plot_int_45, color=cols['CO2'],
             linestyle='-', linewidth=1.2, label='interior')
    ax1.plot(plot_masses_45, plot_atm_45, color=cols['H2O'],
             linestyle='-', linewidth=1.2, label='atmosphere')
    ax1.plot(plot_masses_45, plot_ejec_45, color=cols['N2'],
             linestyle='-', linewidth=1.2, label='not accreted')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit_int_60 = np.polyfit(interp_mass_60, interp_int_60, 2)
    fit_atm_60 = np.polyfit(interp_mass_60, interp_atm_60, 2)
    fit_ejec_60 = np.polyfit(interp_mass_60, interp_ejec_60, 2)

    plot_masses_60 = np.linspace(4e21, 4e22, 50)
    plot_masses_60 = plot_masses_60 * 1e-22

    plot_int_60 = (fit_int_60[0] * plot_masses_60 ** 2) + \
                  (fit_int_60[1] * plot_masses_60) + fit_int_60[2]

    plot_atm_60 = (fit_atm_60[0] * plot_masses_60 ** 2) + \
                  (fit_atm_60[1] * plot_masses_60) + fit_atm_60[2]

    plot_ejec_60 = (fit_ejec_60[0] * plot_masses_60 ** 2) + \
                   (fit_ejec_60[1] * plot_masses_60) + fit_ejec_60[2]

    for i in range(len(plot_int_60)):
        for item in [plot_int_60, plot_atm_60, plot_ejec_60]:
            if item[i] < 0.:
                item[i] = 0.

    ax2.plot(plot_masses_60, plot_int_60, color=cols['CO2'],
             linestyle='-', linewidth=1.2, label='interior')
    ax2.plot(plot_masses_60, plot_atm_60, color=cols['H2O'],
             linestyle='-', linewidth=1.2, label='atmosphere')
    ax2.plot(plot_masses_60, plot_ejec_60, color=cols['N2'],
             linestyle='-', linewidth=1.2, label='not accreted')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # LINE LABELS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ax1.text(x=plot_masses_30[0], y=0.80 * plot_int_30[0], s='interior',
    #          fontsize=9, color=cols['CO2'], rotation=-27, ha='left',
    #          va='bottom')
    # ax1.text(x=interp_mass_45[0], y=1.15 * interp_atm_45[0], s='atmosphere',
    #          fontsize=9, color=cols['H2O'], rotation=12, ha='left',
    #          va='bottom')
    # ax1.text(x=2.8, y=0.13, s='not accreted',
    #          fontsize=9, color=cols['N2'], rotation=20, ha='left',
    #          va='bottom')

    handles = []
    handles.append(lines.Line2D([0.], [0.], label='interior',
                                color='k', marker=int_mark,
                                linestyle='-'))
    handles.append(lines.Line2D([0.], [0.], label='atmosphere',
                                color=cols['H2O'],  marker=atm_mark,
                                linestyle='-'))
    handles.append(lines.Line2D([0.], [0.], label='not accreted',
                                color=cols['N2'],  marker=ejec_mark,
                                linestyle='-'))

    ax1.legend(handles=handles, loc='upper right', fontsize=8)

    plt.savefig(f"{dir_path}/figures/figure_5.pdf", dpi=200)
    # plt.show()