from copy import deepcopy as dcop
import csv
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

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
def plot_figure_4():
    """Plot Figure 4 from Itcovitz et al. (2022).

    Impact-generated melt mass or vapor mass as a function of impactor mass or energy of impact.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # STANDARD VALUES IMPACTOR
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # mass [kg]
    m_imp_stan = 2e22
    # radius [m]
    r_imp_stan = 0.5 * 1e3 * eq_melt.impactor_diameter(m_imp_stan, 'E')
    # escape velocity [km s-1]
    v_esc_stan = eq_melt.escape_velocity(gC.m_earth, m_imp_stan, gC.r_earth, r_imp_stan)
    # escape velocity [km s-1]
    v_imp_stan = 2.0 * v_esc_stan
    # impact angle
    theta_stan = 45.
    # impact energy
    energy_stan = 0.5 * m_imp_stan * gC.m_earth / (m_imp_stan + gC.m_earth) * \
                  (1e3 * v_imp_stan) ** 2.
    # modified specific energy of impact
    [q_s_stan, _] = eq_melt.specific_energy(
        gC.m_earth, 
        m_imp_stan, 
        v_imp_stan,
        np.sin(np.pi * theta_stan / 180.)
    )

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SET UP FIGURE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.figure(figsize=(7.5, 3.5), dpi=local_dpi)
    plt.subplots_adjust(left=.08, right=.9, top=.9, bottom=.13, wspace=.05)
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
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # MASS-MELT SUBPLOT
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax0.set_xlabel('Impactor Mass /kg')
    ax0.set_xlim([5e21, 2e23])
    ax0.set_xscale('log')

    ax0.axvline(x=2e22, color='grey', linestyle='--', linewidth=1.)

    ax0.set_ylabel('Melt Mass /kg')
    ax0.set_yscale('log')
    ax0.set_ylim([9.5e21, 2.5e24])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ENERGY-MELT SUBPLOT
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1.set_xlabel('Energy of Impact /J')
    ax1.set_xscale('log')
    ax1.set_xlim([3e29, 2.5e31])

    # vertical line indicating standard impact energy
    ax1.axvline(x=energy_stan, color='grey', linestyle='--', linewidth=1.)
    ax1.text(x=energy_stan, y=1.5e22, s='standard', color='grey',
             rotation=270, fontsize=8)

    # ax1.text(x=3.5e29, y=2e24, s='Melt Mass /kg', fontsize=9,
    #          ha='left', va='top', rotation=90)
    ax1.set_yscale('log')
    ax1.set_ylim([9.5e21, 2.5e24])
    ax1.tick_params(axis='y', which='both', left=True, right=True,
                    labelleft=False, labelright=False)

    # move subplot
    ax0_pos = dcop(ax0.get_position())
    ax1_pos = dcop(ax1.get_position())
    ax1_pos.x0 = ax0_pos.x1
    ax1_pos.x1 = ax0_pos.x1 + (ax0_pos.x1 - ax0_pos.x0)
    ax1.set_position(ax1_pos)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ENERGY-VAPOUR SUBPLOT
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2.set_xlabel('Energy of Impact /J')
    ax2.set_xscale('log')
    ax2.set_xlim([3e29, 2.5e31])

    ax2.axvline(x=energy_stan, color='grey', linestyle='--', linewidth=1.)

    ax2.set_ylabel('Vapour Mass /kg')
    ax2.yaxis.set_label_position("right")
    ax2.set_yscale('log')
    ax2.set_ylim([9.5e20, 2.5e23])
    ax2.tick_params(axis='y', which='both', left=True, right=True,
                    labelleft=False, labelright=True)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # DATA VALUES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    m_melt_00, m_melt_30, m_melt_45, m_melt_60 = [], [], [], []

    energy_00, energy_30, energy_45, energy_60 = [], [], [], []
    mass_00, mass_30, mass_45, mass_60 = [], [], [], []
    q_s_00, q_s_30, q_s_45, q_s_60 = [], [], [], []

    file = f"{dir_path}/reduced_atmospheres/data/melt_masses.txt"
    with open(file, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        next(data, None)
        next(data, None)
        for row in data:
            if len(row) in [14, 1]:
                continue

            # target and impactor masses
            m_t, m_i = float(row[1]), float(row[2])
            # target and impactor radii
            r_t = 0.5 * 1e3 * eq_melt.impactor_diameter(m_t, 'E')
            r_i = 0.5 * 1e3 * eq_melt.impactor_diameter(m_i, 'E')
            # mutual escape velocity
            v_esc = eq_melt.escape_velocity(m_t, m_i, r_t, r_i)
            # impact velocity
            v_imp = float(row[3]) * v_esc
            # impact angle
            theta = float(row[4])

            # energy of impact
            E_imp = float(row[5])

            # specific energy of impact
            [Q_S, _] = eq_melt.specific_energy(
                m_t, 
                m_i, 
                v_imp,
                np.sin(np.pi * theta / 180.)
            )
            # forsterite reservoir masses
            M_MELT = float(row[6])
            M_SCF = float(row[7])
            M_VAP = float(row[8])
            M_SCF_ATM = float(row[9])
            M_ATM = float(row[10])
            M_DISC = float(row[11])

            # # print phase reservoir masses
            # print("\nIMPACTOR MASS = %.2e kg" % m_i)
            # print("     Melt Mass = %.2e kg" % M_MELT)
            # print("      SCF Mass = %.2e kg" % M_SCF)
            # print("   Vapour Mass = %.2e kg" % M_VAP)
            # print("SCF Atmos Mass = %.2e kg" % M_SCF_ATM)
            # print("     Disc Mass = %.2e kg" % M_DISC)

            # what we count as melt mass
            # m_melt = (M_MELT + M_SCF - M_SCF_ATM) / gC.m_earth_mantle
            m_melt = M_MELT + M_SCF - M_SCF_ATM

            # what we count as vapour mass
            m_vap = M_VAP + M_SCF_ATM

            # print("\nWHAT WE CALL MELT & VAPOUR")
            # print("  melt mass = %.2e kg" % m_melt)
            # print("vapour mass = %.2e kg" % m_vap)

            if theta == 0.:
                col = cols['H2']
                energy_00.append(E_imp)
                mass_00.append(m_i)
                q_s_00.append(Q_S)
                m_melt_00.append(m_melt + m_vap)
            elif theta == 30.:
                col = cols['H2O']
                energy_30.append(E_imp)
                mass_30.append(m_i)
                q_s_30.append(Q_S)
                m_melt_30.append(m_melt + m_vap)
            elif theta == 45.:
                col = cols['CO2']
                energy_45.append(E_imp)
                mass_45.append(m_i)
                q_s_45.append(Q_S)
                m_melt_45.append(m_melt + m_vap)
            elif theta == 60.:
                col = cols['N2']
                energy_60.append(E_imp)
                mass_60.append(m_i)
                q_s_60.append(Q_S)
                m_melt_60.append(m_melt + m_vap)

            if float(row[3]) == 1.1:
                mark = '+'
            elif float(row[3]) == 1.5:
                mark = 'o'
            elif float(row[3]) == 2.0:
                mark = 's'

            # plot melt mass against impactor mass
            if float(row[3]) != 1.5:
                ax0.plot(m_i, m_melt, marker=mark, markersize=5, color=col)

            # plot melt mass against impact energy
            ax1.plot(E_imp, m_melt, marker=mark, markersize=5, color=col)

            # plot vapour mass against impact energy
            ax2.plot(E_imp, m_vap, marker=mark, markersize=5, color=col)

    # sys.exit()

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # REGRESSION LINES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit_00 = np.polyfit(np.log10(energy_00), np.log10(m_melt_00), 1)
    fit_30 = np.polyfit(np.log10(energy_30), np.log10(m_melt_30), 1)
    fit_45 = np.polyfit(np.log10(energy_45), np.log10(m_melt_45), 1)
    fit_60 = np.polyfit(np.log10(energy_60), np.log10(m_melt_60), 1)

    plot_e_00 = np.logspace(np.log10(min(energy_00)), np.log10(max(energy_00)),
                             100, endpoint=True, base=10.)
    plot_e_30 = np.logspace(np.log10(min(energy_30)), np.log10(max(energy_30)),
                             100, endpoint=True, base=10.)
    plot_e_45 = np.logspace(np.log10(min(energy_45)), np.log10(max(energy_45)),
                             100, endpoint=True, base=10.)
    plot_e_60 = np.logspace(np.log10(min(energy_60)), np.log10(max(energy_60)),
                             100, endpoint=True, base=10.)

    plot_00, plot_30, plot_45, plot_60 = [], [], [], []
    for i in range(len(plot_e_00)):
        plot_00.append(10. ** (fit_00[0] * np.log10(plot_e_00[i]) + fit_00[1]))
        plot_30.append(10. ** (fit_30[0] * np.log10(plot_e_30[i]) + fit_30[1]))
        plot_45.append(10. ** (fit_45[0] * np.log10(plot_e_45[i]) + fit_45[1]))
        plot_60.append(10. ** (fit_60[0] * np.log10(plot_e_60[i]) + fit_60[1]))

    # ax1.plot(plot_e_00, plot_00, color=cols['H2'], linewidth=1.5)
    # ax1.plot(plot_e_30, plot_30, color=cols['H2O'], linewidth=1.5)
    ax1.plot(plot_e_45, plot_45, color=cols['CO2'], linewidth=1.5)
    # ax1.plot(plot_e_60, plot_60, color=cols['N2'], linewidth=1.5)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # LEGEND
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    handles = []
    handles.append(lines.Line2D([0.], [0.], label='$0^\circ$',
                                color=cols['H2'],  linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='$30^\circ$',
                                color=cols['H2O'], linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='$45^\circ$',
                                color=cols['CO2'], linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='$60^\circ$',
                                color=cols['N2'], linestyle='-', linewidth=2))
    ax0.legend(handles=handles, loc='lower left', fontsize=7, ncol=4,
               bbox_to_anchor=(-0.01, 1.01))

    handles = []
    handles.append(lines.Line2D([0.], [0.], label='$1.1\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='+'))
    handles.append(lines.Line2D([0.], [0.], label='$1.5\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='o'))
    handles.append(lines.Line2D([0.], [0.], label='$2.0\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='s'))
    ax2.legend(handles=handles, loc='lower right', fontsize=7, ncol=3,
               bbox_to_anchor=(1.01, 1.01))

    plt.savefig(f"{dir_path}/figures/figure_4.pdf", dpi=200)
    # plt.show()