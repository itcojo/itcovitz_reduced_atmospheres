import csv
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
def plot_figure_2(energy=True, mass=False, specific=False):
    """
    Plots impact-generated melt mass as a function of either impact energy,
    impactor mass, or modified specific energy of impact (Stewart+, 2015).

    Parameters
    ----------
    energy : bool
        Dictates whether impact energy is used as the x-axis for plotting.
    mass : bool
        Dictates whether impactor mass is used as the x-axis for plotting.
    specific : bool
        Dictates whether modified specific energy of impact is used as the
        x-axis for plotting.

    Returns
    -------

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
    [q_s_stan, _] = eq_melt.specific_energy(gC.m_earth, m_imp_stan, v_imp_stan,
                                            np.sin(np.pi * theta_stan / 180.))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SET UP FIGURE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.figure(figsize=(3, 3.5), dpi=local_dpi)
    plt.subplots_adjust(wspace=0.4, left=0.22, right=0.95, top=0.98,
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
        ax0 = fig.add_subplot(111)
        ax0.minorticks_on()

    # plot input
    if energy:
        ax0.set_xlabel('Energy of Impact /J')
        ax0.set_xscale('log')
        ax0.set_xlim([3e29, 2.5e31])

        # vertical line indicating standard impact energy
        ax0.axvline(x=energy_stan, color='grey', linestyle='--', linewidth=1.)
        ax0.text(x=energy_stan, y=1.5e22, s='standard', color='grey',
                 rotation=270, fontsize=8)

    if mass:
        ax0.set_xlabel('Impactor Mass /kg')
        ax0.set_xscale('log')

    if specific:
        ax0.set_xlabel('Specific Energy of Impact /J.kg$^{-1}$')
        ax0.set_xscale('log')

        # vertical line indicating standard specify impact energy
        ax0.axvline(x=q_s_stan, color='grey', linestyle='--', linewidth=1.)
        ax0.text(x=q_s_stan, y=2.5e22, s='standard', color='grey', rotation=270,
                 fontsize=8)

    ax0.set_ylabel('Melt Mass /kg')
    ax0.set_yscale('log')
    ax0.set_ylim([9.5e21, 2.5e24])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # DATA VALUES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    m_melt_00, m_melt_30, m_melt_45, m_melt_60 = [], [], [], []

    energy_00, energy_30, energy_45, energy_60 = [], [], [], []
    mass_00, mass_30, mass_45, mass_60 = [], [], [], []
    q_s_00, q_s_30, q_s_45, q_s_60 = [], [], [], []

    file = dir_path + '/reduced_atmospheres/data/melt_masses.txt'
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
            r_i = 0.5 * 1e3 * eq_melt.impactor_diameter(m_t, 'E')
            # mutual escape velocity
            v_esc = eq_melt.escape_velocity(m_t, m_i, r_t, r_i)
            # impact velocity
            v_imp = float(row[3]) * v_esc
            # impact angle
            theta = float(row[4])

            # energy of impact
            E_imp = float(row[5])

            # specific energy of impact
            [Q_S, _] = eq_melt.specific_energy(m_t, m_i, v_imp,
                                               np.sin(np.pi * theta / 180.))
            # forsterite reservoir masses
            M_MELT = float(row[6])
            M_SCF = float(row[7])
            M_VAP = float(row[8])
            M_SCF_ATM = float(row[9])
            M_ATM = float(row[10])
            M_DISC = float(row[11])

            # what we count as melt mass
            # m_melt = (M_MELT + M_SCF - M_SCF_ATM) / gC.m_earth_mantle
            m_melt = M_MELT + M_SCF - M_SCF_ATM

            if theta == 0.:
                col = cols['H2']
                energy_00.append(E_imp)
                mass_00.append(m_i)
                q_s_00.append(Q_S)
                m_melt_00.append(m_melt)
            elif theta == 30.:
                col = cols['H2O']
                energy_30.append(E_imp)
                mass_30.append(m_i)
                q_s_30.append(Q_S)
                m_melt_30.append(m_melt)
            elif theta == 45.:
                col = cols['CO2']
                energy_45.append(E_imp)
                mass_45.append(m_i)
                q_s_45.append(Q_S)
                m_melt_45.append(m_melt)
            elif theta == 60.:
                col = cols['N2']
                energy_60.append(E_imp)
                mass_60.append(m_i)
                q_s_60.append(Q_S)
                m_melt_60.append(m_melt)

            if float(row[3]) == 1.1:
                mark = '+'
            elif float(row[3]) == 1.5:
                mark = 'o'
            elif float(row[3]) == 2.0:
                mark = 's'

            # plot melt mass against specific impact energy
            if energy:
                ax0.plot(E_imp, m_melt, marker=mark, markersize=5, color=col)
            if mass:
                ax0.plot(m_i, m_melt, marker=mark, markersize=5, color=col)
            if specific:
                ax0.plot(Q_S, m_melt, marker=mark, markersize=5, color=col)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # REGRESSION LINES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if energy:
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

        # ax0.plot(plot_e_00, plot_00, color=cols['H2'], linewidth=1.5)
        # ax0.plot(plot_e_30, plot_30, color=cols['H2O'], linewidth=1.5)
        ax0.plot(plot_e_45, plot_45, color=cols['CO2'], linewidth=1.5)
        # ax0.plot(plot_e_60, plot_60, color=cols['N2'], linewidth=1.5)

    if specific:
        fit_00 = np.polyfit(np.log10(q_s_00), np.log10(m_melt_00), 1)
        fit_30 = np.polyfit(np.log10(q_s_30), np.log10(m_melt_30), 1)
        fit_45 = np.polyfit(np.log10(q_s_45), np.log10(m_melt_45), 1)
        fit_60 = np.polyfit(np.log10(q_s_60), np.log10(m_melt_60), 1)

        plot_qs_00 = np.logspace(np.log10(min(q_s_00)), np.log10(max(q_s_00)),
                                 100, endpoint=True, base=10.)
        plot_qs_30 = np.logspace(np.log10(min(q_s_30)), np.log10(max(q_s_30)),
                                 100, endpoint=True, base=10.)
        plot_qs_45 = np.logspace(np.log10(7e3), np.log10(5e5),
                                 100, endpoint=True, base=10.)
        plot_qs_60 = np.logspace(np.log10(min(q_s_60)), np.log10(max(q_s_60)),
                                 100, endpoint=True, base=10.)

        plot_00, plot_30, plot_45, plot_60 = [], [], [], []
        for i in range(len(plot_qs_00)):
            plot_00.append(10. ** (fit_00[0] * np.log10(plot_qs_00[i]) + fit_00[1]))
            plot_30.append(10. ** (fit_30[0] * np.log10(plot_qs_30[i]) + fit_30[1]))
            plot_45.append(10. ** (fit_45[0] * np.log10(plot_qs_45[i]) + fit_45[1]))
            plot_60.append(10. ** (fit_60[0] * np.log10(plot_qs_60[i]) + fit_60[1]))

        # ax0.plot(plot_qs_00, plot_00, color=cols['H2'], linewidth=1.5)
        # ax0.plot(plot_qs_30, plot_30, color=cols['H2O'], linewidth=1.5)
        ax0.plot(plot_qs_45, plot_45, color=cols['CO2'], linewidth=1.5)
        # ax0.plot(plot_qs_60, plot_60, color=cols['N2'], linewidth=1.5)

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
    handles.append(lines.Line2D([0.], [0.], label='$1.1\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='+'))
    handles.append(lines.Line2D([0.], [0.], label='$1.5\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='o'))
    handles.append(lines.Line2D([0.], [0.], label='$2.0\,v_\mathrm{esc}$',
                                color='k', linestyle='', marker='s'))

    ax0.legend(handles=handles, loc='upper left', fontsize=7, ncol=2,
               columnspacing=0.4)

    plt.savefig(dir_path + '/figures/figure_2.pdf', dpi=200)
    # plt.show()