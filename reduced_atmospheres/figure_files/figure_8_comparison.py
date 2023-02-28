import h5py
import matplotli.axes as plt_axes
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
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
def flatten(a_list: list) -> list:
    """Flattens list of lists into single list.

    Args: 
        a_list (list): List of lists to be flattened.

    Returns:
        long_list : Flattened list.

    """
    return [item for sublist in a_list for item in sublist]


def move_axes_across(ax0 : plt_axes, ax1: plt_axes) -> None:
    """Adjust horizontal axis pair so that they touch each other.
    
    Args: 
        ax0 (plt_axes): Left axis in horizontal pair.
        ax1 (plt_axes): Right axis in horizontal pair.
        
    """
    # location of lower left and upper right corners of ax0
    ax0_pos = ax0.get_position()
    ax0_points = ax0_pos.get_points()
    # location of lower left and upper right corners of ax1
    ax1_pos = ax1.get_position()
    ax1_points = ax1_pos.get_points()

    # horizontal difference between axes
    hor_diff = ax0_points[1][0] - ax1_points[0][0]

    # move the left corner of ax1 leftwards to the centre of ax0
    ax1_points[0][0] += 0.5 * hor_diff

    # move the right side of ax1 left by the same amount, so that it
    # keeps the same shape
    ax1_points[1][0] += 0.5 * hor_diff

    ax1_pos.set_points(ax1_points)
    ax1.set_position(ax1_pos)


def move_axes_up(ax0 : plt_axes, ax1: plt_axes) -> None:
    """Adjust vertical axis pair so that they touch each other.
    
    Args: 
        ax0 (plt_axes): Upper axis in horizontal pair.
        ax1 (plt_axes): Lower axis in horizontal pair.
        
    """
    # location of lower left and upper right corners of ax0
    ax0_pos = ax0.get_position()
    ax0_points = ax0_pos.get_points()
    # location of lower left and upper right corners of ax1
    ax1_pos = ax1.get_position()
    ax1_points = ax1_pos.get_points()

    # vertical difference between axes
    vert_diff = ax0_points[0][1] - ax1_points[1][1]

    # move the upper corner of ax1 upwards to the lower corner of ax0
    ax1_points[1][1] += vert_diff

    # move the lower corner of ax1 upwards by the same amount, so that it
    # keeps the same shape
    ax1_points[0][1] += vert_diff

    ax1_pos.set_points(ax1_points)
    ax1.set_position(ax1_pos)


def plot_figure_8() -> None:
    """Plot hydrogen abundances and oxygen fugacities under the different model scenarios before and after melt-atmosphere equilibration.

    """
    masses = np.logspace(np.log10(2.00e21), np.log10(2.44e22), 30, base=10.,
                         endpoint=True)

    labels = ['1A_', '1B_', '2_', '3A_', '3B_']
    plot_mass_bas = [[] for _ in range(len(labels))]
    plot_mass_per = [[] for _ in range(len(labels))]
    h2_bas = [[] for _ in range(len(labels))]
    h2_per = [[] for _ in range(len(labels))]
    fo2_atm_bas = [[] for _ in range(len(labels))]
    fo2_atm_per = [[] for _ in range(len(labels))]
    fo2_melt_bas = [[] for _ in range(len(labels))]
    fo2_melt_per = [[] for _ in range(len(labels))]

    h2_bas_fixed = [[] for _ in range(len(labels))]
    h2_per_fixed = [[] for _ in range(len(labels))]
    fo2_atm_bas_fixed = [[] for _ in range(len(labels))]
    fo2_atm_per_fixed = [[] for _ in range(len(labels))]
    fo2_melt_bas_fixed = [[] for _ in range(len(labels))]
    fo2_melt_per_fixed = [[] for _ in range(len(labels))]

    p_tot, iw_vals = [], []

    fac = 1e-4 / (4. * np.pi * gC.r_earth**2.)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Grab Data (this study)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    for idx in range(len(labels)):
        for m_imp in masses:
            try:
                # file name
                var_str = "%.2e" % m_imp
                var_str = var_str.replace('.', '_')
                var_str = var_str.replace('+', '')
                dir_mass = dir_path + '/output/pre_erratum/m_imps/'
                file = dir_mass + 'basalt_' + labels[idx] + var_str

                # read data - with dissolution
                with h5py.File(file + '.hdf5', 'r') as f:
                    # unpack trackers
                    h2_bas[idx].append(np.array(list(f['atmos/h2'])))
                    fo2_atm_bas[idx].append(np.array(list(f['atmos/fo2'])))
                    fo2_melt_bas[idx].append(np.array(list(f['melt/fo2'])))

                    # track which impactor masses successfully converged
                    plot_mass_bas[idx].append(m_imp)
            except:
                pass

    for idx in range(len(labels)):
        for m_imp in masses:
            try:
                # file name
                var_str = "%.2e" % m_imp
                var_str = var_str.replace('.', '_')
                var_str = var_str.replace('+', '')
                dir_mass = dir_path + '/output/pre_erratum/m_imps/'
                file = dir_mass + 'peridotite_' + labels[idx] + var_str

                # read data - with dissolution
                with h5py.File(file + '.hdf5', 'r') as f:
                    # unpack trackers
                    h2_per[idx].append(np.array(list(f['atmos/h2'])))
                    fo2_atm_per[idx].append(np.array(list(f['atmos/fo2'])))
                    fo2_melt_per[idx].append(np.array(list(f['melt/fo2'])))

                    # track which impactor masses successfully converged
                    plot_mass_per[idx].append(m_imp)

                    if labels[idx] == '1A_':
                        p_tot.append(list(f['atmos/p_tot'])[-1])
                        iw_vals.append(
                            float(eq_melt.fo2_iw(f['temp'][()], p_tot[-1])))
            except:
                pass

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Grab Data (this study - with erratum)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    for idx in range(len(labels)):
        for m_imp in masses:
            try:
                # file name
                var_str = "%.2e" % m_imp
                var_str = var_str.replace('.', '_')
                var_str = var_str.replace('+', '')
                dir_mass = dir_path + '/output/m_imps/'
                file = dir_mass + 'basalt_' + labels[idx] + var_str

                # read data - with dissolution
                with h5py.File(file + '.hdf5', 'r') as f:
                    # unpack trackers
                    h2_bas_fixed[idx].append(np.array(list(f['atmos/h2'])))
                    fo2_atm_bas_fixed[idx].append(np.array(list(f['atmos/fo2'])))
                    fo2_melt_bas_fixed[idx].append(np.array(list(f['melt/fo2'])))

            except:
                pass

    for idx in range(len(labels)):
        for m_imp in masses:
            try:
                # file name
                var_str = "%.2e" % m_imp
                var_str = var_str.replace('.', '_')
                var_str = var_str.replace('+', '')
                dir_mass = dir_path + '/output/m_imps/'
                file = dir_mass + 'peridotite_' + labels[idx] + var_str

                # read data - with dissolution
                with h5py.File(file + '.hdf5', 'r') as f:
                    # unpack trackers
                    h2_per_fixed[idx].append(np.array(list(f['atmos/h2'])))
                    fo2_atm_per_fixed[idx].append(np.array(list(f['atmos/fo2'])))
                    fo2_melt_per_fixed[idx].append(np.array(list(f['melt/fo2'])))

            except:
                pass

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Grab Data (Zahnle+, 2020)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    Z_dir = dir_path + '/reduced_atmospheres/data/Zahnle_2020_pCO2=100/'

    Z_masses = [24.2, 24.4, 24.6, 24.8, 25, 25.2]

    # temperatures
    z_temps = np.zeros(len(Z_masses))

    # column densities
    z_n_h2, z_n_h2o = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_n_co2, z_n_n2 = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_n_co, z_n_ch4 = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_n_nh3, z_n_tot = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))

    # partial pressures
    z_p_h2, z_p_h2o = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_p_co2, z_p_n2 = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_p_co, z_p_ch4 = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))
    z_p_nh3, z_p_tot = np.zeros(len(Z_masses)), np.zeros(len(Z_masses))

    for i in range(len(Z_masses)):
        # impactor mass [g, sort of]
        M = Z_masses[i]

        # which row to start at? (i.e., temperature | row 51 = 1500 K)
        n_start = 51

        # column density file
        file_col = 'Mi=' + str(M) + '/IW_Mi=' + str(M) + \
                   '_pCO2=100_pH2O=500_nobuffer.outcolumns'
        with open(Z_dir + file_col, 'r') as f:
            for _ in range(8 + n_start):  # move to top row of results
                next(f)
            for row in f:
                # get rid of white spaces (i.e., separate columns)
                split_row = row.split()

                z_temps[i] = split_row[8]
                z_n_tot[i] = split_row[9]
                z_n_h2o[i] = split_row[10]
                z_n_h2[i] = split_row[11]
                z_n_co[i] = split_row[12]
                z_n_co2[i] = split_row[13]
                z_n_ch4[i] = split_row[14]
                z_n_n2[i] = split_row[15]
                z_n_nh3[i] = split_row[16]

                break

        # partial pressures
        file_pre = 'Mi=' + str(M) + '/IW_Mi=' + str(M) + \
                   '_pCO2=100_pH2O=500_nobuffer.out'
        with open(Z_dir + file_pre, 'r') as f:
            for _ in range(8 + n_start):  # move to top row of results
                next(f)
            for row in f:
                # get rid of white spaces (i.e., separate columns)
                split_row = row.split()

                z_temps[i] = split_row[8]
                z_p_tot[i] = split_row[9]
                z_p_h2o[i] = split_row[10]
                z_p_h2[i] = split_row[11]
                z_p_co[i] = split_row[12]
                z_p_co2[i] = split_row[13]
                z_p_ch4[i] = split_row[14]
                z_p_n2[i] = split_row[15]
                z_p_nh3[i] = split_row[16]

                break

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # scaled impactor masses for x-axis
    masses_z = np.zeros(len(Z_masses))
    for j in range(len(Z_masses)):
        string_mass = str(Z_masses[j])
        string_mass = string_mass.split(sep='.')
        if len(string_mass) == 1:
            M = 10. ** (float(string_mass[0]))
        elif len(string_mass) == 2:
            M = float(string_mass[1]) * 10. ** (float(string_mass[0]))

        masses_z[j] = M * 1e-25  # [10^22 kg]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Set Up Plot Layout
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # set up plot layout
    mosaic = [['1B_atm', '2_atm'],
              ['1B_fo2', '2_fo2'],
              ['3A_atm', '3B_atm'],
              ['3A_fo2', '3B_fo2']]
    flat_labels = flatten(mosaic)

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(6.5, 8.5), dpi=local_dpi)
    plt.subplots_adjust(left=0.11, right=0.7, bottom=0.05, top=0.97,
                        hspace=0.15, wspace=0.02)
    plt.rcParams.update({'font.size': 10})

    # move axes up
    move_axes_up(axs['1B_atm'], axs['1B_fo2'])
    move_axes_up(axs['2_atm'], axs['2_fo2'])

    move_axes_up(axs['3A_atm'], axs['3A_fo2'])
    move_axes_up(axs['3B_atm'], axs['3B_fo2'])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Axes Parameters - Specific
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # X-AXIS
    # --- --- --- --- --- ---
    xticks = [2e21, 5e21, 1e22, 2e22]
    xticks = [X * 1e-22 for X in xticks]

    masses = [M * 1e-22 for M in masses]

    # x-axis properties, minor ticks, and margins
    for ax in flat_labels:
        axs[ax].set_xlim([0.17, 3.3])
        axs[ax].set_xscale('log')
        axs[ax].set_xticks(xticks)
        axs[ax].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        axs[ax].tick_params(axis='both', which='both', direction='in')

        axs[ax].axvline(x=2., color=cols['CO2'], linewidth=1.)

        axs[ax].minorticks_on()
        axs[ax].margins(0.05)

    # no x-ticks and labels on atmosphere subplots
    for ax in ['1B_atm', '2_atm', '3A_atm', '3B_atm']:
        axs[ax].tick_params(axis='x', which='both', top=True, bottom=True,
                            labeltop=False, labelbottom=False)
    # x-ticks and labels on fugacity subplots
    for ax in ['1B_fo2', '2_fo2', '3A_fo2', '3B_fo2']:
        axs[ax].tick_params(axis='x', which='both', top=True, bottom=True,
                            labeltop=False, labelbottom=True)
    # x-axis labels
    for ax in ['3A_fo2', '3B_fo2']:
        axs[ax].set_xlabel('Impactor Mass /10$^{22}$ kg')

    # --- --- --- --- --- ---
    # Y-AXIS
    # --- --- --- --- --- ---
    # y-ticks and labels on left column
    for ax in ['1B_atm', '1B_fo2', '3A_atm', '3A_fo2']:
        axs[ax].tick_params(axis='y', which='both', left=True, right=True,
                            labelleft=True, labelright=False)
    # y-ticks and labels on right column
    for ax in ['2_atm', '2_fo2']:
        axs[ax].tick_params(axis='y', which='both', left=True, right=True,
                            labelleft=False, labelright=False)
    for ax in ['3B_atm', '3B_fo2']:
        axs[ax].tick_params(axis='y', which='both', left=True, right=True,
                            labelleft=False, labelright=False)
    # y-axis labels
    for ax in ['1B_atm', '3A_atm']:
        axs[ax].set_ylabel('Column Density \n/moles cm$^{-2}$', fontsize=9)
    for ax in ['1B_fo2', '3A_fo2']:
        axs[ax].set_ylabel('$\log_{10}($fO$_2)$', fontsize=9)

    # y-axis limits and scales
    axs['1B_atm'].set_ylim([2e1, 6e4])
    # axs['1B_atm'].set_ylim([-1000, 3e4])
    axs['1B_atm'].set_yscale('log')
    axs['1B_fo2'].set_ylim([-10.85, -2.8])

    for ax in ['2_atm', '3A_atm', '3B_atm']:
        axs[ax].set_ylim(axs['1B_atm'].get_ylim())
        axs[ax].set_yscale('log')
    for ax in ['2_fo2', '3A_fo2', '3B_fo2']:
        axs[ax].set_ylim(axs['1B_fo2'].get_ylim())

    # --- --- --- --- --- ---
    # COLUMN TITLES
    # --- --- --- --- --- ---
    titles = ['Model 1B', 'Model 2', 'Model 3A', 'Model 3B']
    titled_axes = ['1B_atm', '2_atm', '3A_atm', '3B_atm']
    for idx in range(len(titled_axes)):
        axs[titled_axes[idx]].text(x=0.01, y=1.04, s=titles[idx], ha='left',
                                   transform=axs[titled_axes[idx]].transAxes,
                                   fontsize=9)

    # --- --- --- --- --- ---
    # STANDARD IMPACTOR MASS
    # --- --- --- --- --- ---
    for ax in flat_labels:
        axs[ax].axvline(x=2., color='grey', linewidth=1.)

    # label for Standard impactor mass line
    axs['1B_atm'].text(x=2., y=1e3, s='standard', fontsize=8, color='grey',
                      rotation=270, ha='left', va='top')

    # --- --- --- --- --- ---
    # LEGEND
    # --- --- --- --- --- ---
    handles = []
    handles.append(lines.Line2D([0.], [0.], color=cols['H2'], label='H$_2$ original',
                                linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['NH3'], linestyle='-',
                                label='fO$_2$ Atmos', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='-',
                                label='fO$_2$ Melt', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['N2'], label='Erratum',
                                linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='',
                                marker='', label='', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle=':',
                                label='fiducial', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='--',
                                label='post-impact', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='-',
                                label='post-equilibration', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='',
                                label='Zahnle+ (2020)', marker='s'))

    axs['2_atm'].legend(handles=handles, loc='upper left', ncol=1, fontsize=8,
                        bbox_to_anchor=(1.03, 1.1))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Model 1A Comparison Lines
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # masses successfully calculated for Model 1A
    masses = [M * 1e-22 for M in plot_mass_bas[0]]

    for ax in ['1B_atm', '2_atm', '3A_atm', '3B_atm']:
        # hydrogen
        axs[ax].plot(masses, [h2_bas[0][i][-1] * fac for i in range(len(masses))],
                     color=cols['H2'], linestyle=':', linewidth=1)

    for ax in ['1B_fo2', '2_fo2', '3A_fo2', '3B_fo2']:
        # fO2 atmos
        axs[ax].plot(masses, [fo2_atm_bas[0][i][-1] for i in range(len(masses))],
                     color=cols['NH3'], linestyle=':', linewidth=1)
        # fO2 melt
        axs[ax].plot(masses, [fo2_melt_bas[0][i][-1] for i in range(len(masses))],
                     color=cols['CO2'], linestyle=':', linewidth=1)
        axs[ax].plot(masses, [fo2_melt_per[0][i][-1] for i in range(len(masses))],
                     color=cols['CO2'], linestyle=':', linewidth=1)

    for ax in ['1B_fo2', '2_fo2', '3B_fo2']:
        axs[ax].plot(masses, [item - 2. for item in iw_vals],
                     color='grey', linestyle='--', linewidth=1)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Zahnle+ (2020) Data
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    for ax in ['1B_atm', '2_atm', '3A_atm', '3B_atm']:
        axs[ax].plot(masses_z, z_n_h2, marker='s', ms=4, linestyle='',
                     color=cols['H2'])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Model 1B
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # masses successfully calculated for Model 1B
    masses = [M * 1e-22 for M in plot_mass_bas[1]]

    # final H2
    axs['1B_atm'].plot(masses, [h2_bas[1][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)

    # --- --- --- --- --- ---
    # final atmosphere fO2
    axs['1B_fo2'].plot(masses, [fo2_atm_bas[1][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)

    # final melt phase fO2
    axs['1B_fo2'].plot(masses, [fo2_melt_bas[1][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)
    axs['1B_fo2'].plot(masses, [fo2_melt_per[1][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)

    # labels
    # print("ΔFMQ = 0 has raw fO2 of %.2f" % fo2_melt_bas[1][0][-1])
    # print("  IW = 0 has raw fO2 of %.2f" % iw_vals[0])
    # sys.exit()

    axs['1B_fo2'].text(masses[0], 1.01 * fo2_melt_bas[1][0][-1], s='B',
                       color=cols['CO2'], va='top')
    axs['1B_fo2'].text(masses[0], 1.01 * fo2_melt_per[1][0][-1], s='P',
                       color=cols['CO2'], va='top')
    axs['1B_fo2'].text(1.8, 1.01 * fo2_melt_bas[1][0][-1], s='ΔFMQ = 0',
                       fontsize=7, color='grey', va='top', ha='right')
    axs['1B_fo2'].text(masses[0], iw_vals[0] - 2.1, s='ΔIW = -2',
                       fontsize=7, color='grey', va='top', ha='left')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Model 2
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # masses successfully calculated for Model 2
    masses = [M * 1e-22 for M in plot_mass_bas[2]]

    # initial H2
    axs['2_atm'].plot(masses, [h2_bas[2][i][0] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='--', linewidth=1)

    # final H2
    axs['2_atm'].plot(masses, [h2_bas[2][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)
    axs['2_atm'].plot(masses, [h2_per[2][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)

     # final H2 (ERRATUM)
    axs['2_atm'].plot(masses, [h2_bas_fixed[2][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)
    axs['2_atm'].plot(masses, [h2_per_fixed[2][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)

    # labels
    axs['2_atm'].text(x=2.45, y=0.88 * h2_bas[2][-1][-1] * fac, s='B',
                      color=cols['H2'], ha='left', va='center')
    axs['2_atm'].text(x=2.45, y=1.17 * h2_per[2][-1][-1] * fac, s='P',
                      color=cols['H2'], ha='left', va='center')

    # --- --- --- --- --- ---
    # init atmosphere fO2
    axs['2_fo2'].plot(masses, [fo2_atm_bas[2][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)
    axs['2_fo2'].plot(masses, [fo2_atm_per[2][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)

    # init melt phase fO2
    axs['2_fo2'].plot(masses, [fo2_melt_bas[2][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)
    axs['2_fo2'].plot(masses, [fo2_melt_per[2][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)

    # final atmosphere fO2
    axs['2_fo2'].plot(masses, [fo2_atm_bas[2][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)
    axs['2_fo2'].plot(masses, [fo2_atm_per[2][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)

    # final melt phase fO2
    axs['2_fo2'].plot(masses, [fo2_melt_bas[2][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)
    axs['2_fo2'].plot(masses, [fo2_melt_per[2][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)

    # final melt phase fO2 (ERRATUM)
    axs['2_fo2'].plot(masses, [fo2_melt_bas_fixed[2][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1)
    axs['2_fo2'].plot(masses, [fo2_melt_per_fixed[2][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1)

    # labels
    axs['2_fo2'].text(x=2.45, y=fo2_melt_bas[2][-1][0], s='B',
                      color=cols['CO2'], ha='left', va='center')
    axs['2_fo2'].text(x=2.45, y=fo2_melt_per[2][-1][0], s='P',
                      color=cols['CO2'], ha='left', va='center')

    axs['2_fo2'].text(x=2.45, y=fo2_melt_bas[2][-1][-1], s='B',
                      color=cols['CO2'], ha='left', va='center')
    axs['2_fo2'].text(x=2.45, y=fo2_melt_per[2][-1][-1], s='P',
                      color=cols['CO2'], ha='left', va='center')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Model 3A
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # masses successfully calculated for Model 3A
    masses = [M * 1e-22 for M in plot_mass_bas[3]]

    # initial H2
    axs['3A_atm'].plot(masses, [h2_bas[3][i][0] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='--', linewidth=1)
    axs['3A_atm'].plot(masses, [h2_per[3][i][0] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='--', linewidth=1)

    # final H2
    axs['3A_atm'].plot(masses, [h2_bas[3][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)
    axs['3A_atm'].plot(masses, [h2_per[3][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)

    # final H2 (ERRATUM)
    axs['3A_atm'].plot(masses, [h2_bas_fixed[3][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)
    axs['3A_atm'].plot(masses, [h2_per_fixed[3][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)

    # labels
    axs['3A_atm'].text(x=2.6, y=0.80 * h2_bas[3][-1][-1] * fac, s='B',
                       color=cols['H2'], ha='left', va='center')
    axs['3A_atm'].text(x=2.6, y=1.10 * h2_per[3][-1][-1] * fac, s='P',
                       color=cols['H2'], ha='left', va='center')

    # --- --- --- --- --- ---
    # init atmosphere fO2
    axs['3A_fo2'].plot(masses, [fo2_atm_bas[3][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)
    axs['3A_fo2'].plot(masses, [fo2_atm_per[3][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)

    # init melt phase fO2
    axs['3A_fo2'].plot(masses, [fo2_melt_bas[3][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)
    axs['3A_fo2'].plot(masses, [fo2_melt_per[3][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)

    # # init melt phase fO2 (ERRATUM)
    # axs['3A_fo2'].plot(masses, [fo2_melt_bas_fixed[3][i][0] for i in range(len(masses))],
    #                    color=cols['N2'], linestyle='--', linewidth=1)
    # axs['3A_fo2'].plot(masses, [fo2_melt_per_fixed[3][i][0] for i in range(len(masses))],
    #                    color=cols['N2'], linestyle='--', linewidth=1)

    # final atmosphere fO2
    axs['3A_fo2'].plot(masses, [fo2_atm_bas[3][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)
    axs['3A_fo2'].plot(masses, [fo2_atm_per[3][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)

    # final melt phase fO2
    axs['3A_fo2'].plot(masses, [fo2_melt_bas[3][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)
    axs['3A_fo2'].plot(masses, [fo2_melt_per[3][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)

    # final melt phase fO2 (ERRATUM)
    axs['3A_fo2'].plot(masses, [fo2_melt_bas_fixed[3][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)
    axs['3A_fo2'].plot(masses, [fo2_melt_per_fixed[3][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)

    # labels
    axs['3A_fo2'].text(x=0.2, y=fo2_melt_bas[3][0][0], s='B',
                       color=cols['CO2'], ha='left', va='bottom')
    axs['3A_fo2'].text(x=0.2, y=1.01 * fo2_melt_per[3][0][0], s='P',
                       color=cols['CO2'], ha='left', va='top')

    axs['3A_fo2'].text(x=2.6, y=fo2_melt_bas[3][-1][-1], s='B',
                       color=cols['CO2'], ha='left', va='center')
    axs['3A_fo2'].text(x=2.6, y=1.01 * fo2_melt_per[3][-1][-1], s='P',
                       color=cols['CO2'], ha='left', va='center')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # Model 3B
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # masses successfully calculated for Model 3B
    masses = [M * 1e-22 for M in plot_mass_bas[4]]

    # initial H2
    # axs['3B_atm'].plot(masses, [h2_bas[4][i][0] * fac for i in range(len(masses))],
    #                    color=gC.custom_colors[0], linestyle='--', linewidth=1)
    axs['3B_atm'].plot(masses, [h2_per[4][i][0] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='--', linewidth=1)

    # final H2
    axs['3B_atm'].plot(masses, [h2_bas[4][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)
    axs['3B_atm'].plot(masses, [h2_per[4][i][-1] * fac for i in range(len(masses))],
                       color=cols['H2'], linestyle='-', linewidth=1)

    # final H2 (ERRATUM)
    axs['3B_atm'].plot(masses, [h2_bas_fixed[4][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)
    axs['3B_atm'].plot(masses, [h2_per_fixed[4][i][-1] * fac for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)

    # labels
    axs['3B_atm'].text(x=2.6, y=h2_bas[4][-1][-1] * fac, s='B',
                       color=cols['H2'], ha='left', va='center')
    axs['3B_atm'].text(x=2.6, y=h2_per[4][-1][-1] * fac, s='P',
                       color=cols['H2'], ha='left', va='center')

    # --- --- --- --- --- ---
    # init atmosphere fO2
    axs['3B_fo2'].plot(masses, [fo2_atm_bas[4][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)
    axs['3B_fo2'].plot(masses, [fo2_atm_per[4][i][0] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='--', linewidth=1)

    # init melt phase fO2
    axs['3B_fo2'].plot(masses, [fo2_melt_bas[4][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)
    axs['3B_fo2'].plot(masses, [fo2_melt_per[4][i][0] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='--', linewidth=1)

    # # init melt phase fO2 (ERRATUM)
    # axs['3B_fo2'].plot(masses, [fo2_melt_bas_fixed[4][i][0] for i in range(len(masses))],
    #                    color=cols['N2'], linestyle='--', linewidth=1)
    # axs['3B_fo2'].plot(masses, [fo2_melt_per_fixed[4][i][0] for i in range(len(masses))],
    #                    color=cols['N2'], linestyle='--', linewidth=1)

    # final atmosphere fO2
    axs['3B_fo2'].plot(masses, [fo2_atm_bas[4][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)
    axs['3B_fo2'].plot(masses, [fo2_atm_per[4][i][-1] for i in range(len(masses))],
                       color=cols['NH3'], linestyle='-', linewidth=1)

    # final melt phase fO2
    axs['3B_fo2'].plot(masses, [fo2_melt_bas[4][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)
    axs['3B_fo2'].plot(masses, [fo2_melt_per[4][i][-1] for i in range(len(masses))],
                       color=cols['CO2'], linestyle='-', linewidth=1)

    # final melt phase fO2 (ERRATUM)
    axs['3B_fo2'].plot(masses, [fo2_melt_bas_fixed[4][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)
    axs['3B_fo2'].plot(masses, [fo2_melt_per_fixed[4][i][-1] for i in range(len(masses))],
                       color=cols['N2'], linestyle='-', linewidth=1, alpha=0.5)

    # labels
    axs['3B_fo2'].text(x=2.6, y=fo2_melt_bas[4][-1][0], s='B',
                       color=cols['CO2'], ha='left', va='center')
    axs['3B_fo2'].text(x=0.2, y=1.01 * fo2_melt_per[4][0][0], s='P',
                       color=cols['CO2'], ha='left', va='top')

    axs['3B_fo2'].text(x=2.6, y=fo2_melt_bas[4][-1][-1], s='B',
                       color=cols['CO2'], ha='left', va='center')
    axs['3B_fo2'].text(x=2.6, y=1.01 * fo2_melt_per[4][-1][-1], s='P',
                       color=cols['CO2'], ha='left', va='center')

    # --- --- --- --- --- --- --- ---
    # text description of models
    axs['2_fo2'].text(x=4.1, y=-3.0, s='P - peridotitic melt')
    axs['2_fo2'].text(x=4.1, y=-4.0, s='B - basaltic melt')

    axs['2_fo2'].text(x=4.1, y=-5.7, s='(Fiducial)')
    axs['2_fo2'].text(x=4.1, y=-6.4, s='  \u2A2F melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-7.1, s='  \u2A2F iron distribution')

    axs['2_fo2'].text(x=4.1, y=-8.4, s='(Model 1B)')
    axs['2_fo2'].text(x=4.1, y=-9.1, s='  \u2A2F melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-9.8, s='  \u2713 iron distribution')

    axs['2_fo2'].text(x=4.1, y=-11.1, s='(Model 2)')
    axs['2_fo2'].text(x=4.1, y=-11.8, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-12.5, s='  \u2A2F iron distribution')

    axs['2_fo2'].text(x=4.1, y=-13.8, s='(Model 3A)')
    axs['2_fo2'].text(x=4.1, y=-14.5, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-15.2, s='  \u2713 iron distribution')
    axs['2_fo2'].text(x=4.1, y=-15.9, s='  [interior iron in melt]')

    axs['2_fo2'].text(x=4.1, y=-17.2, s='(Model 3B)')
    axs['2_fo2'].text(x=4.1, y=-17.9, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-18.6, s='  \u2713 iron distribution')
    axs['2_fo2'].text(x=4.1, y=-19.3, s='  [interior iron in other]')
    
    plt.savefig(f"{dir_path}/figures/figure_8_comparison.pdf", dpi=200)
    # plt.show()
