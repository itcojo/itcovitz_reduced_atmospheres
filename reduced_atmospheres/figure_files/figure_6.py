import sys

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


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def flatten(LIST):
    """
    Flattens list of lists into single list.

    Parameters
    ----------
    LIST : list
        List of lists to be flattened.

    Returns
    -------

    """
    return [item for sublist in LIST for item in sublist]


def move_axes_across(ax0, ax1):
    """
    ax0 should be to the left of ax1.
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


def move_axes_up(ax0, ax1):
    """
    ax0 should be above ax1.
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


def plot_figure_6():
    """
    Parameters
    ----------

    Returns
    -------

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

    p_tot, iw_vals = [], []

    fac = 1e-4 / (4. * np.pi * gC.r_earth**2.)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Grab Data
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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
                dir_mass = dir_path + '/output/m_imps/'
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

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Set Up Plot Layout
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Axes Parameters - Specific
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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
    axs['1B_atm'].set_yscale('log')
    axs['1B_fo2'].set_ylim([-14.9, -7.3])

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
    handles.append(lines.Line2D([0.], [0.], color=cols['H2'], label='H$_2$',
                                linestyle='-', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['NH3'], linestyle='-',
                                label='fO$_2$ Atmos', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='-',
                                label='fO$_2$ Melt', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='',
                                marker='',  label='', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle=':',
                                label='fiducial', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='--',
                                label='post-impact', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='-',
                                label='post-equilibration', linewidth=2))
    # handles.append(lines.Line2D([0.], [0.], color=cols['CO2'], linestyle='',
    #                             marker='x', markersize=4, label='no H2O partitioning'))

    axs['2_atm'].legend(handles=handles, loc='upper left', ncol=1, fontsize=8,
                        bbox_to_anchor=(1.03, 1.01))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Model 1A Comparison Lines
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Model 1B
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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
    axs['1B_fo2'].text(masses[0], 1.01 * fo2_melt_bas[1][0][-1], s='B',
                       color=cols['CO2'], va='top')
    axs['1B_fo2'].text(masses[0], 1.01 * fo2_melt_per[1][0][-1], s='P',
                       color=cols['CO2'], va='top')
    axs['1B_fo2'].text(1.8, 1.01 * fo2_melt_bas[1][0][-1], s='ΔFMQ = 0',
                       fontsize=7, color=cols['CO2'], va='top', ha='right')
    axs['1B_fo2'].text(masses[0], iw_vals[0] - 2.1, s='ΔIW = -2',
                       fontsize=7, color='grey', va='top', ha='left')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Model 2
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

    # labels
    axs['2_fo2'].text(x=2.45, y=fo2_melt_bas[2][-1][0], s='B',
                      color=cols['CO2'], ha='left', va='center')
    axs['2_fo2'].text(x=2.45, y=fo2_melt_per[2][-1][0], s='P',
                      color=cols['CO2'], ha='left', va='center')

    axs['2_fo2'].text(x=2.45, y=fo2_melt_bas[2][-1][-1], s='B',
                      color=cols['CO2'], ha='left', va='center')
    axs['2_fo2'].text(x=2.45, y=fo2_melt_per[2][-1][-1], s='P',
                      color=cols['CO2'], ha='left', va='center')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Model 3A
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

    # labels
    axs['3A_fo2'].text(x=0.2, y=fo2_melt_bas[3][0][0], s='B',
                       color=cols['CO2'], ha='left', va='bottom')
    axs['3A_fo2'].text(x=0.2, y=1.01 * fo2_melt_per[3][0][0], s='P',
                       color=cols['CO2'], ha='left', va='top')

    axs['3A_fo2'].text(x=2.6, y=fo2_melt_bas[3][-1][-1], s='B',
                       color=cols['CO2'], ha='left', va='center')
    axs['3A_fo2'].text(x=2.6, y=1.01 * fo2_melt_per[3][-1][-1], s='P',
                       color=cols['CO2'], ha='left', va='center')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Model 3B
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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
    axs['2_atm'].text(x=4.1, y=5e1, s='P - peridotitic melt')
    axs['2_atm'].text(x=4.1, y=1e1, s='B - basaltic melt')

    axs['2_fo2'].text(x=4.1, y=-9.7, s='(Fiducial)')
    axs['2_fo2'].text(x=4.1, y=-10.4, s='  \u2A2F melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-11.1, s='  \u2A2F iron distribution')

    axs['2_fo2'].text(x=4.1, y=-12.4, s='(Model 1B)')
    axs['2_fo2'].text(x=4.1, y=-13.1, s='  \u2A2F melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-13.8, s='  \u2713 iron distribution')

    axs['2_fo2'].text(x=4.1, y=-15.1, s='(Model 2)')
    axs['2_fo2'].text(x=4.1, y=-15.8, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-16.5, s='  \u2A2F iron distribution')

    axs['2_fo2'].text(x=4.1, y=-17.8, s='(Model 3A)')
    axs['2_fo2'].text(x=4.1, y=-18.5, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-19.2, s='  \u2713 iron distribution')
    axs['2_fo2'].text(x=4.1, y=-19.9, s='  [interior iron in melt]')

    axs['2_fo2'].text(x=4.1, y=-21.2, s='(Model 3B)')
    axs['2_fo2'].text(x=4.1, y=-21.9, s='  \u2713 melt-atmosphere')
    axs['2_fo2'].text(x=4.1, y=-22.6, s='  \u2713 iron distribution')
    axs['2_fo2'].text(x=4.1, y=-23.3, s='  [interior iron in other]')

    plt.savefig(dir_path + '/figures/figure_6.pdf', dpi=200)
    # plt.show()
