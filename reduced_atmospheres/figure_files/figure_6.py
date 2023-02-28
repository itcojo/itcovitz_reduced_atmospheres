import csv
import sys

import h5py
import matplotlib
import matplotlib.gridspec as gridspec
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


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def plot_figure_6() -> None:
    """Plot the post-impact state of the system.

    Show the composition of the atmosphere, the oxygen fugacity of the impact-generated melt phase, and the quantity of metallic iron left.

    """
    # range of impact masses [kg]
    impactor_masses = np.logspace(np.log10(2.00e21), np.log10(2.44e22), 30, base=10., endpoint=True)

    # calculate post-impact atmospheres
    h2, h2o, co2, n2, co, ch4, pressures = [], [], [], [], [], [], []
    h2_i, h2o_i, co2_i, n2_i, pressures_i = [], [], [], [], []
    fo2_atmos, fo2_bas, fo2_per, fe_bas, fe_per = [], [], [], [], []

    # successfully converged impactor masses
    masses_per, masses_bas = [], []

    # convert from moles to column density
    fac = 1e-4 / (4. * np.pi * gC.r_earth ** 2.)

    for m_imp in impactor_masses:
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # File Name
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        var_string = f"{m_imp:.2e}"
        var_string = var_string.replace('.', '_')
        var_string = var_string.replace('+', '')

        # read data - peridotite
        file = dir_path + '/output/m_imps/peridotite_3A_'
        with h5py.File(file + var_string + '.hdf5', 'r') as f:
            temp = f['temp'][()]

            h2.append(np.array(list(f['atmos/h2']))[0] * fac)
            h2o.append(np.array(list(f['atmos/h2o']))[0] * fac)
            co2.append(np.array(list(f['atmos/co2']))[0] * fac)
            n2.append(np.array(list(f['atmos/n2']))[0] * fac)
            co.append(np.array(list(f['atmos/co']))[0] * fac)
            ch4.append(np.array(list(f['atmos/ch4']))[0] * fac)
            pressures.append(np.array(list(f['atmos/p_tot']))[0])

            h2o_i.append(np.array(list(f['initial/values']))[0] * fac)
            h2_i.append(np.array(list(f['initial/values']))[1] * fac)
            n2_i.append(np.array(list(f['initial/values']))[2] * fac)
            co2_i.append(np.array(list(f['initial/values']))[3] * fac)
            pressures_i.append(f['initial/pressure'][()])

            fo2_atmos.append(np.array(list(f['atmos/fo2']))[0])
            fo2_per.append(np.array(list(f['melt/fo2']))[0])
            fe_per.append(np.array(list(f['metal/fe']))[0])
            masses_per.append(m_imp)

        # read data - basalt
        file = dir_path + '/output/m_imps/basalt_3A_'
        with h5py.File(file + var_string + '.hdf5', 'r') as f:
            fo2_bas.append(np.array(list(f['melt/fo2']))[0])
            fe_bas.append(np.array(list(f['metal/fe']))[0])
            masses_bas.append(m_imp)

    # partial pressures
    p_h2, p_h2o, p_co2, p_n2, p_co, p_ch4, p_tot = [], [], [], [], [], [], []
    p_h2_i, p_h2o_i, p_co2_i, p_n2_i, p_tot_i = [], [], [], [], []
    for j in range(len(h2)):
        n_tot = h2[j] + h2o[j] + co2[j] + n2[j] + co[j] + ch4[j]
        n_tot_i = h2_i[j] + h2o_i[j] + co2_i[j] + n2_i[j]

        p_h2.append(1e-5 * pressures[j] * h2[j] / n_tot)
        p_h2o.append(1e-5 * pressures[j] * h2o[j] / n_tot)
        p_co2.append(1e-5 * pressures[j] * co2[j] / n_tot)
        p_n2.append(1e-5 * pressures[j] * n2[j] / n_tot)
        p_co.append(1e-5 * pressures[j] * co[j] / n_tot)
        p_ch4.append(1e-5 * pressures[j] * ch4[j] / n_tot)
        p_tot.append(1e-5 * pressures[j])

        p_h2_i.append(1e-5 * pressures_i[j] * h2_i[j] / n_tot_i)
        p_h2o_i.append(1e-5 * pressures_i[j] * h2o_i[j] / n_tot_i)
        p_co2_i.append(1e-5 * pressures_i[j] * co2_i[j] / n_tot_i)
        p_n2_i.append(1e-5 * pressures_i[j] * n2_i[j] / n_tot_i)
        p_tot_i.append(1e-5 * pressures_i[j])

    # FMQ buffer for various P & T
    fmq_line = np.zeros(len(masses_per))
    for k in range(len(fmq_line)):
        fmq_line[k] = eq_melt.fo2_fmq(temp, pressures[k])

    # IW buffer for various P & T
    iw_line, iw_line_2 = np.zeros(len(masses_per)), np.zeros(len(masses_per))
    for kk in range(len(iw_line)):
        iw_line[kk] = eq_melt.fo2_iw(temp, pressures[kk])
        iw_line_2[kk] = eq_melt.fo2_iw(temp, pressures[kk]) - 2.

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # plot
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.figure(figsize=(6.5, 4.5), dpi=local_dpi)
    matplotlib.rcParams.update({'font.size': 9})
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
    margins = 0.05

    gs = gridspec.GridSpec(5, 5)
    plt.subplots_adjust(top=0.88, left=0.1, right=0.98, bottom=0.1,
                        hspace=0.04, wspace=1.)

    with sns.axes_style(plot_params):
        ax0 = fig.add_subplot(gs[:3, :3])
        ax0.minorticks_on()
        ax0.margins(margins)

        ax1 = fig.add_subplot(gs[3:, :3])
        ax1.minorticks_on()
        ax1.margins(margins)

        ax2 = fig.add_subplot(gs[:3, 3:])
        ax2.minorticks_on()
        ax2.margins(margins)

        ax3 = fig.add_subplot(gs[3:, 3:])
        ax3.minorticks_on()
        ax3.margins(margins)

    # mass limits
    masses = [mass * 1e-22 for mass in masses_per]

    # mass ticks
    xticks = [2e21, 5e21, 1e22, 2e22]
    xticks = [X * 1e-22 for X in xticks]

    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_xlim([0.17, 3.2])
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='both', direction='in')

    ax0.text(x=3., y=4e4, s='(a)', ha='right', va='top')
    ax1.text(x=3., y=625, s='(b)', ha='right', va='top')
    ax2.text(x=0.215, y=-3.1, s='(c)', va='top', ha='center')
    ax3.text(x=0.215, y=8e22, s='(d)', ha='center')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # standard value impactor mass
    ax0.axvline(x=2., color='grey', linestyle='--', linewidth=1.3)
    ax0.text(x=2., y=7e1, s='standard', fontsize=7, color='grey',
             rotation=270)

    ax0.plot(masses, ch4, color=cols['CH4'], marker='o', markersize=3)

    ax0.plot(masses, n2_i, color=cols['N2'], linestyle='', marker='x', markersize=3)
    ax0.plot(masses, n2, color=cols['N2'], marker='o', markersize=3)

    ax0.plot(masses, co2_i, color=cols['CO2'], linestyle='', marker='x', markersize=3)
    ax0.plot(masses, co2, color=cols['CO2'], marker='o', markersize=3)

    ax0.plot(masses, co, color=cols['CO'], marker='o', markersize=3)

    ax0.plot(masses, h2o_i, color=cols['H2O'], linestyle='', marker='x', markersize=3)
    ax0.plot(masses, h2o, color=cols['H2O'], marker='o', markersize=3)

    ax0.plot(masses, h2_i, color=cols['H2'], linestyle='', marker='x', markersize=3)
    ax0.plot(masses, h2, color=cols['H2'], marker='o', markersize=3)

    ax0.set_xscale('log')
    ax0.set_xticks(xticks)
    ax0.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax0.set_ylabel('Column Density /moles cm$^{-2}$')
    ax0.set_ylim([15., 5e4])
    ax0.set_yscale('log')

    handles = []
    handles.append(lines.Line2D([0.], [0.], label='H$_2$O', color=cols['H2O'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='H$_2$', color=cols['H2'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='CO$_2$', color=cols['CO2'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='N$_2$', color=cols['N2'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='CO', color=cols['CO'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='CH4', color=cols['CH4'],
                                linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='No Atmos LTE', color='grey',
                                linestyle='', marker='x', markersize=3))
    handles.append(lines.Line2D([0.], [0.], label='Atmos LTE', color='grey',
                                linestyle='-', marker='o'))

    ax0.legend(handles=handles, loc='lower left', ncol=4, fontsize=8,
               bbox_to_anchor=(-0.01, 1.01))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # standard value impactor mass
    ax1.axvline(x=2., color='grey', linestyle='--', linewidth=1.3)

    # ax1.plot(masses, p_ch4, color=cols['CH4'])
    # ax1.plot(masses, p_n2, color=cols['N2'])
    ax1.plot(masses, p_co2, color=cols['CO2'], marker='o', markersize=3)
    # ax1.plot(masses, p_co, color=cols['CO'])
    ax1.plot(masses, p_h2o, color=cols['H2O'], marker='o', markersize=3)
    ax1.plot(masses, p_h2, color=cols['H2'], marker='o', markersize=3)
    ax1.plot(masses, p_tot, color='grey')

    ax1.set_xlabel('Impactor Mass /10$^{22}$kg')
    ax1.set_xscale('log')
    ax1.set_xticks(xticks)
    ax1.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax1.set_ylabel('Partial Pressure /bar')
    ax1.set_ylim([-50., 650.])

    ax1.text(x=1.9, y=610, s='total pressure', fontsize=7, color='grey',
             rotation=-7, ha='right', va='top')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # standard value impactor mass
    ax2.axvline(x=2., color='grey', linestyle='--', linewidth=1.3)

    # atmosphere fO2
    ax2.plot(masses, fo2_atmos, color=cols['NH3'], linewidth=1.1)
    ax2.text(x=1.75, y=-6.6, s='Atmosphere', color=cols['NH3'],
             fontsize=7, rotation=-25, ha='right', va='bottom')

    # melt phase fO2
    ax2.plot(masses, fo2_bas, color=cols['CO2'], linewidth=1.1)
    ax2.text(x=0.208, y=-8.9, s='Basalt', color=cols['CO2'],
             fontsize=7, ha='left', va='bottom')

    ax2.plot(masses, fo2_per, color=cols['CO2'], linewidth=1.1)
    ax2.text(x=0.2, y=-10.3, s='Peridotite', color=cols['CO2'],
             fontsize=7, ha='left', va='top')

    # FMQ comparison line
    ax2.plot(masses, fmq_line, color='grey', linestyle=':')
    ax2.text(x=1.8, y=fmq_line[-1], s='\u0394FMQ=0', color='grey',
             fontsize=8, ha='right', va='bottom')

    # IW comparison line
    ax2.plot(masses, iw_line, color='grey', linestyle=':')
    ax2.text(x=masses[0], y=iw_line[0], s='\u0394IW=0', color='grey',
             fontsize=8, ha='left', va='bottom')

    # IW-2 comparison line
    ax2.plot(masses, iw_line_2, color='grey', linestyle=':')
    ax2.text(x=1.8, y=iw_line_2[0], s='\u0394IW=-2', color='grey',
             fontsize=8, ha='right', va='bottom')

    ax2.set_xscale('log')
    ax2.set_xticks(xticks)
    ax2.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.tick_params(axis="x", bottom=True, top=True, labeltop=True,
                    labelbottom=False)

    ax2.set_ylabel('$\log_{10}$(fO$_2$)')
    ax2.set_ylim([-10.85, -2.8])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # standard value impactor mass
    ax3.axvline(x=2., color='grey', linestyle='--', linewidth=1.3)

    # iron in melt phase
    idxs = [i for i, e in enumerate(fe_bas) if e != 0]
    ax3.plot([masses[i] for i in idxs], [fe_bas[i] for i in idxs],
             color='k', linewidth=1.)
    ax3.text(x=0.7, y=4e21, s='Basalt', color='k',
             fontsize=7, rotation=18, ha='left', va='bottom')

    ax3.plot(masses, fe_per, color='k', linewidth=1.)
    ax3.text(x=0.205, y=1e22, s='Peridotite', color='k',
             fontsize=7, rotation=18, ha='left', va='bottom')

    ax3.set_xlabel('Impactor Mass /10$^{22}$kg')
    ax3.set_xscale('log')
    ax3.set_xticks(xticks)
    ax3.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax3.set_ylabel('Fe /moles')
    ax3.set_ylim([6e20, 1.5e23])
    ax3.set_yscale('log')

    plt.savefig(f"{dir_path}/figures/figure_6.pdf", dpi=200)
    # plt.show()
