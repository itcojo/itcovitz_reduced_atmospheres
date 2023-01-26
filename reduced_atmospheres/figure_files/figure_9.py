from copy import deepcopy as dcop
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


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def plot_figure_9():
    """Plot Figure 9 from Itcovitz et al. (2022).

    Oxygen fugacity of the mantle before and after remixing with the impacted-effected melt phase.
    
    """
    # impactor masses`
    masses = np.logspace(np.log10(2e21), np.log10(2.44e22), 30, base=10., endpoint=True)

    # convert moles to column densities
    fac = 1e-4 / (4. * np.pi * gC.r_earth**2.)

    # standard values
    m_imp, v_imp, theta = 2e22, 20.7, 45.  # [kg], [km s-1], [deg]
    temp, p_tot, h2o_init, tol = 1900., 450. * 1e5, 0.05, 1e-5  # [K], [Pa], [wt%]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SCALING BETWEEN fO2 AND FERRIC-TO-IRON RATIOS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # PERIDOTITE
    # --- --- --- --- --- ---
    fe_max = 0.18
    iron_ratios_per, fo2_vals_per = np.arange(0.01, fe_max, 0.02), []

    # mass of the magma [kg]
    m_mag = eq_melt.impact_melt_mass(m_imp, v_imp, theta)

    for ratio in iron_ratios_per:
        # calculate moles from wt% prescription (includes adding H2O)
        n_melt = eq_melt.peridotite_comp_by_fe_ratio(m_mag, ratio, gC.klb, h2o_init)

        oxides = {}
        for mol in list(n_melt.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                oxides[mol] = n_melt[mol]

        fo2_melt = eq_melt.calc_peridotite_fo2(
            n_melt['Fe2O3'], 
            n_melt['FeO'],
            oxides, 
            temp, 
            p_tot
        )

        fo2_vals_per.append(fo2_melt - eq_melt.fo2_fmq(temp, p_tot))

    # --- --- --- --- --- ---
    # BASALT
    # --- --- --- --- --- ---
    iron_ratios_bas = []
    fo2_vals_bas = [val for val in fo2_vals_per]

    for fo2_val in [val for val in fo2_vals_per]:
        # calculate moles from wt% prescription (includes adding H2O)
        n_melt = eq_melt.basalt_comp_by_fo2(
            m_mag, 
            'FMQ', 
            fo2_val, 
            gC.basalt,
            0.05, 
            p_tot, 
            temp
        )
        
        oxides = {}
        for mol in list(n_melt.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                oxides[mol] = n_melt[mol]

        fe2o3, feo = dcop(n_melt['Fe2O3']), dcop(n_melt['FeO'])

        iron_ratios_bas.append(2. * fe2o3 / (2. * fe2o3 + feo))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # IMPORT DATA FROM MODELS 3A
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # BASALT
    # --- --- --- --- --- ---
    ferric_bas_init, ferric_bas_fin = [], []
    fo2_bas_init, fo2_bas_fin = [], []
    m_melt_bas = []
    for item in masses:
        # file name
        var_str = f"{item:.2e}"
        var_str = var_str.replace('.', '_')
        var_str = var_str.replace('+', '')
        dir_mass = dir_path + '/output/m_imps/'
        file = dir_mass + 'basalt_3A_' + var_str

        # read data
        with h5py.File(file + '.hdf5', 'r') as f:
            temp = f['temp'][()]
            p_tot = np.array(list(f['atmos/p_tot']))[-1]

            fe_3 = np.array(list(f['melt/fe2o3']))
            fe_2 = np.array(list(f['melt/feo']))
            fe_0 = np.array(list(f['metal/fe']))
            ferric_bas_init.append(2. * fe_3[0] / (2. * fe_3[0] + fe_2[0] + fe_0[0]))
            ferric_bas_fin.append(2. * fe_3[-1] / (2. * fe_3[-1] + fe_2[-1] + fe_0[-1]))

            fo2_melt = np.array(list(f['melt/fo2']))
            fo2_bas_init.append(fo2_melt[0] - eq_melt.fo2_fmq(temp, p_tot))
            fo2_bas_fin.append(fo2_melt[-1] - eq_melt.fo2_fmq(temp, p_tot))

            m_melt_bas.append(list(f['melt/mass'])[-1])

    # --- --- --- --- --- ---
    # PERIDOTITE
    # --- --- --- --- --- ---
    ferric_per_init, ferric_per_fin = [], []
    fo2_per_init, fo2_per_fin = [], []
    m_melt_per = []
    for item in masses:
        # file name
        var_str = f"{item:.2e}"
        var_str = var_str.replace('.', '_')
        var_str = var_str.replace('+', '')
        dir_mass = dir_path + '/output/m_imps/'
        file = dir_mass + 'peridotite_3A_' + var_str

        # read data
        with h5py.File(file + '.hdf5', 'r') as f:
            temp = f['temp'][()]
            p_tot = np.array(list(f['atmos/p_tot']))[-1]

            fe_3 = np.array(list(f['melt/fe2o3']))
            fe_2 = np.array(list(f['melt/feo']))
            fe_0 = np.array(list(f['metal/fe']))
            ferric_per_init.append(2. * fe_3[0] / (2. * fe_3[0] + fe_2[0] + fe_0[0]))
            ferric_per_fin.append(2. * fe_3[-1] / (2. * fe_3[-1] + fe_2[-1] + fe_0[-1]))

            fo2_melt = np.array(list(f['melt/fo2']))
            fo2_per_init.append(fo2_melt[0] - eq_melt.fo2_fmq(temp, p_tot))
            fo2_per_fin.append(fo2_melt[-1] - eq_melt.fo2_fmq(temp, p_tot))

            m_melt_per.append(list(f['melt/mass'])[-1])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # REMIXING OF IMPACT-GENERATED MELT PHASES WITH BSE UNMELTED MANTLE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    mixed_ferric_per, mixed_ferric_bas = [], []
    for i in range(len(masses)):
        # mass of the magma [kg]
        m_mag = eq_melt.impact_melt_mass(masses[i], v_imp, theta)

        # mass of unmelted mantle [kg]
        m_whole_mantle = (1. - 0.3201) * gC.m_earth
        m_sol = m_whole_mantle - m_mag

        # molar composition of solid mantle
        n_solid = eq_melt.peridotite_comp_by_fe_ratio(m_sol, 0.05, gC.klb, h2o_init)

        # ferric-to-iron ratio of solid mantle
        fe3_frac_solid = 2. * n_solid['Fe2O3'] / \
                         (2. * n_solid['Fe2O3'] + n_solid['FeO'])

        # mix together using mass fractions --- --- --- --- --- --- --- --- --- ---
        mix_per = (m_melt_per[i] * ferric_per_fin[i] + m_sol * fe3_frac_solid) / \
                  (m_melt_per[i] + m_sol)
        mixed_ferric_per.append(mix_per)

        mix_bas = (m_melt_bas[i] * ferric_bas_fin[i] + m_sol * fe3_frac_solid) / \
                  (m_melt_bas[i] + m_sol)
        mixed_ferric_bas.append(mix_bas)

        # print("\n>>> impactor mass : %.2e kg" % masses[i])
        # print("*** peridotite ***")
        # print(">>>     melt mass = %.2e kg" % m_melt_per[i])
        # print(">>>    solid mass = %.2e kg" % m_sol)
        # print(">>>   fe3/fe melt = %.5f" % ferric_per_fin[i])
        # print(">>>  fe3/fe solid = %.5f" % fe3_frac_solid)
        # print(">>>  fe3/fe mixed = %.5f" % mix_per)
        # print("*** basalt ***")
        # print(">>>     melt mass = %.2e kg" % m_melt_bas[i])
        # print(">>>    solid mass = %.2e kg" % m_sol)
        # print(">>>   fe3/fe melt = %.5f" % ferric_bas_fin[i])
        # print(">>>  fe3/fe solid = %.5f" % fe3_frac_solid)
        # print(">>>  fe3/fe mixed = %.5f" % mix_bas)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # FIGURE SET-UP
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # scale impactor masses
    masses = [mass * 1e-22 for mass in masses]

    # mass ticks
    xticks = [2e21, 5e21, 1e22, 2e22]
    xticks = [X * 1e-22 for X in xticks]

    fig = plt.figure(figsize=(7., 3.5), dpi=local_dpi)
    plt.subplots_adjust(left=0.11, right=0.89, bottom=0.15, top=0.9, wspace=0.2)
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
        'font.sans-serif': ['Arial', 'DejaVu Sans',
                            'Liberation Sans',
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
        ax0 = fig.add_subplot(121)  # basalt
        ax1 = fig.add_subplot(122)  # peridotite

    for ax in [ax0, ax1]:
        ax.set_xlim([0.17, 3.2])
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='both', direction='in')

    # X-AXIS : impactor mass --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax0.set_xlabel('Impactor Mass /10$^{22}$ kg', fontsize=12)

    ax1.set_xlabel('Impactor Mass /10$^{22}$ kg', fontsize=12)

    # Y-AXIS PERIDOTITE --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax0.set_ylabel('Fe$^{3+}$ / $\Sigma$ Fe', fontsize=12)
    ax0.set_yticks(iron_ratios_per)
    ax0.set_yticklabels([f"{val:.3f}" for val in iron_ratios_per])
    ax0.set_ylim([0., fe_max])
    ax0.tick_params(axis='y', which='both', left=True, right=False,
                    labelleft=True, labelright=False)

    ax00 = ax0.twinx()
    ax00.text(x=list(ax0.get_xlim())[1], y=1.01 * list(ax0.get_ylim())[1],
              s=" $\Delta$FMQ", color=cols['H2'])

    ax00.set_ylim(ax0.get_ylim())
    ax00.set_yticks(ax0.get_yticks())
    yticklabels = [f"{val:.2f}" for val in fo2_vals_per]
    ax00.set_yticklabels(yticklabels, color=cols['H2'])

    ax0.set_title('Peridotite')

    ax0.axhline(y=0.05, color='k', linestyle=':')
    ax0.plot(masses, ferric_per_fin, color=cols['H2'], linestyle='--')
    ax0.plot(masses, mixed_ferric_per, color='k', linestyle='-')

    # Y-AXIS BASALT --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_yticks(ax0.get_yticks())
    yticklabels = [f"{val:.2f}" for val in fo2_vals_bas]
    ax1.set_yticklabels(yticklabels)

    ax11 = ax1.twinx()
    ax11.set_ylabel('Fe$^{3+}$ / $\Sigma$ Fe', fontsize=12)
    ax11.set_ylim(ax0.get_ylim())
    ax11.set_yticks(ax0.get_yticks())
    ax11.set_yticklabels([f"{val:.3f}" for val in iron_ratios_bas])
    ax11.tick_params(axis='y', which='both', left=False, right=True,
                     labelleft=False, labelright=True)

    ax1.set_title('Basalt')

    ax11.axhline(y=0.05, color='k', linestyle=':')
    ax11.plot(masses, ferric_bas_fin, color=cols['H2'], linestyle='--')
    ax11.plot(masses, mixed_ferric_bas, color='k', linestyle='-')

    # --- --- --- --- --- ---
    # LABELS
    # --- --- --- --- --- ---
    ax0.text(masses[0], ferric_per_fin[0], s='(3A)', color=cols['H2'],
             va='bottom', ha='left')
    ax0.text(masses[0], 0.93 * mixed_ferric_per[0], s='(3A)', color=cols['CO2'],
             va='top', ha='left')

    ax11.text(masses[0], ferric_bas_fin[0], s='(3A)', color=cols['H2'],
             va='bottom', ha='left')
    ax11.text(masses[0], 0.93 * mixed_ferric_bas[0], s='(3A)', color=cols['CO2'],
             va='top', ha='left')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # IMPORT DATA FROM MODELS 3B
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impactor masses
    masses = np.logspace(np.log10(2e21), np.log10(2.44e22), 30, base=10., endpoint=True)

    # --- --- --- --- --- ---
    # BASALT
    # --- --- --- --- --- ---
    ferric_bas_init, ferric_bas_fin = [], []
    fo2_bas_init, fo2_bas_fin = [], []
    m_melt_bas = []
    for item in masses:
        # file name
        var_str = f"{item:.2e}" 
        var_str = var_str.replace('.', '_')
        var_str = var_str.replace('+', '')
        dir_mass = dir_path + '/output/m_imps/'
        file = dir_mass + 'basalt_3B_' + var_str

        # read data
        with h5py.File(file + '.hdf5', 'r') as f:
            temp = f['temp'][()]
            p_tot = np.array(list(f['atmos/p_tot']))[-1]

            fe_3 = np.array(list(f['melt/fe2o3']))
            fe_2 = np.array(list(f['melt/feo']))
            fe_0 = np.array(list(f['metal/fe']))
            ferric_bas_init.append(2. * fe_3[0] /
                                   (2. * fe_3[0] + fe_2[0] + fe_0[0]))
            ferric_bas_fin.append(2. * fe_3[-1] /
                                  (2. * fe_3[-1] + fe_2[-1] + fe_0[-1]))

            fo2_melt = np.array(list(f['melt/fo2']))
            fo2_bas_init.append(fo2_melt[0] - eq_melt.fo2_fmq(temp, p_tot))
            fo2_bas_fin.append(fo2_melt[-1] - eq_melt.fo2_fmq(temp, p_tot))

            m_melt_bas.append(list(f['melt/mass'])[-1])

    # --- --- --- --- --- ---
    # PERIDOTITE
    # --- --- --- --- --- ---
    ferric_per_init, ferric_per_fin = [], []
    fo2_per_init, fo2_per_fin = [], []
    m_melt_per = []
    for item in masses:
        # file name
        var_str = f"{m_imp:.2e}"
        var_str = var_str.replace('.', '_')
        var_str = var_str.replace('+', '')
        dir_mass = dir_path + '/output/m_imps/'
        file = dir_mass + 'peridotite_3B_' + var_str

        # read data
        with h5py.File(file + '.hdf5', 'r') as f:
            temp = f['temp'][()]
            p_tot = np.array(list(f['atmos/p_tot']))[-1]

            fe_3 = np.array(list(f['melt/fe2o3']))
            fe_2 = np.array(list(f['melt/feo']))
            fe_0 = np.array(list(f['metal/fe']))
            ferric_per_init.append(2. * fe_3[0] /
                                   (2. * fe_3[0] + fe_2[0] + fe_0[0]))
            ferric_per_fin.append(2. * fe_3[-1] /
                                  (2. * fe_3[-1] + fe_2[-1] + fe_0[-1]))

            fo2_melt = np.array(list(f['melt/fo2']))
            fo2_per_init.append(fo2_melt[0] - eq_melt.fo2_fmq(temp, p_tot))
            fo2_per_fin.append(fo2_melt[-1] - eq_melt.fo2_fmq(temp, p_tot))

            m_melt_per.append(list(f['melt/mass'])[-1])

    # --- --- --- --- --- ---
    # REMIXING OF IMPACT-GENERATED MELT PHASES WITH BSE UNMELTED MANTLE
    # --- --- --- --- --- ---
    mixed_ferric_per, mixed_ferric_bas = [], []
    for i in range(len(masses)):
        # mass of the magma [kg]
        m_mag = eq_melt.impact_melt_mass(masses[i], v_imp, theta)

        # mass of unmelted mantle [kg]
        m_whole_mantle = (1. - 0.3201) * gC.m_earth
        m_sol = m_whole_mantle - m_mag

        # molar composition of solid mantle
        n_solid = eq_melt.peridotite_comp_by_fe_ratio(m_sol, 0.05, gC.klb, h2o_init)

        # ferric-to-iron ratio of solid mantle
        fe3_frac_solid = 2. * n_solid['Fe2O3'] / \
                         (2. * n_solid['Fe2O3'] + n_solid['FeO'])

        # mix together using mass fractions --- --- --- --- --- --- --- --- --- ---
        mix_per = (m_melt_per[i] * ferric_per_fin[i] + m_sol * fe3_frac_solid) / \
                  (m_melt_per[i] + m_sol)
        mixed_ferric_per.append(mix_per)

        mix_bas = (m_melt_bas[i] * ferric_bas_fin[i] + m_sol * fe3_frac_solid) / \
                  (m_melt_bas[i] + m_sol)
        mixed_ferric_bas.append(mix_bas)

    # scale impactor masses
    masses = [mass * 1e-22 for mass in masses]

    ax0.plot(masses, ferric_per_fin, color=cols['H2'], linestyle='--')
    ax0.plot(masses, mixed_ferric_per, color='k', linestyle='-')

    ax11.plot(masses, ferric_bas_fin, color=cols['H2'], linestyle='--')
    ax11.plot(masses, mixed_ferric_bas, color='k', linestyle='-')

    # --- --- --- --- --- ---
    # LABELS
    # --- --- --- --- --- ---
    ax0.text(masses[0], ferric_per_fin[0], s='(3B)', color=cols['H2'],
             va='bottom', ha='left')
    ax0.text(masses[-1], 1.09 * mixed_ferric_per[-1], s='(3B)', color=cols['CO2'],
             va='bottom', ha='right')

    ax11.text(masses[0], ferric_bas_fin[0], s='(3B)', color=cols['H2'],
             va='bottom', ha='left')
    ax11.text(masses[-1], 1.03 * mixed_ferric_bas[-1], s='(3B)', color=cols['CO2'],
             va='bottom', ha='right')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # LEGEND
    # --- --- --- --- --- ---
    handles = []
    handles.append(lines.Line2D([0.], [0.], label='unmelted mantle',
                                color='k', linestyle=':', linewidth=2))
    handles.append(lines.Line2D([0.], [0.],
                                label='melt after atmosphere equilibration',
                                color=cols['H2'], linestyle='--', linewidth=2))
    handles.append(lines.Line2D([0.], [0.], label='bulk mantle',
                                color='k', linestyle='-', linewidth=2))
    ax0.legend(handles=handles, loc='upper right', fontsize=8)

    plt.savefig(f"{dir_path}/figures/figure_9.pdf", dpi=200)
    # plt.show()
