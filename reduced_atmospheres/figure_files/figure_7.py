import sys
from copy import deepcopy as dcop
import matplotlib.gridspec as gs
from matplotlib.patches import ConnectionPatch
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
def atmos_init_adap(mass_imp, vel_imp, init_ocean, p_atmos, temp, fe_frac,
                    sys_id, imp_comp='E'):
    """
    Adapted version of 'atmos_init' from 'equilibrate_melt', specifically made
    for plotting the figure.

    Parameters
    ----------
    mass_imp : float [kg]
        Mass of the impactor.
    vel_imp : float [km s-1]
        Impact velocity.
    init_ocean : float [Earth Oceans]
        Initial amount of water on the planet receiving impact.
    p_atmos : dict
        Composition of the atmosphere.
        Keys (str) full formulae of molecules.
        Values (float) partial pressure of each species [bar].
    temp : float [K]
        Temperature of the atmosphere before impact.
    sys_id : str
        Label of the atmosphere-magma system ('system_id'), used as file names
    imp_comp : str
        Impactor composition indicator ('C': carbonaceous chondrite,
        'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite,
        'E': enstatite chondrite, 'F': iron meteorite)

    Returns
    -------
    p_atmos (updated)
    n_atmos : dict
        Composition of the atmosphere.
        Keys (str) full formulae of molecules.
        Values (float) number of moles of species in atmosphere.
    p_list : list
        Partial pressure dictionaries at each stage of the calculations.
        [initial, erosion, ocean vaporisation, impactor vaporisation,
    n_list : list
        Moles dictionaries at each stage of the calculations.

    """
    p_init = dcop(p_atmos)  # initial input atmosphere
    for mol in list(p_init.keys()):
        p_init[mol] = p_init[mol] * 1e5

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # pre-impact atmosphere
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # convert bars to Pa
    for mol in list(p_atmos.keys()):
        p_atmos[mol] = p_atmos[mol] * 1e5

    # total pressure
    p_tot = np.sum(list(p_atmos.values()))

    # a.x = b matrix equation, where x is a list of molecules' moles
    mols = list(p_atmos.keys())
    a = np.zeros((len(mols), len(mols)))
    for ii in range(len(a[0])):  # columns
        for i in range(len(a) - 1):  # rows
            if i == ii:
                a[i, ii] = (p_tot / p_atmos[mols[i]]) - 1
            else:
                a[i, ii] = -1
        # final row
        a[-1, ii] = gC.mol_phys_props(mols[ii])[0] * 1e-3 / gC.u

    b = np.zeros(len(mols))
    b[len(mols) - 1] = 4. * np.pi * gC.r_earth ** 2. * p_tot / gC.g

    solve = np.linalg.solve(a, b)

    # unpack solution into moles dictionary
    n_atmos = {}
    for j in range(len(mols)):
        n_atmos[mols[j]] = solve[j]

    n_init = dcop(n_atmos)  # initial input atmosphere

    # atmosphere mass
    m_atm = 0.
    for mol in list(p_atmos.keys()):
        m_atm += n_atmos[mol] * gC.common_mol_mass[mol]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # atmospheric erosion by the impact
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impactor diameter [km]
    d_imp = eq_melt.impactor_diameter(mass_imp, imp_comp)

    [X_ejec, n_atmos] = eq_melt.atmos_ejection(n_atmos, mass_imp, d_imp,
                                               vel_imp,  param=0.7)

    # recalculate pressures
    [p_atmos, _] = eq_melt.update_pressures(n_atmos)

    p_erosion, n_erosion = dcop(p_atmos), dcop(n_atmos)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impactor vaporisation of volatiles
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    h2o_degas = eq_melt.vaporisation(mass_imp, imp_comp)

    if 'H2O' in list(n_atmos.keys()):
        n_atmos['H2O'] += h2o_degas
    else:
        n_atmos['H2O'] = h2o_degas

    # recalculate pressures
    [p_atmos, _] = eq_melt.update_pressures(n_atmos)

    p_degas, n_degas = dcop(p_atmos), dcop(n_atmos)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # injection of H2O into the atmosphere (vaporisation only)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # convert units of Earth Oceans into moles
    EO = 1.37e21  # [kg]
    EO_moles = EO / gC.common_mol_mass['H2O']  # [moles]
    init_h2o = init_ocean * EO_moles  # [moles]

    # add H2O and H2 into the atmosphere
    if 'H2O' in list(n_atmos.keys()):
        n_atmos['H2O'] += init_h2o
    else:
        n_atmos['H2O'] = init_h2o

    # recalculate pressures
    [p_atmos, _] = eq_melt.update_pressures(n_atmos)

    p_ocean, n_ocean = dcop(p_atmos), dcop(n_atmos)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # interaction of the impactor iron with the atmosphere
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # wt% of impactor mass is iron used to reduce oceans
    # Fe + H2O --> FeO + H2
    init_reduce_mass = fe_frac * gC.iron_wt[imp_comp] * mass_imp  # [kg]
    init_reduce_moles = init_reduce_mass / gC.common_mol_mass['Fe']  # [moles]

    if init_reduce_moles > init_h2o + n_atmos['CO2']:
        print("More Fe than H2O + CO2 for impactor mass = %.2e." % mass_imp)
        sys.exit()
    elif init_reduce_moles > init_h2o:
        print("More Fe than H2O for impactor mass = %.2e." % mass_imp)
    else:
        # add H2O and H2 into the atmosphere
        n_atmos['H2O'] -= init_reduce_moles
        n_atmos['H2'] = init_reduce_moles

    # recalculate pressures
    [p_atmos, _] = eq_melt.update_pressures(n_atmos)

    p_iron, n_iron = dcop(p_atmos), dcop(n_atmos)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # FastChem Equilibrium Calculations
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    pre_fc = dcop(n_atmos)

    # abundances for FastChem
    abund = eq_melt.calc_elem_abund(n_atmos)

    # prepare FastChem config files
    eq_melt.write_fastchem(dir_path + '/reduced_atmospheres/data/FastChem/' +
                           sys_id, abund, temp,
                           float(np.sum(list(p_atmos.values()))))

    # run automated FastChem
    eq_melt.run_fastchem_files(sys_id)

    # read FastChem output
    [p_atmos, n_atmos] = eq_melt.read_fastchem_output(sys_id)

    p_chem, n_chem = dcop(p_atmos), dcop(n_atmos)

    return p_atmos, n_atmos,\
           [p_init, p_erosion, p_degas, p_ocean, p_iron, p_chem],\
           [n_init, n_erosion, n_degas, n_ocean, n_iron, n_chem]


def zero_to_nan(array_1d):
    """
    Change all zeroes in input array to numpy nans.

    Parameters
    ----------
    array_1d : list

    Returns
    -------

    """
    for idx in range(len(array_1d)):
        if array_1d[idx] == 0.:
            array_1d[idx] = np.nan

    return array_1d


def plot_figure_7():
    """
    Plots partial pressures of species from pre-impact to post-equilibration.

    Parameters
    ----------

    Returns
    -------

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SET UP FIGURE
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.figure(figsize=(6.5, 3.), dpi=local_dpi)
    plt.subplots_adjust(left=0.1, right=0.92, top=0.93, bottom=0.25,
                        wspace=0)
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

    grid = gs.GridSpec(1, 2, width_ratios=[2, 1])

    with sns.axes_style(plot_params):
        ax0 = plt.subplot(grid[0])
        ax0.minorticks_on()
        ax0.set_title('Impact Processing', fontsize=7)

        ax1 = plt.subplot(grid[1])
        ax1.minorticks_on()
        ax1.set_title('Melt-Atmosphere Equilibration', fontsize=7)

    x_vals = np.arange(6)
    ax0.set_xticks(x_vals)
    ax0.set_xticklabels(
        ['initial \nconditions', 'atmospheric \nerosion', 'mantle volatiles',
         'ocean \nvaporisation', 'iron interaction',
         'thermochemical \nequilibrium'],
        rotation=45., ha='right', fontsize=7)
    ax0.set_xlim([-0.5, 5.5])

    ax0.set_ylabel('Partial Pressures /bar')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # SET UP PROCESSING
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # range of impact masses [kg]
    impactor_masses = np.logspace(np.log10(2e21), np.log10(2.7e22), 30,
                                  base=10., endpoint=True)

    # find which impactor mass is closest to 2e22 kg
    m_imp = impactor_masses[(np.abs(impactor_masses - 2e22)).argmin()]

    # empties for initial conditions calculations
    # (i)   initial values,
    # (ii)  after atmospheric erosion,
    # (iii) after addition of mantle volatiles,
    # (iv)  after ocean vaporisation,
    # (v)   after iron interaction,
    # (vi)  after thermochemical equilibrium
    h2, h2o, co2, n2 = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    co, ch4, nh3, p_tot = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)

    # file name
    var_string = "%.2e" % m_imp
    var_string = var_string.replace('.', '_')
    var_string = var_string.replace('+', '')

    s_id = 'test_figure_' + var_string

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # INITIAL CONDITIONS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    temp = 1500.  # temperature of atmosphere [K]

    theta = 45.  # impact angle [degrees]

    imp_comp = 'E'  # impactor composition

    impactor_diam = eq_melt.impactor_diameter(m_imp, imp_comp)  # [km]

    v_esc = eq_melt.escape_velocity(gC.m_earth, m_imp, gC.r_earth,
                                    0.5 * impactor_diam * 1e3)  # [km s-1]
    v_imp = 2. * v_esc  # [km s-1]

    N_oceans = 1.85  # [EO]
    pCO2 = 100  # [bars]
    pN2 = 2.  # [bars]
    init_atmos = {'CO2': pCO2, 'N2': pN2}

    # water content of the magma [wt%]
    H2O_init = 0.05

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ATMOSPHERE THROUGHOUT ATMOSPHERIC PROCESSING
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # iron distribution
    [X_fe_atm, _, _] = eq_melt.available_iron(m_imp, v_imp, theta, 'E')

    [_, _, P_LIST, _] = atmos_init_adap(m_imp, v_imp, N_oceans,
                                        dcop(init_atmos), temp, X_fe_atm,
                                        sys_id=s_id, imp_comp='E')
    # sort output
    for i in range(len(P_LIST)):
        p_tot[i] = np.sum(list(P_LIST[i].values())) * 1e-5  # [bar]

        for mol in list(P_LIST[i].keys()):
            if mol == 'H2':
                h2[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'H2O':
                h2o[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'CO2':
                co2[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'N2':
                n2[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'CO':
                co[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'CH4':
                ch4[i] = P_LIST[i][mol] * 1e-5
            elif mol == 'H3N':
                nh3[i] = P_LIST[i][mol] * 1e-5

    # change zeros to nans
    h2 = zero_to_nan(h2)
    h2o = zero_to_nan(h2o)
    co2 = zero_to_nan(co2)
    n2 = zero_to_nan(n2)
    co = zero_to_nan(co)
    ch4 = zero_to_nan(ch4)
    nh3 = zero_to_nan(nh3)

    # plotting
    ax0.plot(x_vals, h2, color=cols['H2'], label='H$_2$',
             linestyle='', marker='o')
    ax0.plot(x_vals, h2o, color=cols['H2O'], label='H$_2$O',
             linestyle='', marker='o')
    ax0.plot(x_vals[:-1], co2[:-1], color=cols['CO2'], label='CO$_2$',
             linestyle='', marker='o')
    ax0.plot(x_vals[:-1], n2[:-1], color=cols['N2'], label='N$_2$',
             linestyle='', marker='o')

    ax0.plot(x_vals[-1] - 0.05, co2[-1], color=cols['CO2'],
             linestyle='', marker='o', ms=3)
    ax0.plot(x_vals, co, color=cols['CO'], label='CO',
             linestyle='', marker='o', ms=3)
    ax0.plot(x_vals-0.05, ch4, color=cols['CH4'], label='CH$_4$',
             linestyle='', marker='o', ms=3)
    ax0.plot(x_vals[-1], n2[-1], color=cols['N2'], label='N$_2$',
             linestyle='', marker='o', ms=3)
    ax0.plot(x_vals+0.05, nh3, color=cols['NH3'], label='NH$_3$',
             linestyle='', marker='o', ms=3)

    ax0.plot(x_vals, p_tot, color='grey', label='Total \nPressure')

    ax0.legend(loc='upper left', fontsize=6)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # MELT-ATMOSPHERE INTERACTIONS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    comp_per = gC.klb  # peridotite composition of the silicate melt [wt%]
    iron_ratio = 0.05  # Fe^3+ / Σ Fe in the impact-generated melt phase

    # # equilibrate magma with atmosphere
    # [trackers, _, _] = eq_melt.eq_melt_peridotite(
    #     m_imp, v_imp, theta, imp_comp, N_oceans, init_atmos, comp_per, H2O_init,
    #     iron_ratio, temp, '3A', partition=True, chem=False, tol=1e-5, sys_id=s_id)

    [trackers, _, _] = eq_melt.eq_melt_basalt(
        m_imp, v_imp, theta, imp_comp, N_oceans, init_atmos, gC.basalt,
        H2O_init, 'FMQ', 0.0, temp, '3A',  partition=True, chem=False,
        tol=1e-5, sys_id=s_id)

    h2_ma, h2o_ma, co2_ma = trackers[0], trackers[1], trackers[2]
    n2_ma, co_ma, ch4_ma = trackers[3], trackers[4], trackers[5]
    nh3_ma, n_tot_ma, p_tot_ma = trackers[6], trackers[16], trackers[7]

    for j in range(len(h2_ma)):
        h2_ma[j] = 1e-5 * p_tot_ma[j] * h2_ma[j] / n_tot_ma[j]
        h2o_ma[j] = 1e-5 * p_tot_ma[j] * h2o_ma[j] / n_tot_ma[j]
        co2_ma[j] = 1e-5 * p_tot_ma[j] * co2_ma[j] / n_tot_ma[j]
        n2_ma[j] = 1e-5 * p_tot_ma[j] * n2_ma[j] / n_tot_ma[j]
        co_ma[j] = 1e-5 * p_tot_ma[j] * co_ma[j] / n_tot_ma[j]
        ch4_ma[j] = 1e-5 * p_tot_ma[j] * ch4_ma[j] / n_tot_ma[j]
        nh3_ma[j] = 1e-5 * p_tot_ma[j] * nh3_ma[j] / n_tot_ma[j]

        p_tot_ma[j] = 1e-5 * p_tot_ma[j]

    limit = 3  # how many steps to show?
    ax1.plot(np.arange(limit), h2_ma[1:limit+1], color=cols['H2'],
             linestyle='', marker='s')
    ax1.plot(np.arange(limit), h2o_ma[1:limit+1], color=cols['H2O'],
             linestyle='', marker='s')

    # partial pressures
    ax1.plot(np.arange(limit)-0.025, co2_ma[1:limit + 1], color=cols['CO2'],
             linestyle='', marker='s', markersize=3)
    ax1.plot(np.arange(limit)+0.025, co_ma[1:limit + 1], color=cols['CO'],
             linestyle='', marker='s', markersize=3)
    ax1.plot(np.arange(limit)-0.05, ch4_ma[1:limit + 1], color=cols['CH4'],
             linestyle='', marker='s', markersize=3)
    ax1.plot(np.arange(limit)+0.05, nh3_ma[1:limit + 1], color=cols['NH3'],
             linestyle='', marker='s', markersize=3)
    ax1.plot(np.arange(limit), n2_ma[1:limit + 1], color=cols['N2'],
             linestyle='', marker='s', markersize=3)

    # total atmospheric pressure
    ax1.plot(np.arange(limit), p_tot_ma[1:limit+1], color='grey')

    x_labels = ['Step 1 \n(chemical)', 'Step 1 \n(partitioning)',
                'Step 2 \n(chemical)', 'Step 2 \n(partitioning)']
    ax1.set_xticks(np.arange(limit))
    ax1.set_xlim([-0.5, limit + 0.5])
    ax1.set_xticklabels(x_labels[:limit], rotation=315, ha='center', fontsize=7)

    ax1.set_yticks(ax0.get_yticks())
    ax1.set_ylim(ax0.get_ylim())
    ax1.tick_params(axis='y', which='both', left=True, right=True,
                    labelleft=False, labelright=True)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # CONNECTION PATCHES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # equilibration dots
    dots = np.arange(limit-0.5, limit+0.5, 0.25)
    ax1.plot(dots, np.ones(len(dots)) * h2_ma[limit - 1], color=cols['H2'],
             linestyle='', marker='o', markersize=2)
    ax1.text(x=dots[-1] + 0.05, y=1.2 * h2_ma[limit - 1],
             s='repeat to\nequilibrium', color=cols['H2'], fontsize=6,
             ha='right', va='bottom')

    p_xy = (x_vals[-1], p_tot[-1])
    p_ma_xy = (0, p_tot_ma[1])
    con = ConnectionPatch(xyA=p_xy, xyB=p_ma_xy,
                          coordsA="data", coordsB="data",
                          axesA=ax0, axesB=ax1, color="grey", linewidth=1.5)
    fig.add_artist(con)
    ax1.text(x=0.25, y=460, s='H$_2$O partitioning', fontsize=6, color='grey',
             ha='left', va='top', rotation=295)

    h2_xy = (x_vals[-1] + 0.1, 120)
    h2_ma_xy = (-0.2, 270)
    con = ConnectionPatch(xyA=h2_xy, xyB=h2_ma_xy,
                          coordsA="data", coordsB="data", arrowstyle='->',
                          axesA=ax0, axesB=ax1, color=cols['H2'])
    fig.add_artist(con)
    ax0.text(x=5.38, y=200, s='redox chemistry', fontsize=6, ha='right',
             color=cols['H2'])

    # h2o_xy = (x_vals[-1], h2o[-1])
    # h2o_ma_xy = (0, h2o_ma[1])
    # con = ConnectionPatch(xyA=h2o_xy, xyB=h2o_ma_xy,
    #                       coordsA="data", coordsB="data",
    #                       axesA=ax0, axesB=ax1, color=cols['H2O'])
    # fig.add_artist(con)

    plt.savefig(dir_path + '/figures/figure_7.pdf', dpi=200)
    # plt.show()