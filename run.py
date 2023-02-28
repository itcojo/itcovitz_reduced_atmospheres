import h5py
import numpy as np
from pprint import pprint
import sys

import reduced_atmospheres
import reduced_atmospheres.constants as constants
import reduced_atmospheres.equilibrate_melt as eq_melt
gC = constants.Constants()  # global constants
dir_path = reduced_atmospheres.dir_path

# color-blind friendly colour palette
wong = gC.color_wong
cols = {'H2O': wong[2], 'H2': wong[-2], 'CO2': wong[0], 'N2': wong[3],
        'CO': wong[1], 'CH4': wong[-3], 'NH3': wong[-1], 'fO2_A': 'grey',
        'fO2_M': 'k'}

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONTROL CENTER FOR RUNNING MODELS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# should calculations be carried out?
# [model outputs are saved automatically as h5py files]
model_run = False

# are we using the pre_erratum version of the code?
pre_erratum = ''
if 'erratum' in Repository('.').head.shorthand:
    pre_erratum = 'pre_erratum_'

# which model version is running?
# model_version = '1A'
# model_version = '1B'
# model_version = '2'
model_version = '3A'
# model_version = '3B'

# basaltic or peridotitic melt phase?
basalt = False
peridotite = True

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONTROL CENTER FOR PLOTTING
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# figure booleans correspond to figures from paper.
figure_4 = False
figure_5 = False
figure_6 = False
figure_7 = False
figure_8 = False
figure_9 = False

# figure bolleans for figures in erratum, showing updates from original publication
figure_6_comparison = False
figure_8_comparison = True
figure_9_comparison = True

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# SUGGESTED RANGES OF INITIAL CONDITIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# impactor mass
m_imps = np.logspace(np.log10(2.00e21), np.log10(2.44e22), 30, base=10., endpoint=True)

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

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# STANDARD VALUES
temp = 1900.  # system temperature [K]

# m_imp = 2e22  # impactor mass [kg]

theta = 45.  # degrees

imp_type = 'E'  # composition of impactor (see 'constants')

vaporised_oceans = 1.85  # surface H2O oceans [EO]

pCO2 = 100.  # initial CO2 partial pressure [bars]

pN2 = 2.  # initial N2 partial pressure [bars]

initial_water = 0.05  # water content of the impact-generated melt [wt%]

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# RUN MODELS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
if (basalt and peridotite) or (not basalt and not peridotite):
    print(">>> One of either basalt or peridotite must be True.")
    sys.exit()

if model_run:
    # # EXAMPLE OF CHANGING VARIABLE OTHER THAN IMPACTOR MASS
    # for temp in temps:
    #     # file name --- --- --- --- ---
    #     var_string = "%.0d" % temp
    #     var_string = var_string.replace('.', '_')
    #     var_string = var_string.replace('+', '')
    #
    #     folder = 'temps'

    for m_imp in m_imps:
        # file name --- --- --- --- ---
        var_string = "%.2e" % m_imp
        var_string = var_string.replace('.', '_')
        var_string = var_string.replace('+', '')

        # where to save output
        folder = 'm_imps'
        if basalt:
            save_file = dir_path + '/output/' + pre_erratum + folder + '/basalt_' + \
                        model_version + '_' + var_string
        if peridotite:
            save_file = dir_path + '/output/' + pre_erratum + folder + '/peridotite_' + \
                        model_version + '_' + var_string

        # FastChem system ID
        system_id = 'calc_multi_' + var_string

        # initial conditions values --- --- --- --- --- ---
        init_atmos = {'CO2': pCO2, 'N2': pN2}

        d_imp = eq_melt.impactor_diameter(m_imp, imp_type)  # [km]

        v_esc = eq_melt.escape_velocity(  # escape velocity [km s-1]
            gC.m_earth, 
            m_imp, 
            gC.r_earth,
            0.5 * d_imp * 1e3
        )
        v_imp = 2. * v_esc  # impact velocity [km s-1]

        # BASALT --- --- --- --- --- ---
        if basalt:
            comp_bas = gC.basalt  # composition of the silicate melt [wt%]
            fO2_buffer, fO2_diff = 'FMQ', 0.0  # fO2 of the generated melt phase

            # equilibrate magma with atmosphere
            [trackers, p_init, n_init] = eq_melt.eq_melt_basalt(
                m_imp, 
                v_imp, 
                theta, 
                imp_type, 
                vaporised_oceans, 
                init_atmos,
                comp_bas, 
                initial_water, 
                fO2_buffer, 
                fO2_diff, 
                temp,
                model_version, 
                partition=True, 
                chem=False, 
                tol=1e-5,
                sys_id=system_id
            )  

        # PERIDOTITE --- --- --- --- --- ---
        if peridotite:
            comp_per = gC.klb  # composition of the silicate melt [wt%]
            iron_ratio = 0.05  # Fe^3+ / Σ Fe in the impact-generated melt phase

            # equilibrate magma with atmosphere
            [trackers, p_init, n_init] = eq_melt.eq_melt_peridotite(
                m_imp, 
                v_imp, 
                theta, 
                imp_type, 
                vaporised_oceans, 
                init_atmos,
                comp_per, 
                initial_water, 
                iron_ratio, 
                temp, 
                model_version,
                partition=True, 
                chem=False, 
                tol=1e-5, 
                sys_id=system_id
            )

        # unpack atmosphere before thermochemical equilibrium is applied
        species = ['H2O', 'H2', 'N2', 'CO2']
        init_p_tot = np.sum(list(p_init[-2].values()))
        values = []
        for item in species:
            values.append(n_init[-2][item])

        # write data --- --- --- --- ---
        with h5py.File(save_file + '.hdf5', 'w') as f:
            f.create_dataset('temp', data=[temp])

            f.create_dataset('initial/species', data=species)
            f.create_dataset('initial/values', data=values)
            f.create_dataset('initial/pressure', data=[init_p_tot])

            f.create_dataset('atmos/h2', data=trackers[0])
            f.create_dataset('atmos/h2o', data=trackers[1])
            f.create_dataset('atmos/co2', data=trackers[2])
            f.create_dataset('atmos/n2', data=trackers[3])
            f.create_dataset('atmos/co', data=trackers[4])
            f.create_dataset('atmos/ch4', data=trackers[5])
            f.create_dataset('atmos/nh3', data=trackers[6])
            f.create_dataset('atmos/n_tot', data=trackers[16])
            f.create_dataset('atmos/p_tot', data=trackers[7])
            f.create_dataset('atmos/fo2', data=trackers[8])
            f.create_dataset('atmos/mass', data=trackers[9])

            f.create_dataset('melt/h2o', data=trackers[10])
            f.create_dataset('melt/fe2o3', data=trackers[11])
            f.create_dataset('melt/feo', data=trackers[12])
            f.create_dataset('melt/n_tot', data=trackers[17])
            f.create_dataset('melt/fo2', data=trackers[13])
            f.create_dataset('melt/mass', data=trackers[14])

            f.create_dataset('metal/fe', data=trackers[15])

            # NOTE: cannot save dictionaries like this!!

        print('\n\x1b[1;33m>>> Equilibrated system saved to ' + save_file + '\x1b[0m')

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# PLOT FIGURES
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
if figure_4:
    # melt mass plot
    reduced_atmospheres.figure_files.figure_4.plot_figure_4()
    print(">>> Figure 4 generated.")

if figure_5:
    # iron distribution plot
    reduced_atmospheres.figure_files.figure_5.plot_figure_5()
    print(">>> Figure 5 generated.")

if figure_6:
    # post-impact system plot
    reduced_atmospheres.figure_files.figure_6.plot_figure_6()
    # reduced_atmospheres.figure_files.figure_6_B.plot_figure_6B()
    print(">>> Figure 6 generated.")

if figure_7:
    # figure walking through the standard values impact scenario
    reduced_atmospheres.figure_files.figure_7.plot_figure_7()
    print(">>> Figure 7 generated.")

if figure_8:
    # five models comparison plot
    reduced_atmospheres.figure_files.figure_8.plot_figure_8()
    print(">>> Figure 8 generated.")

if figure_9:
    # melt-bulk mantle redox comparison plot
    reduced_atmospheres.figure_files.figure_9.plot_figure_9()
    print(">>> Figure 9 generated.")

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# COMPARATIVE FIGURES
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
if figure_6_comparison:
    # post-impact system plot
    reduced_atmospheres.figure_files.figure_6_comparison.plot_figure_6()
    # reduced_atmospheres.figure_files.figure_6_B.plot_figure_6B()
    print(">>> Figure 6 comparison generated.")

if figure_8_comparison:
    # five models comparison plot
    reduced_atmospheres.figure_files.figure_8_comparison.plot_figure_8()
    print(">>> Figure 8 comparison generated.")

if figure_9_comparison:
    # melt-bulk mantle redox comparison plot
    reduced_atmospheres.figure_files.figure_9_comparison.plot_figure_9()
    print(">>> Figure 9 comparison generated.")
