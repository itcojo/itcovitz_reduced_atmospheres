import csv
import numpy as np
import os
from pprint import pprint
import re
import scipy.optimize as opt
import subprocess
import sys
from copy import deepcopy as dcop
from scipy.stats import linregress
from tabulate import tabulate
from typing import Optional, Tuple

# directory where FastChem is installed
import reduced_atmospheres
dir_fastchem = reduced_atmospheres.dir_fastchem

# global constants
gC = reduced_atmospheres.constants.Constants()

# directory where 'itcovitz_reduced_atmospheres/reduced_atmospheres' is located
dir_path = f"{reduced_atmospheres.dir_path}/reduced_atmospheres"


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# READ/WRITE THINGS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def read_fastchem_output(sys_id: str) -> Tuple[dict, dict]:
    """Read FastChem outputs.

    Reads the FastChem output files and creates dictionaries of species' partial pressures and moles [UPDATED FOR FASTCHEM V2].

    Args:
        sys_id (str): Label of the atmosphere-melt system ('system_id'), used as file names.

    Returns:
        p_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) partial pressure of each species [Pa].
        n_atmos (dict):  Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) moles of each species.

    """
    file = f"{dir_path}/data/FastChem/{sys_id}_output.dat"

    data = [i.strip().split() for i in open(file).readlines()]
    
    # get unique list of headers
    headers, skip = [], []
    for idx in range(len(data[0]) - 1):
        if idx in skip:
            pass
        elif '(' in data[0][idx+1]:
            headers.append(data[0][idx] + data[0][idx+1])
            skip.append(idx+1)
        else:
            headers.append(data[0][idx])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    display = False
    if display:
        # display system properties
        table_list = [
            ['Pressure', 'bar', float(data[1][0])],
            ['Temperature', 'K', float(data[1][1])],
            ['n(H)', 'cm-3', float(data[1][2])],
            ['n(g)', 'cm-3', float(data[1][3])],
            ['mean weight', 'u', float(data[1][4])]
        ]
        print('\n')
        print('\x1b[1;31m*** After FastChem ***\x1b[0m')
        print(tabulate(
            table_list,
            tablefmt='orgtbl',
            headers=['System Property', 'Unit', 'Value'],
            floatfmt=("", "", ".5e")
        ))
        print('\n')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # locate where first molecule is
    loc = None
    for ii in range(len(data[0])):
        # m(u) is the last system property before the molecules begin
        if data[0][ii] == 'm' and data[0][ii+1] == '(u)':
            loc = ii
            break
    loc += 1

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # fill in dictionary of species' abundances (mixing ratio)
    species = {}
    for iii in range(loc, len(headers)):
        # remove '1' from species names
        if '1' in headers[iii]:
            mol = headers[iii].replace('1', '')
        else:
            mol = headers[iii]

        # fill in species dictionary
        species[mol] = data[1][iii]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # sort species by their abundances
    mols = list(species.keys())
    vals = [float(item) for item in species.values()]

    sorted_mols = [x for _, x in sorted(zip(vals, mols))]
    sorted_mols.reverse()
    sorted_vals = [y for y, _ in sorted(zip(vals, mols))]
    sorted_vals.reverse()

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # create dictionary of species' partial pressures
    p_atmos = {}  # [Pa]
    for iv in range(len(sorted_mols)):
        if sorted_vals[iv] > 1e-9:
            p_atmos[sorted_mols[iv]] = sorted_vals[iv] * float(data[1][0]) * 1e5

    # a.x = b matrix equation, where x is a list of molecules' moles
    mols = list(p_atmos.keys())
    p_tot = np.sum(list(p_atmos.values()))

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

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if display:
        # print table of species' abundances
        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            if p_atmos[mol] / p_tot > 1e-10:
                table_list.append([
                    mol, 
                    p_atmos[mol] / p_tot,
                    p_atmos[mol] * 1e-5, 
                    n_atmos[mol]
                ])

        print(tabulate(
            table_list, 
            tablefmt='orgtbl', 
            headers=['Species', 'Mixing Ratio', 'Partials /bar', 'Moles'],
             floatfmt=("", ".2e", ".2e", ".2e")
        ))

    return p_atmos, n_atmos


def run_fastchem_files(sys_id: str) -> None:
    """Run FastChem.

    Edit the FastChem files to use the data produced for the current system, and run the FastChem code.

    Args:
        sys_id (str): Label of the atmosphere-melt system ('system_id'), used as file names

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # open config file
    with open(f"{dir_fastchem}/input/config.input", "r") as fc:
        # edit location of PT profile
        data = fc.readlines()

        data[1] = f"{dir_path}/data/FastChem/{sys_id}_PT.dat\n"
        data[4] = f"{dir_path}/data/FastChem/{sys_id}__output.dat\n"
        data[7] = f"{dir_path}/data/FastChem/{sys_id}__monitor.dat\n"
        data[16] = f"{dir_path}/data/FastChem/{sys_id}__abund.dat\n"

    # open config file and edit in changes
    with open(f"{dir_fastchem}/input/config.input", "w") as fc:
        fc.writelines(data)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # open parameters file
    with open(f"{dir_fastchem}/input/parameters.dat", "r") as fc:
        # edit location of PT profile
        data = fc.readlines()
        data[1] = f"{dir_path}/data/FastChem/{sys_id}_abund.dat\n"

    # open parameters file and edit in changes
    with open(f"{dir_fastchem}/input/parameters.dat", "w") as fc:
        fc.writelines(data)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # FastChem bash command
    fastchem_bash = ["./fastchem", "input/config.input"]
    process = subprocess.run(
        fastchem_bash, 
        cwd=dir_fastchem,
        capture_output=True
    )


def write_fastchem(
    path: str, 
    abundances: dict, 
    T: float, 
    P: float,
) -> None:
    """Write FastChem files ready for run.

    Writes the atmosphere elemental abundances, total atmospheric pressure, and temperature, to .dat files readable by FastChem.

    Args:
        path (str): Path to where the files will be saved.
        abundances (dict):  Elemental abundances in the atmosphere. Keys (str) elements. Values (float) abundances in the solar convention.
        T (float): Temperature of the atmosphere, in units of 'K'.            
        P (float): Total pressure in the atmosphere, in units of 'Pa'.

    """
    # abundances
    file_a = open(f"{path}_abund.dat", "w")
    file_a.write("# Chemical composition of a post_impact atmosphere:\n")
    for elem in sorted(list(abundances.keys())):
        # file_a.write(elem + '    ' + '%.10f' % abundances[elem] + '\n')
        file_a.write(f"{elem}    {abundances[elem]:.10f}\n")

    # environment
    file_e = open(f"{path}_PT.dat", "w")
    file_e.write("# Post_impact atmosphere, pressure in bar, temperature in K\n")
    # file_e.write('%.6e' % (P * 1e-5) + '    ' + '%.6e' % T + '\n')
    file_e.write(f"{P * 1e-5:.6e}    {T:.6e}\n")


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# USEFUL CALCULATIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def calc_anhydrous_mass(fe2o3: float, feo: float, oxides: dict) -> float:
    """Calculate the mass of anhydrous components of the silicate melt phase.

    Args:
        fe2o3 (float): Ferric iron in the melt phase, in units of moles.
        feo (float): Ferrous iron in the melt phase, in units of moles.
        oxides (dict): Moles of non-iron species in the melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.

    Returns:
        m_melt (float): Mass of the anhydrous melt, in units of 'kg'.

    """
    m_melt = fe2o3 * gC.common_mol_mass['Fe2O3'] + feo * gC.common_mol_mass['FeO']
    for mol in list(oxides.keys()):
        M = gC.common_mol_mass[mol]
        m_melt += M * oxides[mol]

    return m_melt


def calc_elem_abund(n_atmos: dict) -> dict:
    """Calculate the elemental abundances within a gas. 
    
    Uses the Solar Abundance convention (normalised to A(H) = 12).

    Args:
        n_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) number of moles of species in atmosphere.

    Returns:
        elem_frac (dict): Elemental composition of the atmosphere. Keys (str) elements. Values (float)
    """
    nums = {}  # [mol]

    for mol in list(n_atmos.keys()):
        # separate out molecules into elements, numbers, and symbols
        char = [m for m in re.split(r'([A-Z][a-z]*)', mol) if m]

        for j in range(len(char) - 1):
            # element-number pairing
            if char[j].isalpha() and char[j + 1].isdigit():
                # if element already in 'nums'
                if char[j] in list(nums.keys()):
                    nums[char[j]] += int(char[j + 1]) * n_atmos[mol]
                # if element not yet in 'nums'
                else:
                    nums[char[j]] = int(char[j + 1]) * n_atmos[mol]

            # single element
            elif char[j].isalpha():
                # if element already in 'nums'
                if char[j] in list(nums.keys()):
                    nums[char[j]] += n_atmos[mol]
                # if element not yet in 'nums'
                else:
                    nums[char[j]] = n_atmos[mol]

        if char[(len(char) - 1)].isalpha():
            # if element already in 'nums'
            if char[(len(char) - 1)] in list(nums.keys()):
                nums[char[(len(char) - 1)]] += n_atmos[mol]
            # if element not yet in 'nums'
            else:
                nums[char[(len(char) - 1)]] = n_atmos[mol]

    abund = {}
    for elem in list(nums.keys()):
        abund[elem] = 12. + np.log10(nums[elem] / nums['H'])

    return abund


def escape_velocity(m1: float, m2: float, r1: float, r2: float) -> float:
    """Calculate the mutual escape velocity between two bodies.

    Args:
        m1 (float): Mass of the first body, in units of 'kg'.
        m2 (float): Mass of the second body, in units of 'kg'.
        r1 (float): Radius of the first body, in units of 'm'.
        r2 (float): Radius of the first body, in units of 'm'.

    Returns:
        v_esc (float): Mutual escape velocity, in units of 'km s-1'.
            
    """
    return np.sqrt(2. * gC.G * (m1 + m2) / (r1 + r2)) * 1e-3


def specific_energy(
    m_t: float, 
    m_i: float, 
    v_i: float, 
    b: float,
) -> Tuple[float, float]:
    """Calculate modified specific energy of an impact.

    Args:
        m_i (float): Mass of the impactor, in units of 'kg'.
        m_t (float): Mass of the target, in units of 'kg'.
        v_i (float): Impact velocity, in units of 'km s-1'.            
        b (float): Impact parameter (sin of impact angle).

    Returns:
        Q_S (float): Modified specific energy of impact for target, in units of 'J kg-1'.
        Q_R (float): Specific energy of impact for target, in units of 'J kg-1'.

    """
    m_tot = m_t + m_i  # [kg]
    mu = (m_t * m_i) / m_tot  # reduced mass [kg]
    Q_R = mu * (1e3 * v_i) **2. / (2. * m_tot)  # specific energy
    Q_S = Q_R * (1. + (m_i / m_t)) * (1. - b)  # modified specific energy

    return Q_S, Q_R


def update_pressures(n_atmos: dict) -> Tuple[dict, float]:
    """Update species' pressures in the atmosphere.

    Args:
        n_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) number of moles of species in atmosphere.

    Returns:
        p_atmos (dict): Partial pressures in the atmosphere. Keys (str) full formulae of molecules. Values (float) partial pressure of species in atmosphere, in units of 'Pa'.
        p_tot (float): Total atmospheric pressure, in units of 'Pa'.

    """
    m_atm = 0.  # total mass of the atmosphere
    for mol in list(n_atmos.keys()):
        if mol in list(gC.common_mol_mass.keys()):
            m_atm += n_atmos[mol] * gC.common_mol_mass[mol]
        else:
            m_atm += n_atmos[mol] * gC.mol_phys_props(mol)[0] * gC.N_A

    # total pressure of the atmosphere
    p_tot = gC.g * m_atm / (4. * np.pi * gC.r_earth ** 2.)

    # partial pressures
    p_atmos = {}
    for mol in list(n_atmos.keys()):
        p_atmos[mol] = p_tot * n_atmos[mol] / np.sum(list(n_atmos.values()))

    return p_atmos, p_tot


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CALCULATE FUGACITIES
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def calc_basalt_fo2(
    fe2o3: float, 
    feo: float, 
    oxides: dict, 
    T: float, 
    P: float,
) -> float:
    """Calculate the oxygen fugacity of a basaltic melt composition.

    Args:
        feo2 (float): Moles of Fe2O3 in the melt.
        feo (float): Moles of FeO in the melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        fO2_melt (float): Oxygen fugacity of the melt (log base 10).

    """
    # total anhydrous moles in the melt
    N_melt = fe2o3 + feo + np.sum(list(oxides.values()))

    # fugacities
    if fe2o3 == 0.:
        fO2_melt = fo2_f91_rh12(feo / N_melt, T, P)
    else:
        fO2_KC = fo2_kc91(fe2o3, feo, oxides, T, P)
        fo2_iw = fo2_f91_rh12(feo / N_melt, T, P)
        fO2_melt = np.max([fO2_KC, fo2_iw])

    return fO2_melt


def calc_peridotite_fo2(
    fe2o3: float, 
    feo: float, 
    oxides: dict, 
    T: float, 
    P: float,
    tol: Optional[float] = 1e-5,
) -> float:
    """Calculate the oxygen fugacity of a peridotitic melt composition.

    Args:
        feo2 (float): Moles of Fe2O3 in the melt.
        feo (float): Moles of FeO in the melt.
        oxides (dict): Moles of non-iron species in the melt. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.
        tol (optional, float): Tolerance in what is classified as 'zero moles'.

    Returns:
        fO2_melt (float): Oxygen fugacity of the melt (log base 10).

    """
    # total anhydrous moles in the melt
    N_melt = fe2o3 + feo + np.sum(list(oxides.values()))

    # fugacities
    if np.abs(fe2o3) < tol:
        fO2_melt = fo2_f91_rh12(feo / N_melt, T, P)
    else:
        # fO2_KC = fo2_kc91(fe2o3, feo, oxides, T, P)
        fO2_sos = fo2_sossi(fe2o3, feo, oxides, T, P)
        fO2_iw = fo2_f91_rh12(feo / N_melt, T, P)
        fO2_melt = np.max([fO2_sos, fO2_iw])

    return fO2_melt


def calc_ph2o(h2o_mag: float, m_melt: float) -> float:
    """Calculate the partial pressure of H2O at melt-atmosphere equilibrium.

    Follows the prescription of Carroll and Holloway (1994).

    Args: 
        h2o_mag (float): Moles of H2O in the melt.
        m_melt (float): Mass of the melt, in units of 'kg'.

    Returns:
        p_H2O (float): Predicted partial pressure of H2O, in units of 'Pa'.

    """
    m_frac_H2O = h2o_mag * gC.common_mol_mass['H2O'] / m_melt
    p_H2O = (m_frac_H2O / 6.8e-8) ** (1. / 0.7)  # [Pa]
    return p_H2O


def fo2_atm(h2: float, h2o: float, T: float) -> float:
    """Calculate the oxygen fugacity of an atmosphere.

    Uses the equilibrium between hydrogen and oxygen, and water to determine fO2. Uses the formulation of Ohmoto and Kerrick (1977).

    Equilibrium form:  H2 + 0.5 O2 <=> H2O

    Args:
        h2 (float): Moles of H2 in the atmosphere.
        h2o (float): Moles of H2O in the atmosphere.
        T (float):  Temperature of the atmosphere, in units of 'K'.

    Returns:
        log10_fO2 (float): Oxygen fugacity of the atmosphere (log base 10).

    """
    # assume ideal gas, so gas activities are equal to mixing ratios
    log10_K = (12510 / T) - (0.979 * np.log10(T)) + 0.483
    log10_fO2 = -2 * (log10_K + np.log10(h2 / h2o))

    return log10_fO2


def fo2_iw(T: float, P: float) -> float:
    """Calculate the oxygen fugacity of the reference IW buffer.
    
    Uses the pure/theoretical iron-wustite equilibrium buffer of Frost (1991).

    Args:
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        log10_fO2 (float): Oxygen fugacity of the IW buffer (log base 10).
    """
    a, b, c = -27489, 6.702, 0.055  # empirical constants
    log10_fO2 = (a / T) + b + c * ((P * 1e-5 - 1) / T)

    return log10_fO2


def fo2_fmq(T: float, P: float) -> float:
    """Calculate the oxygen fugacity of the reference FMQ buffer.

    Uses the pure/theoretical fayalite-magnetite-quartz equilibrium buffer of Frost (1991).

    Args:
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        log10_fO2 (float): Oxygen fugacity of the FMQ buffer (log base 10).
    """
    a, b, c = -25096.3, 8.735, 0.110  # empirical constants
    log10_fO2 = (a / T) + b + c * ((P * 1e-5 - 1) / T)

    return log10_fO2


def fo2_f91_rh12(feo: float, T: float, P: float) -> float:
    """Calculate the oxygen fugacity of the melt. 
    
    The pure/theoretical iron-wustite equilibrium buffer of Frost (1991) is used, with the addition of a term accounting for the change in FeO to Fe ratio from Righter & Ghiorso, 2012.

    Args:
        feo (float): Moles of FeO in the melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        log10_fO2 (float): Oxygen fugacity of the melt buffer (log base 10).

    """
    a, b, c = -27489, 6.702, 0.055  # empirical constants
    fe = 0.98  # how 'pure' the metal phase is in iron
    log10_fO2 = (a / T) + b + c * ((P * 1e-5 - 1) / T) - 2 * np.log10(fe / feo)

    return log10_fO2


def fo2_kc91(
    fe2o3: float, 
    feo: float, 
    oxides: dict, 
    T: float, 
    P: float
) -> float:
    """ Calculate the oxygen fugacity of the melt using the formulation of Kress & Carmichael (1991).

    Args:
        feo2 (float): Moles of Fe2O3 in the melt.
        feo (float): Moles of FeO in the melt.
        oxides (dict): Moles of non-iron species in the melt. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        fO2_melt (float): Oxygen fugacity of the melt (log base 10).

    """
    [A, B, C, D, E, F, G, H] = kc_consts(dcop(fe2o3), dcop(feo), oxides, T, P)

    # natural log
    log_fO2 = (np.log(fe2o3 / feo) - B - C - D - E - F - G - H) / A
    # log base 10
    log10_fO2 = log_fO2 / np.log(10)

    return log10_fO2


def fo2_sossi(
    fe2o3: float, 
    feo: float, 
    oxides: dict, 
    T: float,
    P: float
) -> float:
    """Calculate the oxygen fugacity of the melt phase using the formulation of Sossi+ (2020).

    Args:
        feo2 (float): Moles of Fe2O3 in the melt.
        feo (float): Moles of FeO in the melt.
        oxides (dict): Moles of non-iron species in the melt. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        fO2_melt (float): Oxygen fugacity of the melt (log base 10).

    """
    IW = fo2_iw(T, P)  # log10

    d_IW = (1. / 0.252) * (np.log10(2 * fe2o3 / feo) + 1.530)  # log10

    fO2 = IW + d_IW

    return fO2


def kc_consts(
    fe2o3: float, 
    feo: float, 
    oxides: dict, 
    T: float, 
    P: float
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return constants for the Kress and Carmichael (1991) parameterisation of the FMQ buffer.

    Args:
        feo2 (float): Moles of Fe2O3 in the melt.
        feo (float): Moles of FeO in the melt.
        oxides (dict): Moles of non-iron species in the melt. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.
        T (float): Temperature of the melt, in units of 'K'.
        P (float): Pressure in the melt, in units of 'Pa'.

    Returns:
        constants (tuple, floats): KC91 parameterisation constants.

    """
    # empirical constants
    a, b, c, e, f, g, h = 0.196, 1.1492e4, -6.675, -3.36, -7.01e-7, -1.54e-10, 3.85e-17
    T_0 = 1673  # [K]
    d = {
        'Al2O3': -2.243,
        'CaO': 3.201, 
        'Fe2O3': -1.828, 
        'FeO': -1.828,
        'K2O': 6.215, 
        'Na2O': 5.854
    }

    # total moles in the melt, must be anhydrous
    N_melt = fe2o3 + feo + float(np.sum(list(oxides.values())))
    # list of molecules in the melt
    melt_strs = list(oxides.keys()) + ['Fe2O3', 'FeO']
    # calculate molecules' contributions to the 'd-term'
    common = [mol for mol in melt_strs if mol in list(d.keys())]

    A = a
    B = b / T
    C = c
    D = 0.
    for mol in common:
        if mol in list(oxides.keys()):
            D += d[mol] * oxides[mol] / N_melt
        elif mol == 'Fe2O3':
            D += d['Fe2O3'] * fe2o3 / N_melt
        elif mol == 'FeO':
            D += d['FeO'] * feo / N_melt
    E = e * (1 - (T_0 / T) - np.log(T / T_0))
    F = f * P / T
    G = g * (((T - T_0) * P) / T)
    H = h * P ** 2 / T

    return A, B, C, D, E, F, G, H


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# INITIAL CONDITIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def atmos_ejection(
    n_atmos: dict,
    m_imp: float, 
    d_imp: float, 
    vel_imp: float, 
    angle: Optional[bool] = False, 
    param: Optional[bool] = False, 
    ocean_erosion: Optional[bool] = False, 
    h2o_rat: Optional[float] = None,
) -> Tuple[float, dict]:
    """Calculate the atmospheric mass ejection as a result of the impact. 
    
    Uses the precription of Kegerreis+ (2020b). Assumes an Earth-like target.

    Args:
        n_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) number of moles of species in atmosphere.
        m_imp (float): Mass of the impactor, in units of 'kg'.
        d_imp (float): Radius of the impactor, in units of 'km'.
        vel_imp (float): Impact velocity, in units of 'km s-1'.
        angle (optional, float): Impact angle, in units of degrees. Can alternatively provide 'param'.
        param (optional, float): Impact parameter. Can alternatively provide 'angle'.
        ocean_erosion (optional, bool): Dictates whether the effects of the ocean are taken into consideration in atospheric erosion, in line with Genda and Abe (2005). If 'True', must also provide a value for 'h2o_rat'.
        h2o_rat  (optional, float): The mass ratio of the atmospheric H2O to the oceanic H2O on the target.

    Returns:
        X (float): Mass fraction of the atmosphere which is removed.
        n_kept (dict): Composition of the atmosphere remaining after ejection. Keys (str) full formulae of molecules. Values (float) number of moles of species in atmosphere.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Densities
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Earth
    vol_earth = ((4. / 3.) * np.pi * gC.r_earth ** 3.)   # [m3]
    rho_earth = gC.m_earth / vol_earth  # [kg m-3]

    # impactor
    r_imp = 1e3 * (0.5 * d_imp)  # [m]
    vol_imp = (4. / 3.) * np.pi * r_imp ** 3.  # [m3]
    rho_imp = m_imp / vol_imp  # [kg m-3]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Interacting Mass Fraction
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impact parameter
    if not angle:
        if not param:
            print("Must provide impact angle or parameter.")
            sys.exit()
        else:
            b = param
    else:
        b = np.sin(angle/180. * np.pi)

    # interacting height [m]
    d = (r_imp + gC.r_earth) * (1 - b)

    # volume of impactor cap [m3]
    vol_imp_cap = (np.pi/3.) * (d ** 2.) * (3. * r_imp - d)
    # volume of earth cap [m3]
    vol_earth_cap = (np.pi / 3.) * (d ** 2.) * (3. * gC.r_earth - d)

    # interacting mass fraction
    f = (rho_earth * vol_earth_cap + rho_imp * vol_imp_cap) / (rho_earth * vol_earth + rho_imp * vol_imp)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Calculate Atmosphere Mass Fraction Lost
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    v_esc = escape_velocity(m_imp, gC.m_earth, r_imp, gC.r_earth)  # [km s-1]

    # Kegerreis+ (2020b) prescription
    X = 0.64 * (
        (m_imp / gC.m_earth)**0.5 * 
        (rho_imp / rho_earth)**0.5 *
        (vel_imp / v_esc)**2. * 
        f
    ) ** 0.65

    # print("\nAtmos Fraction Removed (No Ocean Effect) = %.2f %%" % (X * 100.))

    if ocean_erosion:  # Genda and Abe (2005) prescription
        # atmos mass / ocean mass
        ratios = [1. / 300., 1. / 100., 1. / 30., 1. / 10., 1. / 3., 1.]
        ratios_log10 = [np.log10(item) for item in ratios]
        # % atmos mass loss
        X_atm = [36.7, 30.0, 21.1, 12.2, 4.3, 0.4]

        # best fit - linear regression
        [slope, intercept, r_val, _, _] = linregress(ratios_log10, X_atm)

        # estimated fraction of atmosphere lost
        X_percent = slope * np.log10(h2o_rat) + intercept
        X = X_percent / 100.

        print(f"\nAtmos Fraction Removed (+ Ocean Effect) = {X:.2e}")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Remove Atmosphere Moles
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    n_kept = {}
    for mol in list(n_atmos.keys()):
        M = gC.common_mol_mass[mol]  # [kg mol-1]
        old_mass = M * n_atmos[mol]  # [kg]

        new_mass = (1. - X) * old_mass  # [kg]
        n_kept[mol] = new_mass / M

    return [X, n_kept]


def atmos_init(
    mass_imp: float, 
    vel_imp: float, 
    init_ocean: float, 
    p_atmos: dict,
    temp: float, 
    fe_frac: float,
    sys_id: str, 
    imp_comp: Optional[str] = 'E', 
    display: Optional[bool] = False,
) -> Tuple[dict, dict, list, list]:
    """Calculate atmospheric composition after initial impact processes. 

    Predict the partial pressures and moles of H2, H2O, CO2, and N2 in the atmosphere after impact.
        - from the given partial pressures, we calculate the moles of each species in the atmosphere
        - we then carry out atmospheric erosion by mass ejection by the impactor, using the prescription of Kegerreis+ (2020b)
        - the impactor then vaporises the surface oceans, and is able to reduce some/all of this H2O to H2 using the iron in its core
        - FastChem is then used to find the equilibrium composition of such an atmosphere

    Args:
        mass_imp (float): Mass of the impactor, in units of 'kg'.
        vel_imp (float): Impact velocity, in units of 'km s-1'.
        init_ocean (float): Initial amount of water on the planet receiving impact, in units of Earth Oceans.
        p_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) partial pressure of each species, in units of 'bar'.
        temp (float): Temperature of the atmosphere before impact, in units of 'K'.
        fe_frac (float): Fraction of the impactor's iron inventory which is available to reduce the vaporised steam oceans.
        sys_id (str): Label of the atmosphere-melt system ('system_id'), used as file names
        imp_comp (optional, str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite,  'F': iron meteorite)
        display (optional, bool): Whether to print out results tables during calculations.

    Returns:
        p_atmos (dict): Update of input parameter 'p_atmos'.
        n_atmos (dict): Composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) number of moles of species in atmosphere.
        p_list (list):  Partial pressure dictionaries at each stage of the calculations, [initial, erosion, ocean vaporisation, impactor vaporisation, chemical equilibrium].
        n_list (list):  Molar composition dictionaries at each stage of the calculations, [initial, erosion, ocean vaporisation, impactor vaporisation, chemical equilibrium].

    """
    p_init = dcop(p_atmos)  # initial input atmosphere
    if display:
        print("\n--- --- --- --- ---")
        print(f"{'Mass of Impactor':20s} : {mass_imp:.2e} kg")
        print(f"{'Velocity of Impact':20s} : {vel_imp:.2f} km/s")
        print(f"{'Vaporised Oceans':20s} : {init_ocean:.2f} EO")
        print(f"{'pCO2':20s} : {p_init['CO2']:.2f} bar")
        print(f"{'pN2':20s} : {p_init['N2']:.2f} bar")

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

    if display:
        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            table_list.append([
                mol, 
                p_atmos[mol] / p_tot,
                p_atmos[mol] * 1e-5, 
                n_atmos[mol]
            ])
        print('\n')
        print('\x1b[1;34m*** Initial Atmosphere ***\x1b[0m')
        print(tabulate(
            table_list, 
            tablefmt='orgtbl', 
            headers=['Species', 'Mixing Ratio', 'Partial /bar', 'Moles'],
            floatfmt=("", ".5f", ".2f", ".2e")
        ))
        print(f"\n>>> Total Atmos Mass : {m_atm:.2e} kg")

    # relative mass of ocean and atmosphere
    r_mass = m_atm / (1.37e21 * init_ocean)  # [kg]

    # print(f">>> Atmos/Ocean : {r_mass:.2e}")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # atmospheric erosion by the impact
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impactor diameter [km]
    d_imp = impactor_diameter(mass_imp, imp_comp)

    [X_ejec, n_atmos] = atmos_ejection(
        n_atmos, 
        mass_imp, 
        d_imp, 
        vel_imp,
        param=0.7,
        h2o_rat=r_mass
    )

    # recalculate pressures
    [p_atmos, _] = update_pressures(n_atmos)

    p_erosion, n_erosion = dcop(p_atmos), dcop(n_atmos)

    if display:
        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            table_list.append([
                mol, p_atmos[mol] / p_tot,
                p_atmos[mol] * 1e-5, n_atmos[mol]
            ])
        print('\n')
        print('\x1b[1;33m*** After Erosion ***\x1b[0m')
        print(tabulate(
            table_list, 
            tablefmt='orgtbl', 
            headers=['Species', 'Mixing Ratio', 'Partial /bar', 'Moles'],
            floatfmt=("", ".5f", ".2f", ".2e")
        ))
        print(f"\n>>> Mass fraction of atmosphere removed : {X_ejec:.3f}")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # impactor vaporisation of volatiles
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    h2o_degas = vaporisation(mass_imp, imp_comp)

    if 'H2O' in list(n_atmos.keys()):
        n_atmos['H2O'] += h2o_degas
    else:
        n_atmos['H2O'] = h2o_degas

    # recalculate pressures
    [p_atmos, _] = update_pressures(n_atmos)

    p_degas, n_degas = dcop(p_atmos), dcop(n_atmos)

    if display:
        print(f"\n>>> Impactor Type : {imp_comp}")
        print(f">>> {'Moles of H2O Added':20s} : {h2o_degas:.2e}\n")

        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            table_list.append([
                mol,
                p_atmos[mol] / p_tot,
                p_atmos[mol] * 1e-5,
                n_atmos[mol]
            ])

        print('\x1b[1;34m*** After Volatiles from Mantles ***\x1b[0m')
        print(tabulate(
            table_list, 
            tablefmt='orgtbl',
            headers=['Species', 'Mixing Ratio', 'Partial /bar', 'Moles'],
            floatfmt=("", ".5f", ".2f", ".2e")
        ))

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # injection of H2O and H2 into the atmosphere (vaporisation and reducing)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # convert units of Earth Oceans into moles
    EO = 1.37e21  # [kg]
    EO_moles = EO / gC.common_mol_mass['H2O']  # [moles]
    init_h2o = init_ocean * EO_moles  # [moles]

    if display:
        print(f"\n>>>Iron fraction available to reduce oceans = {fe_frac:.3f}")

    # wt% of impactor mass is iron used to reduce oceans
    # Fe + H2O --> FeO + H2
    init_reduce_mass = fe_frac * gC.iron_wt[imp_comp] * mass_imp  # [kg]
    init_reduce_moles = init_reduce_mass / gC.common_mol_mass['Fe']  # [moles]

    if init_reduce_moles > init_h2o + n_atmos['CO2']:
        print("More Fe than H2O + CO2 for impactor mass = {mass_imp:.2e} kg")
        sys.exit()

    if init_reduce_moles > init_h2o:
        print("More Fe than H2O for impactor mass = {mass_imp:.2e} kg")
        sys.exit()
    else:
        # add H2O into the atmosphere
        if 'H2O' in list(n_atmos.keys()):
            n_atmos['H2O'] += init_h2o - init_reduce_moles
        else:
            n_atmos['H2O'] = init_h2o - init_reduce_moles

        # add H2 into the atmosphere
        if 'H2' in list(n_atmos.keys()):
            n_atmos['H2'] += init_reduce_moles
        else:
            n_atmos['H2'] = init_reduce_moles

    # recalculate pressures
    [p_atmos, _] = update_pressures(n_atmos)

    p_ocean, n_ocean = dcop(p_atmos), dcop(n_atmos)

    if display:
        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            table_list.append([
                mol, 
                p_atmos[mol] / p_tot,
                p_atmos[mol] * 1e-5, 
                n_atmos[mol]
            ])
        print('\n')
        print('\x1b[1;32m*** After Ocean Vaporisation & Iron Reduction ***\x1b[0m')
        print(tabulate(
            table_list,
            tablefmt='orgtbl',
            headers=['Species', 'Mixing Ratio', 'Partial /bar', 'Moles'],
            floatfmt=("", ".5f", ".2f", ".2e")
        ))

    # print(f"\n>>> log(fO2) of the atmosphere: {fo2_atm(n_atmos['H2'], n_atmos['H2O'], temp)}")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # FastChem Equilibrium Calculations
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    pre_fc = dcop(n_atmos)

    # abundances for FastChem
    abund = calc_elem_abund(n_atmos)

    # prepare FastChem config files
    write_fastchem(
        f"{dir_path}/data/FastChem/{sys_id}", 
        abund,
        temp, 
        float(np.sum(list(p_atmos.values())))
    )

    # run automated FastChem
    run_fastchem_files(sys_id)

    # read FastChem output
    [p_atmos, n_atmos] = read_fastchem_output(sys_id)

    p_chem, n_chem = dcop(p_atmos), dcop(n_atmos)

    if display:
        table_list = []
        p_tot = np.sum(list(p_atmos.values()))
        for mol in list(p_atmos.keys()):
            # mixing ratio
            mr = p_atmos[mol] / p_tot

            if mol in list(pre_fc.keys()):
                if mr > 1e-5:
                    table_list.append([
                        mol,
                        p_atmos[mol] / p_tot,
                        p_atmos[mol] * 1e-5, 
                        n_atmos[mol],
                        pre_fc[mol]
                    ])
            elif mol not in list(pre_fc.keys()):
                if mr > 1e-5:
                    table_list.append([
                        mol,
                        p_atmos[mol] / p_tot,
                        p_atmos[mol] * 1e-5, 
                        n_atmos[mol],
                        0.
                    ])

        print('\n')
        print('\x1b[1;31m*** After FastChem ***\x1b[0m')
        print(tabulate(
            table_list, 
            tablefmt='orgtbl',
            headers=['Species', 'Mixing Ratio', 'Partial /bar', 'Moles', 'Moles (pre)'],
            floatfmt=("", ".5f", ".2f", ".2e", ".2e")
        ))

        print(f"\n>>> Total atmospheric pressure : {1e-5 * np.sum(list(p_atmos.values())):.2f}")

    return p_atmos, n_atmos, [p_init, p_erosion, p_degas, p_ocean, p_chem], [n_init, n_erosion, n_degas, n_ocean, n_chem]


def available_iron(
    m_imp: float, 
    vel_imp: float, 
    angle: float, 
    imp_comp: str, 
    max_hse: Optional[bool] = False,
) -> Tuple[float, float, float]:
    """Determine where iron from the impactor ends up within the target.

    How much of the iron from the impactor core is made available to the atmosphere for the reduction of the vaporised surface oceans?

    NOTE: interpolation is only carried out as a function of impactor mass. In this version of the code, therefore, impact velocity and angles must be one of the modelled values (see exceptions).

    Args: 
        m_imp (float): Mass of the impactor, in units of 'kg'.
        vel_imp (float): Impact velocity, in units of 'km s-1'.
        angle (float): Impact angle, in units of degrees.
        imp_comp (str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite, 'F': iron meteorite)
        max_hse (optional, bool): Whether the maximum HSE impactor is calculated and displayed, as calculated from the iron distribution (i.e., scaled from 2e22 kg (Bottke+, 2010) by iron escaping the system).

    Returns:
        X_atm_out (float): Fraction of impactor iron accreted by the target atmosphere.
        X_int_out (float): Fraction of impactor iron accreted by the target interior.
        X_ej_out (float): Fraction of the impactor iron not accreted by the target.

    """
    # --- Checks --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    if angle not in [0., 30., 45., 60.]:
        print(">>> Given impact angle not simulated for iron distribution. Must be one of [0, 30, 45, 60] degrees.")
        sys.exit()

    r_imp = 1e3 * 0.5 * impactor_diameter(m_imp, imp_comp)  # [m]
    v_esc = escape_velocity(gC.m_earth, m_imp, gC.r_earth, r_imp)  # [km s-1]

    if vel_imp / v_esc not in [1.1, 1.5, 2.0]:
        print(">>> Given impact velocity not simulated for iron distribution. Must be one of [1.1, 1.5, 2.0] v_esc.")
        sys.exit()

    # --- Import Data --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
    masses, vels, thetas = [], [], []
    X_int, X_surf, X_atm, X_ejec = [], [], [], []

    impact_m_earth = 5.9127e+24

    with open(f"{dir_path}/data/iron_distributions.txt", "r") as file:
        count = -1
        for line in file:
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

    # --- Read Data into Structure --- --- --- --- --- --- --- --- --- --- ---
    completed = False
    interp_mass, interp_int, interp_atm, interp_ejec = [], [], [], []
    for i in range(len(masses)):
        # assume that we are using 2 v_esc as the impact velocity
        # HARDCODED - CHANGE IF NECESSARY
        if vels[i] == vel_imp / v_esc:
            completed = True
            # carry out interpolation for 45 degree impacts
            if thetas[i] == angle:
                interp_mass.append(masses[i])
                interp_int.append(X_int[i])
                interp_atm.append(X_atm[i])
                interp_ejec.append(X_ejec[i])

    # --- Fits --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit_int = np.polyfit(interp_mass, interp_int, 2)
    fit_atm = np.polyfit(interp_mass, interp_atm, 2)
    fit_ejec = np.polyfit(interp_mass, interp_ejec, 2)

    # --- Output --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    X_int_out = (fit_int[0] * m_imp ** 2) + (fit_int[1] * m_imp) + fit_int[2]

    X_atm_out = (fit_atm[0] * m_imp ** 2) + (fit_atm[1] * m_imp) + fit_atm[2]

    X_ej_out = (fit_ejec[0] * m_imp ** 2) + (fit_ejec[1] * m_imp) + fit_ejec[2]
    X_ej_out = max(0., X_ej_out)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if max_hse:
        # function to minimise
        def fe_accrete(m_imp, ejected):
            X_ejec_hse = (ejected[0] * m_imp ** 2) + (ejected[1] * m_imp) + ejected[2]
            X_ejec_hse = max(0., X_ejec_hse)

            return np.abs((1. - X_ejec_hse) * m_imp - 2e22)

        output = opt.minimize_scalar(
            fe_accrete, 
            args=fit_ejec, 
            tol=1e-5,
            method='bounded', 
            bounds=[2e21, 1e23]
        )
        print(f">>> Maximum HSE Impactor = {output.x:.3e} kg")

    return X_atm_out, X_int_out, X_ej_out


def basalt_comp_by_fo2(
    m_melt: float, 
    buffer: str, 
    relative: float, 
    init_comp: dict, 
    H2O_init: float, 
    P: float, 
    T: float,
) -> dict:
    """Create a basaltic melt with the given oxygen fugacity.
    
    Vary the composition of a given basaltic melt phase such that the oxygen fugacity is the input value relative to the input mineral buffer,

    Args: 
        m_elt (float): Mass of the melt phase, in units of 'kg'.
        buffer (str): Mineral buffer against which we are measuring fO2 (possible values: 'IW', 'FMQ').
        relative (float): Log units of fO2 above/below the stated mineral buffer.
        init_comp (dict): Initial composition of the melt phase. (Keys) strings of each molecule. (Values) wt% of each molecule.
        H2O_init (float): Initial water content of the melt melt phase, in units of 'wt%'.
        P (float): System pressure, in units of 'Pa'.
        T (float): System temperature, in units of 'K'.

    Returns:
        n (dict): Composition of the melt phase at the desired fO2. (Keys) strings of each molecule. (Values) moles of each molecule.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # add 'H2O_init' wt% H2O to the melt and scale old wt%
    wt = {'H2O': H2O_init}
    for mol in list(init_comp.keys()):
        if mol != 'H2O':
            wt[mol] = init_comp[mol] * ((100 - wt['H2O']) / 100)

    M = {}
    for mol in list(wt.keys()):
        M[mol] = gC.common_mol_mass[mol]  # [kg mol-1]

    for mol in ['Fe2O3', 'FeO']:
        if mol not in list(init_comp.keys()):
            init_comp[mol], wt[mol], M[mol] = 0., 0., gC.mol_phys_props(mol)[0]

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # calculate number of moles for each species
    n = {}  # [moles]
    for mol in list(wt.keys()):
        n[mol] = m_melt * wt[mol] * 1e-2 / M[mol]  # [moles]

    oxides = {}
    for mol in list(n.keys()):
        if mol not in ['Fe2O3', 'FeO', 'H2O']:
            oxides[mol] = n[mol]

    # current oxygen fugacity
    fO2 = fo2_kc91(n['Fe2O3'], n['FeO'], oxides, T, P)

    # oxygen fugacity we want to achieve
    if buffer.lower() == 'iw':
        if relative <= -2.:
            print(">>> Code not set up to start with metal-saturated impact-generate melt. Please start with melt fO2 > IW - 2.")
            sys.exit()
        else:
            comp_fO2 = fo2_iw(T, P) + relative
    elif buffer.lower() in ['fmq', 'fqm', 'qfm']:
        comp_fO2 = fo2_fmq(T, P) + relative
    else:
        print(">>> Not set up to take buffers other than IW or FMQ.")
        sys.exit()

    # if already at the user input melt phase fO2
    if np.abs(np.abs(fO2) - np.abs(comp_fO2)) < 1e-5:
        print(">>> Melt phase already produced at user input fO2.")
        return n

    # if more oxidised than the desired melt phase fO2
    elif fO2 > comp_fO2:
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # relax the melt phase down to input fO2 value
        # [set up in such a way that user cannot request fO2 <= IW - 2
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        complete_target = False  # have we reached the target fO2?

        Fe2O3, FeO = dcop(n['Fe2O3']), dcop(n['FeO'])  # unpack dictionary

        relax_frac = 0.09  # relaxation factor
        while not complete_target:
            # keep copies of variables which change
            Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)

            # make a step, reducing ferric to ferrous
            FeO += 2. * Fe2O3 * relax_frac
            Fe2O3 -= Fe2O3 * relax_frac

            # --- --- --- --- --- --- --- --- --- ---
            # ensure we don't use more Fe2O3 than we have
            if Fe2O3 < 0.:
                # reset to start of 'IW2' loop
                Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)

                # diminish frac of reducing power used
                relax_frac = 0.1 * relax_frac

                continue

            # --- --- --- --- --- --- --- --- --- ---
            # calculate fO2 of melt phase
            log_fO2_KC = fo2_kc91(Fe2O3, FeO, oxides, T, P)

            # if we've gone below the target fO2
            if log_fO2_KC < comp_fO2:
                # reset to start of 'IW2' loop
                Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)

                # diminish frac of reducing power used
                relax_frac = 0.1 * relax_frac

                continue

            # if we've reached the target fO2
            if np.abs(np.abs(log_fO2_KC) - np.abs(comp_fO2)) < 1e-5:
                complete_target = True
                # print(">>> System reached target fO2.")
                break

        # replace dictionary values
        n['Fe2O3'], n['FeO'] = Fe2O3, FeO

        return n

    # if more reduced than the desired melt phase fO2
    elif fO2 < comp_fO2:
        print('>>> Need to write melt phase oxidation.')
        sys.exit()


def peridotite_comp_by_fe_ratio(
    m_melt: float, 
    ratio: float, 
    init_comp: dict, 
    H2O_init: float,
) -> dict:
    """Create a peridotitic melt with the given oxygen fugacity.

    Varies the composition of a given melt phase such that the ferric-to-iron ratio is the desired value.

    Ars:
        m_melt (float): Mass of the melt phase, in units of 'kg'.
        ratio (float): Molar ratio of Fe2O3 to total Fe in melt phase (usually Fe2O3 + FeO).
        init_comp (dict): Initial composition of the melt phase. (Keys) strings of each molecule. (Values) wt% of each molecule.
        H2O_init (float): Initial water content of the melt melt phase, in units of 'wt%'.

    Returns:
        n (dict): Composition of the melt phase at the desired fO2. (Keys) strings of each molecule. (Values) moles of each molecule.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # add 'H2O_init' wt% H2O to the melt and scale old wt%
    wt = {'H2O': H2O_init}
    for mol in list(init_comp.keys()):
        if mol != 'H2O':
            wt[mol] = init_comp[mol] * ((100 - wt['H2O']) / 100)

    # calculate number of moles for each species
    n = {}  # [moles]
    for mol in list(wt.keys()):
        n[mol] = m_melt * wt[mol] * 1e-2 / gC.common_mol_mass[mol]

    # check that both Fe2O3 and FeO are present in composition
    irons = ['Fe2O3', 'FeO']
    for mol in irons:
        if mol not in list(init_comp.keys()):
            init_comp[mol], wt[mol], n[mol] = 0., 0., 0.

    # split iron between Fe2O3 and FeO according to initial condition 'ratio'
    total_iron = 2 * n['Fe2O3'] + n['FeO']
    fe2o3 = 0.5 * ratio * total_iron
    feo = (1. - ratio) * total_iron
    n['Fe2O3'], n['FeO'] = fe2o3, feo

    return n


def impactor_diameter(m_imp: float, imp_comp: str) -> float:
    """Calculate impactor diameter from its mass.

    An iron core and silicate mantle are determined by the given impactor composition. The core, and the body as a whole, are assumed to be spherical.

    Args:
        m_imp (float): Mass of impactor, in units of 'kg'.
        imp_comp (str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite, 'F': iron meteorite)

    Returns:
        d_imp (float): Diameter of impactor, in units of 'km'.

    """
    # check composition in mass fractions --- --- --- --- --- --- --- --- ---
    rho_forsterite = 3.27 * 1e3  # [kg m-3]
    rho_iron = 7.87 * 1e3  # [kg m-3]

    # size of core
    vol_core = gC.iron_wt[imp_comp] * m_imp / rho_iron  # [m3]
    r_core = (3. * vol_core / (4. * np.pi)) ** (1/3)  # [m]

    # size of mantle
    vol_mantle = (1. - gC.iron_wt[imp_comp]) * m_imp / rho_forsterite

    r_imp = (r_core ** 3. + (3. * vol_mantle / (4. * np.pi))) ** (1/3)

    return 2. * r_imp * 1e-3


def impact_melt_mass(m_imp: float, vel_imp: float, angle: float) -> float:
    """Calculate the mass of silicate melt phase generated in a given impact.
    
    Interpolation is carried as a function of modified specific impact energy.

    Args:
        m_imp (float): Mass of the impactor, in units of 'kg'.
        vel_imp (float): Impact velocity, in units of 'km s-1'.
        angle (float): Impact angle, in units of degrees.

    Returns:
        m_melt_out (float): Mass of impact-generated silicate melt phase, in units of 'kg'.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # CHECKS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if angle not in [0., 30., 45., 60.]:
        print(">>> Selected angle not simulated (melt mass calculations).")
        sys.exit()

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # DATA VALUES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    q_s_interp, mass_interp = [], []

    with open(f"{dir_path}/data/melt_masses.txt", "r") as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        next(data, None)
        next(data, None)
        for row in data:
            if len(row) in [14, 1]:
                continue
            if float(row[4]) != angle:
                continue

            # target and impactor masses
            m_t, m_i = float(row[1]), float(row[2])
            # target and impactor radii
            r_t = 0.5 * 1e3 * impactor_diameter(m_t, 'E')
            r_i = 0.5 * 1e3 * impactor_diameter(m_i, 'E')
            # mutual escape velocity
            v_esc = escape_velocity(m_t, m_i, r_t, r_i)
            # impact velocity
            v_imp = float(row[3]) * v_esc
            # impact angle
            theta = float(row[4])

            # modified specific energy of impact
            [Q_S, _] = specific_energy(m_t, m_i, v_imp, np.radians(theta))

            # forsterite reservoir masses
            M_MELT = float(row[6])  # melt in the planet + atmos
            M_SCF = float(row[7])  # supercritical fluid in the planet + atmos
            M_VAP = float(row[8])  # vapour in the planet + atmos (no disk or ejecta)
            M_SCF_ATM = float(row[9])  # supercritical fluid in the atmos
            M_ATM = float(row[10])  # all phases in the atmosphere
            M_DISC = float(row[11])  # all phases in the disk

            # what we count as planetary melt mass
            m_melt = M_MELT + M_SCF - M_SCF_ATM

            # what we count as vapour mass
            m_vap = M_VAP + M_SCF_ATM

            q_s_interp.append(Q_S)
            mass_interp.append(m_melt + m_vap)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # REGRESSION LINES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fit = np.polyfit(np.log10(q_s_interp), np.log10(mass_interp), 1)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # INTERPOLATION USING INPUT VALUES
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    [Q_S_in, _] = specific_energy(
        gC.m_earth, 
        m_imp, 
        vel_imp,
        np.sin(np.pi * angle / 180.)
    )

    m_melt_out = 10. ** (fit[0] * np.log10(Q_S_in) + fit[1])

    return m_melt_out


def vaporisation(m_imp: float, imp_comp: str) -> float:
    """
    Calculates the amount of water vapour degassed from the impactor's rocky
    mantle.

    Args:
        m_imp (float): Impactor mass, in units of 'kg'.
        imp_comp (str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite, 'F': iron meteorite)

    Returns:
        n_h2o (float): Moles of degassed water vapour.

    """
    types = ['C', 'L', 'H', 'E', 'F']
    if imp_comp.upper() not in types:
        print('Impactor composition not valid in vaporisation calculations.')
        sys.exit()

    # mass of water
    m_h2o = m_imp * (1 - gC.iron_wt[imp_comp]) * gC.h2o_wt[imp_comp]

    return m_h2o / gC.common_mol_mass['H2O']


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# REDOX CALCULATIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def add_iron_to_basalt(
    n_fe: float, 
    n_mo: dict, 
    T: float, 
    P: float, 
    tol: float,
) -> Tuple[dict, float, float]:
    """Add iron from the impactor core which did not interact with the atmosphere into the melt phase, and equilibrate.

    Args: 
        n_fe (float): Moles of iron leftover from the impactor that is to be sequesterd into the mantle.
        n_mo (dict): Composition of the bulk magma ocean silicate melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of each species in melt.
        T (float): Temperature of the system, in units of 'K'.
        P (float): Total pressure of the system, in units of 'Pa'.
        tol (float): Tolerance on moles of Fe designated as 'zero'.

    Returns:
        n_mo_new (dict): New composition of the bulk magma ocean silicate melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of each species in melt.
        Fe (float): Moles of metallic iron leftover after equilibration.
        m_melt (float): Mass of the melt phase after equilibration.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # if reducing of the melt phase using impactor iron does not
    # take the melt to below ΔIW-2, then the system is not metal
    # saturated, and the metal phase can be fully oxidised
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # copy of inputs
    n_mo_og, n_fe_og = dcop(n_mo), dcop(n_fe)

    # non-ferric metal oxides
    oxides = {}
    for mol in list(n_mo.keys()):
        if mol not in ['Fe2O3', 'FeO', 'H2O']:
            oxides[mol] = n_mo[mol]

    # attempt to use up all impactor iron in reducing the melt phase
    if n_mo['Fe2O3'] > n_fe:
        # Fe + Fe2O3 --> 3FeO
        n_mo['Fe2O3'] -= n_fe
        n_mo['FeO'] += 3. * n_fe
        n_fe = 0.

        # if redox does not take the melt packet fO2 below ΔIW-2, melt is
        # metal unsaturated, and no other treatment is needed
        fO2 = calc_basalt_fo2(n_mo['Fe2O3'], n_mo['FeO'], oxides, T, P)

        if fO2 - (fo2_iw(T, P) - 2) > tol:
            m_melt = 0.
            for mol in list(n_mo.keys()):
                m_melt += n_mo[mol] * gC.common_mol_mass[mol]

            return dcop(n_mo), n_fe, m_melt

        # if redox takes us to metal saturated, return to original state
        else:
            n_mo, n_fe = dcop(n_mo_og), dcop(n_fe_og)
    else:
        print(">>> More impactor iron than ferric iron in the melt phase.")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # carry out the melt phase reduction until the fO2 parametrisations
    # equal one another
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # unpack original values
    Fe2O3, FeO, Fe = dcop(n_mo['Fe2O3']), dcop(n_mo['FeO']), dcop(n_fe)

    complete, relax_frac = False, 0.07
    while not complete:
        # keep copies of system state before iteration
        Fe2O3_og, FeO_og, Fe_og = dcop(Fe2O3), dcop(FeO), dcop(Fe)

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # Fe + Fe2O3 --> 3FeO
        Fe2O3 -= Fe2O3 * relax_frac
        FeO += 3. * Fe2O3 * relax_frac
        Fe -= Fe2O3 * relax_frac

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # make sure we haven't used more impactor iron than available
        if Fe < 0.:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if we've used all of the available impactor iron
        if Fe < tol:
            complete = True
            print(">>> COMPLETE - used up iron.")
            break

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # make sure we haven't used more melt phase Fe2O3 than available
        if Fe2O3 < 0.:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if we've used all of the available melt phase Fe2O3
        if Fe2O3 < tol:
            complete = True
            print(">>> COMPLETE - used up ferric iron.")
            break

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # anhydrous melt moles
        N_melt_anhydrous = Fe2O3 + FeO + np.sum(list(oxides.values()))

        # melt phase fO2
        kc91_fo2 = fo2_kc91(Fe2O3, FeO, oxides, T, P)
        f91_fo2 = fo2_f91_rh12(FeO / N_melt_anhydrous, T, P)

        # make sure we haven't gone below where fO2 parameterisations are equal
        if kc91_fo2 < f91_fo2:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if fO2 parameterisations are equal
        if np.abs(kc91_fo2 - f91_fo2) < tol:
            complete = True
            print(">>> COMPLETE - reached fo2 equality.")
            break

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # reform values
    n_melt_out = dcop(n_mo)
    n_melt_out['Fe2O3'], n_melt_out['FeO'] = Fe2O3, FeO

    m_melt = 0.
    for mol in list(n_melt_out.keys()):
        m_melt += n_melt_out[mol] * gC.common_mol_mass[mol]

    return dcop(n_melt_out), Fe, m_melt


def add_iron_to_peridotite(
    n_fe: float, 
    n_mo: dict, 
    T: float, 
    P: float, 
    tol: float,
) -> Tuple[dict, float, float]:
    """Add iron from the impactor core which did not interact with the atmosphere into the melt phase, and equilibrate.

    Args:
        n_fe (float): Moles of iron leftover from the impactor that is to be sequestered into the mantle.
        n_mo (dict): Composition of the bulk magma ocean silicate melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of each species in melt.
        T (float): Temperature of the system, in units of 'K'.
        P (float): Total pressure of the system, in units of 'Pa'.
        tol (float): Tolerance on amount of Fe designated as 'zero'.

    Returns:
        n_mo_new (dict): New composition of the bulk magma ocean silicate melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of each species in melt.
        Fe (float): Moles of metallic iron leftover after equilibration.
        m_melt (float): Mass of the melt phase after equilibration.

    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # if oxidising all of the metal takes the melt phase above ΔIW-2,
    # then the system is no longer metal saturated, and the metal phase
    # can be fully oxidised.
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # copy of inputs
    n_mo_og, n_fe_og = dcop(n_mo), dcop(n_fe)

    # non-ferric metal oxides
    oxides = {}
    for mol in list(n_mo.keys()):
        if mol not in ['Fe2O3', 'FeO', 'H2O']:
            oxides[mol] = n_mo[mol]

    # attempt to use up all impactor iron in reducing the melt phase
    if n_mo['Fe2O3'] > n_fe:
        # Fe + Fe2O3 --> 3FeO
        n_mo['Fe2O3'] -= n_fe
        n_mo['FeO'] += 3. * n_fe
        n_fe = 0.

        # if redox does not take the melt packet fO2 below ΔIW-2, melt is
        # metal unsaturated, and no other treatment is needed
        fO2 = calc_peridotite_fo2(n_mo['Fe2O3'], n_mo['FeO'], oxides, T, P)

        if fO2 - (fo2_iw(T, P) - 2) > tol:
            m_melt = 0.
            for mol in list(n_mo.keys()):
                m_melt += n_mo[mol] * gC.common_mol_mass[mol]

            return dcop(n_mo), n_fe, m_melt

        # if redox takes us to metal saturated, return to original state
        else:
            n_mo, n_fe = dcop(n_mo_og), dcop(n_fe_og)
    else:
        print(">>> More impactor iron than ferric iron in the melt phase.")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # carry out the melt phase reduction until the fO2 parametrisations
    # equal one another
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # unpack original values
    Fe2O3, FeO, Fe = dcop(n_mo['Fe2O3']), dcop(n_mo['FeO']), dcop(n_fe)

    complete, relax_frac = False, 0.07
    while not complete:
        # keep copies of system state before iteration
        Fe2O3_og, FeO_og, Fe_og = dcop(Fe2O3), dcop(FeO), dcop(Fe)

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # Fe + Fe2O3 --> 3FeO
        Fe2O3 -= Fe2O3 * relax_frac
        FeO += 3. * Fe2O3 * relax_frac
        Fe -= Fe2O3 * relax_frac

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # make sure we haven't used more impactor iron than available
        if Fe < 0.:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if we've used all of the available impactor iron
        if Fe < tol:
            complete = True
            print(">>> COMPLETE - used up iron.")
            break

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # make sure we haven't used more melt phase Fe2O3 than available
        if Fe2O3 < 0.:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if we've used all of the available melt phase Fe2O3
        if Fe2O3 < tol:
            complete = True
            print(">>> COMPLETE - used up ferric iron.")
            break

        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # anhydrous melt moles
        N_melt_anhydrous = Fe2O3 + FeO + np.sum(list(oxides.values()))

        # melt phase fO2
        sossi_fo2 = fo2_sossi(Fe2O3, FeO, oxides, T, P)
        f91_fo2 = fo2_f91_rh12(FeO / N_melt_anhydrous, T, P)

        # make sure we haven't gone below where fO2 parameterisations are equal
        if sossi_fo2 < f91_fo2:
            # return to state at start of iteration
            Fe2O3, FeO, Fe = dcop(Fe2O3_og), dcop(FeO_og), dcop(Fe_og)

            # diminish frac of native iron used
            relax_frac = 0.1 * relax_frac

            continue

        # if fO2 parameterisations are equal
        if np.abs(sossi_fo2 - f91_fo2) < tol:
            complete = True
            print(">>> COMPLETE - reached fo2 equality.")
            break

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # reform values
    n_melt_out = dcop(n_mo)
    n_melt_out['Fe2O3'], n_melt_out['FeO'] = Fe2O3, FeO

    m_melt = 0.
    for mol in list(n_melt_out.keys()):
        m_melt += n_melt_out[mol] * gC.common_mol_mass[mol]

    return dcop(n_melt_out), Fe, m_melt


def equilibrate_H2O(
    fe2o3: float, 
    feo: float, 
    h2o_mag: float, 
    h2: float, 
    h2o_atm: float, 
    oxides: dict, 
    co2: float, 
    n2: float, 
    co: float,
    ch4: float, 
    nh3: float,
    m_melt: float, 
    tol: float
):
    """Dissolve H2O into the melt. 
    
    This will affect the fO2 of the atmosphere but not the fO2 of the melt under the current prescription of KC91.

    Args:
        fe2o3 (float): Moles of ferric iron in the melt phase.
        feo (float): Moles of ferrous iron in the melt phase.
        h2o_mag (float): Moles of H2O in the melt phase.
        h2 (float): Moles of H2 in the atmosphere.
        h2o_atm (float): Moles of H2O in the atmosphere.
        oxides (dict): Moles of non-iron species in the melt phase. Keys (str) full formulae of molecules. Values (float) number of moles of species in melt.
        co2 (float): Moles of CO2 in the atmosphere.
        n2 (float): Moles of N2 in the atmosphere.
        co (float): Moles of CO in the atmosphere.
        ch4 (float): Moles of CH4 in the atmosphere.
        nh3 (float): Moles of NH3 in the atmosphere.
        m_melt (float): Mass of melt packet, in units of 'kg'.
        tol (float): Tolerance on fO2 convergence (absolute).

    Returns:
        h2o_atm (float): Moles of H2O in the atmosphere at equilibrium.
        h2o_mag (float): Moles of H2O in the melt phase at equilibrium.
        m_melt (float):  Mass of melt packet at equilibrium, in units of 'kg'.
        
    """
    h2o_display = False

    # update pressures
    [_, P] = update_pressures({'H2': h2, 'H2O': h2o_atm, 'CO2': co2,
                               'N2': n2, 'CO': co, 'CH4': ch4, 'NH3': nh3})

    # total moles in the atmosphere
    N_atm = h2 + h2o_atm + co2 + n2 + co + ch4 + nh3

    # initial state
    XH2O = h2o_mag * gC.common_mol_mass['H2O'] / m_melt
    p_H2O_mag = (XH2O / 6.8e-8) ** (1. / 0.7)
    p_H2O_atm = P * h2o_atm / N_atm

    # modifies the amount of water vapour in the atmosphere we try to
    # dissolve at any one time (necessary as without modifier, sometimes the
    # melt tries to absorb more H2O than exists in the atmosphere)
    modifier = .07
    modifier_og = dcop(modifier)

    # track iterations
    complete, step, fails = False, 1, 0
    while not complete:
        # keep originals
        fe2o3_og, feo_og, h2o_mag_og = dcop(fe2o3), dcop(feo), dcop(h2o_mag)
        h2_og, h2o_atm_og = dcop(h2), dcop(h2o_atm)
        m_melt_og = dcop(m_melt)

        # total moles and pressure in the atmosphere
        N_atm = h2 + h2o_atm + co2 + n2 + co + ch4 + nh3

        [_, P] = update_pressures({'H2': h2, 'H2O': h2o_atm, 'CO2': co2,
                                   'N2': n2, 'CO': co, 'CH4': ch4, 'NH3': nh3})

        # mass fractions in melt packet
        m_frac_old = dcop({
            'H2O': h2o_mag * gC.common_mol_mass['H2O'] / m_melt,
            'Fe2O3': fe2o3 * gC.common_mol_mass['Fe2O3'] / m_melt,
            'FeO': feo * gC.common_mol_mass['FeO'] / m_melt
        })

        for mol in list(oxides.keys()):  # non-iron oxides
            M = gC.common_mol_mass[mol]  # [kg.mol-1]
            m_frac_old[mol] = oxides[mol] * M / m_melt

        # partial pressure of H2O in the atmosphere [Pa]
        p_H2O = P * h2o_atm / N_atm

        # predicted mass fraction of H2O in the melt (Lebrun+, 2013)
        m_frac_H2O = 6.8e-8 * (p_H2O ** 0.7)

        # change melt packet mass to account for new H2O
        m_melt_new = (m_melt * (1. - m_frac_old['H2O'])) / (1 - m_frac_H2O)

        # calculate new mass fractions in melt packet - scale the old ones
        m_frac_new = dcop({'H2O': m_frac_H2O})
        for mol in list(m_frac_old.keys()):
            if mol != 'H2O':
                m_frac_new[mol] = m_frac_old[mol] * m_melt / m_melt_new

        # moles of H2O dissolved into melt packet (can be negative)
        d_H2O = modifier * ((m_melt_new * m_frac_H2O / gC.common_mol_mass['H2O']) - h2o_mag)

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # H2O is Outgassed from the Magma
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        if d_H2O < 0.:
            # print(">>> H2O outgassed from the melt.")

            # is there enough H2O in the melt to outgas 'd_H2O'?
            # YES, degas H2O into atmosphere --- --- ---
            if h2o_mag - np.abs(d_H2O) > tol:
                h2o_mag -= np.abs(d_H2O)
                h2o_atm += np.abs(d_H2O)

                m_melt = fe2o3 * gC.common_mol_mass['Fe2O3']
                m_melt += feo * gC.common_mol_mass['FeO']
                m_melt += h2o_mag * gC.common_mol_mass['H2O']
                for mol in list(oxides.keys()):
                    m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # NO, we are H2O limited --- --- ---
            elif h2o_mag - np.abs(d_H2O) < tol:
                # revert to pre-iteration values
                fe2o3, feo = dcop(fe2o3_og), dcop(feo_og)
                h2o_mag, h2o_atm = dcop(h2o_mag_og), dcop(h2o_atm_og)
                m_melt = dcop(m_melt_og)

                # change modifier
                modifier = modifier * 0.5
                # return to start of while loop
                fails += 1

                continue

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # H2O is Drawn Down into the Magma
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        elif d_H2O > 0.:
            # print(">>> H2O drawn down into the melt.")

            # is there enough H2O in the atmosphere to draw down 'd_H2O'?
            # YES, dissolve H2O into melt --- --- ---
            if h2o_atm - d_H2O > tol:
                h2o_mag += d_H2O
                h2o_atm -= d_H2O

                m_melt = fe2o3 * gC.common_mol_mass['Fe2O3']
                m_melt += feo * gC.common_mol_mass['FeO']
                m_melt += h2o_mag * gC.common_mol_mass['H2O']
                for mol in list(oxides.keys()):
                    m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # NO, we are H2O limited --- --- ---
            elif h2o_atm - d_H2O < tol:
                # revert to pre-iteration values
                fe2o3, feo = dcop(fe2o3_og), dcop(feo_og)
                h2o_mag, h2o_atm = dcop(h2o_mag_og), dcop(h2o_atm_og)
                m_melt = dcop(m_melt_og)

                # change modifier
                modifier = modifier * 0.5
                # return to start of while loop
                fails += 1
                
                continue

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # Test for completion
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        [_, P] = update_pressures({
            'H2': h2, 
            'H2O': h2o_atm, 
            'CO2': co2,
            'N2': n2, 
            'CO': co, 
            'CH4': ch4, 
            'NH3': nh3
        })

        N_atm = h2 + h2o_atm + co2 + n2 + co + ch4 + nh3
        p_H2O_atm = P * h2o_atm / N_atm
        XH2O = h2o_mag * gC.common_mol_mass['H2O'] / m_melt
        p_H2O_mag = (XH2O / 6.8e-8) ** (1. / 0.7)

        stop = 99999
        if step > stop:
            print("Reached max. (" + str(stop) + ") dissolution steps.")
            sys.exit()

        if np.abs(p_H2O_atm - p_H2O_mag) < tol:
            if h2o_display:
                print("\x1b[36;1m>>> H2O Partitioning \x1b[0m")
                print(f">>> H2O (melt) Molar Fraction : {XH2O:.2e}")
                print(f">>>           New pH2O (melt) : {p_H2O_mag * 1e-5:.2f} bar")
                print(f">>>          New pH2O (atmos) : {p_H2O_atm * 1e-5:.2f} bar")
                print(f">>>           Melt Phase Mass : {m_melt:.2e}kg \n")

            return h2o_atm, h2o_mag, m_melt
        else:
            step += 1
            modifier = dcop(modifier_og)
            fails = 0


def eq_melt_basalt(
    m_imp: float, 
    v_imp: float, 
    theta: float, 
    imp_comp: str, 
    N_oceans: float, 
    init_atmos: dict, 
    wt_mo: dict,
    H2O_init: float, 
    buffer: str, 
    rel_fO2: float, 
    T: float,
    model_version: str, 
    partition: bool,
    chem: bool, 
    tol: float, 
    sys_id: str,
):
    """Equilibrate the atmosphere with a basaltic melt phase.

    Args:
        m_imp (float): Mass of impactor, in units of 'kg'. 
        v_imp (float): Impact velocity, in units of 'km s-1'.
        theta (float): Impact angle, in units of degrees.
        imp_comp (str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite, 'F': iron meteorite)
        N_oceans (float): Initial amount of water on the planet, in units of Earth Oceans.
        init_atmos (dict): Initial composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) partial pressure of each species [bar].
        wt_mo (dict): Composition of the body of melt. Keys (str) full formulae of molecules. Values (float) wt% of species in the body of melt.
        H2O_init (float): Initial water content of the melt phase, in units of 'wt%'.
        buffer (str):  Mineral buffer against which we are measuring fO2 (e.g., 'IW', 'FMQ').
        rel_fO2 (float): Log units of fO2 above/below the stated mineral buffer.
        T (float): Temperature, in units of 'K'.
        model_version (str): Which model version runs (e.g., 1A, 1B, 2, 3A, 3B).
        partition (bool): Whether H2O dissolution/outgassing takes place.
        chem (bool): Whether atmospheric chemistry takes place.
        tol (float): Tolerance in the difference between the atmosphere and the melt fO2 at completion (i.e., fo2_atmos * (1 +/- tol) in relation to fO2_melt)
        sys_id (str): Identifier for the system. Used in file labelling.

    Returns:
        trackers (list): Atmospheric abundances, fO2 & masses for the atmosphere and melt phase, total atmospheric pressure, and iron abundances, for all steps of the equilibration.
        P_LIST (list): Composition of the atmosphere throughout the initial conditions calculations. Each item in list is a dictionary similar to 'p_atmos'
        N_LIST (list): Composition of the atmosphere throughout the initial conditions calculations. Each item in list is a dictionary similar to 'n_atmos'

    """
    # --- Checks, Bools, and Trackers --- --- --- --- --- --- --- --- --- ---
    print("*** BASALT ***")

    # display values in command line as code proceeds?
    display = False

    # check for valid model version
    if model_version not in ['1A', '1B', '2', '3A', '3B']:
        print("\x1b[1;31m>>> Model version not valid. Please select one of [1A, 1B, 2, 3A, 3B].\x1b[0m")
        sys.exit()

    n_h2_track, n_h2o_atm_track = [], []
    n_co2_track, n_n2_track, n_co_track = [], [], []
    n_ch4_track, n_nh3_track = [], []
    n_h2o_mag_track, fe2o3_track, feo_track = [], [], []
    pressure_track = []
    fo2_atm_track, fo2_mag_track = [], []
    m_mag_track, m_atm_track, fe_track = [], [], []
    atm_moles_track, melt_moles_track = [], []

    # --- Iron Distribution --- --- --- --- --- --- --- --- --- --- --- --- --
    [X_fe_atm, X_fe_int, X_fe_ejec] = available_iron(
        m_imp, 
        v_imp, 
        theta, 
        imp_comp=imp_comp
    )

    # fiducial model iron distribution
    if model_version in ['1A', '2']:
        print(">>> FIDUCIAL IRON DISTRIBUTION ACTIVE.")
        X_fe_atm, X_fe_int = 1., 0.
    elif model_version in ['1B', '3B']:
        print(">>> NO IRON DEPOSITED IN MELT PHASE.")
        X_fe_int = 0.
    else:
        print(">>> ALL INTERIOR IRON DEPOSITED INTO MELT PHASE.")

    if display:
        print(f">>> Iron into the atmosphere : {X_fe_atm:.5f}")
        print(f">>> Iron into the interior : {X_fe_int:.5f}")
        print(f">>> Iron not accreted by target : {X_fe_ejec:.5f}")

    # --- Initial Atmospheric Composition --- --- --- --- --- --- --- --- ---
    [P_ATMOS_IC, N_ATMOS_IC, P_LIST, N_LIST] = atmos_init(
        m_imp, 
        v_imp, 
        N_oceans, 
        init_atmos, 
        T, 
        X_fe_atm,
        sys_id=sys_id, 
        imp_comp=imp_comp, 
        display=display
    )

    # select which species to include going forward
    n_atmos = {}
    for mol in list(P_ATMOS_IC.keys()):
        if mol in gC.gas_species:
            n_atmos[mol] = N_ATMOS_IC[mol]
        elif mol == 'H3N':
            n_atmos['NH3'] = N_ATMOS_IC[mol]
    for mol in gC.gas_species:
        if mol not in list(P_ATMOS_IC.keys()):
            n_atmos[mol] = 0.

    [p_atmos, _] = update_pressures(n_atmos)

    # total atmospheric pressure [Pa]
    p_tot = float(np.sum(list(p_atmos.values())))

    # starting oxygen fugacity of the atmosphere
    fo2_atmos = fo2_atm(n_atmos['H2'], n_atmos['H2O'], T)
    fo2_atmos_og = dcop(fo2_atmos)

    # --- Initial Magma Composition --- --- --- --- --- --- --- --- --- --- ---
    # mass of the melt [kg]
    m_mag = impact_melt_mass(m_imp, v_imp, 45.)
    m_melt_from_impact = dcop(m_mag)  # store melt phase mass for display

    # calculate moles from wt% prescription (includes adding H2O)
    n_melt = basalt_comp_by_fo2(
        m_mag, 
        buffer, 
        rel_fO2,
        wt_mo, 
        H2O_init,
        p_tot, 
        T
    )
    n_melt_from_impact = dcop(n_melt)  # store melt phase for display

    if model_version in ['2', '3A', '3B']:
        # partition iron from impactor between atmosphere and melt
        fe_total = gC.iron_wt[imp_comp] * m_imp / gC.common_mol_mass['Fe']
        # add FeO from the oxidised impactor iron into the initial melt phase
        feo_atmos = X_fe_atm * fe_total
        n_melt['FeO'] += feo_atmos

        # add in metallic iron from the unoxidised impactor iron
        fe_melt = X_fe_int * fe_total
        [n_melt, n_metal_bulk, m_mag] = add_iron_to_basalt(
            fe_melt, 
            dcop(n_melt), 
            T, 
            p_tot, 
            tol
        )

    else:
        # ensure variable creation
        n_metal_bulk = 0.

    if display:
        # starting oxygen fugacity of the melt
        melt_oxides = {}
        for mol in list(n_melt_from_impact.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                melt_oxides[mol] = n_melt_from_impact[mol]

        fO2_melt_imp = calc_basalt_fo2(
            n_melt_from_impact['Fe2O3'],
            n_melt_from_impact['FeO'],
            melt_oxides, 
            T, 
            p_tot
        )
        fO2_melt_comb = calc_basalt_fo2(
            n_melt['Fe2O3'], 
            n_melt['FeO'],
            melt_oxides, 
            T, 
            p_tot
        )

        fmq = fo2_fmq(T, p_tot)  # current FMQ fO2
        iw = fo2_iw(T, p_tot)  # current FMQ fO2

        print(f"\n>>> Total Atmospheric Presure : {p_tot * 1e-5:.2f} bar")

        print(f"\n>>> Impact-Generated Melt : {m_melt_from_impact:.5e} kg")
        print(f">>>                       : {m_melt_from_impact / gC.m_earth_mantle:.5f} Earth Mantle")
        print(f">>>     FeO Added to Melt : {feo_atmos:.5e} moles")
        print(f">>>         Combined Melt : {m_mag:.5e} kg")
        print(f">>>    Pre-Iron Melt ΔFMQ : {fO2_melt_imp - fmq:+.5f}")
        print(f">>>   Post-Iron Melt ΔFMQ : {fO2_melt_comb - fmq:+.5f}")
        print(f">>>                  ΔIW  : {fO2_melt_comb - iw:+.5f}")
        print(f">>>      Atmospheric ΔFMQ : {fo2_atmos_og - fmq:+.5f}")
        print(f">>>                  ΔIW  : {fo2_atmos_og - iw:+.5f}")

        print(f"\n>>>        Remaining Iron : {n_metal_bulk:.5e} moles")

    # keep iron mass separate from the mass of the silicate phase
    m_metal_bulk = n_metal_bulk * gC.common_mol_mass['Fe']

    # --- Equilibration Calculations --- --- --- --- --- --- --- --- --- --- -
    complete_full, step = False, -1
    while not complete_full:
        # --- Preparation --- --- --- --- ---
        # refresh (deepcopy) dictionaries to avoid pythonic issues
        n_melt, n_atmos = dcop(n_melt), dcop(n_atmos)
        p_tot, p_atmos = dcop(p_tot), dcop(p_atmos)

        step += 1
        n_termintate = 2000
        term_display = False
        if step == n_termintate:
            print(f"\x1b[1;31m>>> Equilibration terminated at {n_termintate} steps.\x1b[0m")

            if term_display:
                oxides = {}
                for mol in list(n_melt.keys()):
                    if mol not in ['Fe2O3', 'FeO', 'H2O']:
                        oxides[mol] = n_melt[mol]

                fO2_melt = calc_basalt_fo2(
                    n_melt['Fe2O3'],
                    n_melt['FeO'], 
                    oxides, 
                    T, 
                    p_tot
                )

                p_H2O_mag = calc_ph2o(n_melt['H2O'], m_mag)

                table_list = [
                    ['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                    ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, p_tot)) - np.abs(fO2_melt), 
                    np.abs(fo2_iw(T, p_tot)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]
                ]
                headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

                print('\n')
                print(f"\x1b[36;1m>>> Step {step}\x1b[0m")
                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
                ))
                print('\n')

                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                p_total = np.sum(list(p_atmos.values()))
                n_melt_total = np.sum(list(n_melt.values()))

                headers = ['Species', 'p [bar]', 'Mixing Ratio', 'Moles', 'Mole Fraction']
                table_list = []
                for mol in list(n_atmos.keys()):
                    table_list.append([
                        mol, 
                        p_atmos[mol] * 1e-5,
                        p_atmos[mol] / p_total,
                        n_atmos[mol], None
                    ])
                table_list.append([None, None, None, None, None])
                for mol in list(n_melt.keys()):
                    if mol == 'H2O':
                        table_list.append([
                            mol, 
                            p_H2O_mag * 1e-5, 
                            None,
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                    else:
                        table_list.append([
                            mol, 
                            None, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                table_list.append([None, None, None, None, None])
                table_list.append(['Fe', None, None, n_metal_bulk, None])

                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
                ))

            sys.exit()

            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST

        # --- display melt and atmosphere composition --- --- --- --- --- ---
        if display:
            oxides_disp = {}
            for mol in list(n_melt.keys()):
                if mol not in ['Fe2O3', 'FeO', 'H2O']:
                    oxides_disp[mol] = n_melt[mol]

            fO2_melt = calc_basalt_fo2(
                n_melt['Fe2O3'], 
                n_melt['FeO'],
                oxides_disp, 
                T, 
                p_tot
            )

            p_H2O_mag = calc_ph2o(n_melt['H2O'], m_mag)

            table_list = [
                ['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, p_tot)) - np.abs(fO2_melt),
                np.abs(fo2_iw(T, p_tot)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]
            ]
            headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

            print('\n')
            print(f"\x1b[36;1m>>> Step {step}\x1b[0m")
            print(tabulate(
                table_list, 
                tablefmt='orgtbl', 
                headers=headers,
                floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
            ))
            print('\n')

            # --- display partial pressures, mixing ratios and mole fractions
            p_total = np.sum(list(p_atmos.values()))
            n_melt_total = np.sum(list(n_melt.values()))

            headers = ['Species', 'p [bar]', 'Mixing Ratio', 'Moles', 'Mole Fraction']
            table_list = []
            for mol in list(n_atmos.keys()):
                table_list.append([
                    mol, 
                    p_atmos[mol] * 1e-5,
                    p_atmos[mol] / p_total,
                    n_atmos[mol] / (4.e4 * np.pi * gC.r_earth ** 2.), 
                    None
                ])
            table_list.append([None, None, None, None, None])
            for mol in list(n_melt.keys()):
                if mol == 'H2O':
                    table_list.append([
                        mol, 
                        p_H2O_mag * 1e-5, 
                        None,
                        n_melt[mol], 
                        n_melt[mol] / n_melt_total
                    ])
                else:
                    table_list.append([
                        mol, 
                        None, 
                        None, 
                        n_melt[mol],
                        n_melt[mol] / n_melt_total
                    ])
            table_list.append([None, None, None, None, None])
            table_list.append(['Fe', None, None, n_metal_bulk, None])

            print(tabulate(
                table_list, 
                tablefmt='orgtbl', 
                headers=headers,
                floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
            ))

            m_total = 0.
            for mol in list(n_atmos.keys()):
                m_total += n_atmos[mol] * gC.common_mol_mass[mol]
            for mol in list(n_melt.keys()):
                m_total += n_melt[mol] * gC.common_mol_mass[mol]

            print(f"\n>>> Total Mass in System : {m_total:.15e} \n")

        # --- fO2 Equilibration & Dissolving --- --- --- --- --- --- --- --- --
        # unpack Species (avoids issues with dictionaries)
        FeO, Fe2O3 = dcop(n_melt['FeO']), dcop(n_melt['Fe2O3'])
        H2O_mag = dcop(n_melt['H2O'])
        oxides = {}
        for mol in list(n_melt.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                oxides[mol] = n_melt[mol]

        N_oxides = np.sum(list(oxides.values()))

        Fe = dcop(n_metal_bulk)

        H2O_atm, H2 = dcop(n_atmos['H2O']), dcop(n_atmos['H2'])
        CO2, N2 = dcop(n_atmos['CO2']), dcop(n_atmos['N2'])
        CO, CH4 = dcop(n_atmos['CO']), dcop(n_atmos['CH4'])
        NH3 = dcop(n_atmos['NH3'])

        # --- Initial Conditions --- --- --- --- --- --- --- --- --- --- --- -
        # total moles in the melt
        N_melt = Fe2O3 + FeO + H2O_mag + N_oxides

        # total mass in melt phase
        m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
        m_melt += FeO * gC.common_mol_mass['FeO']
        m_melt += H2O_mag * gC.common_mol_mass['H2O']
        for mol in list(oxides.keys()):
            m_melt += oxides[mol] * gC.common_mol_mass[mol]

        # total mass in atmosphere
        m_atm = H2 * gC.common_mol_mass['H2']
        m_atm += H2O_atm * gC.common_mol_mass['H2O']
        m_atm += CO2 * gC.common_mol_mass['CO2']
        m_atm += N2 * gC.common_mol_mass['N2']
        m_atm += CO * gC.common_mol_mass['CO']
        m_atm += CH4 * gC.common_mol_mass['CH4']
        m_atm += NH3 * gC.common_mol_mass['NH3']

        # pressures
        [_, P] = update_pressures({
            'H2': H2, 
            'H2O': H2O_atm, 
            'CO2': CO2,
            'N2': N2, 
            'CO': CO, 
            'CH4': CH4,
            'NH3': NH3
        })
        # fugacities
        fo2_atmos = fo2_atm(H2, H2O_atm, T)
        fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)

        # --- Tracking --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # atmosphere
        n_h2_track.append(H2)
        n_h2o_atm_track.append(H2O_atm)
        n_co2_track.append(CO2)
        n_n2_track.append(N2)
        n_co_track.append(CO)
        n_ch4_track.append(CH4)
        n_nh3_track.append(NH3)
        pressure_track.append(P)
        fo2_atm_track.append(fo2_atmos)
        m_atm_track.append(m_atm)

        # melt phase
        n_h2o_mag_track.append(H2O_mag)
        fe2o3_track.append(Fe2O3)
        feo_track.append(FeO)
        fo2_mag_track.append(fO2_melt)
        m_mag_track.append(m_melt)

        # total moles
        atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
        melt_moles_track.append(N_melt)

        # metal phase
        fe_track.append(Fe)

        if model_version in ['1A', '1B']:
            print(">>> RETURNED WITH NO MELT-ATMOSPHERE INTERACTIONS.")
            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST

        # --- Checks --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
        if np.isnan(fo2_atmos) or np.isnan(fO2_melt):
            print(">>> NaN encountered.")
            sys.exit()

        # --- Reduce the Melt Phase --- --- --- --- --- --- --- --- --- --- ---
        if fO2_melt > fo2_atmos:
            if display:
                print("\x1b[1;31m>>> Atmosphere is more reduced than melt."
                      "\x1b[0m")

            complete_fO2 = False

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # relax the system down to fO2 = ΔIW-2 by reducing only Fe2O3
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            if fO2_melt - (fo2_iw(T, P) - 2) > tol:
                complete_IW2 = False  # have we reached ΔIW-2?

                relax_frac = 0.7  # relaxation factor
                while not complete_IW2:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # are we H2-limited or Fe2O3-limited
                    if H2 < Fe2O3:
                        delta = H2
                    else:
                        delta = Fe2O3

                    # make a step, reducing ferric to ferrous
                    FeO += 2 * delta * relax_frac
                    H2 -= delta * relax_frac
                    H2O_atm += delta * relax_frac
                    Fe2O3 -= delta * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use more Fe2O3 than we have
                    if Fe2O3 < 0.:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # calculate fO2 of melt phase
                    log_fO2_KC = fo2_kc91(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of IW buffer
                    log_fo2_iw = fo2_iw(T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching ΔIW-2
                    if np.abs(np.abs(log_fO2_KC) - np.abs(log_fo2_atm)) < tol:
                        complete_IW2 = True
                        complete_fO2 = True
                        print(">>> System reached equilibrium before IW-2.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've reached ΔIW-2
                    if np.abs(np.abs(log_fO2_KC) -
                              np.abs(log_fo2_iw - 2)) < tol:
                        complete_IW2 = True
                        print(">>> System reached IW-2.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've gone too far below ΔIW-2
                    if log_fO2_KC < log_fo2_iw - 2:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've got a melt more reduced than atmosphere
                    if log_fO2_KC < log_fo2_atm:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # relax the system down to where the parametrisations of 'KC91'
            # and 'F91_adj' predict equal fO2 values, reducing Fe2O3 & FeO
            # simultaneously
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # total anhydrous moles in melt
            N_melt = Fe2O3 + FeO + N_oxides
            # fO2 calculated by KC91 prescription
            KC91 = fo2_kc91(Fe2O3, FeO, oxides, T, P)
            # fO2 calculated by Frost 91 prescription
            F91 = fo2_f91_rh12(FeO / N_melt, T, P)

            # check we are above the equality point
            if KC91 - F91 > tol and not complete_fO2:
                complete_equal = False  # have we reached equal fO2

                relax_frac = 0.07
                while not complete_equal:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # --- --- --- --- --- --- --- --- --- ---
                    # change in Fe2O3
                    d_Fe2O3 = - H2 * relax_frac

                    # ensure we don't have ΔFe2O3 > Fe2O3
                    if np.abs(d_Fe2O3) > Fe2O3:
                        # reset to start of 'equal' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # update pressures
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # total moles in melt
                    N_melt = Fe2O3 + FeO + H2O_mag + N_oxides

                    # molar fractions
                    x_fe2o3, x_feo = Fe2O3 / N_melt, FeO / N_melt

                    # 'a' coefficient from KC91 prescription
                    a_KC91 = 0.196

                    # coefficient linking change in X_FeO to change in X_Fe2O3
                    epsilon = (x_feo / x_fe2o3) * (1. + 1.828 * x_fe2o3) / (1. + 2. * a_KC91 - 1.828 * x_feo)

                    # total change in FeO (combined FQM and IW reactions)
                    zeta = (N_melt - Fe2O3 + epsilon * FeO) / (Fe2O3 + epsilon * (N_melt - FeO))

                    # execute changes in moles
                    Fe2O3 += d_Fe2O3
                    FeO += zeta * d_Fe2O3
                    Fe += - (2. + zeta) * d_Fe2O3
                    H2 += (3. + zeta) * d_Fe2O3
                    H2O_atm += - (3. + zeta) * d_Fe2O3

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'equal' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # total anhydrous moles in the melt
                    N_melt = Fe2O3 + FeO + N_oxides

                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # fO2 calculated by KC91 prescription
                    log_fO2_KC = fo2_kc91(Fe2O3, FeO, oxides, T, P)
                    # fO2 calculated by Frost 91 prescription
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching prescription equality
                    if np.abs(np.abs(log_fO2_KC) - np.abs(log_fo2_atm)) < tol:
                        complete_fO2 = True
                        print(">>> System equilibrated before prescription"
                              " equality.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've reached fO2 equality
                    if np.abs(np.abs(log_fO2_KC) - np.abs(log_fO2_F91)) < tol:
                        complete_equal = True
                        print(">>> System reached KC91 = F91.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've gone too far below fO2 equality
                    if log_fO2_KC < log_fO2_F91:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've got a melt more reduced than atmosphere
                    if log_fO2_KC < log_fo2_atm:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # reduce FeO to Fe only until we reach melt-atmosphere equilibrium
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            if not complete_fO2:
                # have we reached zero FeO
                complete_FeO = False

                relax_frac = 0.07
                while not complete_FeO:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO -= H2 * relax_frac
                    H2 -= H2 * relax_frac
                    Fe += H2 * relax_frac
                    H2O_atm += H2 * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more FeO than we have
                    if FeO < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # total anhydrous moles in the melt
                    N_melt = Fe2O3 + FeO + N_oxides

                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # fO2 calculated by Frost 91 prescription
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # ensure we don't go too far in fO2 space
                    if log_fO2_F91 < log_fo2_atm:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    if np.abs(np.abs(log_fO2_F91) - np.abs(log_fo2_atm)) < tol:
                        complete_fO2 = True
                        print(">>> System equilibrated before zero FeO.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've reached effectively no more reducing power
                    if H2 < tol:
                        complete_fO2 = True
                        print(">>> No more reducing power.")
                        sys.exit()

                    # if we've reached effectively no more FeO in the melt
                    if FeO < tol:
                        complete_fO2 = True
                        print(">>> No more ferrous iron.")
                        sys.exit()

        # --- Oxidise the Melt Phase --- --- --- --- --- --- --- --- --- --- --
        if fO2_melt < fo2_atmos:
            # print("\x1b[1;36m>>> Atmosphere is more oxidised than melt."
            #       "\x1b[0m")

            complete_fO2 = False

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # if we are more reduced than IW-2, oxidise Fe to FeO only
            # until KC91 and F91 fO2 parametrisations are equal
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # anhydrous moles in melt
            N_melt = Fe2O3 + FeO + N_oxides

            # update atmospheric pressure
            [_, P] = update_pressures({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                'N2': N2, 
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            })

            log_fo2_kc91 = fo2_kc91(Fe2O3, FeO, oxides, T, P)
            log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)

            if log_fO2_F91 > log_fo2_kc91 and not complete_fO2:
                # print("\n\x1b[1;35m>>> Below fO2 parameterisation equality."
                #       "\x1b[0m")

                complete_equal = False
                relax_frac = 0.07

                while not complete_equal:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)
                    Fe_og = dcop(Fe)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO += Fe * relax_frac
                    H2 += Fe * relax_frac
                    Fe -= Fe * relax_frac
                    H2O_atm -= Fe * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use up more Fe than we have
                    if Fe < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more atmospheric H2O than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    log_fo2_kc91 = fo2_kc91(Fe2O3, FeO, oxides, T, P)
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)

                    # if we've gone too far above fO2 equality
                    if log_fo2_kc91 > log_fO2_F91:
                        # reset to start of loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of oxidising power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached fO2 equality
                    if np.abs(np.abs(log_fo2_kc91) - np.abs(log_fO2_F91)) < tol:
                        complete_equal = True
                        print(">>> System reached KC91 = F91.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # calculate fO2 of the melt phase
                    fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    fo2_atmos = fo2_atm(H2, H2O_atm, T)

                    # if we've got a melt more oxidised than atmosphere
                    if fO2_melt > fo2_atmos:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching ΔIW-2
                    if np.abs(np.abs(fO2_melt) - np.abs(fo2_atmos)) < tol:
                        complete_metal = True
                        complete_fO2 = True
                        print(">>> System equilibrated before KC91 = F91.")
                        break

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # if we are still more reduced than IW-2, oxidise Fe to FeO
            # and FeO to Fe2O3 simultaneously until we reach IW-2
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # update atmospheric pressure
            [_, P] = update_pressures({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                'N2': N2, 
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            })
            # melt phase fO2
            fo2_melt = fo2_kc91(Fe2O3, FeO, oxides, T, P)

            if fo2_melt < fo2_iw(T, P) - 2. and not complete_fO2:
                # print("\x1b[1;35m>>> Below IW-2.\x1b[0m")

                # try to get rid of all iron
                complete_IW, relax_frac = False, 1.0
                while not complete_IW:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)
                    Fe_og = dcop(Fe)

                    # --- --- --- --- --- --- --- --- --- ---
                    # change in Fe2O3
                    d_Fe = - Fe * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # total moles in melt
                    N_melt = Fe2O3 + FeO + H2O_mag + N_oxides

                    # molar fractions
                    x_fe2o3, x_feo = Fe2O3 / N_melt, FeO / N_melt

                    # 'a' coefficient from KC91 prescription
                    a_KC91 = 0.196

                    # coefficient linking change in X_FeO to change in X_Fe2O3
                    epsilon = (x_feo / x_fe2o3) * (1. + 1.828 * x_fe2o3) / (1. + 2. * a_KC91 - 1.828 * x_feo)

                    # total change in FeO (combined FQM and IW reactions)
                    zeta = (N_melt - Fe2O3 + epsilon * FeO) / (Fe2O3 + epsilon * (N_melt - FeO))

                    # execute changes in moles
                    Fe2O3 += - 1. / (2. + zeta) * d_Fe
                    FeO += - zeta / (2. + zeta) * d_Fe
                    Fe += d_Fe
                    H2 += - (3. + zeta) / (2. + zeta) * d_Fe
                    H2O_atm += (3. + zeta) / (2. + zeta) * d_Fe

                    # --- --- --- --- --- --- --- --- --- ---
                    # update atmospheric pressure
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm,
                        'CO2': CO2, 
                        'N2': N2, 
                        'CO': CO,
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # ensure we don't use up more atmospheric H2O than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more Fe than we have
                    if Fe < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # calculate fO2 of the melt phase
                    fO2_melt = fo2_kc91(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    fo2_atmos = fo2_atm(H2, H2O_atm, T)

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've got a melt more oxidised than atmosphere
                    if fO2_melt > fo2_atmos:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching ΔIW-2
                    if np.abs(np.abs(fO2_melt) - np.abs(fo2_atmos)) < tol:
                        complete_IW = True
                        complete_fO2 = True
                        print(">>> System equilibrated before IW-2.")
                        break

                    # if rid of all iron
                    if Fe < tol:
                        complete_IW = True
                        break

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # we should be at or more oxidised than IW-2
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            if not complete_fO2:
                # print("\x1b[1;35m>>> Above IW-2.\x1b[0m")

                complete_ox = False
                relax_frac = 0.00001
                while not complete_ox:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO -= 2. * H2O_atm * relax_frac
                    H2 += H2O_atm * relax_frac
                    Fe2O3 += H2O_atm * relax_frac
                    H2O_atm -= H2O_atm * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more oxidising power than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more FeO than we have
                    if FeO < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # fO2 calculated by Frost 91 prescription
                    log_fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # if we've got a atmosphere more reduced than melt
                    if log_fO2_melt > log_fo2_atm:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    if np.abs(np.abs(log_fO2_melt) - np.abs(log_fo2_atm)) < tol:
                        complete_ox = True
                        complete_fO2 = True
                        break

        # --- H2O Dissolution --- --- --- --- --- --- --- --- --- --- --- --- -
        if partition:
            # print(">>> Carrying out H2O partitioning.")

            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # metal phase
            fe_track.append(Fe)

            # --- Partitioning --- --- ---
            [H2O_atm, H2O_mag, m_mag] = equilibrate_H2O(
                Fe2O3, 
                FeO, 
                H2O_mag, 
                H2, 
                H2O_atm, 
                oxides,
                CO2, 
                N2, 
                CO, 
                CH4, 
                NH3, 
                m_mag, 
                tol
            )

            # reform dictionaries
            n_atmos = {
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2, 
                'N2': N2,
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            }

        # --- H2O Dissolution --- --- --- --- --- --- --- --- --- --- --- --- -
        if chem:
            # print(">>> Enforcing atmospheric thermochemical equilibrium.")

            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # metal phase
            fe_track.append(Fe)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # abundances for FastChem
            abund = calc_elem_abund({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                'N2': N2, 
                'CO': CO, 
                'CH4': CH4,
                'NH3': NH3
            })
            # prepare FastChem config files
            new_id = f"{sys_id}inside_{step}"
            write_fastchem(f"{dir_path}/data/FastChem/{new_id}", abund, T, P)
            # run automated FastChem
            run_fastchem_files(new_id)
            # read FastChem output
            [_, n_atmos_chem] = read_fastchem_output(new_id)

            # unpack dictionary
            n_atmos = {}
            for mol in list(n_atmos_chem.keys()):
                if mol in gC.gas_species:
                    n_atmos[mol] = n_atmos_chem[mol]
                elif mol == 'H3N':
                    n_atmos['NH3'] = n_atmos_chem[mol]
            for mol in gC.gas_species:
                if mol not in list(n_atmos_chem.keys()):
                    n_atmos[mol] = 0.

            # print(">>> Atmospheric chemistry complete.")

        # --- Test for Convergence --- --- --- --- --- --- --- --- --- --- ---
        if not chem and not partition:
            # print(">>> No partitioning or thermochemistry taking place.")

            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # metal phase
            fe_track.append(Fe)

            # recreate atmosphere dictionary
            n_atmos = {
                'H2': H2, 
                'H2O': H2O_atm,
                'CO2': CO2, 
                'N2': N2,
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            }

        # update pressures
        [p_atmos, P] = update_pressures(n_atmos)

        # oxygen fugacities
        fO2_melt = calc_basalt_fo2(Fe2O3, FeO, oxides, T, P)
        fo2_atmos = fo2_atm(n_atmos['H2'], n_atmos['H2O'], T)

        n_melt = dcop(oxides)
        n_melt['Fe2O3'] = Fe2O3
        n_melt['FeO'] = FeO
        n_melt['H2O'] = H2O_mag

        n_metal_bulk = dcop(Fe)

        # YES - finished
        if np.abs(fO2_melt - fo2_atmos) < tol:
            # --- --- --- --- --- ---
            if display:
                m_melt = 0.  # [kg]
                for mol in list(n_melt.keys()):
                    m_melt += n_melt[mol] * gC.common_mol_mass[mol]

                p_H2O_mag = calc_ph2o(n_melt['H2O'], m_melt)

                table_list = [['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                              ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, P)) - np.abs(fO2_melt),
                               np.abs(fo2_iw(T, P)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]]
                headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

                print('\n')
                print('\x1b[36;1m>>> Terminus Step\x1b[0m')
                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
                ))
                print('\n')

                # display partial pressures, mixing ratios and mole fractions
                n_melt_total = np.sum(list(n_melt.values()))

                headers = ['Species', 'p [bar]', 'Mixing Ratio', 'Moles', 'Mole Fraction']
                table_list = []
                for mol in list(n_atmos.keys()):
                    table_list.append([
                        mol, 
                        p_atmos[mol] * 1e-5, 
                        p_atmos[mol] / P,
                        n_atmos[mol] / (4.e4 * np.pi * gC.r_earth ** 2.),
                        None
                    ])
                table_list.append([None, None, None, None, None])
                for mol in list(n_melt.keys()):
                    if mol == 'H2O':
                        table_list.append([
                            mol, 
                            p_H2O_mag * 1e-5, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                    else:
                        table_list.append([
                            mol, 
                            None, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                table_list.append([None, None, None, None, None])
                table_list.append(['Fe', None, None, n_metal_bulk, None])

                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
                ))

                m_total = 0.
                for mol in list(n_atmos.keys()):
                    m_total += n_atmos[mol] * gC.common_mol_mass[mol]
                for mol in list(n_melt.keys()):
                    m_total += n_melt[mol] * gC.common_mol_mass[mol]

                print(f"\n>>> Total Mass in System : {m_total:.15e} \n")

            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST
        else:
            continue


def eq_melt_peridotite(
    m_imp: float, 
    v_imp: float, 
    theta: float, 
    imp_comp: str, 
    N_oceans: float, 
    init_atmos: dict,
    wt_mo: dict, 
    H2O_init: float, 
    iron_ratio: float, 
    T: float, 
    model_version: str,
    partition: bool, 
    chem: bool, 
    tol: float, 
    sys_id: str,
) -> Tuple[list, list, list]:
    """Equilibrate the atmosphere with a peridotitic melt phase.

    Args:
        m_imp (float): Mass of impactor, in units of 'kg'. 
        v_imp (float): Impact velocity, in units of 'km s-1'.
        theta (float): Impact angle, in units of degrees.
        imp_comp (str): Impactor composition indicator ('C': carbonaceous chondrite, 'L': ordinary (L) chondrite, 'H': ordinary (H) chondrite, 'E': enstatite chondrite, 'F': iron meteorite)
        N_oceans (float): Initial amount of water on the planet, in units of Earth Oceans.
        init_atmos (dict): Initial composition of the atmosphere. Keys (str) full formulae of molecules. Values (float) partial pressure of each species [bar].
        wt_mo (dict): Composition of the body of melt. Keys (str) full formulae of molecules. Values (float) wt% of species in the body of melt.
        H2O_init (float): Initial water content of the melt phase, in units of 'wt%'.
        iron_ratio (float): Molar ratio of Fe2O3 to total iron in melt phase (usually Fe2O3 + FeO).
        T (float): Temperature, in units of 'K'.
        model_version (str): Which model version runs (e.g., 1A, 1B, 2, 3A, 3B).
        partition (bool): Whether H2O dissolution/outgassing takes place.
        chem (bool): Whether atmospheric chemistry takes place.
        tol (float): Tolerance in the difference between the atmosphere and the melt fO2 at completion (i.e., fo2_atmos * (1 +/- tol) in relation to fO2_melt)
        sys_id (str): Identifier for the system. Used in file labelling.

    Returns:
        trackers (list): Atmospheric abundances, fO2 & masses for the atmosphere and melt phase, total atmospheric pressure, and iron abundances, for all steps of the equilibration.
        P_LIST (list): Composition of the atmosphere throughout the initial conditions calculations. Each item in list is a dictionary similar to 'p_atmos'
        N_LIST (list): Composition of the atmosphere throughout the initial conditions calculations. Each item in list is a dictionary similar to 'n_atmos'

    """
    # --- Checks, Bools, and Trackers --- --- --- --- --- --- --- --- --- ---
    print("*** PERIDOTITE ***")

    # display values in command line as code proceeds?
    display = False

    # check for valid model version
    if model_version not in ['1A', '1B', '2', '3A', '3B']:
        print("\x1b[1;31m>>> Model version not valid. Please select one of"
              "[1A, 1B, 2, 3A, 3B].\x1b[0m")
        sys.exit()

    n_h2_track, n_h2o_atm_track = [], []
    n_co2_track, n_n2_track, n_co_track = [], [], []
    n_ch4_track, n_nh3_track = [], []
    n_h2o_mag_track, fe2o3_track, feo_track = [], [], []
    pressure_track = []
    fo2_atm_track, fo2_mag_track = [], []
    m_mag_track, m_atm_track, fe_track = [], [], []
    atm_moles_track, melt_moles_track = [], []

    # --- Iron Distribution --- --- --- --- --- --- --- --- --- --- --- --- --
    [X_fe_atm, X_fe_int, X_fe_ejec] = available_iron(
        m_imp,
        v_imp, 
        theta, 
        imp_comp=imp_comp
    )

    # fiducial model iron distribution
    if model_version in ['1A', '2']:
        print(">>> FIDUCIAL IRON DISTRIBUTION ACTIVE.")
        X_fe_atm, X_fe_int = 1., 0.
    elif model_version in ['1B', '3B']:
        print(">>> NO IRON DEPOSITED IN MELT PHASE.")
        X_fe_int = 0.
    else:
        print(">>> ALL INTERIOR IRON DEPOSITED INTO MELT PHASE.")

    if display:
        print(f">>> Iron into the Atmosphere : {X_fe_atm:.5f}")
        print(f">>> Iron into the Melt Phase : {X_fe_int:.5f}")
        print(f">>>     Iron Escaping System : {X_fe_ejec:.5f}")

    # --- Initial Atmospheric Composition --- --- --- --- --- --- --- --- ---
    [P_ATMOS_IC, N_ATMOS_IC, P_LIST, N_LIST] = atmos_init(
        m_imp,
        v_imp, 
        N_oceans, 
        init_atmos, 
        T,
        X_fe_atm,
        sys_id=sys_id, 
        imp_comp=imp_comp, 
        display=display
    )

    # select which species to include going forward
    n_atmos = {}
    for mol in list(P_ATMOS_IC.keys()):
        if mol in gC.gas_species:
            n_atmos[mol] = N_ATMOS_IC[mol]
        elif mol == 'H3N':
            n_atmos['NH3'] = N_ATMOS_IC[mol]
    for mol in gC.gas_species:
        if mol not in list(P_ATMOS_IC.keys()):
            n_atmos[mol] = 0.

    [p_atmos, _] = update_pressures(n_atmos)

    # total atmospheric pressure [Pa]
    p_tot = float(np.sum(list(p_atmos.values())))

    # starting oxygen fugacity of the atmosphere
    fo2_atmos = fo2_atm(n_atmos['H2'], n_atmos['H2O'], T)
    fo2_atmos_og = dcop(fo2_atmos)

    # --- Initial Magma Composition --- --- --- --- --- --- --- --- --- --- ---
    # mass of the melt [kg]
    m_mag = impact_melt_mass(m_imp, v_imp, theta)
    m_melt_from_impact = dcop(m_mag)  # store melt phase mass for display

    # calculate moles from wt% prescription (includes adding H2O)
    n_melt = peridotite_comp_by_fe_ratio(m_mag, iron_ratio, wt_mo, H2O_init)
    n_melt_from_impact = dcop(n_melt)  # store melt phase for display

    if model_version in ['2', '3A', '3B']:
        # partition iron from impactor between atmosphere and melt
        fe_total = gC.iron_wt[imp_comp] * m_imp / gC.common_mol_mass['Fe']
        # add FeO from the oxidised impactor iron into the initial melt phase
        feo_atmos = X_fe_atm * fe_total
        n_melt['FeO'] += feo_atmos

        # add in metallic iron from the unoxidised impactor iron
        fe_melt = X_fe_int * fe_total
        [n_melt, n_metal_bulk, m_mag] = add_iron_to_peridotite(
            fe_melt, 
            dcop(n_melt), 
            T, 
            p_tot, 
            tol
        )
    else:
        # ensure variable creation
        n_metal_bulk = 0.

    if display:
        # starting oxygen fugacity of the melt
        melt_oxides = {}
        for mol in list(n_melt_from_impact.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                melt_oxides[mol] = n_melt_from_impact[mol]

        fO2_melt_imp = calc_peridotite_fo2(
            n_melt_from_impact['Fe2O3'],
            n_melt_from_impact['FeO'],
            melt_oxides, 
            T, 
            p_tot
        )
        fO2_melt_comb = calc_peridotite_fo2(
            n_melt['Fe2O3'], 
            n_melt['FeO'],
            melt_oxides, 
            T, 
            p_tot
        )

        fmq = fo2_fmq(T, p_tot)  # current FMQ fO2
        iw = fo2_iw(T, p_tot)  # current IW fO2

        print(f"\n>>> Total Atmospheric Presure : {p_tot * 1e-5:.2f} bar")

        print(f"\n>>> Impact-Generated Melt : {m_melt_from_impact:.3e} kg")
        print(f">>>                       : {m_melt_from_impact / gC.m_earth_mantle:.3f} Earth Mantle")
        print(f">>>     FeO Added to Melt : {feo_atmos:.3e} moles")
        print(f">>>         Combined Melt : {m_mag:.3e} kg")
        print(f">>>     Pre-Iron Melt fO2 : FMQ {fO2_melt_imp - fmq:+.2f}")
        print(f">>>                       :  IW {fO2_melt_imp - iw:+.2f}")
        print(f">>>    Post-Iron Melt fO2 : FMQ {fO2_melt_comb - fmq:+.2f}")
        print(f">>>                       :  IW {fO2_melt_comb - iw:+.2f}")
        print(f">>>       Atmospheric fO2 : FMQ {fo2_atmos_og - fmq:+.2f}")
        print(f"\n>>>        Remaining Iron : {n_metal_bulk:.3e} moles")

        fe3_fe = 2. * n_melt['Fe2O3'] / (2. * n_melt['Fe2O3'] + n_melt['FeO'] + n_metal_bulk)
        print(f"\n>>> Fe3+/\u03A3Fe (after iron) : {100. * fe3_fe:.2f} %%")

    # keep iron mass separate from the mass of the silicate phase
    m_metal_bulk = n_metal_bulk * gC.common_mol_mass['Fe']

    # --- Equilibration Calculations --- --- --- --- --- --- --- --- --- --- --
    complete_full, step = False, -1
    while not complete_full:
        # --- Preparation --- --- --- --- ---
        # refresh (deepcopy) dictionaries to avoid pythonic issues
        n_melt, n_atmos = dcop(n_melt), dcop(n_atmos)
        p_tot, p_atmos = dcop(p_tot), dcop(p_atmos)

        step += 1
        n_terminal = 2000
        term_display = False
        if step == n_terminal:
            print("\x1b[1;31m>>> Equilibration terminated at " +
                  str(n_terminal) + " steps.\x1b[0m")

            if term_display:
                oxides = {}
                for mol in list(n_melt.keys()):
                    if mol not in ['Fe2O3', 'FeO', 'H2O']:
                        oxides[mol] = n_melt[mol]

                fO2_melt = calc_basalt_fo2(
                    n_melt['Fe2O3'],
                    n_melt['FeO'], 
                    oxides, 
                    T, 
                    p_tot
                )

                p_H2O_mag = calc_ph2o(n_melt['H2O'], m_mag)

                table_list = [['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                              ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, p_tot)) - np.abs(fO2_melt),
                               np.abs(fo2_iw(T, p_tot)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]]
                headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

                print('\n')
                print('\x1b[36;1m>>> Step ' + str(step) + '\x1b[0m')
                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
                ))
                print('\n')

                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                p_total = np.sum(list(p_atmos.values()))
                n_melt_total = np.sum(list(n_melt.values()))

                headers = ['Species', 'p [bar]', 'Mixing Ratio', 'Moles', 'Mole Fraction']
                table_list = []
                for mol in list(n_atmos.keys()):
                    table_list.append([
                        mol, 
                        p_atmos[mol] * 1e-5,
                        p_atmos[mol] / p_total,
                        n_atmos[mol], 
                        None
                    ])
                table_list.append([None, None, None, None, None])
                for mol in list(n_melt.keys()):
                    if mol == 'H2O':
                        table_list.append([
                            mol, 
                            p_H2O_mag * 1e-5, 
                            None,
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                    else:
                        table_list.append([
                            mol, 
                            None, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                table_list.append([None, None, None, None, None])
                table_list.append(['Fe', None, None, n_metal_bulk, None])

                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
                ))

            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST

        # --- display fO2 and pH2O --- --- --- --- --- --- --- --- --- ---
        if display:
            oxides = {}
            for mol in list(n_melt.keys()):
                if mol not in ['Fe2O3', 'FeO', 'H2O']:
                    oxides[mol] = n_melt[mol]

            fO2_melt = calc_peridotite_fo2(
                n_melt['Fe2O3'],
                n_melt['FeO'], 
                oxides, 
                T, 
                p_tot
            )

            p_H2O_mag = calc_ph2o(n_melt['H2O'], m_mag)

            table_list = [['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                          ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, p_tot)) - np.abs(fO2_melt),
                           np.abs(fo2_iw(T, p_tot)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]]
            headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

            print('\n')
            print('\x1b[36;1m>>> Step ' + str(step) + '\x1b[0m')
            print(tabulate(
                table_list, 
                tablefmt='orgtbl', 
                eaders=headers,
                floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
            ))
            print('\n')

            # --- display partial pressures, mixing ratios and mole fractions
            p_total = np.sum(list(p_atmos.values()))
            n_melt_total = np.sum(list(n_melt.values()))
            fe_total = 2. * n_melt['Fe2O3'] + n_melt['FeO'] + n_metal_bulk
            fac = 1e-4 / (4. * np.pi * gC.r_earth**2.)

            headers = ['Species', 'P_atmos\n[bar]', 'Mixing\nRatio', 'Moles', 'Iron Fraction\n(Ions)']
            table_list = []
            for mol in list(n_atmos.keys()):
                table_list.append([
                    mol, 
                    p_atmos[mol] * 1e-5,
                    p_atmos[mol] / p_total,
                    n_atmos[mol] * fac, 
                    None
                ])
            table_list.append([None, None, None, None, None])
            for mol in list(n_melt.keys()):
                if mol == 'H2O':
                    table_list.append([
                        mol, 
                        p_H2O_mag * 1e-5, 
                        None,
                        n_melt[mol], 
                        None
                    ])
                elif mol == 'Fe2O3':
                    table_list.append([
                        mol, 
                        None, 
                        None, 
                        n_melt[mol],
                        2. * n_melt[mol] / fe_total
                    ])
                elif mol == 'FeO':
                    table_list.append([
                        mol, 
                        None, 
                        None, 
                        n_melt[mol],
                        n_melt[mol] / fe_total
                    ])
                else:
                    table_list.append([
                        mol, 
                        None, 
                        None, 
                        n_melt[mol], 
                        None
                    ])
            table_list.append([None, None, None, None, None])
            table_list.append(['Fe', None, None, n_metal_bulk, n_metal_bulk / fe_total])

            print(tabulate(
                table_list, 
                tablefmt='orgtbl', 
                headers=headers,
                floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
            ))

            m_total = 0.
            for mol in list(n_atmos.keys()):
                m_total += n_atmos[mol] * gC.common_mol_mass[mol]
            for mol in list(n_melt.keys()):
                m_total += n_melt[mol] * gC.common_mol_mass[mol]
            print(f"\n>>> Total Mass in System : {m_total:.15e} \n")

            fe3_fe = 2. * n_melt['Fe2O3'] / (2. * n_melt['Fe2O3'] + n_melt['FeO'] + n_metal_bulk)
            print(f"\n>>> Fe3+/\u03A3Fe : {100. * fe3_fe:.3f} %%")

        # --- fO2 Equilibration & Dissolving --- --- --- --- --- --- --- --- --
        # unpack species (avoids issues with dictionaries)
        FeO, Fe2O3 = dcop(n_melt['FeO']), dcop(n_melt['Fe2O3'])
        H2O_mag = dcop(n_melt['H2O'])
        oxides = {}
        for mol in list(n_melt.keys()):
            if mol not in ['Fe2O3', 'FeO', 'H2O']:
                oxides[mol] = n_melt[mol]
        N_oxides = np.sum(list(oxides.values()))

        Fe = dcop(n_metal_bulk)

        H2O_atm, H2 = dcop(n_atmos['H2O']), dcop(n_atmos['H2'])
        CO2, N2 = dcop(n_atmos['CO2']), dcop(n_atmos['N2'])
        CO, CH4 = dcop(n_atmos['CO']), dcop(n_atmos['CH4'])
        NH3 = dcop(n_atmos['NH3'])

        # --- Initial Conditions --- --- --- --- --- --- --- --- --- --- --- -
        # total moles in the melt
        N_melt = Fe2O3 + FeO + N_oxides
        N_melt += H2O_mag

        # total mass in melt phase
        m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
        m_melt += FeO * gC.common_mol_mass['FeO']
        m_melt += H2O_mag * gC.common_mol_mass['H2O']
        for mol in list(oxides.keys()):
            m_melt += oxides[mol] * gC.common_mol_mass[mol]

        # total mass in atmosphere
        m_atm = H2 * gC.common_mol_mass['H2']
        m_atm += H2O_atm * gC.common_mol_mass['H2O']
        m_atm += CO2 * gC.common_mol_mass['CO2']
        m_atm += N2 * gC.common_mol_mass['N2']
        m_atm += CO * gC.common_mol_mass['CO']
        m_atm += CH4 * gC.common_mol_mass['CH4']
        m_atm += NH3 * gC.common_mol_mass['NH3']

        # pressures
        [_, P] = update_pressures({
            'H2': H2, 
            'H2O': H2O_atm, 
            'CO2': CO2,
            'N2': N2, 
            'CO': CO, 
            'CH4': CH4, 
            'NH3': NH3
        })

        # fugacities
        fo2_atmos = fo2_atm(H2, H2O_atm, T)
        fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)

        # --- Tracking --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # atmosphere
        n_h2_track.append(H2)
        n_h2o_atm_track.append(H2O_atm)
        n_co2_track.append(CO2)
        n_n2_track.append(N2)
        n_co_track.append(CO)
        n_ch4_track.append(CH4)
        n_nh3_track.append(NH3)
        pressure_track.append(P)
        fo2_atm_track.append(fo2_atmos)
        m_atm_track.append(m_atm)

        # melt phase
        n_h2o_mag_track.append(H2O_mag)
        fe2o3_track.append(Fe2O3)
        feo_track.append(FeO)
        fo2_mag_track.append(fO2_melt)
        m_mag_track.append(m_melt)

        # total moles
        atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
        melt_moles_track.append(N_melt)

        # metal phase
        fe_track.append(Fe)

        if model_version in ['1A', '1B']:
            print(">>> RETURNED WITH NO MELT-ATMOSPHERE INTERACTIONS.")
            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST

        # --- Checks --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
        if np.isnan(fo2_atmos) or np.isnan(fO2_melt):
            print(">>> NaN encountered in fO2 values.")
            sys.exit()

        # --- Reduce the Melt Packet --- --- --- --- --- --- --- --- --- --- --
        if fO2_melt > fo2_atmos:
            # print("\x1b[1;31m>>> Atmosphere is reduced relative to melt.\x1b[0m")

            complete_fO2 = False

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # - relax the system down to fO2 = ΔIW-2 by reducing only Fe2O3
            # - stop if we reach fO2 equilibrium between the melt phase and
            #   the atmosphere
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            if fO2_melt - (fo2_iw(T, P) - 2) > tol:
                complete_IW2 = False  # have we reached ΔIW-2?

                relax_frac = 0.7  # relaxation factor
                while not complete_IW2:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # are we H2-limited or Fe2O3-limited
                    if H2 < Fe2O3:
                        delta = H2
                    else:
                        delta = Fe2O3

                    # make a step, reducing ferric to ferrous
                    FeO += 2 * delta * relax_frac
                    H2 -= delta * relax_frac
                    H2O_atm += delta * relax_frac
                    Fe2O3 -= delta * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use more Fe2O3 than we have
                    if Fe2O3 < 0.:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # calculate fO2 of melt phase
                    log_fO2_sos = fo2_sossi(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of IW buffer
                    log_fo2_iw = fo2_iw(T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've got a melt more reduced than atmosphere
                    if log_fO2_sos < log_fo2_atm:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching ΔIW-2
                    if np.abs(np.abs(log_fO2_sos) - np.abs(log_fo2_atm)) < tol:
                        complete_IW2 = True
                        complete_fO2 = True
                        print(">>> System equilibrated before IW-2.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've gone too far below ΔIW-2
                    if log_fO2_sos < log_fo2_iw - 2:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached ΔIW-2
                    if np.abs(np.abs(log_fO2_sos) - np.abs(log_fo2_iw - 2)) < tol:
                        complete_IW2 = True
                        print(">>> System reached IW-2.")
                        break

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # relax the system down to where the parametrisations of 'Sossi'
            # and 'F91' predict equal fO2 values, reducing Fe2O3 & FeO
            # simultaneously
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # total anhydrous moles in melt
            N_melt = Fe2O3 + FeO + N_oxides
            # fO2 calculated by Sossi prescription
            sossi = fo2_sossi(Fe2O3, FeO, oxides, T, P)
            # fO2 calculated by Frost 91 prescription
            F91 = fo2_f91_rh12(FeO / N_melt, T, P)

            # check we are above the equality point
            if sossi > F91 and not complete_fO2:
                complete_equal = False  # have we reached equal fO2

                relax_frac = 0.07
                while not complete_equal:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # --- --- --- --- --- --- --- --- --- ---
                    # change in Fe2O3
                    d_Fe2O3 = - H2 * relax_frac

                    # ensure we don't have ΔFe2O3 > Fe2O3
                    if np.abs(d_Fe2O3) > Fe2O3:
                        # reset to start of 'equal' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # total moles in melt
                    N_melt = Fe2O3 + FeO + H2O_mag + N_oxides

                    # coefficient linking change in molar fraction of Fe^2+ to
                    # molar fraction of Fe^3+ (ΔX_Fe3+ = alpha * ΔX_Fe2+)
                    alpha = 1.504 * (2. * Fe2O3 / FeO)

                    # coefficient linking change in moles of FeO to change in
                    # moles of Fe2O3 (ΔFeO = β * ΔFe2O3)
                    beta = (N_melt - Fe2O3 + 2. * alpha * FeO) / \
                           (Fe2O3 + 2. * alpha * (N_melt - FeO))

                    # execute changes in moles
                    Fe2O3 += d_Fe2O3
                    FeO += beta * d_Fe2O3
                    Fe += - (2. + beta) * d_Fe2O3
                    H2 += (3. + beta) * d_Fe2O3
                    H2O_atm += - (3. + beta) * d_Fe2O3

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'equal' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # total anhydrous moles in the melt
                    N_melt = Fe2O3 + FeO + N_oxides

                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # fO2 calculated by Sossi prescription
                    log_fO2_sos = fo2_sossi(Fe2O3, FeO, oxides, T, P)
                    # fO2 calculated by Frost 91 prescription
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching prescription equality
                    if np.abs(np.abs(log_fO2_sos) - np.abs(log_fo2_atm)) < tol:
                        complete_fO2 = True
                        print(">>> System equilibrated before prescription"
                              " equality.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've reached fO2 equality
                    if np.abs(np.abs(log_fO2_sos) - np.abs(log_fO2_F91)) < tol:
                        complete_equal = True
                        print(">>> System reached prescription equality.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've gone too far below fO2 equality
                    if log_fO2_sos < log_fO2_F91:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've got a melt more reduced than atmosphere
                    if log_fO2_sos < log_fo2_atm:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            # reduce FeO to Fe only until we reach melt-atmosphere equilibrium
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
            if not complete_fO2:
                # have we reached zero FeO
                complete_FeO = False

                relax_frac = 0.07
                while not complete_FeO:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    Fe_og = dcop(Fe)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO -= H2 * relax_frac
                    H2 -= H2 * relax_frac
                    Fe += H2 * relax_frac
                    H2O_atm += H2 * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use more reducing power than we have
                    if H2 < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more FeO than we have
                    if FeO < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        Fe = dcop(Fe_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # total anhydrous moles in the melt
                    N_melt = Fe2O3 + FeO + N_oxides

                    # update pressures in the atmosphere
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2, 
                        'N2': N2,
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # fO2 calculated by Frost 91 prescription
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)
                    # calculate fO2 of the atmosphere
                    log_fo2_atm = fo2_atm(H2, H2O_atm, T)

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching zero FeO
                    if np.abs(np.abs(log_fO2_F91) - np.abs(log_fo2_atm)) < tol:
                        complete_fO2 = True
                        print(">>> System equilibrated before zero FeO.")
                        break

        # --- Oxidise the Melt Packet --- --- --- --- --- --- --- --- --- --- -
        if fO2_melt < fo2_atmos:
            # print("\x1b[1;36m>>> Atmosphere is oxidised relative to melt.\x1b[0m")

            complete_fO2 = False

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # if we are more reduced than IW-2, oxidise Fe to FeO only
            # until Sossi and F91 fO2 parameterisations are equal
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # anhydrous moles in melt
            N_melt = Fe2O3 + FeO + N_oxides

            # update atmospheric pressure
            [_, P] = update_pressures({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                'N2': N2, 
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            })

            log_fO2_sos = fo2_sossi(Fe2O3, FeO, oxides, T, P)
            log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)

            if log_fO2_F91 > log_fO2_sos and not complete_fO2:
                complete_equal = False
                relax_frac = 0.07
                # print("\n\x1b[1;35m>>> Below fO2 parameterisation equality."
                #       "\x1b[0m")

                while not complete_equal:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)
                    Fe_og = dcop(Fe)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO += Fe * relax_frac
                    H2 += Fe * relax_frac
                    Fe -= Fe * relax_frac
                    H2O_atm -= Fe * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use up more Fe than we have
                    if Fe < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more atmospheric H2O than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    log_fO2_sos = fo2_sossi(Fe2O3, FeO, oxides, T, P)
                    log_fO2_F91 = fo2_f91_rh12(FeO / N_melt, T, P)

                    # if we've gone too far above fO2 equality
                    if log_fO2_sos > log_fO2_F91:
                        # reset to start of loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of oxidising power used
                        d_Fe = 0.1 * d_Fe
                        # relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached fO2 equality
                    if np.abs(np.abs(log_fO2_sos) - np.abs(log_fO2_F91)) < tol:
                        complete_equal = True
                        print(">>> System reached Sossi = F91.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # calculate fO2 of the melt phase
                    fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    fo2_atmos = fo2_atm(H2, H2O_atm, T)

                    # if we've got a melt more oxidised than atmosphere
                    if fO2_melt > fo2_atmos:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    # before reaching ΔIW-2
                    if np.abs(np.abs(fO2_melt) - np.abs(fo2_atmos)) < tol:
                        complete_metal = True
                        complete_fO2 = True
                        print(">>> System equilibrated before Sossi = F91.")
                        break

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # if we are still more reduced than IW-2, oxidise Fe to FeO
            # and FeO to Fe2O3 simultaneously until we reach IW-2
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # update atmospheric pressure
            [_, P] = update_pressures({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                'N2': N2, 
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            })

            # melt phase fO2
            fo2_melt = fo2_sossi(Fe2O3, FeO, oxides, T, P)

            if fo2_melt < fo2_iw(T, P) - 2. and not complete_fO2:
                # print("\x1b[1;35m>>> Below IW-2.\x1b[0m")

                complete_IW, relax_frac = False, 1.0
                while not complete_IW:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)
                    Fe_og = dcop(Fe)

                    # --- --- --- --- --- --- --- --- --- ---
                    # change in Fe
                    d_Fe = - Fe * relax_frac

                    # total moles in melt
                    N_melt = Fe2O3 + FeO + H2O_mag + N_oxides

                    # coefficient linking change in molar fraction of Fe^2+ to
                    # molar fraction of Fe^3+ (ΔX_Fe3+ = alpha * ΔX_Fe2+)
                    alpha = 1.504 * (2. * Fe2O3 / FeO)

                    # coefficient linking change in moles of FeO to change in
                    # moles of Fe2O3 (ΔFeO = β * ΔFe2O3)
                    beta = (N_melt - Fe2O3 + 2. * alpha * FeO) / \
                           (Fe2O3 + 2. * alpha * (N_melt - FeO))

                    # execute changes in moles
                    Fe2O3 += - d_Fe / (2. + beta)
                    FeO += - d_Fe * beta / (2. + beta)
                    Fe += d_Fe
                    H2 += - d_Fe * (3. + beta) / (2. + beta)
                    H2O_atm += d_Fe * (3. + beta) / (2. + beta)

                    # --- --- --- --- --- --- --- --- --- ---
                    # update atmospheric pressure
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2,
                        'N2': N2, 
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # ensure we don't use up more atmospheric H2O than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more Fe than we have
                    if Fe < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached zero iron
                    if Fe < tol:
                        complete_IW = True
                        print(">>> Metallic iron fully oxidised.")
                        break

                    # --- --- --- --- --- --- --- --- --- ---
                    # calculate fO2 of the melt phase
                    fO2_melt = fo2_sossi(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    fo2_atmos = fo2_atm(H2, H2O_atm, T)

                    # --- --- --- --- --- --- --- --- --- ---
                    # if we've got a melt more oxidised than atmosphere
                    if fO2_melt > fo2_atmos:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium before reaching ΔIW-2
                    if np.abs(np.abs(fO2_melt) - np.abs(fo2_atmos)) < tol:
                        complete_IW = True
                        complete_fO2 = True
                        print(">>> System equilibrated before IW-2.")
                        break

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            # we should now be at or more oxidised than IW-2
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            if not complete_fO2:
                # print("\x1b[1;35m>>> Above IW-2.")
                # print(">>> Fe remaining in melt : %.2e \x1b[0m" % Fe)

                complete_ox = False
                relax_frac = 0.0001

                while not complete_ox:
                    # keep copies of variables which change
                    Fe2O3_og, FeO_og = dcop(Fe2O3), dcop(FeO)
                    H2_og, H2O_atm_og = dcop(H2), dcop(H2O_atm)
                    Fe_og = dcop(Fe)

                    # --- --- --- --- --- --- --- --- --- ---
                    FeO -= 2. * FeO * relax_frac
                    Fe2O3 += FeO * relax_frac
                    H2O_atm -= FeO * relax_frac
                    H2 += FeO * relax_frac

                    # --- --- --- --- --- --- --- --- --- ---
                    # ensure we don't use up more atmospheric H2O than we have
                    if H2O_atm < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # ensure we don't use up more FeO than we have
                    if FeO < 0.:
                        # reset to start of 'zero' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # --- --- --- --- --- --- --- --- --- ---
                    # update atmospheric pressure
                    [_, P] = update_pressures({
                        'H2': H2, 
                        'H2O': H2O_atm, 
                        'CO2': CO2,
                        'N2': N2, 
                        'CO': CO, 
                        'CH4': CH4, 
                        'NH3': NH3
                    })

                    # calculate fO2 of the melt phase
                    fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
                    # calculate fO2 of the atmosphere
                    fo2_atmos = fo2_atm(H2, H2O_atm, T)

                    # if we've got a melt more oxidised than atmosphere
                    if fO2_melt > fo2_atmos:
                        # reset to start of 'IW2' loop
                        Fe2O3, FeO = dcop(Fe2O3_og), dcop(FeO_og)
                        H2, H2O_atm = dcop(H2_og), dcop(H2O_atm_og)
                        Fe = dcop(Fe_og)

                        # diminish frac of reducing power used
                        relax_frac = 0.1 * relax_frac

                        continue

                    # if we've reached melt-atmosphere fO2 equilibrium
                    if np.abs(np.abs(fO2_melt) - np.abs(fo2_atmos)) < tol:
                        complete_ox = True
                        complete_fO2 = True

        # --- H2O Dissolution --- --- --- --- --- --- --- --- --- --- --- --- -
        if partition:
            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # metal phase
            fe_track.append(Fe)

            # --- Partitioning --- --- ---
            [H2O_atm, H2O_mag, m_mag] = equilibrate_H2O(
                Fe2O3, 
                FeO, 
                H2O_mag, 
                H2, 
                H2O_atm, 
                oxides,
                CO2, 
                N2, 
                CO, 
                CH4, 
                NH3, 
                m_mag, 
                tol
            )

            # reform dictionaries
            n_atmos = {
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2, 
                'N2': N2,
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            }

        # --- H2O Dissolution --- --- --- --- --- --- --- --- --- --- --- --- -
        if chem:
            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # metal phase
            fe_track.append(Fe)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # abundances for FastChem
            abund = calc_elem_abund({
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2,
                 'N2': N2, 
                 'CO': CO, 
                 'CH4': CH4,
                 'NH3': NH3
            })

            # prepare FastChem config files
            new_id = f"{sys_id}inside_{step}"
            write_fastchem(f"{dir_path}/data/FastChem/{new_id}", abund, T, P)
            # run automated FastChem
            run_fastchem_files(new_id)
            # read FastChem output
            [_, n_atmos_chem] = read_fastchem_output(new_id)

            # unpack dictionary
            n_atmos = {}
            for mol in list(n_atmos_chem.keys()):
                if mol in gC.gas_species:
                    n_atmos[mol] = n_atmos_chem[mol]
                elif mol == 'H3N':
                    n_atmos['NH3'] = n_atmos_chem[mol]
            for mol in gC.gas_species:
                if mol not in list(n_atmos_chem.keys()):
                    n_atmos[mol] = 0.

            # print(">>> Atmospheric chemistry complete.")

        # --- Test for Convergence --- --- --- --- --- --- --- --- --- --- ---
        if not chem and not partition:
            # --- Update Values --- --- ---
            # total moles in the melt
            N_melt = Fe2O3 + FeO + N_oxides
            N_melt += H2O_mag

            # total mass in melt phase
            m_mag = calc_anhydrous_mass(Fe2O3, FeO, oxides)
            m_mag += H2O_mag * gC.common_mol_mass['H2O']

            # total mass in atmosphere
            m_atm = H2 * gC.common_mol_mass['H2']
            m_atm += H2O_atm * gC.common_mol_mass['H2O']
            m_atm += CO2 * gC.common_mol_mass['CO2']
            m_atm += N2 * gC.common_mol_mass['N2']
            m_atm += CO * gC.common_mol_mass['CO']
            m_atm += CH4 * gC.common_mol_mass['CH4']
            m_atm += NH3 * gC.common_mol_mass['NH3']

            # total mass in melt phase
            m_melt = Fe2O3 * gC.common_mol_mass['Fe2O3']
            m_melt += FeO * gC.common_mol_mass['FeO']
            m_melt += H2O_mag * gC.common_mol_mass['H2O']
            for mol in list(oxides.keys()):
                m_melt += oxides[mol] * gC.common_mol_mass[mol]

            # update pressures
            [_, P] = update_pressures(n_atmos)

            # oxygen fugacities
            fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
            fo2_atmos = fo2_atm(H2, H2O_atm, T)

            # --- Tracking --- --- ---
            # atmosphere
            n_h2_track.append(H2)
            n_h2o_atm_track.append(H2O_atm)
            n_co2_track.append(CO2)
            n_n2_track.append(N2)
            n_co_track.append(CO)
            n_ch4_track.append(CH4)
            n_nh3_track.append(NH3)
            pressure_track.append(P)
            fo2_atm_track.append(fo2_atmos)
            m_atm_track.append(m_atm)

            # melt phase
            n_h2o_mag_track.append(H2O_mag)
            fe2o3_track.append(Fe2O3)
            feo_track.append(FeO)
            fo2_mag_track.append(fO2_melt)
            m_mag_track.append(m_melt)

            # total moles
            atm_moles_track.append(H2 + H2O_atm + CO2 + N2 + CO + CH4 + NH3)
            melt_moles_track.append(N_melt)

            # metal phase
            fe_track.append(Fe)

            # recreate atmosphere dictionary
            n_atmos = {
                'H2': H2, 
                'H2O': H2O_atm, 
                'CO2': CO2, 
                'N2': N2,
                'CO': CO, 
                'CH4': CH4, 
                'NH3': NH3
            }

        # update pressures
        [p_atmos, P] = update_pressures(n_atmos)

        # oxygen fugacities
        fO2_melt = calc_peridotite_fo2(Fe2O3, FeO, oxides, T, P)
        fo2_atmos = fo2_atm(n_atmos['H2'], n_atmos['H2O'], T)

        n_melt = dcop(oxides)
        n_melt['Fe2O3'] = Fe2O3
        n_melt['FeO'] = FeO
        n_melt['H2O'] = H2O_mag

        n_metal_bulk = dcop(Fe)

        # YES - finished
        if np.abs(fO2_melt - fo2_atmos) < tol:
            # --- --- --- --- --- ---
            if display:
                m_melt = 0.  # [kg]
                for mol in list(n_melt.keys()):
                    m_melt += n_melt[mol] * gC.common_mol_mass[mol]

                p_H2O_mag = calc_ph2o(n_melt['H2O'], m_melt)

                table_list = [['Atmosphere', fo2_atmos, None, None, 1e-5 * p_atmos['H2O']],
                              ['Bulk Magma', fO2_melt, np.abs(fo2_fmq(T, P)) - np.abs(fO2_melt),
                               np.abs(fo2_iw(T, P)) - np.abs(fO2_melt), p_H2O_mag * 1e-5]]
                headers = ['', 'log10(fO2)', '\u0394FMQ', '\u0394IW', 'pH2O [bar]']

                print('\n')
                print('\x1b[36;1m>>> Terminus Step\x1b[0m')
                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", ".5f", ".5f", ".5f", "09.5f")
                ))
                print('\n')

                # display partial pressures, mixing ratios and mole fractions
                n_melt_total = np.sum(list(n_melt.values()))

                headers = ['Species', 'p [bar]', 'Mixing Ratio', 'Moles', 'Mole Fraction']
                table_list = []
                for mol in list(n_atmos.keys()):
                    table_list.append([
                        mol, 
                        p_atmos[mol] * 1e-5, 
                        p_atmos[mol] / P,
                        n_atmos[mol] / (4.e4 * np.pi * gC.r_earth ** 2.),
                        None
                    ])
                table_list.append([None, None, None, None, None])
                for mol in list(n_melt.keys()):
                    if mol == 'H2O':
                        table_list.append([
                            mol, 
                            p_H2O_mag * 1e-5, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                    else:
                        table_list.append([
                            mol, 
                            None, 
                            None, 
                            n_melt[mol],
                            n_melt[mol] / n_melt_total
                        ])
                table_list.append([None, None, None, None, None])
                table_list.append(['Fe', None, None, n_metal_bulk, None])

                print(tabulate(
                    table_list, 
                    tablefmt='orgtbl', 
                    headers=headers,
                    floatfmt=("", "09.5f", ".5f", ".5e", ".5f")
                ))

                m_total = 0.
                for mol in list(n_atmos.keys()):
                    m_total += n_atmos[mol] * gC.common_mol_mass[mol]
                for mol in list(n_melt.keys()):
                    m_total += n_melt[mol] * gC.common_mol_mass[mol]

                print("\n>>> Total Mass in System : {m_total:.15e} \n")

            return [n_h2_track, n_h2o_atm_track, n_co2_track, n_n2_track,
                    n_co_track, n_ch4_track, n_nh3_track, pressure_track,
                    fo2_atm_track, m_atm_track, n_h2o_mag_track, fe2o3_track,
                    feo_track, fo2_mag_track, m_mag_track, fe_track,
                    atm_moles_track, melt_moles_track], P_LIST, N_LIST
        else:
            continue


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
if __name__ == "__main__":
    print("\x1b[1;31m>>> equilibrate_melt.py not meant to be run as __main__."
          "\nPlease run from separate script.\x1b[0m")
    sys.exit()
