import numpy as np
import re
import sys
from typing import Tuple


class Constants:
    def __init__(self):
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # CONSTANTS IN SI UNITS
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        self.c = 299792458  # Speed of Light [m s-1]
        self.g = 9.81  # Earth gravity [m s-2]
        self.G = 6.67e-11  # Gravitational constant [m3 kg-1 s-2]
        self.h = 6.62607004e-34  # Planck [m2 kg s-1]
        self.k_B = 1.38064852e-23  # Boltzmann [m2 kg s-2 K-1]
        self.m_p = 1.6726219e-27  # Proton mass [kg]
        self.N_A = 6.02214086e23  # Avagadro [atoms mol-1]
        self.R = 8.31446261815324  # Gas constant [J K-1 mol-1]
        self.sig = 5.670374419e-8  # Stefan-Boltzmann [W m-2 K-4]
        self.u = 1.66053906660e-27  # Atomic Mass Unit [kg]

        # Earth Oceans - The Volume of Earth's Ocean (Charette & Smith, 2015)
        self.eo_vol = 1.33238e9 * 1e9  # [m3]
        self.eo_mass = self.eo_vol * 997  # [kg], 997 kg m-3

        # Basaltic Composition
        self.basalt = {'SiO2': 49.5, 'Al2O3': 13.18, 'Fe2O3': 3.18, 'FeO': 6.85,
                       'MnO': 0.15, 'MgO': 9.98, 'CaO': 12.34, 'Na2O': 2.18,
                       'K2O': 0.93, 'TiO2': 1.01, 'P2O5': 0.25}

        # Peridotite Composition
        # (https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JB091iB09p09367)
        self.klb = {'SiO2': 44.48, 'TiO2': 0.16, 'Al2O3': 3.59, 'FeO': 8.10,
                    'MnO': 0.12, 'MgO': 39.22, 'CaO': 3.44, 'Na2O': 0.3,
                    'K2O': 0.02, 'P2O5': 0.03, 'Cr2O3': 0.31, 'NiO': 0.25}

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # SOLAR SYSTEM PARAMETERS IN SI UNITS
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        self.AU = 1.5e11  # 1 AU [m]

        self.r_earth = 6.371E6  # Earth Radius [m]
        self.m_earth = 5.899e24  # Earth Mass [kg]
        self.m_earth_mantle = 4.0e24  # Earth Mantle Mass [kg]
        self.g_earth = 9.81  # Earth Surface Gravity [m.s-1]

        self.r_venus = 6.052e6  # Venus Radius [m]
        self.m_venus = 4.8675e24  # Venus Mass [kg]
        self.g_venus = 8.87  # Venus Surface Gravity [m.s-1]

        self.m_lunar = 7.34767309e22  # Lunar Mass [kg] = 18.01528
        self.m_ceres = 8.958e20  # Ceres Mass [kg]
        self.m_vesta = 2.589e20  # Vesta Mass [kg]

        self.m_sun = 1.9e30  # Sun Mass [kg]
        self.l_sun = 3.828E26  # Sun Luminosity [J s-1]
        self.r_sun = 696E6  # Sun Radius [m]
        self.T_sun = 5778.0  # Sun Temperature [K]

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # PERIODIC TABLE
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        self.elements = np.array(
            ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
             'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
             'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
             'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
             'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
             'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
             'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
             'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U'], dtype='str')
        self.atomic_num = dict(zip(self.elements,
                                     np.arange(len(self.elements))))
        # https://www.qmul.ac.uk/sbcs/iupac/AtWt/
        self.atomic_mass = dict(zip(self.elements, self.u * np.array(
            [1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998,
             20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45,
             39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938,
             55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.630, 74.922,
             78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906,
             95.95, 97., 101.07, 102.905, 106.42, 107.8682, 112.414, 114.818,
             118.710, 121.760, 127.60, 126.904, 131.293, 132.905, 137.327,
             138.905, 140.116, 140.908, 144.242, 145, 150.36, 151.964, 157.25,
             158.925, 162.500, 164.930, 167.259, 168.934, 173.045, 174.967,
             178.486, 180.948, 183.84, 186.207, 190.23, 192.217, 195.084,
             196.967, 200.592, 204.38, 207.2, 208.980, 209, 210, 222, 223, 226,
             227, 232.038, 231.036, 238.029])))  # [kg]
        self.atomic_mass['Nu'] = 0.
        self.atomic_mass['e'] = 0.
        self.atomic_mass['J'] = 0.

        # Slater, J.C., 1964. Atomic radii in crystals. The Journal of Chemical
        # Physics, 41(10), pp.3199-3204.
        self.atomic_rad = dict(zip(self.elements, 1e-10 * np.array(
            [0.25, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.80, 1.50, 1.25,
             1.10, 1.00, 1.00, 2.20, 1.80, 1.0, 1.40, 1.35, 1.40, 1.40, 1.40,
             1.35, 1.35, 1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 2.35, 2.00,
             1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55,
             1.45, 1.45, 1.40, 1.40, 2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85,
             1.85, 1.85, 1.80, 1.75, 1.78, 1.75, 1.75, 1.75, 1.73, 1.75, 1.55,
             1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, 1, 90, 1.60, 1.90,
             np.nan, np.nan, 2.15, 1.95, 1.80, 1.80, 1.75, 1.75, 1.75,
             1.75])))  # [m]
        self.atomic_rad['Nu'] = 0.
        self.atomic_rad['e'] = 0.
        self.atomic_rad['J'] = 0.

        # important gaseous species
        self.gas_species = ['H2O', 'H2', 'N2', 'CO2', 'CO', 'CH4', 'NH3']

        # commonly used molar masses [kg mol-1]
        self.__common_mols = ['Fe2O3', 'FeO', 'Fe', 'H2O', 'H2', 'N2', 'CO2',
                              'CO', 'CH4', 'NH3', 'H3N']
        for mol in list(self.basalt.keys()):
            if mol not in self.__common_mols:
                self.__common_mols.append(mol)
        for mol in list(self.klb.keys()):
            if mol not in self.__common_mols:
                self.__common_mols.append(mol)

        self.common_mol_mass = {}
        for mol in list(self.__common_mols):
            self.common_mol_mass[mol] = self.mol_phys_props(mol)[0] * self.N_A

        # mass fraction of composition which is metallic iron
        self.iron_wt = {'C': 0., 'L': 0.05, 'H': 0.15, 'E': 0.333, 'F': 1.}
        # mass fraction of composition which is volatile H2O
        self.h2o_wt = {'C': 10 / 100, 'L': 1 / 100, 'H': 0.5 / 100,
                       'E': 0.1 / 100, 'F': 0.}

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # USEFUL PLOTTING THINGS
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # MATLAB COLOURS, see https://bit.ly/2VJIQKZ
        self.ML_colours = [(0, 0.4470, 0.7410),
                           (0.8500, 0.3250, 0.0980),
                           (0.9290, 0.6940, 0.1250),
                           (0.4940, 0.1840, 0.5560),
                           (0.4660, 0.6740, 0.1880),
                           (0.3010, 0.7450, 0.9330),
                           (0.6350, 0.0780, 0.1840)]

        # COLOURBLIND FRIENDLY COLOURS,
        # see Wong, B. (2011). Color blindness. nature methods, 8(6), 441-442.
        self.color_ibm = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
        self.color_wong = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                           '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # METHODS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    def mol_phys_props(self, mol: str) -> Tuple[float, float]:
        """Calculate physical properties of the given molecule, namely the mass and the diameter.

        Args:
            mol (str): Molecule for which calculations will be performed.

        Returns:
            mass (float): Mass of the molecule, in units of 'kg'.
            diam (float): Diameter of the molecule, in units of 'm'.

        """
        # remove ion markers from molecules' string labels
        mol = mol.strip('+')
        mol = mol.strip('-')

        # initialise mass
        mass = 0.  # [kg]
        diam = 0.  # [m]

        # separate out molecule into elements, numbers, and symbols
        char = [m for m in re.split(r'([A-Z][a-z]*)', mol) if m]
        for j in range(len(char) - 1):
            # element-number pairing
            if char[j].isalpha() and char[j + 1].isdigit():
                mass += self.atomic_mass[char[j]] * float(char[j + 1])
                diam += 2 * self.atomic_rad[char[j]] * float(char[j + 1])
            # single element - do not account for condensation indicator
            elif char[j].isalpha():
                mass += self.atomic_mass[char[j]]
                diam += 2 * self.atomic_rad[char[j]]
        if char[(len(char) - 1)].isalpha():
            mass += self.atomic_mass[char[(len(char) - 1)]]
            diam += 2 * self.atomic_rad[char[(len(char) - 1)]]

        return mass, diam


if __name__ == "__main__":
    print("\x1b[1;31m>>> constants.py is not meant to be run as __main__."
          "\nPlease run from separate script.\x1b[0m")

    sys.exit()