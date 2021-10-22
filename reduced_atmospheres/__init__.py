import os

# directory where 'itcovitz_reduced_atmospheres' is located
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 1)[0]

# directory where FastChem is installed
dir_fastchem = '/home/itcojo/FastChem'

import reduced_atmospheres.constants
import reduced_atmospheres.equilibrate_melt
import reduced_atmospheres.figure_files
import reduced_atmospheres.figure_files.figure_2
import reduced_atmospheres.figure_files.figure_3
import reduced_atmospheres.figure_files.figure_4
import reduced_atmospheres.figure_files.figure_4_B
import reduced_atmospheres.figure_files.figure_5
import reduced_atmospheres.figure_files.figure_6
import reduced_atmospheres.figure_files.figure_7


