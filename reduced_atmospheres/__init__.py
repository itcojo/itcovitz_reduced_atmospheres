import os

# directory where 'itcovitz_reduced_atmospheres' is located
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 1)[0]

# directory where FastChem is installed
dir_fastchem = '/Users/itcojo/FastChem'

import reduced_atmospheres.constants
import reduced_atmospheres.equilibrate_melt
import reduced_atmospheres.figure_files
import reduced_atmospheres.figure_files.figure_4
import reduced_atmospheres.figure_files.figure_5
import reduced_atmospheres.figure_files.figure_6
import reduced_atmospheres.figure_files.figure_6_B
import reduced_atmospheres.figure_files.figure_7
import reduced_atmospheres.figure_files.figure_8
import reduced_atmospheres.figure_files.figure_9
