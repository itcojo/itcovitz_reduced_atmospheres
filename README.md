# Reduced atmospheres of post-impact worlds: The early Earth

## Code Accompanyment, by J. P. Itcovitz


This code accompanies, and can be used to reproduce the results and figures of, Itcovitz et al. (2021) (doi:URL). The code is presented as a Python module, with an accompanying yml for reproducibility of the Python environment. 

## Code Structure

The module is structured as follows:
* `run.py` acts as a control center for the code, with the script being used for running models and generating figures. From here,
  * the Model version can be selected (see Figure 1 and Section 5.2 of paper),
  * the melt phase composition can be selected (basaltic or peridotitic),
  * the initial conditions can be varied (a suggested range of initial conditions for the target and impactor are given), and
  * the 7 paper figures (not including Figure 1) can be created.

* `reduced_atmospheres` is where the majority of the code exists, and be broken down as follows:
  * `equilibrate_melt.py` contains all of the functions used for impact processing of the system, as well as functions used for melt-atmosphere interactions;
  * `constants.py` contains useful global constants for calculations (e.g., mass of the Earth, Avogadro's constant);
  * `data` contains the melt mass and iron distribution results from our GADGET2 SPH (Springel 2005, Ćuk and Stewart 2012) simulations, and is also where data generated as part of the FastChem (Stock et al. 2018) calculations is stored; and
  * `figure_files` contains files used to produce each of the figures from Itcovitz et al. (2021).

* `output` is where the model outputs are saved as `h5py` files. Values that are stored for each step of calculations include: atmospheric abundances (moles), total atmospheric pressure (Pa), atmosphere oxygen fugacity (log10), total atmospheric mass (kg), abundances of species in the silicate melt phase (moles), oxygen fugacity of the melt phase (log10), mass of the melt phase (kg), and amount of metallic iron hosted by the melt (moles).

* `figures` is where figures are saved to once generated.

## Getting Started

The module almost exclusively contains relative pathways. As such, it is important that path pointers are correct. The first thing to check after install is that `dir_path` in `reduced_atmospheres/__init__.py` points to the general install folder (`INSTALL_DIRECTORY/itcovitz_reduced_atmospheres`), and **not** to the subdirectory `reduced_atmospheres`.

FastChem (Stock et al. 2018) must be installed in order to perform initial conditions calcualtions for the model. [This repository](https://github.com/exoclime/FastChem) contains the latest release at the time of writing. The model must be told the install directory for FastChem via the `dir_fastchem` variable in `reduced_atmospheres/__init__.py`. The model edits the FastChem input and config files of its own accord. However, the verbose level of FastChem can be lowered or raised by the user (FastChem/input/config.input) - the recommended verbose level is 0 in order to avoid cluttering the console.


## Run Your First Model

In `run.py`, ensure that `model_run` is set as **True**. Select which Model should be run, and set the melt composition in the same region of the script. The impactor mass is automatically set as the model parameter to be varied, over ~ an order of magnitude from the maximum HSE impactor. An example of how to best vary other model parameters is given in the code. The model output will be saved in the form `COMPOSITION_MODEL_MASS.hdf5`.










