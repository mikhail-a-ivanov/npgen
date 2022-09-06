# npgen
A set of scripts for generating inorganic nanomaterials and their topology files for running atomistic molecular dynamics simulations

# Instructions on how to set up slab and nanoparticle generation

**Make sure you have Anaconda3/Miniconda3 before you begin!**

Miniconda is preferable as it does not contain any unnecessary packages. You can get it here:
https://docs.conda.io/en/latest/miniconda.html

Create a separate conda environment with all the necessary packages using the provided `yml` recipe:

1. `conda env create -f npgen.yml`

Activate the new environment:

2. `conda activate npgen`

Install the packages for generating nanomaterials:

3. `./install-npgen.sh`

Now everything is set up! Generate slabs, nanoparticles, hydrate their surfaces and create the `itp` files

## Example for a small (3x3x3 nm) anatase (101) slab

Generate the slab:

1. `./scripts/slabs/generate_surface_slab.py -uc scripts/slabs/TiO2-anatase-unitcell.dat -t 30 -l 30 -p 1 0 1`

**Check if the generated slab is normal to Z axis! Otherwise try varying its thickness.**

Hydrate the surface:

*Optionally provide a custom config.wet file with the -c flag (can be found in the mizzle source folder)*

2. `mizzler Anatase_101_T4_Ti864_O1728.pdb`

Generate the topology:

*FF-bones.itp forcefield file can be found in the bones source folder*

3. `flesh-out -c Anatase_101_T4_Ti864_O1728_wet.pdb`

Resulting output files include a `gro` configuration file, `itp` topology file as well as a template for the `top` file. 

Use in combination with the `FF-bones.itp` file to start MD simulations in `GROMACS`.
