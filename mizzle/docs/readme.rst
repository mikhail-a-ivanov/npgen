======
ReadMe
======

Mizzler has one mandatory input which is a file in the pdb format. The default
name for the resulting file is inputfilename_wet.pdb. The main module for the 
wetting algorithm is called `Wetter`. It is possible to directly import this 
module into an existing project. An example of this is found under the Wetter 
module documentation.

Usage
-----

**Example**::

    $ ./mizzler structure.pdb

**Optional flags**:

-s  Non-verbose mode
-o  (filename.pdb) Output filename 
-c  (filename.wet) Supply custom config file 
--check  [none, metal, all (default)] How to remove low coordinated atoms. 'none' -> do not remove any atoms, 'metal' -> only remove low coordianted metal atoms, all -> remove all low coordinated atoms. 
--log  (output log file) Print minimization progress to 'minimization.log'
-solver  [L-BFGS-B (default), SLSQP] Solver for the minimization
-maxiter  (max iterations) Max iterations for the minimizer

For partially wetted structures the `-check` flag should be set to `metal` to prevent removal.

Wetting Options
---------------

**Configuration file**

The configuration file specifies the fractions in which each metal atom type
should be hydrated. All Nmax - 1 coordinated metal atoms are labeled `surface`
and Nmax - 2 atoms are labeled `defect`, where Nmax is the maximum coordination
(or bulk coordination) specified in `metals.data`. There's also a possibility
to choose the residue names for the resulting pdb file.

config.wet::

    atom Ti: surface
        water: 1
        hydroxyl: 0

    atom Ti: defect
        water: 0.5
        hydroxyl: 0.5

    OHresname: HYD
    OH2resname: SOL

    end

    Lines after the `end` keyword will be ignored. The format of `config.wet`
    is not "cut in stone" and the tabs are optional.

**Metal data file**

In the metal data file (*metals.data*), properties of metal atoms are 
specified, like bulk coordination, bond angles and bond lengths to 
solvate molecules.

metals.data::

    Ti:
        Nmax: 6
        d_MOH: 2
        d_MOH2: 2.2
        <MOH: 115
    Fe:
        Nmax: 6
        d_MOH: 2
        d_MOH2: 2.2
    ...