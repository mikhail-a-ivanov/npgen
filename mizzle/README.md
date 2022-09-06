# Mizzle

## Motivation:
Metal oxides in the presence of water are covered in a hydration shell. To
accurately perform simulations on such systems this fact needs to be taken into
account. The motivation for this program lies therein and automatically adds 
hydration shells to arbitrary metal oxide surfaces.

## Description:
Hydrates metal oxides in the pdb format with OH2 and OH. In `config.wet` the
user specifies which metals to hydrate and in what ratio. Custom config files
are included using the `-c` flag, if no config file is supplied the default
config file is used. Defult name for the output pdb file is 
`inputfilename_wet.pdb`Hydrating a crystal containing different metals is
supported.

# Modules
The program consists of four main modules:
	*Wetter
	*AtomCreator
	*pdbExplorer
	*potential

## Wetter
Calculates directional vectors which are the oxygen -> metal center
vectors. From the sum of directional vectors of each center the coordinates
where oxygen should be positioned is calculated.

## AtomCreator
The AtomCreator module holds methods for creating molecules according to
specified coordinates and directional vectors from the Wetter
module. Rotational matrices are set up and aligns the molecules accordning to
the directional vectors.


## pdbExplorer
This module handles writing and reading from the supplied .pdb file. It also
handles the removal of low coordinated atoms (Nmax - 3, where Nmax is the
maximum coordination of the metal) using the remove_lower_coordinated method at
the start of the program.

## potential
Contains the objective function to be optimized and the Jacobian.

# Installation instructions:
Clone repository and install with pip
```
pip install .
```

## Development install
```
pip install -e .
```

## Rebuild Cython libraries
```
python setup.py build_ext --inplace
```

# Usage
To run the program with default settings, only a pdb-file is required as input
```bash
./mizzler input.pdb
```
optional flags:
```
-s (silent) -o (ouput filename) -c (config file) --check [none, metal, all] --log(output log file) -solver [L-BFGS-B, SLSQP] -maxiter (max iterations) -print_dist (output min and max distances in resulting pdb file.)
```

# Examples
## Config file
`config.wet` (This is also the default config file)
```
atom Ti: surface
	water: 1.0
	hydroxyl: 0.0

atom Ti: defect
	water: 0.5
	hydroxyl: 0.5
end
resname SOL
```

```bash
./mizzler input.pdb -c config.wet
```
will put water on all Nmax-1 (surface) coordinated Ti atoms, each Nmax-2 (defect) coordinated atom will have a 50/50 chance of being hydrated with water and/or hydroxyl.
# Other notes
## Non-standard packages:
	*numpy
	*scipy
	*pandas
	*radish

