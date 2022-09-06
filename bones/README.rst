# The BONES module
Construct BOnd NEtwork Similarity force fields based on ab initio data! Two
scripts are provided. `bonify` is for developers to construct force field
templates from ab initio data. `flesh-out` is for users to construct molecular
topologies from the templates.

## Installation
The module is installed with pip. Issue 
```
pip install 
```
in project directory. Use `--user` flag for local user install.

## Usage
### bonify
Net atomic charges (NACs) and net atomic volumes (NAVs) are used to identify
and calculate partial charges and Lennard-Jones parameters for atom types with
similar bond networks. The force field templates are stored in easy-access JSON
formats.

### flesh-out
The force field templates are used to map given atomic coordinates to proper
atom types. Molecular topologies are written to standard GROMACS format.


