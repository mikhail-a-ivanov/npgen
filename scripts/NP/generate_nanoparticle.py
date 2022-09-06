#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Create a spherical nanoparticle.
"""

import sys
import argcomplete
import argparse
import numpy as np
import mdtraj as md
from itertools import islice
from operator import itemgetter as ig

# pymatgen
from pymatgen.core.structure import Molecule
from pymatgen.core.surface import Lattice,Structure

from IPython import embed

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def parse_planes(planes):
    try:
        h,k,l = map(int, list(planes))
        return (h,k,l)
    except:
        print(planes)
        raise argparse.ArgumentTypeError("Error in plane specification")
         
def main(argv=None):
    if argv is None:
        argv = sys.argv

    argparser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=CustomFormatter)
    
    # Options
    argparser.add_argument("unitcell",
                           default="atoms.dat", metavar="atoms.dat",
                           help=("Input: Atom oxidation states and (fractional) coordinates"
                                 "and size of the unitcell"))
    argparser.add_argument("-o","--output", default="NP.pdb",
                           help="Output: Nanoparticle coordinates")
    argparser.add_argument("-r", "--radius", type=float, default=15,
                           help="Param: Maximum radius of the NP (in Angstrom)")
    argparser.add_argument("--resname", default="NPW",
                           help="Param: Residue name in output PDB")

    # Parse arguments
    argcomplete.autocomplete(argparser)
    args = argparser.parse_args()
    
    # Read unitcell
    print("Reading unit cell data from {}...".format(args.unitcell))
    
    with open(args.unitcell) as f:
        mname, N = f.readline().split()

        oxidation_states = f.readline().split()[1:]
        oxidation_states = dict([e.split("=") for e in oxidation_states])
        oxidation_states = {k: int(v) for (k,v) in oxidation_states.items()}

        atoms = list(islice(f, int(N)))
        cell = list(islice(f, 3))
        
    # Convert to floats, arrays and tuples
    N = int(N)
    cell = np.array([l.split() for l in cell]).astype(float)
    atoms = [[ig(0)(a.split()), tuple(map(float, ig(1,2,3)(a.split())))]
             for a in atoms]

    # Create bulk material, set the unit cell and scale the atomic positions
    # accordingly
    lattice = Lattice(cell)
    species,coords = zip(*atoms)
    material = Structure(lattice, species, coords, validate_proximity=True)
    # Add oxidation states
    material.add_oxidation_state_by_element(oxidation_states)
    
    print("The unit cell lattice is (Angstrom):")
    print("\n".join([" "*10 + l for l in  material.lattice.__str__().splitlines()]))
    
    print("")

    print("Creating spherical nanoparticle with {} Angstrom radius...".format(args.radius))

    # Create nanoparticle and save as PDB
    sites = material.get_sites_in_sphere([0,0,0], args.radius)

    NP = Molecule.from_sites(sites)
    NP.to(fmt="pdb", filename=args.output)
        
    print("")
        
    # Modify this PDB to obey the mdtraj convention
    fr = md.load(args.output, standard_names=False)
    top, bondlist = fr.top.to_dataframe()

    # Add simulation box
    fr.unitcell_vectors = 2*fr.xyz[0].ptp(axis=0).max()*np.eye(3).reshape(1,3,3)

    msg = "The simulation box size is Lx={:.3f}, Ly={:.3f}, Lz={:.3f} Angstrom."
    print(msg.format(*(10*fr.unitcell_lengths.reshape(-1))))
    print("")

    # Put nanoparticle in the center of the box
    fr.xyz = fr.xyz + fr.unitcell_lengths/2

    # Rename atom and residues
    fr.top = md.Topology.from_dataframe(
        top.assign(serial=1+np.arange(len(top)),
                   name=top.element,
                   resSeq=1,
                   resName=args.resname,
                   chainID=0,
                   segmentID="")
    )

    # Update file
    fr.save(args.output, force_overwrite=True)

    print("Writing nanoparticle to {}....".format(args.output))
    print("")
    
    print("Goodbye!")

if __name__ == "__main__":
    sys.exit(main())
