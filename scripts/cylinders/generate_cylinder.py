#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Generate cylinders from the surface slabs based on the unit cell of crystalline
materials.
"""
import sys
import os
import argcomplete
import argparse
import numpy as np
import itertools as it
from itertools import islice
from operator import itemgetter as ig
import mdtraj as md
# pymatgen
from pymatgen.core.structure import Molecule
from pymatgen.core.surface import Lattice,Structure,SlabGenerator
from pymatgen.core.operations import SymmOp  
from IPython import embed

# Global variables
cryst_fmt = "{:6s}{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f}{:>11s}{:4d}\n"

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def main(argv=None):
    if argv is None:
        argv = sys.argv

    argparser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=CustomFormatter)
    
    # Options
    argparser.add_argument("-uc", "--unitcell", 
                           default="atoms.dat", metavar="atoms.dat",
                           help=("Input: Atom oxidation states, (fractional) coordinates"
                                 "and size of the unitcell"))
    argparser.add_argument("-o", "--output", 
                           default=None, metavar="material.pdb",
                           help="Output: Surface slab (pdb)")
    argparser.add_argument("-p", "--plane", default=[1,0,1], type=int,
                           metavar="[k,l,m]", nargs=3,
                           help="Param: Miller indices for cleavage plane")
    argparser.add_argument("-l", "--length", default=70, type=float,
                            help="Param: Cylinder length (Angstrom).")
    argparser.add_argument("-d", "--diameter", default=40, type=float, 
                           help="Param: Cylinder diameter (Angstrom).")
    argparser.add_argument("-v", "--vacuum", type=float, default=10, metavar="d",
                            help="Param: Vacuum thickness in X and Y directions (Angstrom)")
    argparser.add_argument("-s", "--symmetrize", action="store_true", 
                            help=("Param: Make sure that top and bottom layers are equivalent "
                                  "(note: stoichometry may change)"))
    argparser.add_argument("--resname", default=None, 
                           help="Param: Residue name for material (default is material name)")
    argparser.add_argument("-n", "--dry-run", action="store_true", 
                           help="Param: Run program without producing output files")


    # Parse arguments
    argcomplete.autocomplete(argparser)
    args = argparser.parse_args()

    # Read unitcell
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

    # Generate and get all slabs
    slabgen = SlabGenerator(material, args.plane, args.length, 0,
                            primitive=False, lll_reduce=False, center_slab=True)
    slabs = slabgen.get_slabs(symmetrize=args.symmetrize)

    print("Found {} terminations for surface {}...".format(len(slabs), args.plane))
    for (slab_count,slab) in enumerate(slabs):
        if slab.is_polar() and not slab.is_symmetric():
            print("Skipping {}, which is polar or asymmetric".format(slab.composition))
            continue

        # Create supercell
        repeats = [int(np.rint(args.diameter/slab.lattice.a)),
                   int(np.rint(args.diameter/slab.lattice.b)),
                   1]
        slab.make_supercell(repeats)
        # The normal
        slab_n = slab.normal
        # With rectangular unitcell
        slab = slab.get_orthogonal_c_slab()
        # And surface normal aligned with z
        # TODO: Fix corner cases when parallel c==1 and antiparallel c==-1
        v = np.cross(slab_n, np.array([0,0,1]))
        if (v == 0).all():
            c = 0.0
        else:
            c = np.dot(slab_n, np.array([0,0,1]))
        vx = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],    0]])
        # Rotation matrix
        R = np.eye(3) + vx + np.linalg.matrix_power(vx, 2)/(1+c)
        # Rotation operator
        rotator = SymmOp.from_rotation_and_translation(R)  
        # Rotated coordinates
        coords = rotator.operate_multi(slab.cart_coords) - slab.cart_coords
        # Update coordinates
        for (i,r) in enumerate(coords):
            slab.translate_sites(i, r, frac_coords=False)

        msg = "    Termination {}/{}: Composition= {}, Formula={}  ({} atoms in total)."
        print(msg.format(slab_count+1, len(slabs),
                         slab.composition.alphabetical_formula,
                         slab.composition.reduced_formula,
                         len(slab)))
        
        thickness = np.ptp(slab.cart_coords[:,2])
        ouc = slab.oriented_unit_cell
        h = ouc.lattice.c*np.sin(np.radians(ouc.lattice.alpha))
        nlayers = int(np.rint(args.length/h))
        v = args.vacuum # vacuum

        msg = "           The approximate size of the cylinder is {} x {} x {} Angstrom."
        print(msg.format(slab.lattice.a,
                         slab.lattice.b,
                         thickness))
        msg = "           There is {} Angstrom of vacuum in X and Y directions." 
        print(msg.format(v))
        msg = "           The cylinder contains {} layers."
        print(msg.format(nlayers))

        # Write output to file
        if not args.dry_run:
            if args.output: 
                root,ext = os.path.splitext(args.output)
            else:
                root = mname
            
            # Find the slab atoms that are outside the cylinder
            # The atoms that satisfy the following condition (x^2 + y^2 > r^2)
            # after centering X and Y are cut

            system = Molecule.from_sites(slab)
            
            # Define X, Y, r and l
            X = system.cart_coords.T[0] - system.center_of_mass[0] # X coordinates
            Y = system.cart_coords.T[1] - system.center_of_mass[1] # Y coordinates
            r = args.diameter / 2 # radius
            l = args.length # length
            
            # Build a list of atoms to be cut
            cylinder_cut = ((X**2 + Y**2) >  r**2).astype(np.int)
            cylinder_cut_indices = list(np.where(cylinder_cut == 1)[0])
            
            # Cut the atoms
            print("\n           Cutting cylinder with r = {} and l = {} Angstrom from the original slab along its normal...\r".format(r, l))
            system.remove_sites(indices=cylinder_cut_indices)


            outp = "_".join([root,"".join(map(str,args.plane)),
                             "T{}".format(slab_count+1)] + system.formula.split()) + ".pdb"

            print("\n           Writing the cylinder to {}...\r".format(outp))
    
            # Write PDB file as string
            pdb_file = system.to("pdb")

            # Write PDB file to output
            with open(outp,"w") as fp:
                fp.write(pdb_file)

            # Modify PDB to mdtraj convention
            fr = md.load(outp, standard_names=False)
            top, bondlist = fr.top.to_dataframe()
            
            # Add simulation box and vacuum to X and Y directions
            fr.unitcell_vectors = (slab.lattice.matrix.reshape(1,3,3) + np.array([[v, 0, 0], [0, v, 0], [0, 0, 0]]))/10.0
   
            # Rename atom and residues
            fr.top = md.Topology.from_dataframe(
                top.assign(serial=1+np.arange(len(top)),
                           name=top.element,
                           resSeq=1,
                           resName=mname if args.resname is None else args.resname,
                           chainID=0,
                           segmentID="")
            )
            
            fr.save(outp, force_overwrite=True)
            print("           Writing the cylinder to {}... done.\n".format(outp))
    
if __name__ == "__main__":
    sys.exit(main())
