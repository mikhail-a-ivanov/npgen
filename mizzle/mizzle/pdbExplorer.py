"""Formatting of output pdb-file and preparation of input pdb-file.

"""

from __future__ import print_function
from radish import Topologizer
import numpy as np
import mdtraj as md
from IPython import embed

def append_atoms(file, OHresname, OH2resname, coords=[], elements = [],\
                 numOfOH2=0):
    """Append atoms to pdb file

    Parameters
    ----------
    file : string
        pdb file to which atoms will be appended
    coords : array
        1D Array of coordinates of all atoms
    elements : array
        Array containing element symbols for all atoms
    verbose : boolean
        sets verbosity
    """

    atomList = []
    fileName = file
    float_format = lambda x: "%.3f" % x
    f = open(file, "r")
    content = f.readlines()
    f.close()
    fileExt = fileName.rsplit('.', 1)[1]

    if(fileExt == "pdb"):
        # Get indices for all entries in list 'content' where the substring
        # 'ATOM' occurs
        indices = [ind for ind, line in enumerate(content) if 'ATOM' in line]

        # Get number of atoms
        nrAtoms = len(indices)

        # Variables for the .pdb format
        chainIdentifier = 'A'
        residueSequenceNumber = '2'
        occupancy = '1.00'
        temperatureFactor = '0.00'

        j = 0
        # Prepare list of atoms to be appended to pdb file
        for coord in coords:
            if(j < numOfOH2*3):
                residueName = OH2resname
            else:
                residueName = OHresname

            i = 4
            nrAtoms += 1
            tempString = "ATOM"

            # Format tempString to .pdb format
            while(i <= 80):
                if(i + len(str(nrAtoms)) == 11):
                    tempString += str(nrAtoms)
                    i += len(str(nrAtoms))

                elif(i + len(elements[j]) == 14):
                    tempString += elements[j]
                    i += len(elements[j])

                elif(i + len(residueName) == 20):
                    tempString += residueName
                    i += len(residueName)

                elif(i +len(chainIdentifier) == 22):
                    tempString += chainIdentifier
                    i += 1

                elif(i + len(residueSequenceNumber) == 26):
                    tempString += residueSequenceNumber
                    i += len(residueSequenceNumber)

                elif(i + len(str(float_format(coord[0]))) == 38):
                    tempString += float_format(coord[0])
                    i += len(str(float_format(coord[0])))

                elif(i + len(str(float_format(coord[1]))) == 46):
                    tempString += float_format(coord[1])
                    i += len(str(float_format(coord[1])))

                elif(i + len(str(float_format(coord[2]))) == 54):
                    tempString += float_format(coord[2])
                    i += len(str(float_format(coord[2])))

                elif(i + len(occupancy) == 60):
                    tempString += occupancy
                    i += len(occupancy)

                elif(i + len(temperatureFactor) == 66):
                    tempString += temperatureFactor
                    i += len(temperatureFactor)

                elif(i == 76):
                    tempString += elements[j]
                    i += len(elements[j])

                tempString += " "
                i += 1
            j += 1

            # Append formatted tempString
            atomList.append(tempString)

        # Termination sequence
        if(content[indices[-1] + 1][:3] == 'TER'):
            pass

        # Get old content of file until last atom entry
        new_content = content[:indices[-1] + 1]
        # Append new atoms
        new_content.extend(atomList)
        # Append lines after the final atom entry
        new_content.extend(content[indices[-1] + 1:])

        # Print to file
        file = open(file, 'w')
        for line in new_content:
            file.write("%s\n" % line.rstrip())
        file.close()


def remove_low_coordinated(topol, Nmax, element, silent, check):
    """Removes low coordinated (<Nmax-3) atoms

    Parameters
    ----------
    topol : radish.Topologizer instance
        Topologizer instance of metal oxide crystal
    Nmax : int
        Maximum coordination for element
    element : string
        Element symbol for metal atom
    verbose : boolean
        sets verbosity

    Returns
    -------
    topol
        radish.Topologizer instance with low coordinated atoms removed
    """

    verboseprint = print if not silent else lambda *a, **k: None
    topol.topologize()
    verboseprint("\nRemoving low coordinated atoms....")
    if(check == 'all'):
        remove_metal = True
        remove_oxygen = True
    elif(check == 'none'):
        print("-check flag is set to 'none', will not remove any low coordinated atoms...\n")
        return topol
    elif(check == 'metal'):
        print("Will not remove low coordinated oxygen\n")
        remove_metal = True
        remove_oxygen = False

    while(1):
        indices = []

        #Create inputs for new Topologizer instance
        xyz = topol.trj.xyz[0]
        topology = topol.trj.topology.to_dataframe()

        # Get low coordinated atoms using radish

        if(remove_metal):
            i = 3   # Coordination less than or equal to Nmax-3
            while(i <= Nmax):
                try:
                    centerIndices = topol.extract(element, environment = {'O': Nmax - i})\
                                        .filter("i").squeeze()
                except IndexError:
                    centerIndices = []
                indices.extend(np.atleast_1d(centerIndices))
                i += 1
        
        if(remove_oxygen):
            i = 0
            while i < 2:
                oxygenIndices = topol.extract('O', environment={element: i})\
                                    .filter("i").squeeze()
                indices.extend(np.atleast_1d(oxygenIndices))

                i += 1

        # Get uncoordinated atoms
        topologyIndices = np.array([atom.index for atom in topol.trj.topology.atoms])
        bondgraphIndices = np.array(topol.bondgraph['i'].unique())
        mask = np.ones(len(topologyIndices), np.bool)
        mask[bondgraphIndices] = 0
        indices.extend(topologyIndices[mask])

        if(len(indices) < 1):
            verboseprint("All low coordinated atoms removed\n")
            return topol

        #Remove atoms from topology
        topology = topology[0].drop(topology[0].index[indices])

        #Rewrite indices and serial
        topology.reset_index(drop = True, inplace = True)
        topology['serial'] = topology.index + 1

        #Remove atoms
        indices = sorted(indices, reverse = True)   #Sort in decreasing order
                                                    #to allow loop to find 
                                                    #correct index
        for index in indices:
            xyz = np.vstack((xyz[:index], xyz[index + 1:]))

        verboseprint("Removed " + str(len(indices)) + " low coordinated atoms")
        topology = md.Topology.from_dataframe(topology, bonds = None)
        trajectory = md.Trajectory(xyz, topology)
        trajectory.unitcell_vectors = topol.trj.unitcell_vectors
        topol = Topologizer.from_mdtraj(trajectory)

        if(len(topol.trj.xyz[0]) < 1):
            raise ValueError("Incompatible structure: When removing low coordinated atoms, all atoms where removed.")

        #try:
        topol.topologize()
        #except ValueError:
        #    raise ValueError("Incompatible structure: Failed to remove low coordinated atoms. The input structure seems flawed.")
        