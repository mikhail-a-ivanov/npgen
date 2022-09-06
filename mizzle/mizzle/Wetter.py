"""The main module that contains the methods for the hydration of
metal oxide surfaces.

Example
-------
Using Wetter module in an existing project::

    from mizzle import Wetter
    wet = Wetter('structure.pdb')
    wet.remove_low_coordinated(Nmax, 'element')
    wet.solvate({'Nmax': Nmax, 'element': element, 'coordination': coordination,
                 'OH': hydroxylFrac, 'OH2': waterFrac, 'O':0.05, 'd_MOH2':2.2, 
                 'd_MOH':2, '<MOH':115})
    wet.optimize()
    wet.wet()
    wet.save('outputfile.pdb', 'residuename')

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import random
from radish import Topologizer
from tqdm import tqdm
from scipy.optimize import minimize
import sys
from timeit import default_timer as timer
import mdtraj as md
from IPython import embed
#from scipy.optimize.optimize import _approx_fprime_helper

# mizzle imports
from mizzle import potential
from mizzle.pdbExplorer import append_atoms
from mizzle.pdbExplorer import remove_low_coordinated as rlc
import mizzle.AtomCreator as mac
from mizzle.overlap import shortest_distance
from mizzle.puts import puts

if sys.version_info[0] == 2:
    # workaround for Sphinx autodoc bug
    import __builtin__
    def print(*args, **kwargs):
        __builtin__.print(*args, **kwargs)

class Wetter:
    """Main class

    Parameters
    ----------
    topol : string
        Path of pdb-file.
    silent : bool, optional
        enables/disables silent mode.
    theta : double, optional
        HOH bond angle. Default is 104.5.
    optLog : bool, optional
        Outputs log file from optimization.
    solver : string, optional
        optimization method.
    maxiter : int, optional
        maximum iterations in optimization.

    Attributes
    ----------
    coords : ndarray
        Coordinates of all solvate molecules, set after calling `wet()`.
    elements : array
        Elements of all atoms in solvate, set after calling `wet()`.

    """
    def __init__(self, topol, silent = False, theta=104.5, optLog = False,\
                 solver = 'L-BFGS-B', maxiter=500):

        # Private attributes
        self.__hydVectors = np.empty([0, 3], dtype=float)
        self.__hydCoords = np.empty([0, 3], dtype=float)
        self.__hydCenters = np.empty([0], dtype=float)

        self.__watVectors = np.empty([0, 3], dtype=float)
        self.__watCoords = np.empty([0, 3], dtype=float)
        self.__watCenters = np.empty([0], dtype=float)

        self.__oxyVectors = np.empty([0, 3], dtype=float)
        self.__oxyCoords = np.empty([0, 3], dtype=float)
        self.__oxyCenters = np.empty([0], dtype=float)
        self.__dMOH = []
        self.__dMOH2 = []
        self.__watAngles = []
        self.__hydAngles = []
        self.__numOfOH = 0
        self.__numOfOH2 = 0

        # Input parameters
        self.theta = theta
        self.theta = theta/360*2*np.pi
        self.lowFrac = 1
        self.highWaterFrac = 1
        self.highHydroxylFrac = 1
        self.silent = silent
        self.optLog = optLog
        self.solver = solver

        if(solver == 'L-BFGS-B'):
            self.optOptions={'disp': False, 'ftol':1e-12, 'gtol':1e-12,\
                             'iprint': 0, 'eps': 1.4901161193847656e-8,\
                             'maxiter': maxiter}
            self.prepOptions={'disp': False, 'ftol':1e-10, 'gtol':1e-10,\
                              'iprint': 0, 'eps': 1.4901161193847656e-8,\
                              'maxiter': 100}
        elif(solver == "SLSQP"):
            self.optOptions={'disp': False, 'ftol':1e-10, 'iprint': 0,\
                     'eps': 1.4901161193847656e-8, 'maxiter':maxiter}
            self.prepOptions={'disp': False, 'ftol':1e-10, 'iprint': 0,\
                     'eps': 1.4901161193847656e-8, 'maxiter':100}
        else:
            raise RuntimeError("Unknown solver {}, please choose 'L-BFGS-B'\
                                (default) or 'SLSQP'.".format(solver))

        #Attributes
        # Use radish (RAD-algorithm) to compute the coordination of each atom
        self.topol = Topologizer.from_coords(topol)

        # Program works without box vectors
        if self.topol.trj.unitcell_lengths is None:
            self.boxVectors = np.array([0, 0, 0], dtype=float)
        else:
            self.boxVectors = 10*self.topol.trj.unitcell_lengths.reshape(-1).astype(float)
        self.coords = []
        self.elements = []

        # Set up verbose print function
        self.__verboseprint = print if not silent else lambda *a, **k: None
        self.__verboseputs = puts if not silent else lambda *a, **k: None

        # Log each step from optimization
        self.__optimizationLog = list()

        # Attributes for show_progress()
        self.__i = 0
        self.__j = 0
        self.__centers = None
        self.__centerNeighbours = None
        self.__centerNumNeighbours = None
        self.__potTime = None
        self.__jacPotTime = None

    def __show_progress(self, x):
        """Print progression of optimizer

        """
        if(self.optLog):
            self.__optimizationLog.append(potential.potential(\
                                        x,\
                                        self.__centers,\
                                        self.topol,\
                                        self.__centerNeighbours,\
                                        self.__centerNumNeighbours,\
                                        self.boxVectors,
                                    np.append(self.__dMOH,\
                                                        self.__dMOH2),
                                    np.asarray(np.hstack((self.__numOfOH,\
                                                        self.__numOfOH2)))))

        if(self.__i % 5 == 0 or self.__i == 1):
            self.__verboseprint('\r', end='')
            sys.stdout.flush()
            self.__verboseprint("Iteration: "+str(self.__i), end='')
        self.__i += 1

    def __show_progress_prep(self, x):
        """Print progress of first guess optimizer

        """
        self.__verboseprint('\r', end='')
        sys.stdout.flush()
        self.__verboseprint(str(int(float(self.__i)/float(self.__j)*100+1))+\
                            "% done.", end='')


    def __optimizer(self, coords, centers):
        """Perform the minimization scheme

        Parameters
        ----------
        coords : 3*N array(float)
            Coordinates of all solvate molecules
        centers : array(int)
            Containing all metal centers which has bound solvate in the same
            order as `coords`

        Raises
        ------
        ValueError
            If one or more coordinates are missing centers

        Returns
        -------
        coords
            Optimized oxygen coordinates
        vectors
            Optimized M-O vectors

        """
        self.__numOfOH = len(self.__hydCenters)
        self.__numOfOH2 = len(self.__watCenters)

        vectors = np.empty([0, 3], dtype = float)
        cutoff = 6.0
        M = self.topol.trj.xyz[0][centers]*10

        if(len(M) != len(coords)):
            raise ValueError("Internal error: some solvate molecules were\
                              not assigned to a center.")        

        # Drops duplicates
        centerCoordination = np.array(self.topol.bondgraph.\
                             loc[self.topol.bondgraph['j'].\
                             isin(centers)]['j'].value_counts())
        missing = len(centers) - len(centerCoordination)

        # Readd duplicates
        if(missing > 0):
            centerCoordination = np.append(centerCoordination,\
                                           centerCoordination[\
                                           len(centerCoordination)-missing:])

        centerNeighbours = np.empty([0, 3], dtype = float)
        centerNumNeighbours = np.empty([0], dtype = int)

        # Get neighbours to each center
        self.__verboseprint("Calculating guesses for the optimizer:")
        i = 0
        self.__j = len(centers)
        for center in centers:
            neighbours = md.compute_neighbors(self.topol.trj, cutoff/10.0,\
                                              np.array([center]))

            centerNumNeighbours = np.hstack((centerNumNeighbours,\
                                             len(neighbours[0])))

            centerArray = np.repeat(center, len(neighbours[0]))
            dispArray = np.vstack((neighbours, centerArray))
            dispArray = np.transpose(dispArray)
            dispVectors = md.compute_displacements(self.topol.trj, dispArray,\
                                                   periodic = True)[0]
            neighbourCoords = -dispVectors*10 +\
                              self.topol.trj.xyz[0][center]*10
            centerNeighbours = np.vstack((centerNeighbours, -dispVectors*10 +\
                                          self.topol.trj.xyz[0][center]*10))

            res = minimize(potential.potential, coords[i],
                        args = (np.asarray([center]), self.topol,\
                                np.asarray(np.vstack((neighbourCoords,\
                                np.delete(coords, i, axis=0))), dtype=float),\
                                np.asarray([len(neighbours[0])+\
                                           len(coords)-1]),\
                                self.boxVectors,
                                np.append(self.__dMOH,\
                                                      self.__dMOH2),
                                np.asarray(np.hstack((self.__numOfOH,\
                                                      self.__numOfOH2)))),
                        jac = potential.potential_jac,
                        callback=self.__show_progress_prep,
                        method = self.solver,
                        options=self.prepOptions)
            coords[i] = res.x
            i += 1
            self.__i += 1

        self.__verboseprint("\n")

        self.__i = 0
        self.__centers = centers
        self.__centerNeighbours = centerNeighbours
        self.__centerNumNeighbours = centerNumNeighbours

        # Timings for objective function
        start = timer()
        potential.potential(coords.flatten(), centers, self.topol,\
                            centerNeighbours, centerNumNeighbours,\
                            self.boxVectors,
                                np.append(self.__dMOH,\
                                                      self.__dMOH2),
                                np.asarray(np.hstack((self.__numOfOH,\
                                                      self.__numOfOH2))))
        end = timer()
        self.__potTime = end-start
        self.__verboseprint("Potential takes: " + str(round(end-start, 4)) +\
                          " seconds to calculate")

        start = timer()
        potential.potential_jac(coords.flatten(), centers, self.topol,\
                                centerNeighbours, centerNumNeighbours,\
                                self.boxVectors,
                                np.append(self.__dMOH,\
                                                      self.__dMOH2),
                                np.asarray(np.hstack((self.__numOfOH,\
                                                      self.__numOfOH2))))
        end = timer()
        self.__jacPotTime = end-start
        self.__verboseprint("Potential Jacobian takes: " + str(round(end-start, 4)) +\
              " seconds to calculate\n")

        self.__verboseprint("Minimizing potential with " +\
                          str(len(coords.flatten())) + " free variables...")

        # Run minimization
        self.__start = timer()
        res = minimize(potential.potential, coords,
                        args = (centers, self.topol, centerNeighbours,\
                                centerNumNeighbours, self.boxVectors,
                                np.append(self.__dMOH,\
                                                      self.__dMOH2),
                                np.asarray(np.hstack((self.__numOfOH,\
                                                      self.__numOfOH2)))),
                        jac = potential.potential_jac,
                        method = self.solver,
                        callback = self.__show_progress,
                        options=self.optOptions)

        self.__verboseprint("\n")
        if(res.success):
            if(not self.silent):
                self.__verboseputs.success("Successfully minimized"+\
                                           " potential!\n")
        else:
            self.__verboseputs.warning("Optimization failed... Try with"+\
                                       " different optimizer using the"+\
                                       " -solver flag or increasing max"+\
                                       " iterations with -maxiter\n")

        if(self.optLog):
            file = open('minimization.log', 'w')
            file.write("Iteration:\tFunction value:\n")
            for ind, line in enumerate(self.__optimizationLog):
                file.write(str(ind+1)+"        \t"+str(line)+"\n")
            print("\n")
            print("Output from Minimizer:")
            file.write(str(res) + "\n")
            file.close()

        coords = np.reshape(res.x, (-1, 3))# Since minimization returns flat
                                           # array.

        # Recalculate directional vectors from optimized coordinates
        i = 0
        for coord in coords:
            vectors = np.vstack((vectors, (coord-M[i])/\
                                 np.linalg.norm(coord-M[i])))
            i += 1

        return coords, vectors


    def __get_center_neighbours(self, coordination, center):
        """Construct neighbourgraph where columns are metal centers and rows 
        each atom in the coordination shell

        Parameters
        ----------
        coordination : int
            The coordination for which to construct the neighbourgraph and
            the indices for all such metal centers

        Returns
        -------
        centerIndices
            Indices for all metal centers if there are any, otherwise empty
        neighbourgraph
            Neighbourgraph if any atoms found, empty otherwise
        
        Notes
        -----
        Coordination shell calculated using the Topologizer module from radish.

        """
        try:
            centerIndices = self.topol.extract(center, environment =
                                            {'O': coordination})\
                                            .filter("i").squeeze()
        except IndexError:
            centerIndices = pd.DataFrame()

        print(type(centerIndices))
        if(type(centerIndices) != pd.core.series.Series):
            #Extract has found 0 or 1 centers
            if(centerIndices.size == 0):
                centerIndices = []
            else:
                centerIndices = np.array([centerIndices])
            print("not")

        if(len(centerIndices) > 0):
            neighbourgraph = self.topol.bondgraph.loc[self.topol.\
                    bondgraph['j'].isin(centerIndices)].\
                    sort_values(['j'])[['i', 'j']].pivot(columns= 'j').\
                    apply(lambda x: pd.Series(x.dropna().values)).\
                    apply(np.int64)['i']

            return (centerIndices, neighbourgraph)
        else:
            return ([] ,[])


    def __calculate_pair_vectors(self, coordination,\
                               OH_frac, OH2_frac, center):
        """Calculate coordinates for solvate on for 4-coordinated atoms

        Notes
        -----
        Calculates the sum of vectors from oxygen to 4-coordinated metal
        atoms. These vectors are used as first guesses of the positions for
        the hydration molecules.

        Returns
        -------
        vectors : ndarray(float)
            Directional vectors for each solvate
        coord : ndarray(float)
            Coordinates for oxygen atom in each solvate
        centers : ndarray(int)
            List of each center that each solvate belongs to

        """
        centerIndices, neighbourgraph = self.__get_center_neighbours(\
                                            coordination, center)
        
        if(len(centerIndices) > 0):
            centers = np.empty([0], dtype=int)
            self.__verboseprint(("Found {} centers that are Nmax - 2 = {}-fold"
                               " coordinated\n".format(len(centerIndices),\
                                                      coordination)))


            vectors = np.empty([0, 3], dtype = float)

            if OH_frac + OH2_frac == 1:
                randIndices = random.sample(range(0, len(centerIndices)),
                                            int((OH_frac+OH2_frac)*\
                                            float(len(centerIndices))))
                
            elif OH_frac != 0 and OH2_frac == 0:
                randIndices = random.sample(range(0, len(centerIndices)),
                                            len(centerIndices))
            
            else:
                raise IndexError("No suitable metal centers found (did you specify\
                              the correct atoms in config.wet?)")
            
            indices = centerIndices[randIndices]

            for center in indices:
                points = np.empty([0, 3], dtype=float)
                tempVectors = np.empty([0, 3], dtype=float)
                pairVectors = np.empty([0, 3], dtype=float)
                sumVec = np.array([0, 0, 0], dtype=float)
                points = np.vstack((points, self.topol.trj.xyz[0][center]*10))

                # Get neighbours
                neighbours = np.array(neighbourgraph[center])
                centerArray = np.repeat(center, len(neighbours))

                # Compute displacement vectors
                dispArray = np.vstack((neighbours, centerArray))
                dispArray = np.transpose(dispArray)
                dispVectors = md.compute_displacements(self.topol.trj,\
                                                       dispArray,\
                                                       periodic = True)[0]

                for vector in dispVectors:
                    if(np.linalg.norm(vector) < 0.1):
                        puts.warning("Directional vector less than 0.1,"+\
                              " will not hydrate atom with index: " +\
                              str(center) + ". Please verify structure"+\
                              " before using it.\n")
                        continue
                    vector = vector/np.linalg.norm(vector)
                    tempVectors = np.vstack((tempVectors, vector))
                    sumVec += vector # Sum M-O vectors

                sumVec = sumVec/np.linalg.norm(sumVec)

                for vector in tempVectors:
                    tempVector = sumVec - vector
                    pairVectors = np.vstack((pairVectors, tempVector))

                #Sort vectors with increasing norm
                pairVectors = sorted(pairVectors, 
                                    key = lambda x: np.sqrt(x[0]**2 + x[1]**2\
                                                            + x[2]**2))

                # Choose the two vectors with smallest norm
                pairVec1 = pairVectors[0]/np.linalg.norm(pairVectors[0])
                pairVec2 = pairVectors[1]/np.linalg.norm(pairVectors[1])

                # If there are only 2 neighbours, rotate Pi/2 around
                # directional vector to maximize distance to neighbours
                if(len(tempVectors) == 2):
                    axis = sumVec
                    angle = np.pi/2
                    pairVec1 = np.dot(mac.rotate_around_vec(axis, angle),\
                                      pairVec1)
                    pairVec2 = np.dot(mac.rotate_around_vec(axis, angle),\
                                      pairVec2)

                # Rotate the vectors towards each other
                # (away from center of bulk)
                crossProd = np.cross(pairVec1, pairVec2)
                dotProdVec1 = np.dot(sumVec, pairVec1)
                dotProdVec2 = np.dot(sumVec, pairVec2)
                if(dotProdVec1 < 0):
                    pairVec1 = np.dot(mac.rotate_around_vec(crossProd,\
                                                            np.pi/7), pairVec1)
                    pairVec2 = np.dot(mac.rotate_around_vec(crossProd,\
                                                           -np.pi/7), pairVec2)
                else:
                    angle = -np.pi/7
                    pairVec1 = np.dot(mac.rotate_around_vec(crossProd,\
                                                            np.pi/7), pairVec1)
                    pairVec2 = np.dot(mac.rotate_around_vec(crossProd,\
                                                           -np.pi/7), pairVec2)

                vectors = np.vstack((vectors, pairVec1))
                vectors = np.vstack((vectors, pairVec2))
                centers = np.append(centers, center)
                centers = np.append(centers, center)
                
            return vectors, centers

        else:
            return (np.empty([0, 3], dtype=float),
                    np.empty([0], dtype=int))


    def __calculate_vectors(self, coordination,\
                          OH_frac, OH2_frac, center):
        """Calculate coordinates for solvate on for 5-coordinated atoms

            Notes
            -----
            See calculate_pair_vectors

        """
        vectors = np.empty([0, 3], dtype=float)
        
        vec = np.array([0,0,0])
        tempNeighbour = np.array([0,0,0])
        #Get indices for metal centers with coordination Nmax - 1
        centerIndices, neighbourgraph = self.__get_center_neighbours(\
                                            coordination, center)

        if(len(centerIndices) > 0): # If there are any Nmax-1 centers
            centers = np.empty([0], dtype=int)

            self.__verboseprint(("Found {} centers that are Nmax - 1 = {}-fold"
                               " coordinated.\n".format(len(centerIndices),\
                                                       coordination)))

            #Calculate only for user specified fraction
            randIndices = random.sample(range(0, len(centerIndices)),\
                                        int((OH2_frac + OH_frac)*\
                                            float(len(centerIndices))))
            indices = centerIndices[randIndices]

            #Calculate M-O vectors
            for center in indices:
                vec = [0, 0, 0]

                neighbours = np.array(neighbourgraph[center])
                centerArray = np.repeat(center, len(neighbours))
                dispArray = np.vstack((neighbours, centerArray))
                dispArray = np.transpose(dispArray)

                dispVectors = md.compute_displacements(self.topol.trj,\
                                                       dispArray,\
                                                       periodic = True)[0]

                #for neighbour in neighbourgraph[center]:
                for vector in dispVectors:
                     vector = vector/np.linalg.norm(vector)
                     vec = vec + vector

                #If norm of directional vector too small
                if(np.linalg.norm(vec) < 0.1):
                    puts.warning("Directional vector almost 0, will " +\
                          "not hydrate at atom with index: " +\
                           str(center + 1))
                    print("Probably due to symmetric center...\n")

                else:
                    centers = np.append(centers, center)
                    vec = vec/np.linalg.norm(vec)
                    vectors = np.vstack((vectors, vec))

            return vectors, centers

        else: # Else return empty arrays
            return (np.empty([0, 3], dtype=float),
                    np.empty([0], dtype=int))


    def get_coordinates(self,params):
        """Calculates coordinates of solvate for specified metal center type

        Parameters
        ----------
        params : dict with keys::
            element : string
                Element symbol
            coordination : int
                coordination number of element
            OH_frac : float
                fraction of element atoms that should be hydrated with
                hydroxyl.
            OH2_frac : float
                fraction of element atoms that should be hydrated with
                water.
            Nmax : int
                Maximum coordination number
            dMOH : float
                M-OH bond distance
            dMOH2 : float
                M-OH2 bond distance
            <MOH : float
                MOH angle

        """
        element = params['element']
        coordination = params['coordination']
        OH_frac = params['OH']
        OH2_frac = params['OH2']
        Nmax = params['Nmax']
        dMOH = params['dMOH']
        dMOH2 = params['dMOH2']
        angle = params['<MOH']

        if(coordination == Nmax - 1):
            # Calculate directional vectors
            vectors, centers = self.__calculate_vectors(\
                                           coordination,\
                                           OH_frac, OH2_frac,
                                           element)
            
            
            if OH_frac + OH2_frac == 1:
                # Calculate coordinates from directional vectors and save.
                self.__hydCoords = np.vstack((self.__hydCoords,\
                                            self.topol.trj.xyz[0][centers[\
                                                :int(OH_frac*len(vectors))]]*10+\
                                                vectors[\
                                                :int(OH_frac*len(vectors))]*dMOH))
                self.__hydVectors = np.vstack((self.__hydVectors,\
                                             vectors[:int(OH_frac*len(vectors))]))
                self.__hydCenters = np.hstack((self.__hydCenters,\
                                             centers[:int(OH_frac*len(vectors))]))
                self.__watCoords = np.vstack((self.__watCoords,\
                                        self.topol.trj.xyz[0][centers[\
                                            int(OH_frac*len(vectors)):]]*10+\
                                            vectors[\
                                                int(OH_frac*len(vectors)):]*dMOH2))
                self.__watVectors = np.vstack((self.__watVectors,\
                                             vectors[int(OH_frac*len(vectors)):]))
                self.__watCenters = np.hstack((self.__watCenters,\
                                             centers[int(OH_frac*len(vectors)):]))
                # Save bondlengths and angles
                self.__dMOH = np.append(self.__dMOH, np.repeat(params['dMOH'],\
                                      len(centers[:int(OH_frac*len(vectors))])))
                self.__dMOH2 = np.append(self.__dMOH2, np.repeat(params['dMOH2'],\
                                       len(centers[int(OH_frac*len(vectors)):])))
                self.__watAngles = np.append(self.__watAngles,\
                                        np.repeat(angle,\
                                        len(centers[int(OH_frac*len(vectors)):])))
                self.__hydAngles = np.append(self.__hydAngles,\
                                        np.repeat(angle,\
                                        len(centers[:int(OH_frac*len(vectors))])))
            elif OH_frac != 0 and OH2_frac == 0:
                # Calculate coordinates from directional vectors and save.
                self.__hydCoords = np.vstack((self.__hydCoords,\
                                            self.topol.trj.xyz[0][centers[\
                                                :len(vectors)]]*10+\
                                                vectors[\
                                                :len(vectors)]*dMOH))
                self.__hydVectors = np.vstack((self.__hydVectors,\
                                             vectors[:len(vectors)]))
                self.__hydCenters = np.hstack((self.__hydCenters,\
                                             centers[:len(vectors)]))
                # Save bondlengths and angles
                self.__dMOH = np.append(self.__dMOH, np.repeat(params['dMOH'],\
                                      len(centers[:len(vectors)])))
                self.__hydAngles = np.append(self.__hydAngles,\
                                        np.repeat(angle,\
                                        len(centers[:len(vectors)])))
                

        elif(coordination == Nmax - 2):
            vectors, centers = self.__calculate_pair_vectors(\
                                           coordination,\
                                           OH_frac, OH2_frac,
                                           element)
                
            if(np.isnan(vectors).any()):
                raise ValueError("Some coordinates are NaN, aborting....")

            # Get random indices where to place hydroxyl and calculate
            # coordinates
            
            # By adding step = 2 to the range function, only
            # one OH group will be placed to the Nmax-2 metal center
            
            if len(vectors) != 0:
            
                if OH_frac <= 0.5:
                    randIndicesOH = random.sample(range(0, len(vectors), 2),\
                                                int((OH_frac)*\
                                                    float(len(vectors))))
                else:
                    raise ValueError("Cannot put more than 50% hydroxyl groups to Nmax - 2 sites!")
               
                if OH_frac + OH2_frac == 1:
                    self.__hydCoords = np.vstack((self.__hydCoords,\
                                            self.topol.trj.xyz[0][centers[randIndicesOH]]*10+\
                                            vectors[randIndicesOH]*dMOH))
                    self.__hydVectors = np.vstack((self.__hydVectors,\
                                                   vectors[randIndicesOH]))
                    self.__hydCenters = np.hstack((self.__hydCenters,\
                                                   centers[randIndicesOH]))
                    
                    
                    # Create mask and calculate water coordiantes
                    mask = np.ones(len(vectors), np.bool)
                    mask[randIndicesOH] = 0
                    self.__watCoords = np.vstack((self.__watCoords,\
                                            self.topol.trj.xyz[0][centers[mask]]*10+\
                                            vectors[mask]*dMOH2))
                    self.__watVectors = np.vstack((self.__watVectors, vectors[mask]))
                    self.__watCenters = np.hstack((self.__watCenters, centers[mask]))
        
                    # Save bondlengths and angles
                    self.__dMOH = np.append(self.__dMOH, np.repeat(params['dMOH'],\
                                          len(centers[randIndicesOH])))
                    self.__dMOH2 = np.append(self.__dMOH2, np.repeat(params['dMOH2'],\
                                           len(centers[mask])))
                    self.__watAngles = np.append(self.__watAngles,\
                                                 np.repeat(angle,\
                                                           len(centers[mask])))
                    self.__hydAngles = np.append(self.__hydAngles,\
                                                 np.repeat(angle,\
                                                           len(centers[randIndicesOH])))
                elif OH_frac != 0 and OH2_frac == 0:
                    self.__hydCoords = np.vstack((self.__hydCoords,\
                                            self.topol.trj.xyz[0][centers[randIndicesOH]]*10+\
                                            vectors[randIndicesOH]*dMOH))
                    self.__hydVectors = np.vstack((self.__hydVectors,\
                                                   vectors[randIndicesOH]))
                    self.__hydCenters = np.hstack((self.__hydCenters,\
                                                   centers[randIndicesOH]))
                    
                    # Save bondlengths and angles
                    self.__dMOH = np.append(self.__dMOH, np.repeat(params['dMOH'],\
                                          len(centers[randIndicesOH])))
                    self.__hydAngles = np.append(self.__hydAngles,\
                                                 np.repeat(angle,\
                                                           len(centers[randIndicesOH])))
            

        else:
            raise ValueError('Can only hydrate Nmax - 1 and Nmax - 2 centers.\
                             You tried to hydrate ' + str(Nmax) + ' - ' +\
                             str(coordination-Nmax) + ' centers. To solve this\
                                                       issue edit config.wet.')

    def optimize(self):
        """Maximize distance between added solvate and it's neighbours

        Notes
        -------
        Collects all coordinates and invokes optimization routine

        """
        if(len(self.__hydCoords) < 1 and len(self.__watCoords) < 1):
            raise IndexError("No suitable metal centers found (did you specify\
                              the correct atoms in config.wet?)")

        vectors = np.empty([0, 3], dtype=float)
        coords = np.empty([0, 3], dtype=float)
        centers= np.empty([0], dtype=int)

        #Collect coordinates and centers and call optimize()
        coords = np.vstack((coords, self.__hydCoords))
        coords = np.vstack((coords, self.__watCoords))
        centers = np.hstack((centers, self.__hydCenters.astype(int)))
        centers = np.hstack((centers, self.__watCenters.astype(int)))

        # Optimize
        coords, vectors = self.__optimizer(coords, centers)

        #Set attributes to optimized values
        self.__hydCoords = coords[:len(self.__hydCoords)]
        self.__hydVectors = vectors[:len(self.__hydCoords)]
        self.__watCoords = coords[len(self.__hydCoords):]
        self.__watVectors = vectors[len(self.__hydCoords):]


    def wet(self, print_dist = False):
        """Creates atoms at the calculated coordinates

        Notes
        -----
        Updates the `coords` and `elements` attributes of the `Wetter`
        object.

        """
        # Create solvent molecules at calculated coordinates and rotate
        # accordingly
        hydCoords, hydElements = mac.add_hydroxyl(self.__hydCoords, 
                                                  self.__hydVectors, 
                                                  self.theta,
                                                  self.__hydAngles)
        watCoords, watElements = mac.add_water(self.__watCoords, 
                                               self.__watVectors, 
                                               self.theta,
                                               self.__watAngles)

        coords = np.concatenate((watCoords, hydCoords))
        elements = np.concatenate((watElements, hydElements))
        self.coords = coords
        self.elements = elements
        self.__verboseprint("Generating output...")
        minHDist = 0
        minODist = 0
        minStructDist = 0
        maxStructDist = 0

        # Generate output
        if(print_dist):
            minODist,\
                minHDist,\
                minStructDist,\
                maxStructDist = shortest_distance(coords.astype(float), elements,\
                                                (self.topol.trj.xyz[0]*10).astype(float),\
                                                self.boxVectors.astype(float))
            print("\nShortest O-O distance in solvent: " +\
                                str(minODist) + " Angstrom.")
            print("Shortest H-H distance in solvent: " +\
                                str(minHDist) + " Angstrom.")
            print("Shortest distance between solvent and structure:"\
                                + " " + str(minStructDist) + " Angstrom.")
            print("Longest distance between solvent oxygens and "+\
                                "structure: " + str(maxStructDist) +\
                                " Angstrom.\n")

    def remove_low_coordinated(self, Nmax, element, check = 'all'):
        """Removes low coordinated atoms

        Parameters
        ----------
        Nmax : int
            Maximum coordination for the specified element
        element : string
            Element symbol
        check : {'all', 'metal', 'none'}
            wether to remove all, none or only oxygen low coordinated atoms
        
        Notes
        -----
        Low coordinated atoms are `Nmax - 1` and `Nmax - 2` coordinated

        """
        self.topol = rlc(self.topol, Nmax, element, self.silent, check)

    def save(self, fileWet, OHresname = 'HYD', OH2resname = 'SOL'):
        """Saves the new solvated structure

        Parameters
        ----------
        fileWet : string
            Name of output file
        resname : string
            Residue name for the solvate molecules

        Notes
        -----
        fileWet should have the extension `.pdb` since it's in the pdb format

        """
        self.topol.trj.save(fileWet, force_overwrite = True)
        append_atoms(file = fileWet, coords = self.coords,\
                     elements = self.elements, OHresname = OHresname,\
                     OH2resname = OH2resname, numOfOH2 = len(self.__watCoords))
        self.__verboseprint("Added " + str(len(self.__watCoords)) +\
                            " waters and " + str(len(self.__hydCoords)) +\
                            " hydroxyls to " + fileWet)
