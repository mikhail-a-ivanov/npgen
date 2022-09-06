import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
import mdtraj as md
import sys

# For debugging
#import IPython
#def embed(locs):
#    # Pass the locals explicitly
#    IPython.embed(user_ns=locs)
 
def numpy_replace(arr, mapping):
    """ Replace values in numpy array using dict"""
    sort_idx = np.argsort(np.fromiter(mapping.keys(), dtype=int))
    idx = np.searchsorted(np.fromiter(mapping.keys(), dtype=int), arr, sorter=sort_idx)
    return np.fromiter(mapping.values(), dtype=float)[sort_idx][idx]

def numpy_groupby(arr, col=0):
    """Group rows in array according to values in one column."""
    values = arr[:,col].astype('u4')
    dif = np.ones(values.shape,values.dtype)
    dif[1:] = np.diff(values)
    idx = np.where(dif>0)
    vals = values[idx]
    return idx[0],vals

def compute_neighborlist_rad(trj, dict cutoffs, int frame=0, periodic=True):   
    """Compute RAD_and neighborlist"""
    cdef int i,j,k,n
    cdef float Rc_max
    cdef float rr,rij,rik,rij2,rik2,rijinv2,rikinv2,costjik
    cdef list RADs
    cdef dict neighborlist_rad,neighborlist_rad_and
    cdef dict r_dict,cost_dict,symmetric

    cdef np.ndarray[np.int_t] atomic_numbers,sorted_indices,atom_indices
    cdef np.ndarray[np.uint32_t] atoms
    cdef np.ndarray[np.int_t, ndim=2] pairs,triplets
    cdef np.ndarray[np.float32_t] dists,th,cost
    cdef np.ndarray[np.float64_t] Rc

    # Maximum is twice the largest cutoff
    Rc_max = 2.0*max(cutoffs.values())
    # Load trajectory frame
    fr = trj[frame]

    # Make atomic numbers available
    atomic_numbers = np.array([a.element.atomic_number for a in fr.top.atoms],
                              dtype=np.int)

    # Step 1: Compute neighbors that lie within the maximum cutoff
    neighborlist = md.compute_neighborlist(fr, Rc_max, periodic=periodic)
    # Step 2: Keep only neighbors that fulfill the BONES criterium
    pairs = np.vstack((map(lambda e: np.column_stack((e[0]*np.ones(len(e[1]),
                                                                   dtype=np.int),
                                                      e[1])),
                           enumerate(neighborlist))))
    dists = md.compute_distances(fr,pairs).reshape(-1)

    Rc = numpy_replace(atomic_numbers[pairs],cutoffs).sum(axis=1)
    
    pairs = pairs[dists<Rc,:]
    dists = dists[dists<Rc]

    # Step 3: Keep only neighbors that fulfill the RAD criterium
    # Sort neighbors according to distance
    sorted_indices = np.lexsort((dists,pairs[:,0]))
    
    pairs = pairs[sorted_indices]
    dists = dists[sorted_indices]
    
    atom_indices,atoms = numpy_groupby(pairs, col=0)    
    neighborlist = list(map(lambda x: x[:,1].reshape(-1),np.split(pairs,atom_indices[1:])))

    # Generate atom triplets for central atom and its neighbor pairs
    triplets = np.vstack(([np.column_stack((
        n*np.ones(len(nbs)**2, dtype=np.int),
        np.array(nbs)[np.rollaxis(np.indices((len(nbs),)*2), 0, 3).reshape(-1,2)]
    ))
                           for (n,nbs) in zip(atoms,neighborlist)]))

    # Store atoms as j-i-k
    triplets[:,0], triplets[:,1] = triplets[:,1], triplets[:,0].copy()
    # Compute angles
    th = md.compute_angles(fr,triplets).reshape(-1)
    # Take care of rounding errors
    th[triplets[:,0]==triplets[:,2]] = 0
    th[np.isnan(th)] = np.pi
    
    cost = np.cos(th)

    # Store values as dicts with triplets as keys
    r_dict = {tuple(r): dists[i] for (i,r) in enumerate(pairs)}
    cost_dict = {tuple(r): cost[i] for (i,r) in enumerate(triplets)}

    # RAD loop
    neighborlist_rad = {}
    symmetric = {}
    for (i,neighbors) in zip(atoms,neighborlist):
        RADs = []
        for j in neighbors:
            blocked = 0
            for k in neighbors: 
                costjik = cost_dict[j,i,k]
                
                rij = r_dict[i,j]
                rik = r_dict[i,k]                 
                rij2 = rij*rij
                rik2 = rik*rik
                rijinv2 = 1.0/rij2
                rikinv2 = 1.0/rik2
                                  
                if rijinv2 < rikinv2*costjik:
                    blocked = 1
                    break
            if blocked == 1:
                break
            else:
                RADs.append((j,r_dict[i,j]))
                symmetric[i,j] = 1
        
        neighborlist_rad[i] = RADs

    # Now we have calculated RAD. Screen for RAD_and, i.e., both atoms i and j
    # needs to have the other one in the shell to be kept.
    neighborlist_rad_and = {}
    for (i,neighbors) in neighborlist_rad.items():
        RADs = []
        for (j,rr) in neighbors:
            if (j,i) in symmetric:
                RADs.append((j,rr))
                
        neighborlist_rad_and[i] = RADs
        
    return neighborlist_rad_and

