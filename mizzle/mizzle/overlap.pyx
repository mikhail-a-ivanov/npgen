"""Calculates minimum distance between atoms

"""

import cython
import numpy as np
cimport numpy as np
from radish import Topologizer

FTYPE = np.float32
ITYPE = np.int_
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t
@cython.cdivision(True)
@cython.boundscheck(False)

def shortest_distance(np.ndarray[np.float64_t, ndim=2] solvCoords,
                      np.ndarray elements, np.ndarray[np.float64_t,
                      ndim=2] structCoords,
                      np.ndarray[np.float64_t, ndim=1] boxVectors):
    """Overlap function

    Parameters
    ----------
    solvCoords : numpy ndarray
        Coordinates for solvent atoms
    elements : ndarray
        Array with element symbols
    structCoords : ndarray
        Coordinates for structure coordinates
    boxVectors : ndarray
        Box vectors for PBC

    Returns
    -------
    minODist
        Min distance between solvent oxygens
    minHDist
        Min distance of solvent hydrogens
    minStructDist
        Min solvent - structure distance
    maxStructDist
        Max solvent - structure distance

    """
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef double minODist = 9999.999
    cdef double minHDist = 9999.999
    cdef double minStructDist = 9999.999
    cdef double maxStructDist = 0
    cdef double distance = 0

    cdef np.float64_t tempX = 0
    cdef np.float64_t tempY = 0
    cdef np.float64_t tempZ = 0

    cdef double tempMinStructDist = 9999.999
    cdef int solvateLen = len(solvCoords)
    cdef int structLen = len(structCoords)

    while i < solvateLen:
        j = i + 1
        while j < solvateLen:
            if(elements[i] == 'O' and elements[j] == 'O'):
                tempX = solvCoords[j][0]
                tempY = solvCoords[j][1]
                tempZ = solvCoords[j][2]

                if(boxVectors[0] > 0):
                    if(solvCoords[j][0] - solvCoords[i][0] > boxVectors[0]/2):
                        tempX = solvCoords[j][0] - boxVectors[0]
                    elif(solvCoords[j][0] - solvCoords[i][0] < -boxVectors[0]/2):
                        tempX = solvCoords[j][0] + boxVectors[0]
                    if(solvCoords[j][1] - solvCoords[i][1] > boxVectors[1]/2):
                        tempY = solvCoords[j][1] - boxVectors[1]
                    elif(solvCoords[j][1] - solvCoords[i][1] < -boxVectors[1]/2):
                        tempY = solvCoords[j][1] + boxVectors[1]
                    if(solvCoords[j][2] - solvCoords[i][2] > boxVectors[2]/2):
                        tempZ = solvCoords[j][2] - boxVectors[2]
                    elif(solvCoords[j][2] - solvCoords[i][2] < -boxVectors[2]/2):
                        tempZ = solvCoords[j][2] + boxVectors[2]

                distance = ((solvCoords[i][0] - tempX)**2 + \
                            (solvCoords[i][1] - tempY)**2 + \
                            (solvCoords[i][2] - tempZ)**2)**(1.0/2)
                if(distance < minODist):
                    minODist = distance

            elif(elements[i] == 'H' and elements[j] == 'H'):
                tempX = solvCoords[j][0]
                tempY = solvCoords[j][1]
                tempZ = solvCoords[j][2]

                if(boxVectors[0] > 0):
                    if(solvCoords[j][0] - solvCoords[i][0] > boxVectors[0]/2):
                        tempX = solvCoords[j][0] - boxVectors[0]
                    elif(solvCoords[j][0] - solvCoords[i][0] < -boxVectors[0]/2):
                        tempX = solvCoords[j][0] + boxVectors[0]
                    if(solvCoords[j][1] - solvCoords[i][1] > boxVectors[1]/2):
                        tempY = solvCoords[j][1] - boxVectors[1]
                    elif(solvCoords[j][1] - solvCoords[i][1] < -boxVectors[1]/2):
                        tempY = solvCoords[j][1] + boxVectors[1]
                    if(solvCoords[j][2] - solvCoords[i][2] > boxVectors[2]/2):
                        tempZ = solvCoords[j][2] - boxVectors[2]
                    elif(solvCoords[j][2] - solvCoords[i][2] < -boxVectors[2]/2):
                        tempZ = solvCoords[j][2] + boxVectors[2]

                distance = ((solvCoords[i][0] - tempX)**2 + \
                            (solvCoords[i][1] - tempY)**2 + \
                            (solvCoords[i][2] - tempZ)**2)**(1.0/2)

                if(distance < minHDist):
                    minHDist = distance

            j += 1
        i += 1

    i = 0
    while i < solvateLen:
        tempMinStructDist = 9999.999

        while k < structLen:
            tempX = structCoords[k][0]
            tempY = structCoords[k][1]
            tempZ = structCoords[k][2]

            if(boxVectors[0] > 0):
                if(structCoords[k][0] - solvCoords[i][0] > boxVectors[0]/2):
                    tempX = structCoords[k][0] - boxVectors[0]
                elif(structCoords[k][0] - solvCoords[i][0] < -boxVectors[0]/2):
                    tempX = structCoords[k][0] + boxVectors[0]
                if(structCoords[k][1] - solvCoords[i][1] > boxVectors[1]/2):
                    tempY = structCoords[k][1] - boxVectors[1]
                elif(structCoords[k][1] - solvCoords[i][1] < -boxVectors[1]/2):
                    tempY = structCoords[k][1] + boxVectors[1]
                if(structCoords[k][2] - solvCoords[i][2] > boxVectors[2]/2):
                    tempZ  = structCoords[k][2]  - boxVectors[2]
                elif(structCoords[k][2] - solvCoords[i][2] < -boxVectors[2]/2):
                    tempZ  = structCoords[k][2] + boxVectors[2]

            distance = ((solvCoords[i][0] - tempX)**2 + \
                (solvCoords[i][1] - tempY)**2 + \
                (solvCoords[i][2] - tempZ)**2)**(1.0/2)

            if(distance < tempMinStructDist):
                tempMinStructDist = distance
            k += 1

        if(elements[i] == 'O'):
            if(tempMinStructDist > maxStructDist):
                maxStructDist = tempMinStructDist
        if(tempMinStructDist < minStructDist):
            minStructDist = tempMinStructDist
        i += 1
        k = 0

    return minODist, minHDist, minStructDist, maxStructDist