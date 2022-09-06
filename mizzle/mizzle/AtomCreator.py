"""Creates and alignes solvate coordinates to metal centers

"""

import numpy as np
import random

def __xRotate(vector, angle):
    """Rotates vector around x-axis

    Parameters
    ----------
    vector : array
        Vector/point to rotate
    angle
        Angle by which the vector is rotated

    Returns
    -------
    dotProd
        Rotated vector
    """

    rotMatrix = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],\
                 [0, np.sin(angle), np.cos(angle)]]
    dotProd = np.dot(rotMatrix, vector)
    return (dotProd)

def __skew(v):
    vecs = np.atleast_2d(v)

    # Number of vectors
    N = len(vecs)

    # Create skew matrix
    S = np.array([[np.zeros(N), -vecs[:,2], vecs[:,1]],
                  [vecs[:,2], np.zeros(N), -vecs[:,0]],
                  [-vecs[:,1], vecs[:,0], np.zeros(N)]])
    # Roll back axis
    S = np.rollaxis(S, -1)

    return S

def rotate_around_vec(axis, angle):
    """Rotates around vector

    Returns
    -------
    R[0] : ndarray
        Rotation matrix
    """
    #mag = np.linalg.norm(vec)
    #vec = vec/mag
    theta = np.atleast_1d(angle)
    u = np.atleast_2d(axis)
    u = np.resize(u, len(theta)*3).reshape(-1,3)

    assert theta.ndim == 1

    # As unit vectors
    u = u/np.linalg.norm(u, axis=1, keepdims=True)

    # Build rotation matrix
    I  = np.einsum("n,ab->nab", np.cos(theta), np.eye(3))
    ux = np.einsum("n,nab->nab", np.sin(theta), __skew(u))
    uu = np.einsum("n,na,nb->nab", 1 - np.cos(theta), u, u)

    R = I + ux + uu

    # rotMatrix = np.array([[np.cos(theta)+vec[0]**2(1-np.cos(theta)), vec[0]*vec[1](1-np.cos(theta))-vec[2]*np.sin(theta), vec[0]*vec[2](1-np.cos(theta))+vec[1]*np.sin(theta)],
    #                   [vec[0]*vec[1](1-np.cos(theta))+vec[2]*np.sin(theta), np.cos(theta)+vec[1]**2(1-np.cos(theta)), 2*vec[1]*vec[2]],
    #                   [2*vec[0]*vec[2], 2*vec[1]*vec[2], np.cos(theta)+vec[2]**2(1-np.cos(theta))]])
    return R[0]

def __align(vec1, vec2):
    """Aligns one vector to another

    Parameters
    ----------
    vec1 : array
        vec2 is aligned with vec1
    vec2 : array
        see vec1

    Returns
    -------
    rotMatrix
        Rotational matrix which performs the rotation
    """
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    crossProd = np.cross(vec1, vec2)
    dotProd = np.dot(vec1, vec2)

    if(dotProd < 1.01 and dotProd > 0.99): # Parallell vectors
        rotMatrix = I

    elif(dotProd < -0.99 and dotProd > -1.01): # Anti parallell vectors
        #mag = np.sqrt(crossProd[0]**2 + crossProd[1]**2 + crossProd[2]**2)
        #ortVec = crossProd/mag

        #rotMatrix = -np.array([[-1+2*ortVec[0]**2, 2*ortVec[0]*ortVec[1], 2*ortVec[0]*ortVec[2]],
        #                   [2*ortVec[0]*ortVec[1], -1+2*ortVec[1]**2, 2*ortVec[1]*ortVec[2]],
        #                   [2*ortVec[0]*ortVec[2], 2*ortVec[1]*ortVec[2], -1+2*ortVec[2]**2]])
        rotMatrix = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) #-I


    else:
        skewSym = np.array([[0, -crossProd[2], crossProd[1]],[crossProd[2], 0,\
                             -crossProd[0]],[-crossProd[1], crossProd[0], 0]])
        prod = np.matmul(skewSym, skewSym)*(1/(1+dotProd))
        rotMatrix = np.add(np.add(I, skewSym), prod)

    return rotMatrix

def add_oxygen(coords, vectors):
    """Adds 1-fold coordinated oxygen

    Parameters
    ----------
    coords : array
        coordinates where oxygen should be placed
    vectors : array
        Directional vectors

    Returns
    -------
    atoms
        Oxygen coordinates
    elements
        List of element symbols
    """

    atoms = np.empty([0, 3], dtype=float)
    elements = []

    i=0
    while i < len(coords):
        O = np.array([0, 0, 0])

        #Translate to correct coordinates
        transVector = [coords[i][0] - O[0], coords[i][1] - O[1],\
                                            coords[i][2] - O[2]]
        O = np.array([O[0] + transVector[0], O[1] + transVector[1],\
                                             O[2] + transVector[2]])

        i += 1

        #Save atoms to be added to pdb file
        atoms = np.vstack((atoms, O))
        elements.extend('O')
    return atoms, elements

def add_hydroxyl(coords, vectors, theta, angles):
    """Creates hydroxyl at specified coordinates

    Parameters
    ----------
    coords : array
        coordinates where the oxygen in hydroxyl should be placed
    vectors : array
        Directional vectors along which to place 

    Returns
    -------
    atoms
        Oxygen and hydrogen coordinates
    elements
        List of element symbols
    """

    atoms = np.empty([0, 3], dtype=float)
    elements = []
 
    i=0
    while i < len(coords):
        O = np.array([0, 0, 0])
        H = np.array([np.sin(theta/2), 0, np.cos(theta/2)])

        #No need to rotate O since it lies on the x-axis
        angle = np.pi - np.arccos(np.cos(np.radians(angles[i]))/\
                                  np.cos(np.radians(104.5/2)))
        H = __xRotate(H, angle)

        #Align z axis to the directional vector
        rotMatrix = __align([0, 0, 1], vectors[i])
        O = np.dot(rotMatrix, O)
        H = np.dot(rotMatrix, H)

        #Rotate randomly aling directional vector
        axis = vectors[i]
        randAngle = random.uniform(0.1, 2*np.pi)
        H = np.dot(rotate_around_vec(axis, randAngle), H)

        #Translate to correct coordinates
        transVector = [coords[i][0] - O[0], coords[i][1] - O[1],\
                                            coords[i][2] - O[2]]
        O = np.array([O[0] + transVector[0], O[1] + transVector[1],\
                                             O[2] + transVector[2]])
        H = np.array([H[0] + transVector[0], H[1] + transVector[1],\
                                             H[2] + transVector[2]])

        i += 1

        #Save atoms to be added to pdb file
        atoms = np.vstack((atoms, O))
        elements.append("OF")
        atoms = np.vstack((atoms, H))
        elements.append("HA")
    return atoms, elements

def add_water(coords, vectors, theta, angles):
    """Creates water molecules at specified coordinates

    Parameters
    ----------
    coords : array
        Coordinates where the oxygen in water should be placed
    vectors : array
        Directional vectors

    Returns
    -------
    atoms
        Oxygen and hydrogen coordinates
    elements
        List of element symbols
    """
    atoms = np.empty([0, 3], dtype=float)
    elements = []

    i=0
    while i < len(coords):
        O = np.array([0, 0, 0])
        H1 = np.array([np.sin(theta/2), 0, np.cos(theta/2)])
        H2 = np.array([-np.sin(theta/2), 0, np.cos(theta/2)])

        #No need to rotate O since it lies in the xz plane
        angle = np.pi - np.arccos(np.cos(np.radians(angles[i]))/\
                                         np.cos(np.radians(104.5/2)))
        H1 = __xRotate(H1, angle)
        H2 = __xRotate(H2, angle)

        #Align z axis to the directional vector
        rotMatrix = __align([0, 0, 1], vectors[i])
        H1 = np.dot(rotMatrix, H1)
        H2 = np.dot(rotMatrix, H2)

        #Rotate randomly along directional vector
        axis = vectors[i]
        randAngle = random.uniform(0.1, 2*np.pi)
        H1 = np.dot(rotate_around_vec(axis, randAngle), H1)
        H2 = np.dot(rotate_around_vec(axis, randAngle), H2)

        #Translate to correct coordinates
        transVector = [coords[i][0] - O[0], coords[i][1] - O[1],\
                                            coords[i][2] - O[2]]
        O = np.array([O[0] + transVector[0], O[1] + transVector[1],\
                                             O[2] + transVector[2]])
        H1 = np.array([H1[0] + transVector[0], H1[1] + transVector[1],\
                                               H1[2] + transVector[2]])
        H2 = np.array([H2[0] + transVector[0], H2[1] + transVector[1],\
                                               H2[2] + transVector[2]])

        i += 1


        #Save atoms to be added to pdb file
        atoms = np.vstack((atoms, O))
        elements.append('OH')
        atoms = np.vstack((atoms, H1))
        elements.append('HA')
        atoms = np.vstack((atoms, H2))
        elements.append('HA')


    return atoms, elements
