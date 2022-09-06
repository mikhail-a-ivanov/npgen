import numpy as np

class units:
    # Conversion factors between units
    ANGSTROM_TO_NM = 0.1
    BOHR_TO_NM = 0.052917725
    HARTREE_TO_KJPERMOL = 2625.499

def sort_pairs(x):
    """Independent sort of the rows of an (N x 2)-array

    The rows of the sorted array fulfills xi < xj.

    Parameters
    ----------
    x: ndarray, shape=(N,2)
        Array to be sorted

    Returns
    -------
    out: ndarray, shape=(N,2)
        The sorted array.

    """
    arr = np.asarray(x)
    return np.select([np.tile(arr[:,0] <  arr[:,1], (2,1)).T,
                      np.tile(arr[:,0] >= arr[:,1], (2,1)).T],
                     [arr,
                      arr[:,::-1]])

def sort_triplets(x):
    """Independent sort of the rows of an (N x 3)-array

    The rows of the sorted array fulfills xi < xk.

    Parameters
    ----------
    x: ndarray, shape=(N,3)
        Array to be sorted

    Returns
    -------
    out: ndarray, shape=(N,3)
        The sorted array.

    """
    arr = np.asarray(x)
    return np.select([np.tile(arr[:,0] <  arr[:,2], (3,1)).T,
                      np.tile(arr[:,0] >= arr[:,2], (3,1)).T],
                     [arr,
                      arr[:,::-1]])

def sort_quadruplets(x):
    """Independent sort of the rows of an (N x 4)-array

    The rows of the sorted array fulfills xj < xk.

    Parameters
    ----------
    x: ndarray, shape=(N,4)
        Array to be sorted

    Returns
    -------
    out: ndarray, shape=(N,4)
        The sorted array.

    """
    arr = np.asarray(x)
    return np.select([np.tile(arr[:,1] <  arr[:,2], (4,1)).T,
                      np.tile(arr[:,1] == arr[:,2], (4,1)).T,
                      np.tile(arr[:,1] >  arr[:,2], (4,1)).T],
                     [arr,
                      np.insert(np.sort(arr[:,[0,3]], axis=1),
                                [1],
                                arr[:,[1,2]], axis=1),
                      arr[:,::-1]])

def numpy_replace(arr, mapping):
    """ Replace with dict in numpy array.

    Replace the entries in an array according to some mapping dictionary k: v.

    Parameters
    ----------
    arr: ndarray, shape=(N,M)
        Array to be replaced
    mapping: dict
        Dictionary so that keys are mapped to values in `arr`.

    Returns
    -------
    out: ndarray, shape=(N,M)
        Array with entries mapped by `mapping.`

    """
    sort_idx = np.argsort(np.array(list(mapping.keys())))
    idx = np.searchsorted(np.array(list(mapping.keys())), arr, sorter=sort_idx)
    return np.array(list(mapping.values()))[sort_idx][idx]

def common_rows(x, y):
    """Return boolean array of the rows of x that can be found in y.

    Return an array which elements are `True` for the rows in `x` that can be
    found in `y.`

    Parameters
    ----------
    x: ndarray, shape=(N,2)
        First array
    y: ndarray, shape=(M,2)
        Second array

    Returns
    -------
    out: ndarray, shape=(N,)

    """
    X = np.asarray(x)
    Y = np.asarray(y)

    return (X == Y[:,None]).all(-1).any(0)

def bimodality_coefficient(x):
    """Calculate bimodality coefficient of a data series
    
    Parameters
    ----------
    x: Series
        
    Returns
    -------
    out: scalar
        The bimodality coefficient
    """
    N = len(x)

    if N <= 3:
        return np.nan

    m3 = x.skew()
    m4 = x.kurtosis()

    bi = ( m3**2 + 1 ) / ( m4 + 3 * ( (N - 1)**2 / ( (N - 2) * (N - 3) ) ) )

    return bi - (5.0 / 9.0)
