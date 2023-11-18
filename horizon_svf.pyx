cimport numpy as np
import numpy as np

cdef extern from "horizon_svf_comp.h":
    void horizon_svf_comp(double* vlon, double* vlat, double* clon,
                          double* clat, float* topography_v,
                          np.npy_int32* vertex_of_cell, int vertex, int cell,
                          float* horizon, float* skyview, int nhori, np.npy_int32* mask)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.float64_t, ndim = 1] clon,
                        np.ndarray[np.float64_t, ndim = 1] clat,
                        np.ndarray[np.float32_t, ndim = 1] topography_v,
                        np.ndarray[np.int32_t, ndim = 2] vertex_of_cell,
                        int vertex,
                        int cell,
                        int nhori,
                        np.ndarray[np.int32_t, ndim = 1] mask):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    vlon : ndarray of double
        Array (one-dimensional) with longitude of cell vertices [radian]
    vlat : ndarray of double
        Array (one-dimensional) with latitude of cell vertices [radian]
    clon : ndarray of double
        Array (one-dimensional) with longitude of cell circumcenters [radian]       
    clat : ndarray of double
        Array (one-dimensional) with latitude of cell circumcenters [radian]
    topography_v : ndarray of float
        Array (one-dimensional) with elevation of cell vertices [metre]
    vertex_of_cell : ndarray of int
        Array (two-dimensional) with indices of cell vertices
    vertex : int
        Total number of cell vertices
    cell : int
        Number of cells
    nhori : int
        Number of terrain horizon sampling directions
    mask : ndarray of integers
        Array (one-dimensional) representing a mask for the cells
    Returns
    -------
    horizon : ndarray of float
        Array (two-dimensional) with terrain horizon [degree]
    skyview : ndarray of float
        Array (one-dimensional) with sky view factor [-]"""

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
        horizon = np.empty((nhori, cell), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        skyview = np.empty(cell, dtype=np.float32)
    
    # Ensure that passed arrays are contiguous in memory
    vertex_of_cell = np.ascontiguousarray(vertex_of_cell)

    # Call C++ function and pass arguments
    horizon_svf_comp(&vlon[0], &vlat[0], &clon[0], &clat[0],
                     &topography_v[0], &vertex_of_cell[0, 0], vertex, 
                     cell, &horizon[0, 0], &skyview[0], nhori, &mask[0])

    return horizon, skyview

