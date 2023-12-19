cimport numpy as np
import numpy as np

cdef extern from "horizon_svf_comp.h":
    void horizon_svf_comp(double* vlon, double* vlat, float* topography_v, 
                          int vertex, 
                          double* clon, double* clat,
                          np.npy_int32* vertex_of_cell,
                          int cell,
                          float* horizon, float* skyview, int nhori,
                          int refine_factor,
                          int svf_type)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.float32_t, ndim = 1] topography_v,
                        np.ndarray[np.float64_t, ndim = 1] clon,
                        np.ndarray[np.float64_t, ndim = 1] clat,
                        np.ndarray[np.int32_t, ndim = 2] vertex_of_cell,
                        int nhori,
                        int refine_factor,
                        int svf_type):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    vlon : ndarray of double
        Array with longitude of cell vertices (vertex) [rad]
    vlat : ndarray of double
        Array with latitude of cell vertices (vertex) [rad]
    topography_v : ndarray of float
        Array with elevation of cell vertices (vertex) [m]
    clon : ndarray of double
        Array with longitude of cell circumcenters (cell) [rad]
    clat : ndarray of double
        Array with latitude of cell circumcenters (cell) [rad]
    vertex_of_cell : ndarray of int
        Array with indices of cell vertices. Indices start with 1 according to
        Fortran (3, cell)
    nhori : int
        Number of terrain horizon sampling directions
    refine_factor : int
        Refinement factor that subdivides 'nhori' for more robust results
    svf_type : int
        Method for computing the Sky View Factor (SVF)
            0: Visible sky fraction; pure geometric skyview-factor
            1: SVF for horizontal surface; geometric scaled with sin(horizon)
            2: ?; geometric scaled with sin(horizon)**2
            3: SVF for sloped surface according to HORAYZON

    Returns
    -------
    horizon : ndarray of float
        Array (two-dimensional) with terrain horizon [deg]
    skyview : ndarray of float
        Array (one-dimensional) with sky view factor [-]"""

    # Check consistency and validity of input arguments
    if (vlon.size != vlat.size) or (vlat.size != topography_v.size):
        raise ValueError("Inconsistent lengths of input arrays 'vlon',"
                         "'vlat' and 'topography_v'")
    if (clon.size != clat.size) or (clat.size != vertex_of_cell.shape[1]):
        raise ValueError("Inconsistent lengths of input arrays 'clon',"
                         "'clat', and 'vertex_of_cell'")
    if vertex_of_cell.shape[0] != 3:
        raise ValueError("First dimension of 'vertex_of_cell' must have "
                            + "length 3")
    if (nhori < 4) or (nhori > 3_600):
        raise ValueError("'nhori' must be in the range [4, 3_600]")
    if (vertex_of_cell.min() < 1) or (vertex_of_cell.max() > vlon.size):
        raise ValueError("Indices of 'vertex_of_cell' out of range")
    if (refine_factor < 1) or (refine_factor > 100):
        raise ValueError("'refine_factor' must be in the range [1, 100]")
    if (svf_type < 0) or (svf_type > 3):
        raise ValueError("'svf_type' must be in the range [0, 3]")

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
        horizon = np.empty((nhori, clon.size), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        skyview = np.empty(clon.size, dtype=np.float32)
    
    # Ensure that passed arrays are contiguous in memory
    vertex_of_cell = np.ascontiguousarray(vertex_of_cell)

    # Call C++ function and pass arguments
    horizon_svf_comp(&vlon[0], &vlat[0], &topography_v[0], vlon.size,
                     &clon[0], &clat[0], &vertex_of_cell[0, 0],
                     clon.size, 
                     &horizon[0, 0], &skyview[0], nhori, refine_factor, 
                     svf_type)

    return horizon, skyview

