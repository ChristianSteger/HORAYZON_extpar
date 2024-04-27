cimport numpy as np
import numpy as np

cdef extern from "horizon_svf_comp.h":
    void horizon_svf_comp(double* clon, double* clat, float* hsurf,
                          int num_vertex,
                          np.npy_int32* vertex_of_triangle, int num_triangle,
                          float* horizon, float* skyview,
                          int nhori, int refine_factor, int svf_type)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] clon,
                        np.ndarray[np.float64_t, ndim = 1] clat,
                        np.ndarray[np.float32_t, ndim = 1] hsurf,
                        np.ndarray[np.int32_t, ndim = 2] vertex_of_triangle,
                        int nhori,
                        int refine_factor,
                        int svf_type):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    clon : ndarray of double
        Array with longitude of cell circumcenters (cell) [rad]
    clat : ndarray of double
        Array with latitude of cell circumcenters (cell) [rad]
    hsurf : ndarray of float
        Array with elevation of cell vertices (vertex) [m]
    vertex_of_triangle : ndarray of int
        Array with indices of triangle vertices. Indices start with 1
        according to Fortran (3, cell)
    nhori : int
        Number of terrain horizon sampling directions
    refine_factor : int
        Refinement factor that subdivides 'nhori' for more robust results
    svf_type : int
        Method for computing the Sky View Factor (SVF)
            0: Visible sky fraction; pure geometric skyview-factor
            1: SVF for horizontal surface; geometric scaled with sin(horizon)
            2: ?; geometric scaled with sin(horizon)**2

    Returns
    -------
    horizon : ndarray of float
        Array (two-dimensional) with terrain horizon [deg]
    skyview : ndarray of float
        Array (one-dimensional) with sky view factor [-]"""

    # Check consistency and validity of input arguments
    if (clon.size != clat.size) or (clat.size != hsurf.size):
        raise ValueError("Inconsistent lengths of input arrays 'clon', "
                         "'clat' and 'hsurf'")
    if vertex_of_triangle.shape[0] != 3:
        raise ValueError("First dimension of 'vertex_of_triangle' must "
            + "have length 3")
    if ((vertex_of_triangle.min() < 1)
        or (vertex_of_triangle.max() > clon.size)):
        raise ValueError("Indices of 'vertex_of_triangle' out of range")
    if (nhori < 4) or (nhori > 1440):
        raise ValueError("'nhori' must be in the range [4, 1440]")
    if (refine_factor < 1) or (refine_factor > 50):
        raise ValueError("'refine_factor' must be in the range [1, 50]")
    if (svf_type < 0) or (svf_type > 2):
        raise ValueError("'svf_type' must be in the range [0, 2]")

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
        horizon = np.empty((nhori, clon.size), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        skyview = np.empty(clon.size, dtype=np.float32)

    # Ensure that passed arrays are contiguous in memory
    vertex_of_triangle = np.ascontiguousarray(vertex_of_triangle)

    # Call C++ function and pass arguments
    horizon_svf_comp(&clon[0], &clat[0], &hsurf[0],
                     clon.size,
                     &vertex_of_triangle[0, 0], vertex_of_triangle.shape[1],
                     &horizon[0, 0], &skyview[0],
                     nhori, refine_factor, svf_type)

    return horizon, skyview
