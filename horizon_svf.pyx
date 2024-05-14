cimport numpy as np
import numpy as np

cdef extern from "horizon_svf_comp.h":
    void horizon_svf_comp(double* clon, double* clat, float* hsurf,
                          int num_cell,
                          double* vlon, double* vlat,
                          int num_vertex,
                          np.npy_int32* cells_of_vertex,
                          float* horizon, float* skyview,
                          int nhori,
                          int refine_factor, int svf_type,
                          int grid_type, double ray_org_elev)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] clon,
                        np.ndarray[np.float64_t, ndim = 1] clat,
                        np.ndarray[np.float32_t, ndim = 1] hsurf,
                        np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.int32_t, ndim = 2] cells_of_vertex,
                        int nhori,
                        int refine_factor,
                        int svf_type,
                        int grid_type,
                        float ray_org_elev):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    clon : ndarray of double
        Array with longitude of ICON cell circumcenters
        (number of ICON cells) [rad]
    clat : ndarray of double
        Array with latitude of ICON cell circumcenters
        (number of ICON cells) [rad]
    hsurf : ndarray of float
        Array with elevation of ICON cell circumcenters
        (number of ICON cells) [rad]
    vlon : ndarray of double
        Array with longitude of ICON cell vertices
        (number of ICON vertices) [rad]
    vlat : ndarray of double
        Array with latitude of ICON cell vertices
        (number of ICON vertices) [rad]
    cells_of_vertex : ndarray of int
        Array with indices of ICON cells adjacent to ICON vertices. Indices
        start with 1 according to Fortran (6, number of ICON vertices)
    nhori : int
        Number of terrain horizon sampling directions
    refine_factor : int
        Refinement factor that subdivides 'nhori' for more robust results
    svf_type : int
        Method for computing the Sky View Factor (SVF)
            0: Visible sky fraction; pure geometric skyview-factor
            1: SVF for horizontal surface; geometric scaled with sin(horizon)
            2: ?; geometric scaled with sin(horizon)**2
    grid_type : int
        Triangle mesh construction method
            0: "Building triangle mesh solely from ICON grid circumcenters
               (-> ambiguous triangulation)
            1: Building triangle mesh from ICON grid circumcenters and vertices
               (-> unique triangulation)
    ray_org_elev : double
        Vertical elevation of ray origin above surface [metre]

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
    if (vlon.size != vlat.size):
        raise ValueError("Inconsistent lengths of input arrays 'vlon' and "
                         "'vlat'")
    if cells_of_vertex.shape[0] != 6:
        raise ValueError("First dimension of 'cells_of_vertex' must "
            + "have length 6")
    if not np.all((cells_of_vertex >= 1) & (cells_of_vertex <= clon.size)
        | (cells_of_vertex == -1)):
        raise ValueError("Indices of 'cells_of_vertex' out of range")
    if (nhori < 4) or (nhori > 1440):
        raise ValueError("'nhori' must be in the range [4, 1440]")
    if (refine_factor < 1) or (refine_factor > 50):
        raise ValueError("'refine_factor' must be in the range [1, 50]")
    if (svf_type < 0) or (svf_type > 2):
        raise ValueError("'svf_type' must be in the range [0, 2]")
    if ray_org_elev < 0.1:
        raise TypeError("Minimal allowed value for 'ray_org_elev' is 0.1 m")

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
        horizon = np.empty((nhori, clon.size), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
        skyview = np.empty(clon.size, dtype=np.float32)

    # Ensure that passed arrays are contiguous in memory
    cells_of_vertex = np.ascontiguousarray(cells_of_vertex)

    # Call C++ function and pass arguments
    horizon_svf_comp(&clon[0], &clat[0], &hsurf[0],
                     clon.size,
                     &vlon[0], &vlat[0],
                     vlon.size,
                     &cells_of_vertex[0, 0],
                     &horizon[0, 0], &skyview[0],
                     nhori,
                     refine_factor, svf_type,
                     grid_type, ray_org_elev)

    return horizon, skyview
