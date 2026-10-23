cimport numpy as np
import numpy as np

cdef extern from "horizon_compute.h" namespace "shapes":
    cdef cppclass CppTerrain:
        CppTerrain()
        void initialise(
            float* tri_vert,
            unsigned int* tri_face,
            int num_vert,
            int num_face,
            float* earth_centre,
            float* north_pole,
        ) except +
        void horizon_vertex(
            int num_azim,
            double azim_offset,
            unsigned int* slice_loc,
            float dist_search,
            double ray_offset_factor,
            double horizon_acc,
            double elev_angle_min,
            float* horizon,
            float* clearance_out,
        )
        void horizon_centroid(
            int num_azim,
            unsigned int* slice_loc,
            float dist_search,
            double ray_offset_factor,
            double horizon_acc,
            double elev_angle_min,
            float* horizon,
        )
        void horizon_arbitrary(
            double* coords,
            int num_loc,
            int num_azim,
            float dist_search,
            double height_min,
            double horizon_acc,
            double elev_angle_min,
            float* horizon,
            float* height_above,
        )

cdef class Terrain:

    cdef CppTerrain *thisptr
    # References to triangle mesh arrays (shared with Embree and used in
    # subsequent method calls -> must not be freed while object exists)
    cdef object _tri_vert
    cdef object _tri_face
    # Status of initialisation (only one successful initialisation allowed)
    cdef bint _initialised

    def __cinit__(self):
        self.thisptr = new CppTerrain()
        self._initialised = False

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr

    def initialise(
        self,
        np.ndarray[np.float32_t, ndim = 2] tri_vert,
        np.ndarray[np.uint32_t, ndim = 2] tri_face,
        np.ndarray[np.float32_t, ndim = 1] earth_centre,
        np.ndarray[np.float32_t, ndim = 1] north_pole,
        ):
        """
        Build Embree BVH from triangle mesh representing a terrain surface.

        Parameters
        ----------
        tri_vert : ndarray of float32 (num_vert, 3)
            Cartesian coordinates of Embree triangle mesh vertices [m]
        tri_face : ndarray of uint32 (num_face, 3)
            Indices of the Embree triangle mesh faces
        earth_centre : ndarray of float32 (3)
            Cartesian coordinates of Earth centre [m]
        north_pole : ndarray of float32 (3)
            Cartesian coordinates of North Pole [m]

        Raises
        ------
        ValueError
            If the triangle mesh contains degenerated triangles (zero area)
            or triangles without anti-clockwise winding (viewed from above),
            i.e. triangles whose normal does not point away from the Earth's
            centre.
        RuntimeError
            If the object has already been initialised successfully.

        Notes
        -----
        The arrays 'tri_vert' and 'tri_face' are not copied but shared with
        Embree. A reference to them is kept as long as the object exists.
        They must not be modified after initialisation. An object can only be
        initialised once.
        """

        # Check that object has not been initialised yet
        if self._initialised:
            raise RuntimeError("Terrain already initialised")

        # Check validity of input arguments
        if ((tri_vert.shape[1] != 3) or (tri_face.shape[1] != 3)):
            raise ValueError("Second dimension size of input arrays must be 3")
        if ((tri_face.min() < 0.0)
            or (tri_face.max() > tri_vert.shape[0] - 1)):
            raise ValueError("Indices of 'tri_face' out of range")
        if ((not tri_vert.flags["C_CONTIGUOUS"])
                or (not tri_face.flags["C_CONTIGUOUS"])):
            raise ValueError("Input arrays must be C-contiguous")

        self.thisptr.initialise(
            &tri_vert[0, 0],
            &tri_face[0, 0],
            tri_vert.shape[0],
            tri_face.shape[0],
            &earth_centre[0],
            &north_pole[0],
        )

        # Keep references to triangle mesh arrays alive
        self._tri_vert = tri_vert
        self._tri_face = tri_face
        self._initialised = True

    def horizon_vertex(
        self,
        int num_azim,
        double azim_offset,
        np.ndarray[np.uint32_t, ndim = 1] slice_loc,
        float dist_search,
        double ray_offset_factor,
        double horizon_acc,
        double elev_angle_min,
        bint return_clearance=False,
        ):
        """
        Compute the terrain horizon at the vertices of the triangle mesh.

        Parameters
        ----------
        num_azim : int
            Number of azimuth positions for horizon
        azim_offset : double
            Offset of first azimuth position from 0.0 [deg]
        slice_loc : ndarray of uint32 (2)
            Array with start and stop index of considered vertices
        dist_search : float
            Radial search distance for horizon computation [m]
        ray_offset_factor : double
            Factor for the offset of the ray origin from the triangle vertex
            along the direction with maximal clearance from the adjacent
            triangles [-]
            The offset is computed as
            (ray_offset_factor * FLT_EPSILON * max(abs(tri_vert))) / clearance
            where clearance is the ratio between the minimal distance of the
            offset point to the adjacent triangles and the length of the
            offset (independent of the offset length). The clearance is
            limited to a minimum of 0.1. If no valid direction is found, the
            sphere normal (with clearance = 1.0) is used
        horizon_acc : double
            Accuracy of horizon computation [deg]
        elev_angle_min : double
            Threshold for sampling in negative elevation angle direction [deg]
        return_clearance : bool
            Additionally return the clearance of the offset direction

        Returns
        -------
        horizon : ndarray of float32 (num_loc, num_azim)
            Terrain horizon for selected locations [deg]
        clearance : ndarray of float32 (num_loc)
            Ratio between the minimal distance of the offset ray origin to the
            adjacent triangles and the length of the offset [-]
            The value is reported before clipping to a minimum of 0.1. Values
            are in the range (0.0, 1.0]; NaN indicates that no valid offset
            direction was found. Only returned if 'return_clearance' is True.
        """

        # Check that object has been initialised
        if not self._initialised:
            raise RuntimeError(
                "Terrain not initialised; call 'initialise' first"
            )

        # Check validity of input arguments
        if (num_azim < 4):
            raise ValueError(
                "Number of azimuth position must be at least 4"
            )
        if (azim_offset < -180.0) or (azim_offset > 180.0):
            raise ValueError(
                "Azimuth offset must be in the range [-180.0, +180.0]"
            )
        if (slice_loc[1] <= slice_loc[0]):
            raise ValueError(
                "Indices in 'slice_loc' must be in increasing order"
            )
        if (dist_search < 1_000.0) or (dist_search > 500_000.0):
            raise ValueError(
                "Search distance must be in the range [1'000, 500'000] m"
            )
        if (ray_offset_factor < 1.0):
            raise ValueError(
                "Minimal allowed value for 'ray_offset_factor' is 1.0"
            )
        if (horizon_acc < 0.001) or (horizon_acc > 10.0):
            raise ValueError(
                "Horizon accuracy must be in the range [0.001, 10.0] deg"
            )
        if ((elev_angle_min - (2.0 * horizon_acc)) <= -90.0):
            raise ValueError(
                "Invalid combination of 'elev_angle_min' and 'horizon_acc'"
            )

        # Allocate array for output
        cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
            horizon = np.empty(
                (slice_loc[1] - slice_loc[0], num_azim), dtype=np.float32
            )
        cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] clearance = None
        cdef float* clearance_ptr = NULL
        if return_clearance:
            clearance = np.empty(slice_loc[1] - slice_loc[0], dtype=np.float32)
            clearance_ptr = &clearance[0]

        self.thisptr.horizon_vertex(
            num_azim,
            azim_offset,
            &slice_loc[0],
            dist_search,
            ray_offset_factor,
            horizon_acc,
            elev_angle_min,
            &horizon[0, 0],
            clearance_ptr,
        )

        if return_clearance:
            return np.rad2deg(horizon), clearance
        return np.rad2deg(horizon)

    def horizon_centroid(
        self,
        int num_azim,
        np.ndarray[np.uint32_t, ndim = 1] slice_loc,
        float dist_search,
        double ray_offset_factor,
        double horizon_acc,
        double elev_angle_min,
        ):
        """
        Compute the terrain horizon at the triangle centroids of the mesh.

        Parameters
        ----------
        num_azim : int
            Number of azimuth positions for horizon
        slice_loc : ndarray of uint32 (2)
            Array with start and stop index of considered centroids
        dist_search : float
            Radial search distance for horizon computation [m]
        ray_offset_factor : double
            Factor for the offset of the ray origin from the triangle centroid
            along the triangle normal [-]
            The offset is computed as
            ray_offset_factor * FLT_EPSILON * max(abs(tri_vert))
        horizon_acc : double
            Accuracy of horizon computation [deg]
        elev_angle_min : double
            Threshold for sampling in negative elevation angle direction [deg]

        Returns
        -------
        horizon : ndarray of float32 (num_loc, num_azim)
            Terrain horizon for selected locations [deg]
        """

        # Check that object has been initialised
        if not self._initialised:
            raise RuntimeError(
                "Terrain not initialised; call 'initialise' first"
            )

        # Check validity of input arguments
        if (num_azim < 4):
            raise ValueError(
                "Number of azimuth position must be at least 4"
            )
        if (slice_loc[1] <= slice_loc[0]):
            raise ValueError(
                "Indices in 'slice_loc' must be in increasing order"
            )
        if (dist_search < 1_000.0) or (dist_search > 500_000.0):
            raise ValueError(
                "Search distance must be in the range [1'000, 500'000] m"
            )
        if (ray_offset_factor < 1.0):
            raise ValueError(
                "Minimal allowed value for 'ray_offset_factor' is 1.0"
            )
        if (horizon_acc < 0.001) or (horizon_acc > 10.0):
            raise ValueError(
                "Horizon accuracy must be in the range [0.001, 10.0] deg"
            )
        if ((elev_angle_min - (2.0 * horizon_acc)) <= -90.0):
            raise ValueError(
                "Invalid combination of 'elev_angle_min' and 'horizon_acc'"
            )

        # Allocate array for output
        cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
            horizon = np.empty(
                (slice_loc[1] - slice_loc[0], num_azim), dtype=np.float32
            )

        self.thisptr.horizon_centroid(
            num_azim,
            &slice_loc[0],
            dist_search,
            ray_offset_factor,
            horizon_acc,
            elev_angle_min,
            &horizon[0, 0],
        )

        return np.rad2deg(horizon)

    def horizon_arbitrary(
        self,
        np.ndarray[np.float64_t, ndim = 2, mode = "c"] coords,
        int num_azim,
        float dist_search,
        double height_min,
        double horizon_acc,
        double elev_angle_min,
        ):
        """
        Compute the terrain horizon at arbitrary locations.

        For each location, the height above the triangle mesh is determined
        along the sphere normal. Locations with a height below 'height_min',
        including those below the triangle mesh, are elevated to 'height_min'.
        All other locations remain unchanged.

        Parameters
        ----------
        coords : ndarray of float64 (num_loc, 3)
            Cartesian coordinates of locations in the same coordinate system
            as the triangle mesh vertices [m]
        num_azim : int
            Number of azimuth positions for horizon
        dist_search : float
            Radial search distance for horizon computation [m]
        height_min : double
            Minimal height of locations above the triangle mesh [m]
        horizon_acc : double
            Accuracy of horizon computation [deg]
        elev_angle_min : double
            Threshold for sampling in negative elevation angle direction [deg]

        Returns
        -------
        horizon : ndarray of float32 (num_loc, num_azim)
            Terrain horizon for locations [deg]
            NaN for locations outside of the triangle mesh domain
        height_above : ndarray of float32 (num_loc)
            Height of locations above the triangle mesh before elevating to
            'height_min' [m]
            NaN for locations outside of the triangle mesh domain
        """

        # Check that object has been initialised
        if not self._initialised:
            raise RuntimeError(
                "Terrain not initialised; call 'initialise' first"
            )

        # Check validity of input arguments
        if (coords.shape[1] != 3):
            raise ValueError("Second dimension size of 'coords' must be 3")
        if (coords.shape[0] < 1):
            raise ValueError("At least one location is required")
        if (num_azim < 4):
            raise ValueError(
                "Number of azimuth position must be at least 4"
            )
        if (dist_search < 1_000.0) or (dist_search > 500_000.0):
            raise ValueError(
                "Search distance must be in the range [1'000, 500'000] m"
            )
        if (height_min < 0.5):
            raise ValueError(
                "Minimal allowed value for 'height_min' is 0.5 m"
            )
        if (horizon_acc < 0.001) or (horizon_acc > 10.0):
            raise ValueError(
                "Horizon accuracy must be in the range [0.001, 10.0] deg"
            )
        if ((elev_angle_min - (2.0 * horizon_acc)) <= -90.0):
            raise ValueError(
                "Invalid combination of 'elev_angle_min' and 'horizon_acc'"
            )

        # Allocate arrays for output
        cdef int num_loc = coords.shape[0]
        cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
            horizon = np.empty((num_loc, num_azim), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim = 1, mode = "c"] \
            height_above = np.empty(num_loc, dtype=np.float32)

        self.thisptr.horizon_arbitrary(
            &coords[0, 0],
            num_loc,
            num_azim,
            dist_search,
            height_min,
            horizon_acc,
            elev_angle_min,
            &horizon[0, 0],
            &height_above[0],
        )

        return np.rad2deg(horizon), height_above
