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
        )
        void horizon_vertex(
            int num_azim,
            double azim_offset,
            unsigned int* slice_loc,
            float dist_search,
            double ray_origin_elev,
            double horizon_acc,
            double elev_angle_min,
            float* horizon,
        )
        void horizon_centroid(
            int num_azim,
            unsigned int* slice_loc,
            float dist_search,
            double ray_origin_elev,
            double horizon_acc,
            double elev_angle_min,
            float* horizon,
        )

cdef class Terrain:

    cdef CppTerrain *thisptr

    def __cinit__(self):
        self.thisptr = new CppTerrain()

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
        """

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

    def horizon_vertex(
        self,
        int num_azim,
        double azim_offset,
        np.ndarray[np.uint32_t, ndim = 1] slice_loc,
        float dist_search,
        double ray_origin_elev,
        double horizon_acc,
        double elev_angle_min,
        ):
        """
        Compute the terrain horizon at the vertices of the triangle mesh.

        Parameters
        ----------
        num_azim : int
            Number of azimuth positions for horizon
        azim_offset : double
            Offset of first azimuth position from 0.0 [deg]
        slice_loc : ndarray of float
            Array with start and stop index of considered vertices (2)
        dist_search : float
            Radial search distance for horizon computation [m]
        ray_origin_elev : double
            Vertical elevation of ray origin above surface [m]
        horizon_acc : double
            Accuracy of horizon computation [deg]
        elev_angle_min : double
            Threshold for sampling in negative elevation angle direction [deg]

        Returns
        -------
        horizon : ndarray of float32
            Terrain horizon for selected locations (num_loc, num_azim) [deg]
        """

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
        if (ray_origin_elev < 0.1):
            raise TypeError(
                "Minimal allowed value for 'ray_origin_elev' is 0.1 m"
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

        self.thisptr.horizon_vertex(
            num_azim,
            azim_offset,
            &slice_loc[0],
            dist_search,
            ray_origin_elev,
            horizon_acc,
            elev_angle_min,
            &horizon[0, 0],
        )

        return np.rad2deg(horizon)

    def horizon_centroid(
        self,
        int num_azim,
        np.ndarray[np.uint32_t, ndim = 1] slice_loc,
        float dist_search,
        double ray_origin_elev,
        double horizon_acc,
        double elev_angle_min,
        ):
        """
        Compute the terrain horizon at the triangle centroids of the mesh.

        Parameters
        ----------
        num_azim : int
            Number of azimuth positions for horizon
        slice_loc : ndarray of float
            Array with start and stop index of considered centroids (2)
        dist_search : float
            Radial search distance for horizon computation [m]
        ray_origin_elev : double
            Vertical elevation of ray origin above surface [m]
        horizon_acc : double
            Accuracy of horizon computation [deg]
        elev_angle_min : double
            Threshold for sampling in negative elevation angle direction [deg]

        Returns
        -------
        horizon : ndarray of float32
            Terrain horizon for selected locations (num_loc, num_azim) [deg]
        """

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
        if (ray_origin_elev < 0.1):
            raise TypeError(
                "Minimal allowed value for 'ray_origin_elev' is 0.1 m"
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
            ray_origin_elev,
            horizon_acc,
            elev_angle_min,
            &horizon[0, 0],
        )

        return np.rad2deg(horizon)
