# Description: Developments for C++ HORAYZON EXTPAR code
#
# Author: Christian R. Steger (Christian.Steger@meteoswiss.ch), April 2025

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri

style.use("classic")

# Paths
path_icon = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/"

###############################################################################
# Generate triangle mesh from ICON grid cell circumcenters (and vertices)
###############################################################################

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def triangle_mesh_circ(clon: np.ndarray, clat: np.ndarray, vlon: np.ndarray,
                       vlat: np.ndarray, cells_of_vertex: np.ndarray) \
                        -> np.ndarray:
    """Build triangle mesh solely from ICON grid cell circumcenters (non-unique
    triangulation of hexa- and pentagons; relatively long triangle edges can
    cause artefacts in horizon computation)

    Parameters
    ----------
    clon : ndarray of double
        Array with longitude of ICON cell circumcenters
        (number of ICON cells) [rad]
    clat : ndarray of double
        Array with latitude of ICON cell circumcenters
        (number of ICON cells) [rad]
    vlon : ndarray of double
        Array with longitude of ICON cell vertices
        (number of ICON vertices) [rad]
    vlat : ndarray of double
        Array with latitude of ICON cell vertices
        (number of ICON vertices) [rad]
    cells_of_vertex : ndarray of int
        Array with indices of ICON cells adjacent to ICON vertices.

    Returns
    -------
    vertex_of_triangle : ndarray of int
        Array with indices of triangle vertices (clockwise order)
        (3, number of triangles)"""

    num_vertex = vlon.size
    vertex_of_triangle = np.zeros((3, num_vertex * 4), dtype=np.int32)
    angles = np.empty(6)
    # allocate arrays with maximal possible size
    num_triangle = 0
    for ind_vertex in range(num_vertex):
        num_angles = 0
        for j in range(6):
            ind_cell = cells_of_vertex[j, ind_vertex]
            if (ind_cell != -2):
                angle = np.arctan2(clon[ind_cell] - vlon[ind_vertex],
                                   clat[ind_cell] - vlat[ind_vertex])
                # clockwise angle from positive latitude-axis (y-axis)
                if (angle < 0.0):
                    angle += 2.0 * np.pi
                angles[num_angles] = angle
                num_angles += 1
        if (num_angles >= 3):
            # at least 3 vertices are needed to create one or multiple
            # triangles(s) from the polygon
            ind_sort = np.argsort(angles[:num_angles])
            ind_1 = 1
            ind_2 = 2
            for j in range(num_angles - 2):
                vertex_of_triangle[0, num_triangle] \
                    = cells_of_vertex[ind_sort[0], ind_vertex]
                vertex_of_triangle[1, num_triangle] \
                    = cells_of_vertex[ind_sort[ind_1], ind_vertex]
                vertex_of_triangle[2, num_triangle] \
                    = cells_of_vertex[ind_sort[ind_2], ind_vertex]
                ind_1 += 1
                ind_2 += 1
                num_triangle += 1
        if (ind_vertex + 1) % 100_000 == 0:
            print("First " + str(ind_vertex + 1) + " triangles constructed")
    vertex_of_triangle = vertex_of_triangle[:, :num_triangle]

    return vertex_of_triangle

def triangle_mesh_circ_vert(clon: np.ndarray, clat: np.ndarray,
                            hsurf: np.ndarray, vlon: np.ndarray,
                            vlat: np.ndarray, cells_of_vertex: np.ndarray) \
                                -> tuple:
    """Build triangle mesh from ICON grid cell circumcenters and vertices
    (elevation at vertices is computed as mean from adjacent cell
    circumcenters; triangulation is unique and artefacts are reduced)

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
        (number of ICON cells) [m]
    vlon : ndarray of double
        Array with longitude of ICON cell vertices
        (number of ICON vertices) [rad]
    vlat : ndarray of double
        Array with latitude of ICON cell vertices
        (number of ICON vertices) [rad]
    cells_of_vertex : ndarray of int
        Array with indices of ICON cells adjacent to ICON vertices.

    Returns
    -------
    vertex_of_triangle : ndarray of int
        Array with indices of triangle vertices (clockwise order)
        (3, number of triangles)
    vertex_lon : ndarray of double
        Array with longitude of triangle vertices [rad]
    vertex_lat : ndarray of double
        Array with latitude of triangle vertices [rad]
    vertex_hsurf : ndarray of float
        Array with elevation of triangle vertices [m]"""

    num_vertex = vlon.size
    vertex_of_triangle = np.zeros((3, num_vertex * 6), dtype=np.int32)
    angles = np.empty(6)
    vertex_lon = np.append(clon, np.empty(num_vertex, dtype=clon.dtype))
    vertex_lat = np.append(clat, np.empty(num_vertex, dtype=clat.dtype))
    vertex_hsurf = np.append(hsurf, np.empty(num_vertex, dtype=hsurf.dtype))
    # allocate arrays with maximal possible size
    num_triangle = 0
    ind_add = clon.size
    ind = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int32)
    for ind_vertex in range(num_vertex):
        num_angles = 0
        hsurf_mean = 0.0
        for j in range(6):
            ind_cell = cells_of_vertex[j, ind_vertex]
            if (ind_cell != -2):
                angle = np.arctan2(clon[ind_cell] - vlon[ind_vertex],
                                   clat[ind_cell] - vlat[ind_vertex])
                # clockwise angle from positive latitude-axis (y-axis)
                if (angle < 0.0):
                    angle += 2.0 * np.pi
                angles[num_angles] = angle
                num_angles += 1
                hsurf_mean += hsurf[ind_cell]
        if (num_angles == 6):
            vertex_lon[ind_add] = vlon[ind_vertex]
            vertex_lat[ind_add] = vlat[ind_vertex]
            vertex_hsurf[ind_add] = hsurf_mean / 6.0
            ind_sort = np.argsort(angles[:num_angles])
            for j in range(6):
                vertex_of_triangle[0, num_triangle] \
                    = cells_of_vertex[ind_sort[ind[j]], ind_vertex]
                vertex_of_triangle[1, num_triangle] \
                    = cells_of_vertex[ind_sort[ind[j + 1]], ind_vertex]
                vertex_of_triangle[2, num_triangle] \
                    = ind_add
                num_triangle += 1
            ind_add += 1
        if (ind_vertex + 1) % 100_000 == 0:
            print("First " + str(ind_vertex + 1) + " triangles constructed")
    vertex_of_triangle = vertex_of_triangle[:, :num_triangle]

    return vertex_of_triangle, vertex_lon[:ind_add], vertex_lat[:ind_add], \
        vertex_hsurf[:ind_add]

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Load grid information
    ds = xr.open_dataset(path_icon + "EXTPAR_test/icon_grid_DOM01.nc")
    clon = ds["clon"].values
    clat = ds["clat"].values
    vlon = ds["vlon"].values
    vlat = ds["vlat"].values
    cells_of_vertex = ds["cells_of_vertex"].values - 1 # -2: no adjacent cell
    # parent_cell_index = ds["parent_cell_index"].values
    vertex_of_cell = ds["vertex_of_cell"].values - 1
    ds.close()

    # Load elevation of cell circumcenters
    ds = xr.open_dataset(path_icon + "EXTPAR_test/"
                        + "external_parameter_icon_d2_PR273.nc")
    hsurf = ds["topography_c"].values
    ds.close()

    # Generate triangle meshes
    vertex_of_triangle_circ \
        = triangle_mesh_circ(clon, clat, vlon, vlat, cells_of_vertex)
    vertex_of_triangle_circ_vert, vertex_lon, vertex_lat, vertex_hsurf = \
        triangle_mesh_circ_vert(clon, clat, hsurf, vlon, vlat, cells_of_vertex)

    # Plot meshes
    linewidth = 0.5 # 0.1
    plt.figure()
    triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                    vertex_of_cell.transpose())
    plt.triplot(triangles, color="black", lw=linewidth)
    triangles = tri.Triangulation(np.rad2deg(clon), np.rad2deg(clat),
                                    vertex_of_triangle_circ.transpose())
    plt.triplot(triangles, color="green", lw=linewidth)
    triangles = tri.Triangulation(np.rad2deg(vertex_lon),
                                  np.rad2deg(vertex_lat),
                                  vertex_of_triangle_circ_vert.transpose())
    plt.triplot(triangles, color="red", lw=linewidth)
    file_plot = path_plot + "triangle_meshes.png"
    plt.show()
    # plt.savefig(file_plot, dpi=600)
    # plt.close()

    # os.remove(file_plot)
