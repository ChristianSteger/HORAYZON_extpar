# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as feature
from pyproj import Transformer

mpl.style.use("classic")

# Path to folders
# path_extpar = "/scratch/mch/csteger/EXTPAR_HORAYZON/ICON_grids_EXTPAR/"
root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"
path_extpar = root_IAC + "Miscellaneous/Thesis_supervision/Caterina_Croci/" \
               + "ICON_grids_EXTPAR/"

# Path to Cython/C++ functions
# sys.path.append("/scratch/mch/csteger/HORAYZON_extpar/")
sys.path.append("/Users/csteger/Downloads/HORAYZON_extpar/")
from horizon_svf import horizon_svf_comp_py

###############################################################################
# Functions
###############################################################################

def observer_perspective(lon, lat, elevation, lon_obs, lat_obs, elevation_obs):
    """Transform points to 'observer perspective'. Latitude/longitude -> ECEF
    -> ENU -> spherical coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array with longitude of points [rad]
    lat : ndarray of double
        Array with latitude of points [rad]
    elevation : ndarray of float
        Array with elevation of points [m]
    lon_obs : double
        Longitude of 'observer' [rad]
    lat_obs : double
        Latitude of 'observer' [rad]
    elevation_obs : float
        Elevation of 'observer' [m]

    Returns
    -------
    phi : ndarray of double
        Array with azimuth angles [deg]
    theta : ndarray of double
        Array with elevation angles [deg]
    radius : ndarray of double
        Array with radii [m]"""

    txt = "+proj=pipeline +step +proj=cart +ellps=sphere +R=6371229.0" \
          + " +step +proj=topocentric +ellps=sphere +R=6371229.0" \
          + " +lon_0=" + str(np.rad2deg(lon_obs)) \
          + " +lat_0=" + str(np.rad2deg(lat_obs)) \
          + " +h_0=" + str(elevation_obs)
    t = Transformer.from_pipeline(txt)
    x, y, z = t.transform(np.rad2deg(lon), np.rad2deg(lat), elevation)
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.rad2deg(np.arccos(z / radius))  # zenith angle [90.0, 0.0 deg]
    theta = 90.0 - theta  # elevation angle [0.0, 90 deg]
    phi = np.rad2deg(np.arctan2(x, y))  # azimuth angle [-180.0, +180.0 deg]
    phi[phi < 0.0] += 360.0  # [0.0, 360.0 deg]

    return phi, theta, radius

# -----------------------------------------------------------------------------

def construct_triangle_mesh(clon, clat, vlon, vlat, cells_of_vertex):
    """Building triangle mesh solely from ICON grid circumcenters

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
        Array with indices of ICON cells adjacent to ICON vertices. Indices
        start with 1 according to Fortran (6, number of ICON vertices)

    Returns
    -------
    vertex_of_triangle : ndarray of int
        Array with indices of triangle vertices (3, number of triangles)"""

    num_vertex = vlon.size
    vertex_of_triangle = np.zeros((3, num_vertex * 6), dtype=np.int32)
    angles = np.empty(6)
    # -> allocate arrays with maximal possible size...
    num_triangle = 0
    for ind_vertex in range(num_vertex):
        num_angles = 0
        for j in range(6):
            ind_cell = cells_of_vertex[j, ind_vertex]
            if (ind_cell != -2):
                angle = np.arctan2(clon[ind_cell] - vlon[ind_vertex],
                                   clat[ind_cell] - vlat[ind_vertex])
                if (angle < 0.0):
                    angle += 2.0 * np.pi
                angles[num_angles] = angle
                num_angles += 1
        if (num_angles >= 3):
            ind_sort = np.argsort(angles[:num_angles])
            ind_1 = 1
            ind_2 = 2
            for j in range(num_angles - 2):
                vertex_of_triangle[:, num_triangle] = \
                    np.array([cells_of_vertex[ind_sort[0], ind_vertex],
                            cells_of_vertex[ind_sort[ind_1], ind_vertex],
                            cells_of_vertex[ind_sort[ind_2], ind_vertex]])
                ind_1 += 1
                ind_2 += 1
                num_triangle += 1
        if ind_vertex % 100_000 == 0:
            print("First " + str(ind_vertex) + " triangles constructed")
    vertex_of_triangle = vertex_of_triangle[:, :num_triangle]

    return vertex_of_triangle

def construct_triangle_mesh_m2(clon, clat, hsurf, vlon, vlat, cells_of_vertex):
    """Building triangle mesh from ICON grid circumcenters and vertices

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

    Returns
    -------
    vertex_of_triangle : ndarray of int
        Array with indices of triangle vertices (3, number of triangles)"""

    num_vertex = vlon.size
    vertex_of_triangle = np.zeros((3, num_vertex * 6), dtype=np.int32)
    angles = np.empty(6)
    clon_ext = np.append(clon, np.empty(num_vertex, dtype=clon.dtype) * np.nan)
    clat_ext = np.append(clat, np.empty(num_vertex, dtype=clat.dtype) * np.nan)
    hsurf_ext = np.append(hsurf, np.empty(num_vertex, dtype=hsurf.dtype)
                          * np.nan)
    # -> allocate arrays with maximal possible size...
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
                if (angle < 0.0):
                    angle += 2.0 * np.pi
                angles[num_angles] = angle
                num_angles += 1
                hsurf_mean += hsurf[ind_cell]
        if (num_angles == 6):

            # ICON grid vertices with elevation
            clon_ext[ind_add] = vlon[ind_vertex]
            clat_ext[ind_add] = vlat[ind_vertex]
            hsurf_ext[ind_add] = hsurf_mean / 6.0

            ind_sort = np.argsort(angles[:num_angles])
            for j in range(6):
                vertex_of_triangle[:, num_triangle] = \
                    np.array([cells_of_vertex[ind_sort[ind[j]], ind_vertex],
                              cells_of_vertex[ind_sort[ind[j + 1]],
                                              ind_vertex],
                              ind_add])
                num_triangle += 1
            ind_add += 1
        if ind_vertex % 100_000 == 0:
            print("First " + str(ind_vertex) + " triangles constructed")
    vertex_of_triangle = vertex_of_triangle[:, :num_triangle]

    return vertex_of_triangle, clon_ext[:ind_add], clat_ext[:ind_add], \
        hsurf_ext[:ind_add]

###############################################################################
# Load ICON data
###############################################################################

# -----------------------------------------------------------------------------
# Real data (EXTPAR test domain DOM01)
# -----------------------------------------------------------------------------

# # Load grid information
# file_grid = "EXTPAR_test/icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_grid)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ds.close()
#
# # Load elevation of cell vertices
# file_topo = "EXTPAR_test/topography_buffer_extpar_v5.8_icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# nhori = ds["nhori"].size
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()
#
# # Further settings
# svf_type = 0
# refine_factor = 10

# -----------------------------------------------------------------------------
# Real data (Brigitta)
# -----------------------------------------------------------------------------

# # Load grid information
# file_grid = "Brigitta/domain1_DOM01.nc"
# # file_grid = "Brigitta/domain2_DOM02.nc"
# # file_grid = "Brigitta/domain3_DOM03.nc"
# # file_grid = "Brigitta/domain4_DOM04.nc"
# # file_grid = "Brigitta/domain_switzerland_100m.nc"
# ds = xr.open_dataset(path_extpar + file_grid)
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
# grid_level = ds.attrs["grid_level"]  # (k)
# grid_root = ds.attrs["grid_root"]  # (n)
# ds.close()

# # Load elevation of cell vertices
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain1_DOM01.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain2_DOM02.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain3_DOM03.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain_switzerland_" \
# #             + "100m.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# nhori = ds["nhori"].size
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()

# # Further settings
# nhori = 24
# refine_factor = 10
# # nhori = 240
# # refine_factor = 1
# svf_type = 2

# -----------------------------------------------------------------------------
# Real data (resolutions from ~2km to ~30 m)
# -----------------------------------------------------------------------------

# Load grid information
# file_grid = "Resolutions/icon_grid_res0032m.nc"
# file_grid = "Resolutions/icon_grid_res0130m.nc"
file_grid = "Resolutions/icon_grid_res0519m.nc"
# file_grid = "Resolutions/icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_grid)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
grid_level = ds.attrs["grid_level"]  # (k)
grid_root = ds.attrs["grid_root"]  # (n)
ds.close()

# Load elevation of cell vertices
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0032m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0130m.nc"
file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0519m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_topo)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
# nhori = 24
# refine_factor = 10
nhori = 240
refine_factor = 1
svf_type = 1   # 0, 1, 2

###############################################################################
# Check resolution and topography
###############################################################################

# Approximate grid resolution
d_x = 5050.0 / (grid_root * 2 ** grid_level) * 1000.0 # [m]
print("Approximate grid resolution: %.0f" % d_x + " m")

# Map plot of topography
cmap = plt.get_cmap("terrain")
levels = np.arange(0.0, 3200.0, 200.0)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = mpl.tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.tripcolor(triangles, hsurf, cmap=cmap, norm=norm,
              edgecolors="black", linewidth=0.0)
ax.add_feature(feature.BORDERS.with_scale("10m"),
               linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"),
               linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="black",
                  alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.colorbar()
plt.title("Elevation [m a.s.l]")

###############################################################################
# Compute horizon / sky view factor and visualise data
###############################################################################

# Loop through two different grid types
horizon_gt = {}
skyview_gt = {}
vertex_of_triangle_gt = {}
for grid_type in range(2):

    # Construct triangle mesh
    if grid_type == 0:
        vertex_of_triangle = construct_triangle_mesh(clon, clat, vlon, vlat,
                                                    cells_of_vertex)
    else:
        vertex_of_triangle, clon_ext, clat_ext, hsurf_ext \
            = construct_triangle_mesh_m2(clon, clat, hsurf, vlon, vlat,
                                        cells_of_vertex)
    vertex_of_triangle_gt[grid_type] = vertex_of_triangle

    # # Check grid
    # fig = plt.figure(figsize=(10, 10))  # (width, height)
    # if grid_type == 0:
    #     triangles = mpl.tri.Triangulation(
    #         clon, clat, vertex_of_triangle.transpose()[:50_000, :])
    # else:
    #     triangles = mpl.tri.Triangulation(
    #         clon_ext, clat_ext, vertex_of_triangle.transpose()[:50_000, :])
    # cmap = plt.get_cmap("YlGnBu")
    # levels = np.arange(-0.5, 6.5, 1.0)
    # norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N)
    # temp = np.tile(np.arange(6), 10_000)
    # plt.tripcolor(triangles, temp[:50_000], cmap=cmap, norm=norm,
    #             edgecolors="black", linewidth=0.0)
    # plt.scatter(clon, clat, color="red", s=20)
    # plt.show()

    # Compute horizon and sky view factor
    t_beg = time.perf_counter()
    horizon, skyview = horizon_svf_comp_py(clon, clat, hsurf,
                                        vlon, vlat,
                                        (cells_of_vertex + 1),
                                        nhori, refine_factor, svf_type,
                                        grid_type)
    print("Total elapsed time: %.2f" % (time.perf_counter() - t_beg) + " s")
    horizon_gt[grid_type] = horizon
    skyview_gt[grid_type] = skyview
    print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
        + ", %.5f" % np.max(horizon))
    print("Sky view factor range [-]: %.8f" % np.min(skyview)
        + ", %.8f" % np.max(skyview))

###############################################################################
# Check and compare terrain horizon and sky view factor
###############################################################################

# Map plot of terrain horizon or sky view factor
grid_type = 1

# values = skyview_gt[grid_type]
# # values = skyview_old
# cmap = plt.get_cmap("YlGnBu_r")
# levels = np.arange(0.8, 1.0, 0.005)

values = horizon_gt[grid_type][0, :]
levels = np.arange(0.0, 42.5, 2.5)
cmap = plt.get_cmap("afmhot_r")

norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = mpl.tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.tripcolor(triangles, values, cmap=cmap, norm=norm,
              edgecolors="black", linewidth=0.0)
ax.add_feature(feature.BORDERS.with_scale("10m"),
               linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"),
               linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="black",
                  alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.colorbar()
# ----- click on specific cell in opened plot window to get index of cell -----
for i in range(5):
    pts = plt.ginput(1)
    ind = np.argmin(np.sqrt((pts[0][0] - np.rad2deg(clon)) ** 2
                            + (pts[0][1] - np.rad2deg(clat)) ** 2))
    print(ind)
# -----------------------------------------------------------------------------

# Check terrain horizon for specific (triangle) cell
# ind = 520_528  # 521_721
ind = 783182  # 468426, 644634  # strange: 644639
# azim_old = np.arange(0.0, 360.0, 360.0 / horizon_old.shape[0]) + 7.5
azim = np.arange(0.0, 360.0, 360.0 / horizon.shape[0])
plt.figure(figsize=(15, 5))
# ---------------------------- Mask for triangles -----------------------------
dist_search = 40_000  # search distance for horizon [m]
if grid_type == 0:
    phi, theta, radius = observer_perspective(clon, clat, hsurf,
                                            clon[ind], clat[ind],
                                            hsurf[ind] + 0.1)
else:
    phi, theta, radius = observer_perspective(clon_ext, clat_ext, hsurf_ext,
                                            clon[ind], clat[ind],
                                            hsurf[ind] + 0.1)
mask_dist = (radius[vertex_of_triangle_gt[grid_type]].min(axis=0)
             <= dist_search)
mask_theta = (theta[vertex_of_triangle_gt[grid_type]].max(axis=0) > 0.0)
mask_phi = (np.ptp(phi[vertex_of_triangle_gt[grid_type]], axis=0) < 180.0)
mask = mask_dist & mask_theta & mask_phi
tri_fac = (mask.sum() / mask.size * 100.0)
print("Fraction of plotted triangles: %.2f" % tri_fac + " %")
mask_vertex = np.unique(vertex_of_triangle_gt[grid_type][:, mask].ravel())
# -----------------------------------------------------------------------------
plt.scatter(phi[mask_vertex], theta[mask_vertex], color="sienna", s=20,
            alpha=0.5)
triangles = mpl.tri.Triangulation(
    phi, theta, vertex_of_triangle_gt[grid_type][:, mask].transpose())
plt.triplot(triangles, color="black", linewidth=0.5)
# -----------------------------------------------------------------------------
# plt.plot(azim_old, horizon_old[:, ind], label="old", color="red", lw=2.5)
plt.plot(azim, horizon[:, ind], label="Ray tracing", color="royalblue", lw=2.5)
plt.axis((0.0, 360.0, 0.0, 50.0))
plt.legend(fontsize=12, frameon=False)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
