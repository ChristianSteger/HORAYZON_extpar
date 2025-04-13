# Description: Run HORAYON-extpar on login node
#
# Author: Christian R. Steger (Christian.Steger@meteoswiss.ch), April 2025

import sys
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, rcParams, tri
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as feature
from pyproj import Transformer

style.use("classic")

# Change latex fonts
rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
rcParams["mathtext.default"] = "rm"
rcParams["mathtext.rm"] = "DejaVu Sans"

# Paths
path_icon = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/"

# Path to Cython/C++ functions
sys.path.append("/scratch/mch/csteger/HORAYZON_extpar/")
from horizon_svf import horizon_svf_comp_py

###############################################################################
# Functions
###############################################################################

def observer_perspective(lon: np.ndarray, lat: np.ndarray,
                         elevation: np.ndarray, lon_obs: float, lat_obs: float,
                         elevation_obs:float) -> tuple:
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
    clon_ext : ndarray of double
        Extended array with longitude of ICON cell circumcenters [rad]
    clat_ext : ndarray of double
        Extended array with latitude of ICON cell circumcenters [rad]
    hsurf_ext : ndarray of float
        Extended array with elevation [m]"""

    num_vertex = vlon.size
    vertex_of_triangle = np.zeros((3, num_vertex * 6), dtype=np.int32)
    angles = np.empty(6)
    clon_ext = np.append(clon, np.empty(num_vertex, dtype=clon.dtype))
    clat_ext = np.append(clat, np.empty(num_vertex, dtype=clat.dtype))
    hsurf_ext = np.append(hsurf, np.empty(num_vertex, dtype=hsurf.dtype))
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
            clon_ext[ind_add] = vlon[ind_vertex]
            clat_ext[ind_add] = vlat[ind_vertex]
            hsurf_ext[ind_add] = hsurf_mean / 6.0
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
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
# grid_level = ds.attrs["grid_level"]  # (k)
# grid_root = ds.attrs["grid_root"]  # (n)
# ds.close()

# # Load elevation of cell circumcenters
# file_topo = "EXTPAR_test/topography_buffer_extpar_v5.8_icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# nhori = ds["nhori"].size
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()

# # Further settings
# # nhori = 24
# # refine_factor = 10
# nhori = 240
# refine_factor = 1
# svf_type = 2   # 0, 1, 2

# -----------------------------------------------------------------------------
# Real data (Brigitta)
# -----------------------------------------------------------------------------

# # Load grid information
# # file_grid = "Brigitta/domain1_DOM01.nc"
# # file_grid = "Brigitta/domain2_DOM02.nc"
# # file_grid = "Brigitta/domain3_DOM03.nc"
# # file_grid = "Brigitta/domain4_DOM04.nc"
# file_grid = "Brigitta/domain_switzerland_100m.nc"
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

# # Load elevation of cell circumcenters
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain1_DOM01.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain2_DOM02.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain3_DOM03.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain_switzerland_" \
#              + "100m.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# # nhori = ds["nhori"].size
# # horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# # skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()

# # Further settings
# # nhori = 24
# # refine_factor = 10
# nhori = 240
# refine_factor = 1
# svf_type = 2   # 0, 1, 2
# ray_org_elev = 0.2

# -----------------------------------------------------------------------------
# Real data (resolutions from ~2km to ~30 m)
# -----------------------------------------------------------------------------

# # Load grid information
# # file_grid = "Resolutions/icon_grid_res0032m.nc"
# # file_grid = "Resolutions/icon_grid_res0130m.nc"
# file_grid = "Resolutions/icon_grid_res0519m.nc"
# # file_grid = "Resolutions/icon_grid_res2076m.nc"
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

# # Load elevation of cell circumcenters
# # file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0032m.nc"
# # file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0130m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0519m.nc"
# # file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res2076m.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# ds.close()

# # Further settings
# # nhori = 24
# # refine_factor = 10
# nhori = 240
# refine_factor = 1
# svf_type = 1   # 0, 1, 2
# ray_org_elev = 0.1

# -----------------------------------------------------------------------------
# MeteoSwiss domains (2km, 1km, 500m)
# -----------------------------------------------------------------------------

# Load grid information
file_grid = "/oprusers/osm/opr/data/grid_descriptions/" \
    + "icon_grid_0001_R19B08_mch.nc"  # 1 km
# file_grid = "/store_new/mch/msopr/glori/glori-ch500-nested/grid/" \
#     + "icon_grid_00005_R19B09_DOM02.nc"  # 500 m
ds = xr.open_dataset(file_grid)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
grid_level = ds.attrs["grid_level"]  # (k)
grid_root = ds.attrs["grid_root"]  # (n)
ds.close()

# Load elevation of cell circumcenters
file_extpar = "/scratch/mch/csteger/ExtPar/output/" \
    + "extpar_icon_grid_0001_R19B08_mch_copernicus.nc"  # 1 km
# file_extpar = "/scratch/mch/csteger/ExtPar/output/" \
#     + "extpar_icon_grid_00005_R19B09_DOM02_copernicus.nc"  # 500 m
ds = xr.open_dataset(file_extpar)
hsurf = ds["topography_c"].values.squeeze()  # (num_cell)
horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
ds.close()

# Ensure consistency of input grid and topography
if (clon.size != hsurf.size):
    raise ValueError("Inconsistent number of cells in grid and topography")

# Further settings
nhori = 24
refine_factor = 10
# nhori = 240
# refine_factor = 1
svf_type = 1   # 0, 1, 2
ray_org_elev = 0.1

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
plt.show()

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
                                        grid_type, ray_org_elev)
    print("Total elapsed time: %.2f" % (time.perf_counter() - t_beg) + " s")
    horizon_gt[grid_type] = horizon
    skyview_gt[grid_type] = skyview
    print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
        + ", %.5f" % np.max(horizon))
    print("Sky view factor range [-]: %.8f" % np.min(skyview)
        + ", %.8f" % np.max(skyview))

# Create new EXTPAR file with ray-tracing based terrain horizon and SVF
grid_type = 1
ds = xr.open_dataset(file_extpar)
# ----------------- check correlation -----------------------------------------
# plt.figure()
# plt.scatter(horizon_old.ravel()[::20], horizon_gt[grid_type].ravel()[::20])
# plt.show()
# plt.figure()
# plt.scatter(skyview_old.ravel()[::5], skyview_gt[grid_type].ravel()[::5])
# plt.show()
# -----------------------------------------------------------------------------
ds["HORIZON"].values[:] = horizon_gt[grid_type]
ds["SKYVIEW"].values[:] = skyview_gt[grid_type]
ds.to_netcdf(file_extpar[:-3] + "_ray.nc")
# Check that really overwritten
ds = xr.open_dataset(file_extpar[:-3] + "_ray.nc")
print(np.all(ds["HORIZON"].values == horizon_gt[grid_type]))
print(np.all(ds["SKYVIEW"].values == skyview_gt[grid_type]))
ds.close()

###############################################################################
# Check and compare terrain horizon and sky view factor
###############################################################################

# Map plot of terrain horizon or sky view factor
grid_type = 1
# values = skyview_old
values = skyview_gt[grid_type]
cmap = plt.get_cmap("YlGnBu_r")
levels = np.arange(0.85, 1.0, 0.005)

# values = horizon_gt[grid_type][0, :]
# levels = np.arange(0.0, 37.5, 2.5)
# cmap = plt.get_cmap("afmhot_r")

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
plt.show()

# Check terrain horizon for specific (triangle) cell
grid_type = 1  # used for plotting grid
# all below indices for 500 m grid!
ind = 1909971  # best so far (Mattertal)
# ind = 1901633  # 2nd (Lauterbrunnental)
# ind = np.random.choice(np.where(skyview_gt[grid_type] < 0.8)[0])
azim_old = np.arange(0.0, 360.0, 360.0 / horizon_old.shape[0]) + 7.5
azim = np.arange(0.0, 360.0, 360.0 / horizon.shape[0])
fig = plt.figure(figsize=(15, 5))
# ---------------------------- Mask for triangles -----------------------------
dist_search = 40_000  # search distance for horizon [m]
if grid_type == 0:
    phi, theta, radius = observer_perspective(clon, clat, hsurf,
                                            clon[ind], clat[ind],
                                            hsurf[ind] + ray_org_elev)
else:
    phi, theta, radius = observer_perspective(clon_ext, clat_ext, hsurf_ext,
                                            clon[ind], clat[ind],
                                            hsurf[ind] + ray_org_elev)
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
plt.plot(azim_old, horizon_old[:, ind], label="current", color="red", lw=2.5)
plt.plot(azim, horizon_gt[0][:, ind], label="Ray tracing (grid_type = 0)",
         color="royalblue", lw=1.5, ls="--")
plt.plot(azim, horizon_gt[1][:, ind], label="Ray tracing (grid_type = 1)",
         color="royalblue", lw=2.5)
hori_all = np.hstack((horizon_old[:, ind], horizon_gt[0][:, ind],
                      horizon_gt[1][:, ind]))
plt.axis((0.0, 360.0, 0.0, hori_all.max() * 1.05))
plt.legend(fontsize=12, frameon=False)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
txt = "Latitude: %.3f" % np.rad2deg(clat[ind]) \
    + "$^{\circ}$, longitude: %.3f" % np.rad2deg(clon[ind]) \
    + "$^{\circ}$, elevation: %.0f" % hsurf[ind] + " m"
plt.title(txt, fontsize=12, loc="left")
# plt.show()
plt.savefig("/scratch/mch/csteger/HORAYZON_extpar/"
            + "horizon_ind_" + str(ind) + ".png", dpi=300)
plt.close()
