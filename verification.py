# Description: Verify HORAYZON EXTPAR code
#
# Author: Christian R. Steger (Christian.Steger@meteoswiss.ch), April 2025

import os
import sys
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, rcParams, tri, colors
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as feature
from pyproj import Transformer

from development import triangle_mesh_circ, triangle_mesh_circ_vert

style.use("classic")

# Change latex fonts
rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
rcParams["mathtext.default"] = "rm"
rcParams["mathtext.rm"] = "DejaVu Sans"

# Paths
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
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

###############################################################################
# Load ICON data
###############################################################################

# MeteoSwiss domain (1km, 500m)
# file_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc" # ~1 km
# file_extpar = "MeteoSwiss/extpar_icon_grid_0001_R19B08_mch.nc"
# file_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc" # ~500 m
# file_extpar = "MeteoSwiss/extpar_icon_grid_00005_R19B09_DOM02.nc"

# EXTPAR test domain DOM01
file_grid = "EXTPAR_test/icon_grid_DOM01.nc" # ~2 km
file_extpar = "EXTPAR_test/external_parameter_icon_d2_PR273.nc"

# Domains from Brigitta
# file_grid = "Brigitta/domain1_DOM01.nc" # ~1 km
# file_extpar = "Brigitta/extpar_icon_grid_domain1_DOM01.nc"
# file_grid = "Brigitta/domain2_DOM02.nc" # ~500 m
# file_extpar = "Brigitta/extpar_icon_grid_domain2_DOM02.nc"
# file_grid = "Brigitta/domain3_DOM03.nc" # ~250 m
# file_extpar = "Brigitta/extpar_icon_grid_domain3_DOM03.nc"
# file_grid = "Brigitta/domain4_DOM04.nc" # ~125 m
# file_extpar = "Brigitta/extpar_icon_grid_domain4_DOM04.nc"
# file_grid = "Brigitta/domain_switzerland_200m.nc" # ~200 m
# file_extpar = "Brigitta/extpar_icon_grid_domain_switzerland_200m.nc"

# -----------------------------------------------------------------------------

# Load grid information
ds = xr.open_dataset(path_ige + file_grid)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
grid_level = ds.attrs["grid_level"]  # (k)
grid_root = ds.attrs["grid_root"]  # (n)
ds.close()
print(f"grid_root (n): {grid_root}, grid_level (k): {grid_level}")
grid_res = (5050.0 / (grid_root * 2 ** grid_level)) * 1000.0  # [m]
print(f"Approximate grid resolution: {grid_res:.1f} m")

# Load elevation of cell circumcenters
ds = xr.open_dataset(path_ige + file_extpar)
hsurf = ds["topography_c"].values.squeeze()  # (num_cell)
num_hori = ds["nhori"].size
horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
ds.close()

# Ensure consistency of input grid and topography
if (clon.size != hsurf.size):
    raise ValueError("Inconsistent number of cells in grid and topography")

###############################################################################
# Check ICON grid and topography
###############################################################################

# Map plot of topography
cmap = plt.get_cmap("terrain")
levels = np.arange(0.0, 3200.0, 200.0)
norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
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
# plt.show()
file_plot = path_plot + "ICON_grid_topo.png"
plt.savefig(file_plot , dpi=300)
plt.close()

# os.remove(file_plot)

###############################################################################
# Compute horizon / sky view factor from two different grid types
###############################################################################

# Settings
num_hori = 24 # 240
dist_search = 50_000.0 #  horizon search distance [m]
ray_org_elev = 0.2 # 0.1, 0.2
refine_factor = 10 # 1
svf_type = 2 # 0, 1, 2

# Loop through two different grid types
for grid_type in range(2):

    # Construct triangle mesh
    if grid_type == 0:
        vertex_of_triangle = triangle_mesh_circ(clon, clat, vlon, vlat,
                                                cells_of_vertex)
    else:
        vertex_of_triangle, clon_ext, clat_ext, hsurf_ext \
            = triangle_mesh_circ_vert(clon, clat, hsurf, vlon, vlat,
                                        cells_of_vertex)
    print(vertex_of_triangle.shape)
    print(vertex_of_triangle.sum(axis=1))
    # hash-like string to compare with C++ code

    # # Check grid
    # if vertex_of_triangle.shape[1] < 1_000_000:
    #     fig = plt.figure(figsize=(10, 10))  # (width, height)
    #     if grid_type == 0:
    #         triangles = tri.Triangulation(
    #             clon, clat, vertex_of_triangle.transpose())
    #         plt.triplot(triangles, color="black", linewidth=0.5)
    #     else:
    #         triangles = tri.Triangulation(
    #             clon_ext, clat_ext, vertex_of_triangle.transpose())
    #         temp = np.tile(np.arange(6), vertex_of_triangle.shape[1] // 6)
    #         cmap = plt.get_cmap("YlGnBu")
    #         levels = np.arange(-0.5, 6.5, 1.0)
    #         norm = colors.BoundaryNorm(levels, ncolors=cmap.N)
    #         plt.tripcolor(triangles, temp, cmap=cmap, norm=norm,
    #                     edgecolors="black", linewidth=0.0)
    #     plt.scatter(clon, clat, color="red", s=20)
    #     plt.show()

    # Compute horizon and sky view factor
    t_beg = time.perf_counter()
    horizon, skyview = horizon_svf_comp_py(
        clon, clat, hsurf.astype(np.float64),
        vlon, vlat,
        (cells_of_vertex + 1),
        num_hori, grid_type, dist_search,
        ray_org_elev, refine_factor,
        svf_type)
    print("Total elapsed time: %.2f" % (time.perf_counter() - t_beg) + " s")
    print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
        + ", %.5f" % np.max(horizon))
    print("Sky view factor range [-]: %.8f" % np.min(skyview)
        + ", %.8f" % np.max(skyview))

    # Create new EXTPAR file with ray-tracing based terrain horizon and SVF
    ds = xr.open_dataset(path_ige + file_extpar)
    ds["HORIZON"].values[:] = horizon
    ds["SKYVIEW"].values[:] = skyview
    file_out = file_extpar[:-3] + f"_horayzon_gt_{grid_type}.nc"
    ds.to_netcdf(path_ige + file_out)
    # Check that really overwritten
    ds = xr.open_dataset(path_ige + file_out)
    print(np.all(ds["HORIZON"].values == horizon))
    print(np.all(ds["SKYVIEW"].values == skyview))
    ds.close()

###############################################################################  continue....
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
