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
from pyproj import CRS
from pyproj import Transformer

mpl.style.use("classic")

# Path to folders
path_extpar = "/scratch/mch/csteger/EXTPAR_HORAYZON/ICON_grids_EXTPAR/"
# root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"
# path_extpar = root_IAC + "Miscellaneous/Thesis_supervision/Caterina_Croci/" \
#                + "ICON_grids_EXTPAR/"

# Path to Cython/C++ functions
sys.path.append("/scratch/mch/csteger/HORAYZON_extpar/")
# sys.path.append("/Users/csteger/Downloads/HORAYZON_extpar/")
from horizon_svf import horizon_svf_comp_py

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


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
# # file_grid = "Brigitta/domain1_DOM01.nc"
# # file_grid = "Brigitta/domain2_DOM02.nc"
# # file_grid = "Brigitta/domain3_DOM03.nc"
# file_grid = "Brigitta/domain4_DOM04.nc"
# # file_grid = "Brigitta/domain_switzerland_100m.nc"
# ds = xr.open_dataset(path_extpar + file_grid)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ds.close()
#
# # Load elevation of cell vertices
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain1_DOM01.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain2_DOM02.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain3_DOM03.nc"
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
# # file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain_switzerland_" \
# #             + "100m.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# # nhori = ds["nhori"].size
# # horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# # skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()
#
# # Further settings
# nhori = 24
# refine_factor = 10
# # nhori = 240
# # refine_factor = 1
# svf_type = 3

# -----------------------------------------------------------------------------
# Real data (resolutions from ~2km to ~30 m)
# -----------------------------------------------------------------------------

# Load grid information
# file_grid = "Resolutions/icon_grid_res0032m.nc"
# file_grid = "Resolutions/icon_grid_res0130m.nc"
file_grid = "Resolutions/icon_grid_res0519m.nc"
# file_grid = "Resolutions/icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_grid)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
ds.close()

# Test plot
colors = ["coral", "firebrick", "blueviolet", "royalblue"]
fig = plt.figure(figsize=(6, 10))  # (width, height)
ax = plt.axes()
# for ind_v in np.where((vlon >= 0.07) & (vlon <= 0.0785) 
#                     & (vlat >= 0.77) & (vlat <= 0.7735))[0]:
for ind_v in np.where((vlon >= 0.07) & (vlon <= 0.0773) 
                    & (vlat >= 0.77) & (vlat <= 0.7715))[0]:
    # -------------------------------------------------------------------------
    poly_vert_num = 0
    angles = np.empty(6, dtype=np.float32)
    angles.fill(999.9)
    for ind_c in cells_of_vertex[:, ind_v]:
        if ind_c != -2:
            plt.scatter(clon[ind_c], clat[ind_c], color="black", s=25)
            # plt.text(float(clon[ind_c]) + 3e-06, float(clat[ind_c]) + 3e-06,
            #          str(poly_vert_num), fontsize=12)
            angle = np.rad2deg(np.arctan2(clon[ind_c] - vlon[ind_v],
                                          clat[ind_c] - vlat[ind_v]))
            if angle < 0.0:
                angle += 360.0
            angles[poly_vert_num] = angle
            # plt.text(float(clon[ind_c]) - 15e-06,
            #          float(clat[ind_c]) - 15e-06,
            #          "%.1f" % angle, fontsize=9, color="blue")
            x_poly = vlon[vertex_of_cell[:, ind_c]]
            y_poly = vlat[vertex_of_cell[:, ind_c]]
            poly = plt.Polygon(list(zip(x_poly, y_poly)), facecolor="none",
                    edgecolor="black", lw=1.0)
            ax.add_patch(poly)
            poly_vert_num += 1
    # -------------------------------------------------------------------------
    plt.scatter(vlon[ind_v], vlat[ind_v], color="black", s=150, marker="*")
    if (cells_of_vertex[:, ind_v] != -2).sum() < 3:
        # not enought vertices for a triangle...
        continue
    ind_sort = np.argsort(angles)
    indices = np.array([0, 1, 2])
    for i in range(poly_vert_num - 2):
        x_poly = clon[cells_of_vertex[ind_sort[indices], ind_v]]
        y_poly = clat[cells_of_vertex[ind_sort[indices], ind_v]]
        poly = plt.Polygon(list(zip(x_poly, y_poly)), facecolor=colors[i],
                           edgecolor="black", alpha=0.3)
        ax.add_patch(poly)
        indices[1:] += 1
    # -------------------------------------------------------------------------
fig.savefig("test_plot.png", dpi=250)
plt.close()

# Compute indices of triangle vertices (clon, clat)
temp = ((cells_of_vertex != -2).sum(axis=0) - 2)
num_tri = temp[temp >= 1].sum()
vertex_of_triangle = np.empty((3, num_tri), dtype=np.int32)
n = 0
# for ind_v in np.where((vlon >= 0.07) & (vlon <= 0.0785) 
#                     & (vlat >= 0.77) & (vlat <= 0.7735))[0]:
# for ind_v in np.where((vlon >= 0.07) & (vlon <= 0.0773) 
#                     & (vlat >= 0.77) & (vlat <= 0.7715))[0]:
for ind_v in range(0, vlon.size):
    # -------------------------------------------------------------------------
    if (cells_of_vertex[:, ind_v] != -2).sum() < 3:
        # not enought vertices for a triangle...
        continue
    poly_vert_num = 0
    angles = np.empty(6, dtype=np.float32)
    angles.fill(999.9)
    for ind_c in cells_of_vertex[:, ind_v]:
        if ind_c != -2:
            angle = np.rad2deg(np.arctan2(clon[ind_c] - vlon[ind_v],
                                          clat[ind_c] - vlat[ind_v]))
            if angle < 0.0:
                angle += 360.0
            angles[poly_vert_num] = angle
            poly_vert_num += 1
    # -------------------------------------------------------------------------
    ind_sort = np.argsort(angles)
    indices = np.array([0, 1, 2])
    for i in range(poly_vert_num - 2):
        vertex_of_triangle[:, n] = cells_of_vertex[ind_sort[indices], ind_v]
        indices[1:] += 1
        n += 1
    # -------------------------------------------------------------------------

# Test plot
fig = plt.figure(figsize=(10, 10))  # (width, height)
plt.scatter(clon, clat, color="red", s=20)
triangles = mpl.tri.Triangulation(clon, clat,
                                  vertex_of_triangle[:, :n].transpose())
plt.triplot(triangles, color="black", linewidth=1.0)
plt.axis((0.077, 0.077 + 0.002, 0.77, 0.77 + 0.002))
fig.savefig("triangle_mesh.png", dpi=300)
plt.close()

# Load elevation of cell vertices
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0032m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0130m.nc"
file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0519m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_topo)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
nhori = 24
refine_factor = 10
# nhori = 240
# refine_factor = 1
svf_type = 3

# -----------------------------------------------------------------------------
# Very large and coarse grid -> check transformation of triangle surface
# normals from global to local ENU
# -----------------------------------------------------------------------------

# # Important note:
# # -> 'double ray_org_elev = 0.5' required -> otherwise, infinite loop occurs
#
# file_grid = "child_grid_very_large_coarse.nc"
# ds = xr.open_dataset(path_extpar + file_grid)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ds.close()
# topography_v = np.zeros(vlon.size, dtype=np.float32)
#
# nhori = 24
# refine_factor = 1
# svf_type = 3

# -----------------------------------------------------------------------------
# Artificial Data for testing (small; only 3 cells)
# -----------------------------------------------------------------------------

# vlon = np.deg2rad(np.array([-1.0, 0.0, +1.0, +0.5, -0.5], dtype=np.float64))
# vlat = np.deg2rad(np.array([0.0, 0.0, 0.0, +1.0, +1.0], dtype=np.float64))
# topography_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
# clon = np.deg2rad(np.array([-0.5, 0.0, +0.5], dtype=np.float64))
# clat = np.deg2rad(np.array([+0.5, +0.5, +0.5], dtype=np.float64))
# vertex_of_cell = np.array([[5, 1, 2],
#                            [4, 5, 2],
#                            [4, 2, 3]], dtype=np.int32).transpose() - 1
# nhori = 24
# refine_factor = 1
# svf_type = 0

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(clon, clat, hsurf,
                                       (vertex_of_triangle + 1),
                                       nhori, refine_factor, svf_type)
print("Total elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")

# Check range of computes values
print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
      + ", %.5f" % np.max(horizon))
print("Sky view factor range [-]: %.8f" % np.min(skyview)
      + ", %.8f" % np.max(skyview))

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

# Compare terrain horizon for specific (triangle) cell
ind = 520_528
# azim_old = np.arange(0.0, 360.0, 360.0 / horizon_old.shape[0])
azim = np.arange(0.0, 360.0, 360.0 / horizon.shape[0])  # 7.5
plt.figure(figsize=(15, 5))
# ---------------------------- Cell centres -----------------------------------
dist_search = 40_000  # search distance for horizon [m]
phi, theta, radius = observer_perspective(clon, clat, hsurf,
                                          clon[ind], clat[ind], hsurf[ind])
mask_d = (radius <= dist_search)
plt.scatter(phi[mask_d], theta[mask_d], color="sienna", s=20, alpha=0.5)
mask_c = mask_d & (theta > 3.0)
# ---------------------------- Cell vertices ----------------------------------
phi, theta, radius = observer_perspective(vlon, vlat, topography_v,
                                          clon[ind], clat[ind], hsurf[ind])
plt.scatter(phi, theta, color="black", s=20)
for i in np.where(mask_c)[0]:
    iv0, iv1, iv2 = vertex_of_cell[:, i]
    phi_line = [phi[iv0], phi[iv1], phi[iv2], phi[iv0]]
    theta_line = [theta[iv0], theta[iv1], theta[iv2], theta[iv0]]
    for j in range(3):
        if np.abs(np.diff(phi_line[j:(j + 2)])) < 180.0:
            plt.plot(phi_line[j:(j + 2)], theta_line[j:(j + 2)],
                     color="grey", lw=0.5)
# -----------------------------------------------------------------------------
# plt.plot(azim_old, horizon_old[:, ind], label="old", color="red", lw=2.5)
plt.plot(azim, horizon[:, ind], label="Ray tracing", color="royalblue", lw=2.5)
plt.axis([0.0, 360.0, 0.0, 50.0])
plt.legend(fontsize=12, frameon=False)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")

# Colormap
values = skyview
cmap = plt.get_cmap("YlGnBu_r")
# values = horizon[180, :]  # 0, 12
# cmap = plt.get_cmap("afmhot_r")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(np.percentile(values, 2), np.percentile(values, 98))
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")

# Compare with 'old' values
# values = skyview_old
# values = horizon_old[6, :]

# Map
plt.figure()
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
# for i in range(5):
#     pts = plt.ginput(1)
#     ind = np.argmin(np.sqrt((pts[0][0] - np.rad2deg(clon)) ** 2
#                             + (pts[0][1] - np.rad2deg(clat)) ** 2))
#     print(ind)
# -----------------------------------------------------------------------------

# Map plot of topography
cmap = plt.get_cmap("terrain")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(np.percentile(hsurf, 5), np.percentile(hsurf, 95))
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = mpl.tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.tripcolor(triangles, hsurf, cmap=cmap, norm=norm,
              edgecolors="black", linewidth=0.0)
# -----------------------------------------------------------------------------
plt.scatter(np.rad2deg(vlon[ind_vertices_outer]),
            np.rad2deg(vlat[ind_vertices_outer]),
            c=topography_v[ind_vertices_outer], cmap=cmap, norm=norm, s=20)
# -----------------------------------------------------------------------------
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
