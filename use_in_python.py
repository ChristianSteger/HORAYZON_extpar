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
root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"
path_extpar = root_IAC + "Miscellaneous/Thesis_supervision/Caterina_Croci/"\
              + "ICON_grids_EXTPAR/"

# Path to Cython/C++ functions
sys.path.append("/Users/csteger/Downloads/Semester_Project/")
from horizon_svf import horizon_svf_comp_py

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
# vertex_of_cell = ds["vertex_of_cell"].values  # (3, num_cell; int32)
# ds.close()
#
# # Load elevation of cell vertices
# file_topo = "EXTPAR_test/topography_buffer_extpar_v5.8_icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# nhori = ds["nhori"].size
# topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
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

# Load grid information
# file_grid = "Brigitta/domain1_DOM01.nc"
# file_grid = "Brigitta/domain2_DOM02.nc"
# file_grid = "Brigitta/domain3_DOM03.nc"
file_grid = "Brigitta/domain4_DOM04.nc"
ds = xr.open_dataset(path_extpar + file_grid)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values  # (3, num_cell; int32)
ds.close()

# Load elevation of cell vertices
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain1_DOM01.nc"
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain2_DOM02.nc"
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain3_DOM03.nc"
file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
ds = xr.open_dataset(path_extpar + file_topo)
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# nhori = ds["nhori"].size
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
# nhori = 24
# refine_factor = 10
nhori = 240
refine_factor = 1
svf_type = 3

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
#                            [4, 2, 3]], dtype=np.int32).transpose()
# nhori = 24
# svf_type = 0
# refine_factor = 1

# -----------------------------------------------------------------------------
# Artificial Data for testing (-> 'dummy data' -> only use for testing the
# coordinate transformation performance!)
# -----------------------------------------------------------------------------

# num_vertex = 8_000_000
# num_cell = 20_000_000
# vlon = np.zeros(num_vertex, dtype=np.float64)
# vlat = np.zeros(num_vertex, dtype=np.float64)
# topography_v = np.zeros(num_vertex, dtype=np.float32)
# clon = np.zeros(num_cell, dtype=np.float64)
# clat = np.zeros(num_cell, dtype=np.float64)
# vertex_of_cell = np.zeros((3, num_cell), dtype=np.int32)
# mask = np.ones_like(clon, dtype=np.uint8)
# nhori = 24
# svf_type = 0

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                       clon, clat, vertex_of_cell,
                                       nhori, refine_factor, svf_type)
print("Total elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")

# Check range of computes values
print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
      + ", %.5f" % np.max(horizon))
print("Sky view factor range [-]: %.8f" % np.min(skyview)
      + ", %.8f" % np.max(skyview))

# -----------------------------------------------------------------------------
# Plot output
# -----------------------------------------------------------------------------

# Specific grid cell
ind = 10528
azim_old = np.arange(0.0, 360.0, 360.0 / horizon_old.shape[0])
azim = np.arange(0.0, 360.0, 360.0 / horizon.shape[0])  # 7.5
plt.figure(figsize=(15, 5))
# plt.plot(azim_old, horizon_old[:, ind], label="old", color="black", lw=2.0)
plt.plot(azim, horizon[:, ind], label="ray casting", color="blue", lw=2.0)
# -----------------------------------------------------------------------------
txt = "+proj=pipeline +step +proj=cart +ellps=sphere +R=6371229.0" \
      + " +step +proj=topocentric +ellps=sphere +R=6371229.0" \
      + " +lon_0=" + str(np.rad2deg(clon[ind])) \
      + " +lat_0=" + str(np.rad2deg(clat[ind])) + " +h_0=" + str(hsurf[ind])
t = Transformer.from_pipeline(txt)
# print(t.transform(np.rad2deg(clon[ind]), np.rad2deg(clat[ind]), hsurf[ind]))
# x, y, z = t.transform(np.rad2deg(clon), np.rad2deg(clat), hsurf)
x, y, z = t.transform(np.rad2deg(vlon), np.rad2deg(vlat), topography_v)
r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
theta = np.rad2deg(np.arccos(z / r))  # zenith angle
phi = np.rad2deg(np.arctan2(x, y))  # azimuth angle
phi[phi < 0.0] += 360.0
plt.scatter(phi, 90.0 - theta, color="grey", s=30)
# -----------------------------------------------------------------------------
plt.axis([0.0, 360.0, 0.0, 50.0])
plt.legend()
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")

# Colormap
# values = skyview
# cmap = plt.get_cmap("YlGnBu_r")
values = horizon[180, :]  # 0, 12
cmap = plt.get_cmap("afmhot_r")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(np.percentile(values, 5), np.percentile(values, 95))
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
for i in range(5):
    pts = plt.ginput(1)
    ind = np.argmin(np.sqrt((pts[0][0] - np.rad2deg(clon)) ** 2
                            + (pts[0][1] - np.rad2deg(clat)) ** 2))
    print(ind)

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
mask = (np.rad2deg(vlon) < 11.15)
plt.scatter(np.rad2deg(vlon[mask]), np.rad2deg(vlat[mask]),
            c=topography_v[mask], cmap=cmap, norm=norm, s=20)
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

