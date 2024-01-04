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
from scipy.spatial import KDTree

mpl.style.use("classic")

# Path to folders
# path_extpar = "/home/catecroci/SP_files/" \
#               + "ICON_grids_EXTPAR/"
root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"
path_extpar = root_IAC + "Miscellaneous/Thesis_supervision/Caterina_Croci/" \
               + "ICON_grids_EXTPAR/"

# Path to Cython/C++ functions
sys.path.append("/Users/csteger/Downloads/Semester_Project/")
from horizon_svf import horizon_svf_comp_py

# -----------------------------------------------------------------------------
# Real data (resolutions from ~2km to ~30 m)
# -----------------------------------------------------------------------------

# Load grid information (domain with boundary zone)
# file_grid = "Resolutions/icon_grid_res0032m_bz40km.nc"
# file_grid = "Resolutions/icon_grid_res0130m_bz40km.nc"
# file_grid = "Resolutions/icon_grid_res0519m_bz40km.nc"
file_grid = "Resolutions/icon_grid_res2076m_bz40km.nc"
ds = xr.open_dataset(path_extpar + file_grid)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
edge_of_cell = ds["edge_of_cell"].values - 1  # (3, num_cell)
adjacent_cell_of_edge = ds["adjacent_cell_of_edge"].values - 1  # (2, num_edge)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
ds.close()

# Load elevation of cell vertices (domain with boundary zone)
# file_topo = "topography_buffer_extpar_v5.8_icon_grid_res0032m_bz40km.nc"
# file_topo = "topography_buffer_extpar_v5.8_icon_grid_res0130m_bz40km.nc"
# file_topo = "topography_buffer_extpar_v5.8_icon_grid_res0519m_bz40km.nc"
file_topo = "topography_buffer_extpar_v5.8_icon_grid_res2076m_bz40km.nc"
ds = xr.open_dataset(path_extpar + "Resolutions/" + file_topo)
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
ds.close()

# Get mask with 'inner cells'
ds = xr.open_dataset(path_extpar + file_grid.replace("_bz40km", ""))
clon_in = ds["clon"].values  # (num_cell; float64)
clat_in = ds["clat"].values  # (num_cell; float64)
ds.close()
pts = np.hstack((clon[:, np.newaxis], clat[:, np.newaxis]))
tree = KDTree(pts)
pts_in = np.hstack((clon_in[:, np.newaxis], clat_in[:, np.newaxis]))
dist, indices = tree.query(pts_in)
if np.max(dist) > 0.0:
    raise ValueError("Non-zero distance found")

# Further settings
nhori = 24
refine_factor = 10
svf_type = 3
mask_cell = np.zeros(clon.size, dtype=np.uint8)
mask_cell[indices] = 1

# Find outermost cells and indices of cell vertices
mask_cell_outer = np.zeros(clon.size, dtype=bool)
for i in range(clon.size):
    if np.any(adjacent_cell_of_edge[:, edge_of_cell[:, i]] == -2):
        mask_cell_outer[i] = True
ind_vertices_outer = np.unique(vertex_of_cell[:, np.where(mask_cell_outer)[0]])

# Adjust erroneous elevation values of outermost cell vertices
for i in ind_vertices_outer:
    mask = cells_of_vertex[:, i] != -2
    topography_v[i] = hsurf[cells_of_vertex[:, i][mask]].mean()

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                       clon, clat, (vertex_of_cell + 1),
                                       nhori, refine_factor, svf_type,
                                       mask_cell)
print("Total elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")

# Check range of computes values
print("Terrain horizon range [deg]: %.5f" % np.nanmin(horizon)
      + ", %.5f" % np.nanmax(horizon))
print("Sky view factor range [-]: %.8f" % np.nanmin(skyview)
      + ", %.8f" % np.nanmax(skyview))

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

# Colormap
values = skyview
cmap = plt.get_cmap("YlGnBu_r")
# values = horizon[180, :]  # 0, 12
# cmap = plt.get_cmap("afmhot_r")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
    .tick_values(np.nanpercentile(values, 2), np.nanpercentile(values, 98))
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
