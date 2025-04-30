# Description: Check the effect of increasing 'nhori'
#
# Author: Christian R. Steger (Christian.Steger@meteoswiss.ch), April 2025

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, rcParams, tri
from scipy.interpolate import interp1d

from development import triangle_mesh_circ_vert
from functions import observer_perspective, distance_to_border

style.use("classic")

# Change latex fonts
rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
rcParams["mathtext.default"] = "rm"
rcParams["mathtext.rm"] = "DejaVu Sans"

# Paths
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/plots/"

###############################################################################
# Load ICON data and terrain horizon
###############################################################################

# ~100 m domain from Brigitta
file_grid = "Brigitta/domain4_DOM04.nc" # ~125 m
file_extpar = "Brigitta/extpar_icon_grid_domain4_DOM04.nc"
file_extpar_new = "domain4_DOM04_horayzon.nc"

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

# Load elevation of cell circumcenters and radtopo fields
file = path_ige + file_extpar.split("/")[0] + "/extpar_grid_shift_topo/" \
    + file_extpar.split("/")[1]
ds = xr.open_dataset(file)
hsurf = ds["topography_c"].values.squeeze()  # (num_cell)
num_hori = ds["nhori"].size
horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
ds.close()

# Ensure consistency of input grid and topography
if (clon.size != hsurf.size):
    raise ValueError("Inconsistent number of cells in grid and topography")

# Dictionary with terrain horizon data
data_hori = {}
data_hori["extpar_old"] = horizon_old

# Load horizon computed with EXTPAR-HORAYZON
file = "/scratch/mch/csteger/ExtPar/output/HORAYZON_extpar/" \
    + "domain4_DOM04_horayzon.nc"
ds = xr.open_dataset(file)
data_hori["nhori_24"] = ds["HORIZON"].values.squeeze()
ds.close()
file = "/scratch/mch/csteger/ExtPar/output/HORAYZON_extpar/" \
    + "domain4_DOM04_horayzon_nhori48.nc"
ds = xr.open_dataset(file)
data_hori["nhori_48"] = ds["HORIZON"].values.squeeze()
ds.close()

# Construct triangle mesh
vertex_of_triangle, clon_ext, clat_ext, hsurf_ext \
    = triangle_mesh_circ_vert(clon, clat, hsurf, vlon, vlat,
                                cells_of_vertex)

# Compute distance to triangle mesh border
dist = distance_to_border(clon, clat, vlon, vlat, cells_of_vertex)

###############################################################################
# Compare terrain horizon for specific locations
###############################################################################

# Settings
dist_search = 40_000.0 #  horizon search distance [m]
ray_org_elev = 0.2 # 0.1, 0.2 [m]

# Find locations with largest deviations
azim_24 = np.arange(0.0, 360.0, 360.0 / data_hori["nhori_24"].shape[0])
azim_48 = np.arange(0.0, 360.0, 360.0 / data_hori["nhori_48"].shape[0])
data_cyc = np.vstack((data_hori["nhori_24"], data_hori["nhori_24"][:1, :]))
azim_24_cyc = np.append(azim_24, 360.0)
f_ip = interp1d(azim_24_cyc, data_cyc, axis=0, bounds_error=True)
data_ip = f_ip(azim_48)
diff_abs = np.abs(data_ip - data_hori["nhori_48"]).mean(axis=0)
# diff_abs = np.abs(data_ip - data_hori["nhori_48"]).max(axis=0)

# # Test plot
# ind = 258502
# plt.figure()
# plt.plot(azim_24, data_hori["nhori_24"][:, ind], color="blue")
# plt.scatter(azim_48, data_ip[:, ind], s=20, color="blue")
# plt.plot(azim_48, data_hori["nhori_48"][:, ind], color="red")
# plt.scatter(azim_48, data_hori["nhori_48"][:, ind], s=20, color="red")
# plt.show()

# Check terrain horizon for specific (triangle) cell
icon_res = "ICON@100m"
grid_type = 1  # used for plotting grid
# ------------------ Brigitta 100 m resolution grid ---------------------------
diff_abs[dist < 10.0] = -99.9
ind = int(np.argsort(diff_abs)[-10]) # -1, -10
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 5))
# ---------------------------- Mask for triangles -----------------------------
phi, theta, radius = observer_perspective(clon_ext, clat_ext, hsurf_ext,
                                        clon[ind], clat[ind],
                                        hsurf[ind] + ray_org_elev)
mask_dist = (radius[vertex_of_triangle].min(axis=0)
             <= dist_search)
mask_theta = (theta[vertex_of_triangle].max(axis=0) > 0.0)
mask_phi = (np.ptp(phi[vertex_of_triangle], axis=0) < 180.0)
mask = mask_dist & mask_theta & mask_phi
tri_fac = (mask.sum() / mask.size * 100.0)
print("Fraction of plotted triangles: %.2f" % tri_fac + " %")
mask_vertex = np.unique(vertex_of_triangle[:, mask].ravel())
# -----------------------------------------------------------------------------
plt.scatter(phi[mask_vertex], theta[mask_vertex], color="sienna", s=20,
            alpha=0.5)
triangles = tri.Triangulation(
    phi, theta, vertex_of_triangle[:, mask].transpose())
plt.triplot(triangles, color="black", linewidth=0.5)
# -----------------------------------------------------------------------------
horizon_old = data_hori["extpar_old"][:, ind]
plt.plot(azim_24, horizon_old, label="Fortran algorithm", color="red", lw=2.5)
horizon_24 = data_hori["nhori_24"][:, ind]
plt.plot(azim_24, horizon_24, label="nhori = 24",
         color="royalblue", lw=1.5, ls="--")
horizon_48 = data_hori["nhori_48"][:, ind]
plt.plot(azim_48, horizon_48, label="nhori = 48",
         color="royalblue", lw=2.5)
hori_all = np.hstack((horizon_old, horizon_24, horizon_48))
plt.axis((0.0, 360.0, 0.0, hori_all.max() * 1.05))
plt.legend(fontsize=12, frameon=False)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
txt = "Latitude: %.3f" % np.rad2deg(clat[ind]) \
    + "$^{\circ}$, longitude: %.3f" % np.rad2deg(clon[ind]) \
    + "$^{\circ}$, elevation: %.0f" % hsurf[ind] + " m, " \
    + icon_res
plt.title(txt, fontsize=12, loc="left")
plt.show()

# conclusion: very minor differences in terrain horizon -> increasing
# 'nhori' from 24 to 48 not really worth...
