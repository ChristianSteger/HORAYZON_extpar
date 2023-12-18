# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# svf_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
# ds.close()
#
# # Further settings
# svf_type = 0
# refine_factor = 10

# -----------------------------------------------------------------------------
# Real data (Brigitta)
# -----------------------------------------------------------------------------

# Load grid information
file_grid = "Brigitta/domain4_DOM04.nc"
ds = xr.open_dataset(path_extpar + file_grid)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values  # (3, num_cell; int32)
ds.close()

# Load elevation of cell vertices
file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
ds = xr.open_dataset(path_extpar + file_topo)
# nhori = ds["nhori"].size
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# svf_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
nhori = 24
svf_type = 3
refine_factor = 10

# -----------------------------------------------------------------------------
# Artificial Data for testing 
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
# Artificial Data for testing (-> 'dummy data' -> should only be used for
# testing the coordinate transformation performance)
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

# Test plot
ind = 13345  # 3534  # 18416
plt.figure()
# plt.plot(horizon_old[:, ind], label="old", color="black", lw=2.0)
plt.plot(horizon[:, ind], label="ray casting", color="blue", lw=2.0)
plt.legend()

# -----------------------------------------------------------------------------
# Check stuff for C++ implementation -> remove later...
# -----------------------------------------------------------------------------

# # Check vector rotation
# v = np.array([+0.5, -0.7, +0.7])
# v = v / np.sqrt((v ** 2).sum())  # unit vector
# k = np.array([+13.7, -9.1, -3.7])
# k = k / np.sqrt((k ** 2).sum())  # unit vector
# num = 10_000
# ang_rot = np.deg2rad(360.0 / float(num))
# ang = 0.0
# v_orig = v.copy()
# for i in range(num):
#     v = v * np.cos(ang_rot) + np.cross(k, v) * np.sin(ang_rot) \
#         + k * np.dot(k, v) * (1.0 - np.cos(ang_rot))
# print(np.abs(v - v_orig).max())
# print((v ** 2).sum() - (v_orig ** 2).sum())

# # Check indices for saving in 'horizon'
# num_cell = 3
# azim_num = 24
# temp = np.empty((azim_num * num_cell), dtype=np.int32)
# n = 0
# for i in range(num_cell):
#     for j in range(azim_num):
#         temp[(j * num_cell) + i] = n
#         n += 1
# print(np.array(temp).reshape(azim_num, num_cell))
