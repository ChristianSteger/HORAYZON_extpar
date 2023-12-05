# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# mpl.style.use("classic")

# Path to folders
root_IAC = os.getenv("HOME") + "/Dropbox/IAC/"
path_extpar = root_IAC + "Miscellaneous/Thesis_supervision/Caterina_Croci/"\
              + "ICON_grids_EXTPAR/EXTPAR_test/"

# Path to Cython/C++ functions
sys.path.append("/Users/csteger/Downloads/Semester_Project/")
from horizon_svf import horizon_svf_comp_py

# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

# # Load grid information
# file_grid = "icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_grid)
# vlon = ds["vlon"].values  # (num_vertex; float64)
# vlat = ds["vlat"].values  # (num_vertex; float64)
# clon = ds["clon"].values  # (num_cell; float64)
# clat = ds["clat"].values  # (num_cell; float64)
# vertex_of_cell = ds["vertex_of_cell"].values  # (3, num_cell; int32)
# ds.close()
#
# # Load elevation of cell vertices
# file_topo = "topography_buffer_extpar_v5.8_icon_grid_DOM01.nc"
# ds = xr.open_dataset(path_extpar + file_topo)
# nhori = ds["nhori"].size
# topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
# ds.close()
#
# mask = np.ones_like(clon, dtype=np.uint8)  # (num_cell, uint8)
# svf_type = 1

# -----------------------------------------------------------------------------
# Artificial Data for testing (very large data set...)
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
# svf_type = 1

# -----------------------------------------------------------------------------
# Artificial Data for testing 
# -----------------------------------------------------------------------------

vlon = np.deg2rad(np.array([-1.0, 0.0, +1.0, +0.5, -0.5], dtype=np.float64))
vlat = np.deg2rad(np.array([0.0, 0.0, 0.0, +1.0, +1.0], dtype=np.float64))
topography_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
clon = np.deg2rad(np.array([-0.5, 0.0, +0.5], dtype=np.float64))
clat = np.deg2rad(np.array([+0.5, +0.5, +0.5], dtype=np.float64))
vertex_of_cell = np.array([[5, 1, 2],
                           [4, 5, 2],
                           [4, 2, 3]], dtype=np.int32).transpose()
mask = np.ones_like(clon, dtype=np.uint8)
nhori = 24
svf_type = 1

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                       clon, clat, vertex_of_cell, mask,
                                       nhori, svf_type)
print("Elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")
