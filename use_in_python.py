# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# mpl.style.use("classic")

# Path to folders
path_extpar = "/home/catecroci/SP_files/ICON_grids_EXTPAR/EXTPAR_test/"

from horizon_svf import horizon_svf_comp_py

# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

# Load grid information
file_grid = "icon_grid_DOM01.nc"
ds = xr.open_dataset(path_extpar + file_grid)
# file_grid = "/home/catecroci/SP_files/ICON_tools/child_grid_DOM01_lon_2.0_lat_1.0.nc"
# ds = xr.open_dataset(file_grid) 
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values  # (3, num_cell; int32)
ds.close()

# Load elevation of cell vertices
file_topo = "topography_buffer_extpar_v5.8_icon_grid_DOM01.nc"
ds = xr.open_dataset(path_extpar + file_topo)
# file_topo = "/home/catecroci/SP_files/output/dir_1/topography_buffer.nc"
# ds = xr.open_dataset(file_topo) 
nhori = ds["nhori"].size
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
ds.close()

hor = ds["HORIZON"].values
hor = np.transpose(hor).ravel()
print(hor[0:24])
svf_type = 4 

# -----------------------------------------------------------------------------
# Artificial Data for testing (very large data set...)
# -----------------------------------------------------------------------------

""" num_vertex = 8_000_000
num_cell = 20_000_000
vlon = np.zeros(num_vertex, dtype=np.float64)
vlat = np.zeros(num_vertex, dtype=np.float64)
topography_v = np.zeros(num_vertex, dtype=np.float32)
clon = np.zeros(num_cell, dtype=np.float64)
clat = np.zeros(num_cell, dtype=np.float64)
vertex_of_cell = np.zeros((3, num_cell), dtype=np.int32)
nhori = 24
svf_type = 1 """   

# -----------------------------------------------------------------------------
# Artificial Data for testing 
# -----------------------------------------------------------------------------

""" vlon = np.deg2rad(np.array([-1.0, 0.0, +1.0, +0.5, -0.5], dtype=np.float64))
vlat = np.deg2rad(np.array([0.0, 0.0, 0.0, +1.0, +1.0], dtype=np.float64))
topography_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
clon = np.deg2rad(np.array([-0.5, 0.0, +0.5], dtype=np.float64))
clat = np.deg2rad(np.array([+0.5, +0.5, +0.5], dtype=np.float64))
vertex_of_cell = np.array([[5, 1, 2],
                           [4, 5, 2],
                           [4, 2, 3]], dtype=np.int32).transpose()
nhori = 24
svf_type = 4 """ 

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

vec = [0.861841, -0.504729, 0.0497841]

rotation_degrees = 1
rotation_radians = np.radians(rotation_degrees)
#rotation_axis = np.array([0.999949, -0.00717149, 0.00714197]) # cross prod first
rotation_axis = np.array([0.506103, 0.862249, -0.0196481]) # cross prod second
#rotation_axis = np.array([-0.00694898, 0.0265842, 0.999622]) # norm

rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)
rotated_vec = rotation.apply(vec)

print(rotated_vec)

refine_factor = 5

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                       clon, clat, vertex_of_cell,
                                       nhori, refine_factor, svf_type)
print("Total elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")
