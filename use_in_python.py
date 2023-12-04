# Load modules
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pymap3d

# Path to Cython/C++ functions
sys.path.append("/Users/csteger/Desktop/Test/")

from horizon_svf import horizon_svf_comp_py
#help(horizon_svf_comp_py)  # returns docstring of function

# Load grid data 
path_grid = "/home/catecroci/SP_files/ICON_tools/"
grid_name = "child_grid_DOM01_lon_2.0_lat_1.0.nc"

# Load topography data
path_topo = "/home/catecroci/SP_files/output/dir_1/"
topo_name = "topography_buffer.nc"

# -----------------------------------------------------------
# Artificial Data for testing 
# -----------------------------------------------------------
'''
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
'''
# Create a test grid using a part of the ICON Grid
cell = 14
vertex = 13
nv = 3
nhori = 24
grid = xr.open_dataset(path_grid + grid_name)

vlon = np.empty(vertex, dtype=np.float64)
vlat = np.empty(vertex, dtype=np.float64)
clon = np.empty(cell, dtype=np.float64)
clat = np.empty(cell, dtype=np.float64)
vertex_of_cell_grid= grid["vertex_of_cell"].values
vertex_of_cell = np.empty((nv, cell), dtype=np.int32)  # indices start with 1!


vlon[0] = grid["vlon"][vertex_of_cell_grid[1][1]-1].values
vlat[0] = grid["vlat"][vertex_of_cell_grid[1][1]-1].values
vlon[1] = grid["vlon"][vertex_of_cell_grid[0][0]-1].values
vlat[1] = grid["vlat"][vertex_of_cell_grid[0][0]-1].values
vlon[2] = grid["vlon"][vertex_of_cell_grid[2][20]-1].values
vlat[2] = grid["vlat"][vertex_of_cell_grid[2][20]-1].values
vlon[3] = grid["vlon"][vertex_of_cell_grid[2][22]-1].values
vlat[3] = grid["vlat"][vertex_of_cell_grid[2][22]-1].values
vlon[4] = grid["vlon"][vertex_of_cell_grid[1][659]-1].values
vlat[4] = grid["vlat"][vertex_of_cell_grid[1][659]-1].values
vlon[5] = grid["vlon"][vertex_of_cell_grid[2][591]-1].values
vlat[5] = grid["vlat"][vertex_of_cell_grid[2][591]-1].values
vlon[6] = grid["vlon"][vertex_of_cell_grid[2][589]-1].values
vlat[6] = grid["vlat"][vertex_of_cell_grid[2][589]-1].values
vlon[7] = grid["vlon"][vertex_of_cell_grid[1][589]-1].values
vlat[7] = grid["vlat"][vertex_of_cell_grid[1][589]-1].values
vlon[8] = grid["vlon"][vertex_of_cell_grid[1][625]-1].values
vlat[8] = grid["vlat"][vertex_of_cell_grid[1][625]-1].values
vlon[9] = grid["vlon"][vertex_of_cell_grid[0][659]-1].values
vlat[9] = grid["vlat"][vertex_of_cell_grid[0][659]-1].values
vlon[10] = grid["vlon"][vertex_of_cell_grid[1][1164]-1].values
vlat[10] = grid["vlat"][vertex_of_cell_grid[1][1164]-1].values
vlon[11] = grid["vlon"][vertex_of_cell_grid[1][1168]-1].values
vlat[11] = grid["vlat"][vertex_of_cell_grid[1][1168]-1].values
vlon[12] = grid["vlon"][vertex_of_cell_grid[1][1163]-1].values
vlat[12] = grid["vlat"][vertex_of_cell_grid[1][1163]-1].values




clon[0] = grid["clon"][577].values
clat[0] = grid["clat"][577].values
clon[1] = grid["clon"][589].values
clat[1] = grid["clat"][589].values
clon[2] = grid["clon"][590].values
clat[2] = grid["clat"][590].values
clon[3] = grid["clon"][591].values
clat[3] = grid["clat"][591].values
clon[4] = grid["clon"][592].values
clat[4] = grid["clat"][592].values
clon[5] = grid["clon"][593].values
clat[5] = grid["clat"][593].values
clon[6] = grid["clon"][659].values
clat[6] = grid["clat"][659].values
clon[7] = grid["clon"][1164].values
clat[7] = grid["clat"][1164].values
clon[8] = grid["clon"][1167].values
clat[8] = grid["clat"][1167].values
clon[9] = grid["clon"][1168].values
clat[9] = grid["clat"][1168].values
clon[10] = grid["clon"][1169].values
clat[10] = grid["clat"][1169].values
clon[11] = grid["clon"][1163].values
clat[11] = grid["clat"][1163].values
clon[12] = grid["clon"][1140].values
clat[12] = grid["clat"][1140].values
clon[13] = grid["clon"][625].values
clat[13] = grid["clat"][625].values

topo = xr.open_dataset(path_topo + topo_name)
nhori = topo["nhori"].size
topo_grid = topo["topography_v"].values
topography_v = np.empty(vertex, dtype=np.float32)

# reshape topography_v
topo_grid = topo_grid.ravel()

topography_v[0] = topo_grid[vertex_of_cell_grid[1][1]-1]
topography_v[1] = topo_grid[vertex_of_cell_grid[0][0]-1]
topography_v[2] = topo_grid[vertex_of_cell_grid[2][20]-1]
topography_v[3] = topo_grid[vertex_of_cell_grid[2][22]-1]
topography_v[4] = topo_grid[vertex_of_cell_grid[1][659]-1]
topography_v[5] = topo_grid[vertex_of_cell_grid[2][591]-1]
topography_v[6] = topo_grid[vertex_of_cell_grid[2][589]-1]
topography_v[7] = topo_grid[vertex_of_cell_grid[1][589]-1]
topography_v[8] = topo_grid[vertex_of_cell_grid[1][625]-1]
topography_v[9] = topo_grid[vertex_of_cell_grid[0][659]-1]
topography_v[10] = topo_grid[vertex_of_cell_grid[1][1164]-1]
topography_v[11] = topo_grid[vertex_of_cell_grid[1][1168]-1]
topography_v[12] = topo_grid[vertex_of_cell_grid[1][1163]-1]

vertex_of_cell[0] = np.array([8, 2, 7, 3, 6, 4, 10, 6, 11, 7, 12, 8, 13, 1])
vertex_of_cell[1] = np.array([2, 8, 3, 7, 4, 6, 5, 11, 6, 12, 7, 13, 8, 9])
vertex_of_cell[2] = np.array([1, 7, 2, 6, 3, 5, 6, 10, 7, 11, 8, 12, 9, 8])

# print circumcenters and vertices
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = mpl.tri.Triangulation(vlon, vlat,(vertex_of_cell.transpose() - 1))
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
ax1.triplot(triang, 'bo-', lw=1)
ax1.set_title('Triangles simple grid')
ax1.scatter(clon, clat, s=10, c='r')

mask = np.ones_like(clon, dtype=np.uint8)

ellps = pymap3d.Ellipsoid(semimajor_axis=6371229.0, semiminor_axis=6371229.0)

e, n, u = pymap3d.geodetic2enu(0.184092, 0.814923, 0.0, 0.183779, 0.814593, 0.0, ell=ellps, deg=False) 

'''
grid = xr.open_dataset(path_grid + grid_name)
cell = grid["cell"].size
vertex = grid["vertex"].size
nv = grid["nv"].size
vlon = grid["vlon"].values
vlat = grid["vlat"].values
clon = grid["clon"].values
clat = grid["clat"].values
vertex_of_cell = grid["vertex_of_cell"].values

topo = xr.open_dataset(path_topo + topo_name)
nhori = topo["nhori"].size
topography_v = topo["topography_v"].values

# reshape topography_v
topography_v = topography_v.ravel()
# create mask 
mask = np.ones_like(clon, dtype=np.uint8)
'''
hor = topo["HORIZON"].values
hor = np.transpose(hor).ravel()
print(hor[0:72], "\n")

svf_type = 0

while(svf_type not in [1, 2, 3, 4]):
    svf_type = int(input("How do you want to compute the Sky View Factor? \n \
                    Please select one of the 4 options (numbers from 1 to 4) \n \
                    1 = pure-geometric sky view factor; \n \
                    2 = geometric scaled with sin(horizon); \n \
                    3 = geometric scaked with sin(horizon)**2 \n \
                    4 = Steger et al. SVF computation \n"))

horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                    clon, clat, vertex_of_cell, mask,
                                    nhori, svf_type)

        


