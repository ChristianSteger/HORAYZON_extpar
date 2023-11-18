# Load modules
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl

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


# Test use
cell = 11
vertex = 11
nv = 3
nhori = 24
grid = xr.open_dataset(path_grid + grid_name)

vlon = np.empty(vertex, dtype=np.float64)
vlat = np.empty(vertex, dtype=np.float64)
clon = np.empty(cell, dtype=np.float64)
clat = np.empty(cell, dtype=np.float64)
topography_v = np.empty(vertex, dtype=np.float32)
vertex_of_cell_grid= grid["vertex_of_cell"].values
vertex_of_cell = np.empty((nv, cell), dtype=np.int32)  # indices start with 1!

vlon[0] = grid["vlon"][vertex_of_cell_grid[0][0]-1].values
vlat[0] = grid["vlat"][vertex_of_cell_grid[0][0]-1].values
vlon[1] = grid["vlon"][vertex_of_cell_grid[1][0]-1].values
vlat[1] = grid["vlat"][vertex_of_cell_grid[1][0]-1].values
vlon[2] = grid["vlon"][vertex_of_cell_grid[2][0]-1].values
vlat[2] = grid["vlat"][vertex_of_cell_grid[2][0]-1].values
vlon[3] = grid["vlon"][vertex_of_cell_grid[1][1]-1].values
vlat[3] = grid["vlat"][vertex_of_cell_grid[1][1]-1].values
vlon[4] = grid["vlon"][vertex_of_cell_grid[2][2]-1].values
vlat[4] = grid["vlat"][vertex_of_cell_grid[2][2]-1].values
vlon[5] = grid["vlon"][vertex_of_cell_grid[2][20]-1].values
vlat[5] = grid["vlat"][vertex_of_cell_grid[2][20]-1].values
vlon[6] = grid["vlon"][vertex_of_cell_grid[1][23]-1].values
vlat[6] = grid["vlat"][vertex_of_cell_grid[1][23]-1].values
vlon[7] = grid["vlon"][vertex_of_cell_grid[2][22]-1].values
vlat[7] = grid["vlat"][vertex_of_cell_grid[2][22]-1].values
vlon[8] = grid["vlon"][vertex_of_cell_grid[1][589]-1].values
vlat[8] = grid["vlat"][vertex_of_cell_grid[1][589]-1].values
vlon[9] = grid["vlon"][vertex_of_cell_grid[2][589]-1].values
vlat[9] = grid["vlat"][vertex_of_cell_grid[2][589]-1].values
vlon[10] = grid["vlon"][vertex_of_cell_grid[2][591]-1].values
vlat[10] = grid["vlat"][vertex_of_cell_grid[2][591]-1].values

clon[0] = grid["clon"][0].values
clat[0] = grid["clat"][0].values
clon[1] = grid["clon"][1].values
clat[1] = grid["clat"][1].values
clon[2] = grid["clon"][2].values
clat[2] = grid["clat"][2].values
clon[3] = grid["clon"][20].values
clat[3] = grid["clat"][20].values
clon[4] = grid["clon"][23].values
clat[4] = grid["clat"][23].values
clon[5] = grid["clon"][22].values
clat[5] = grid["clat"][22].values
clon[6] = grid["clon"][577].values
clat[6] = grid["clat"][577].values
clon[7] = grid["clon"][589].values
clat[7] = grid["clat"][589].values
clon[8] = grid["clon"][590].values
clat[8] = grid["clat"][590].values
clon[9] = grid["clon"][591].values
clat[9] = grid["clat"][591].values
clon[10] = grid["clon"][592].values
clat[10] = grid["clat"][592].values


vertex_of_cell[0] = np.array([293, 151, 306, 1, 302, 148, 578, 293, 719, 302, 586])
vertex_of_cell[1] = np.array([1, 306, 151, 293, 148, 302, 293, 578, 302, 719, 303])
vertex_of_cell[2] = np.array([151, 293, 28, 302, 1, 303, 306, 719, 293, 586, 302])

vertex_of_cell[0] = np.array([1, 3, 4, 2, 6, 7, 9, 1, 10, 6, 11])
vertex_of_cell[1] = np.array([2, 4, 3, 1, 7, 6, 1, 9, 6, 10, 8])
vertex_of_cell[2] = np.array([3, 1, 5, 6, 2, 8, 4, 10, 1, 11, 6])

'''triangles_lon = np.empty((cell, 3))
triangles_lat = np.empty((cell, 3))
for i in range(cell):
    triangles_lon[i][:] = vlon[vertex_of_cell[:,i]-1]
    #triangles_lon[i][1] = vlon[vertex_of_cell[1][i]-1]
    #triangles_lon[i][2] = vlon[vertex_of_cell[2][i]-1]
    triangles_lat[i][:] = vlat[vertex_of_cell[:,i]-1]
'''

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
'''
# create mask 
mask = np.ones_like(clon, dtype="int32")

# give the user the chance to change the mask
#print('By default, all grid cells are considered.\n')
#answer = input('Do you want to exclude some of them? (y/n)')



# Check arguments
if ((vlon.shape != vlat.shape) or (clon.shape != clat.shape)):
    raise ValueError("Inconsistent shapes / number of dimensions of "
                        + "input arrays")
if ((vlon.dtype != "float64") or (vlat.dtype != "float64")
        or (clon.dtype != "float64") or (clat.dtype != "float64")):
    raise ValueError("Input array(s) has/have incorrect data type(s)")

horizon, skyview = horizon_svf_comp_py(vlon, vlat, clon, clat, topography_v,
                                       vertex_of_cell, vertex, cell, nhori, mask)







