# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr

# Path to folders
path_extpar = "/scratch/snx3000/csteger/EXTPAR_HORAYZON/ICON_grids_EXTPAR/"
path_out = "/scratch/snx3000/csteger/EXTPAR_HORAYZON/output/"

# Path to Cython/C++ functions
sys.path.append("/scratch/snx3000/csteger/EXTPAR_HORAYZON/Semester_Project/")
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
# file_grid = "Brigitta/domain4_DOM04.nc"
file_grid = "Brigitta/domain_switzerland_100m.nc"
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
# file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain4_DOM04.nc"
file_topo = "Brigitta/topography_buffer_extpar_v5.8_domain_switzerland_" \
            + "100m.nc"
ds = xr.open_dataset(path_extpar + file_topo)
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
# nhori = ds["nhori"].size
# horizon_old = ds["HORIZON"].values.squeeze()  # (nhori, num_cell)
# skyview_old = ds["SKYVIEW"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
nhori = 24
refine_factor = 10
# nhori = 240
# refine_factor = 1
svf_type = 3

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
# Save output to NetCDF file
# -----------------------------------------------------------------------------

ds = xr.Dataset({
    "HORIZON": xr.DataArray(
        data=horizon,
        dims=["nhori", "cell"],
        attrs={
            "units": "deg"
        }
    ),
    "SKYVIEW": xr.DataArray(
        data=skyview,
        dims=["cell"],
        attrs={
            "units": "-"
        }
    )
},
    attrs={"settings": "refine_factor: " + str(refine_factor)
                       + ", svf_type: " + str(svf_type)}
)
file_out = file_grid.split("/")[-1].split(".")[0] + "_horizon_svf.nc"
ds.to_netcdf(path_out + file_out)
