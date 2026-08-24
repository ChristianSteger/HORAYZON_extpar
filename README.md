# HORAYZON_extpar
Computes grid- and subgrid-scale terrain horizon for ICON grids

# Installation

 Create Conda environment:
```bash
conda create -n horayzon_extpar -c conda-forge embree tbb-devel cython setuptools numpy xarray netcdf4 matplotlib cartopy numba scipy pyinterp skyfield ipython pyproj
conda install sgp4 gfortran meson -c conda-forge
```
activate this environment, clone **HORAYZON_extpar** and compile with:
```bash
git clone git@github.com:ChristianSteger/HORAYZON_extpar.git
cd HORAYZON_extpar
python setup.py build_ext --inplace
```

# Miscellaneous notes

Relevant python packages:
- For pip "HORAYZON_extpar" package: embree tbb-devel cython setuptools numpy
- For pre- and post-processing and checking: xarray netcdf4 matplotlib cartopy numba scipy pyinterp skyfield ipython
- For testing: pyproj

Development strategy:
- First develop entire workflow within "HORAYZON_extpar"
- Then move pre- and post-processing part to EXTPAR (and check, test and plot part to separate python scripts)
- Make "HORAYZON_extpar" a Python package that can be installed for EXTPAR with pip

Important:
- The Embree ray tracing performance does not scale well on multi-sockets nodes (noted for large BVH). Better use 64 CPUs on one socket instead of 128 CPUs on two sockets.
