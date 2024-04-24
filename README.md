# HORAYON_extpar

Implement HORAYZON algorithm in pre-processing tool EXTPAR.

# Installation

 Create Conda environment:
```bash
conda create -n horayzon_extpar -c conda-forge embree3 tbb-devel cython numpy xarray netcdf4 matplotlib cartopy pyproj ipython
```
activate this environment, clone **HORAYON_extpar** and compile with:
```bash
git clone git@github.com:ChristianSteger/HORAYZON_extpar.git
cd HORAYZON_extpar
python setup.py build_ext --inplace
```
