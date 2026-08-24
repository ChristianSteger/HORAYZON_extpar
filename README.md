# HORAYZON_extpar
Embree ray tracing interface to compute grid- and subgrid-scale terrain horizon for ICON grids.

# Installation

 Create Conda environment:
```bash
conda create -n horayzon_extpar -c conda-forge embree tbb-devel cython setuptools numpy
```
activate this environment, clone **HORAYZON_extpar** and compile with:
```bash
git clone git@github.com:ChristianSteger/HORAYZON_extpar.git
cd HORAYZON_extpar
python setup.py build_ext --inplace
```

# Miscellaneous notes

Important:
- The Embree ray tracing performance does not scale well on multi-sockets nodes (noted for large BVH). Better use 64 CPUs on one socket instead of 128 CPUs on two sockets.
