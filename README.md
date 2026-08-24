# HORAYZON_extpar
Embree ray tracing interface to compute grid- and subgrid-scale terrain horizon for ICON grids.

# Installation

## From PyPI (Linux x86_64)

```bash
pip install horayzon-extpar
```

This installs a prebuilt wheel with Embree bundled in, plus `tbb` (Intel's
official runtime wheel) as a dependency.

## From source (development)

Either use Conda to provide Embree and TBB:
```bash
conda create -n horayzon_extpar -c conda-forge embree tbb-devel cython setuptools numpy
conda activate horayzon_extpar
```
then point the build at the Conda environment's Embree install and do an
editable install:
```bash
git clone git@github.com:ChristianSteger/HORAYZON_extpar.git
cd HORAYZON_extpar
EMBREE_DIR="$CONDA_PREFIX" pip install -e .
```

Or skip Conda entirely and just `pip install -e .` from a plain venv with
`pip install tbb-devel tbb` — `setup.py` will download and cache the
official Embree release under `.embree_cache/` automatically.

# Miscellaneous notes

Important:
- The Embree ray tracing performance does not scale well on multi-sockets nodes (noted for large BVH). Better use 64 CPUs on one socket instead of 128 CPUs on two sockets.
