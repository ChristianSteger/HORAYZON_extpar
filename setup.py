# Compile with: pip install -e .
#
# Build-time dependencies:
# - 'tbb-devel' (pip) provides TBB headers/CMake config. Under pip's default
#   build isolation it is installed into a throwaway overlay prefix rather
#   than into the real environment, so its headers are located by walking
#   sys.path (see get_tbb_devel_prefix()) instead of assuming sys.prefix.
# - 'tbb' (pip), a normal runtime dependency, provides the plain (non-debug)
#   libtbb.so used for linking, installed into the real environment prefix.
# - Embree has no equivalent split wheel, so its official prebuilt Linux
#   x86_64 release is downloaded and cached under .embree_cache/ (or pointed
#   to via the EMBREE_DIR environment variable for local/offline builds,
#   e.g. a Conda environment's prefix).
#
# At runtime, only 'tbb' (the pip wheel with just the release .so) is a
# separate dependency; libembree4 is bundled directly into this package's
# wheel by 'auditwheel repair' (see .github/workflows/wheels.yml).

import os
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

if sys.platform != "linux":
    raise RuntimeError("This package only supports Linux x86_64.")

EMBREE_VERSION = "4.4.1"
EMBREE_URL = (
    "https://github.com/RenderKit/embree/releases/download/"
    f"v{EMBREE_VERSION}/embree-{EMBREE_VERSION}.x86_64.linux.tar.gz"
)

ROOT_DIR = Path(__file__).parent.resolve()


def get_embree_dir():
    """Return a directory containing Embree's include/ and lib64/ trees,
    downloading and caching the official release if not overridden."""

    override = os.environ.get("EMBREE_DIR")
    if override:
        return Path(override)

    cache_dir = ROOT_DIR / ".embree_cache" / f"embree-{EMBREE_VERSION}"
    header = cache_dir / "include" / "embree4" / "rtcore.h"
    if not header.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "embree.tar.gz"
        print(f"Downloading Embree {EMBREE_VERSION} from {EMBREE_URL} ...")
        urllib.request.urlretrieve(EMBREE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(cache_dir)
        archive.unlink()
    return cache_dir


def get_tbb_devel_prefix():
    """Locate the install prefix used for the 'tbb-devel' wheel.

    Under pip's default build isolation, 'tbb-devel' (a build-system
    requirement) is installed into a throwaway "overlay" prefix rather than
    into sys.prefix, so it has to be found by walking sys.path for a
    site-packages directory and checking its associated prefix for TBB's
    headers, instead of assuming sys.prefix directly.
    """

    candidates = [Path(sysconfig.get_paths()["data"])]
    for entry in sys.path:
        if entry.endswith("site-packages"):
            candidates.append(Path(entry).parent.parent.parent)

    for prefix in candidates:
        if (prefix / "include" / "tbb" / "parallel_for.h").exists():
            return prefix

    raise RuntimeError(
        "Could not locate TBB headers from any of the following "
        f"candidate prefixes: {candidates}. Is 'tbb-devel' installed?"
    )


embree_dir = get_embree_dir()
embree_lib_dir = embree_dir / "lib64"
if not embree_lib_dir.is_dir():
    embree_lib_dir = embree_dir / "lib"

# Headers come from 'tbb-devel', which build isolation confines to a
# throwaway overlay prefix (see get_tbb_devel_prefix()). The plain,
# non-debug libtbb.so used for linking comes from 'tbb' (a normal runtime
# dependency, so pip installs it into the real environment before this
# extension is built) -- tbb-devel's own overlay only ships *_debug.so
# variants, so it must not be used as the library search path.
prefix_include_dir = get_tbb_devel_prefix() / "include"
prefix_lib_dir = Path(sysconfig.get_paths()["data"]) / "lib"

setup(
    ext_modules=cythonize(
        Extension(
            "horayzon_extpar.horizon",
            sources=[
                "src/horayzon_extpar/horizon.pyx",
                "src/horayzon_extpar/horizon_compute.cpp",
            ],
            include_dirs=[
                np.get_include(),
                str(embree_dir / "include"),
                str(prefix_include_dir),
            ],
            library_dirs=[str(embree_lib_dir), str(prefix_lib_dir)],
            runtime_library_dirs=[str(embree_lib_dir), str(prefix_lib_dir)],
            libraries=["embree4", "tbb"],
            extra_compile_args=["-O3"],
            language="c++",
        )
    ),
)
