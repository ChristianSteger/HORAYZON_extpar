import ctypes
import glob
import os
import sysconfig

# libembree4 (bundled into this package's wheel) depends on libtbb.so.12.
# TBB itself ships as a separate pip wheel ('tbb') rather than being bundled
# here, so it must be preloaded with RTLD_GLOBAL before the compiled
# extension is imported -- otherwise the dynamic linker has no reliable way
# to find it, since 'tbb' installs its libraries relative to sys.prefix
# rather than into site-packages.
def _preload_tbb():
    lib_dir = os.path.join(sysconfig.get_paths()["data"], "lib")
    for pattern in ("libtbb.so.12", "libtbb.so"):
        matches = glob.glob(os.path.join(lib_dir, pattern))
        if matches:
            ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
            return
    raise ImportError(
        "Could not locate libtbb (expected under "
        f"'{lib_dir}'). Is the 'tbb' package installed?"
    )


_preload_tbb()

from .horizon import Terrain  # noqa: E402

__all__ = ["Terrain"]
