from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


ext_modules = []
cmdclass = {}

if os.environ.get("ATOMVOXELIZER_BUILD_CPP") == "1":
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    except ImportError:
        pass
    else:
        extension = Pybind11Extension(
            "atomvoxelizer._cpp_backend",
            [str(Path("src") / "atomvoxelizer" / "_cpp_backend.cpp")],
            cxx_std=17,
        )
        extension.optional = True

        ext_modules = [
            extension,
        ]
        cmdclass = {"build_ext": build_ext}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
