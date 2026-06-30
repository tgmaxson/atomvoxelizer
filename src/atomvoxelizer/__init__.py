"""AtomVoxelizer public API."""

from .voxelgrid import VoxelGrid, VoxelGridNumPy
from .analysis import VoxelGridAnalysis, VoxelRegion

__all__ = [
    "VoxelGrid",
    "VoxelGridAnalysis",
    "VoxelGridCuPy",
    "VoxelGridNumPy",
    "VoxelGridNumba",
    "VoxelGridTaichi",
    "VoxelGridTaichiGPU",
    "VoxelRegion",
]


def __getattr__(name):
    if name == "VoxelGridCuPy":
        from .cupy_backend import VoxelGridCuPy

        return VoxelGridCuPy
    if name == "VoxelGridNumba":
        from .numba_backend import VoxelGridNumba

        return VoxelGridNumba
    if name == "VoxelGridTaichi":
        from .taichi_backend import VoxelGridTaichi

        return VoxelGridTaichi
    if name == "VoxelGridTaichiGPU":
        from .taichi_backend import VoxelGridTaichiGPU

        return VoxelGridTaichiGPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
