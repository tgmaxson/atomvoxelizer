"""AtomVoxelizer public API."""

from .voxelgrid import VoxelGrid, VoxelGridNumPy
from .vectorgrid import FieldVoxelGrid, FieldVoxelGridNumPy, VectorVoxelGrid, VectorVoxelGridNumPy
from .analysis import ProbeAccessibleResult, VoxelGridAnalysis, VoxelRegion

__all__ = [
    "VoxelGrid",
    "VoxelGridAnalysis",
    "VoxelGridCuPy",
    "VoxelGridNumPy",
    "VoxelGridNumba",
    "VoxelGridTaichi",
    "VoxelGridTaichiGPU",
    "VoxelRegion",
    "FieldVoxelGrid",
    "FieldVoxelGridNumPy",
    "FieldVoxelGridNumba",
    "ProbeAccessibleResult",
    "VectorVoxelGrid",
    "VectorVoxelGridNumPy",
    "VectorVoxelGridNumba",
]


def __getattr__(name):
    if name in {
        "FieldVoxelGridCuPy",
        "FieldVoxelGridTaichi",
        "FieldVoxelGridTaichiGPU",
        "VectorVoxelGridCuPy",
        "VectorVoxelGridTaichi",
        "VectorVoxelGridTaichiGPU",
    }:
        raise NotImplementedError(f"{name} is not implemented on the field-grid development branch")
    if name == "VoxelGridCuPy":
        from .cupy_backend import VoxelGridCuPy

        return VoxelGridCuPy
    if name == "VoxelGridNumba":
        from .numba_backend import VoxelGridNumba

        return VoxelGridNumba
    if name == "FieldVoxelGridNumba":
        from .numba_field_backend import FieldVoxelGridNumba

        return FieldVoxelGridNumba
    if name == "VectorVoxelGridNumba":
        from .numba_field_backend import VectorVoxelGridNumba

        return VectorVoxelGridNumba
    if name == "VoxelGridTaichi":
        from .taichi_backend import VoxelGridTaichi

        return VoxelGridTaichi
    if name == "VoxelGridTaichiGPU":
        from .taichi_backend import VoxelGridTaichiGPU

        return VoxelGridTaichiGPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
