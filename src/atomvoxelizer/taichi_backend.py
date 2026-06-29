import os
from pathlib import Path

import numpy as np

from .voxelgrid import VoxelGrid, _cached_sphere_offsets

_default_cache = Path.home() / ".cache"
if not os.access(_default_cache, os.W_OK):
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("TI_CACHE_HOME", "/tmp/taichi-cache")

try:
    import taichi as ti
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "VoxelGridTaichi requires Taichi. Install it with `pip install AtomVoxelizer[taichi]`."
    ) from exc


def _init_taichi_cpu_once():
    try:
        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is not None:
            return
    except Exception:
        pass
    ti.init(arch=ti.cpu, offline_cache=False)


_init_taichi_cpu_once()


@ti.kernel
def _set_sphere_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    value: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] = value


@ti.kernel
def _add_sphere_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    value: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] += value


@ti.kernel
def _mul_sphere_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    factor: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] *= factor


@ti.kernel
def _div_sphere_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    divisor: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] /= divisor


@ti.kernel
def _clamp_grid(grid: ti.template(), min_val: ti.f32, max_val: ti.f32):
    for i, j, k in ti.ndrange(grid.shape[0], grid.shape[1], grid.shape[2]):
        value = grid[i, j, k]
        if value < min_val:
            grid[i, j, k] = min_val
        elif value > max_val:
            grid[i, j, k] = max_val


class VoxelGridTaichi(VoxelGrid):
    """Voxel grid with Taichi CPU kernels for mutating sphere operations."""

    backend_name = "taichi-cpu"

    def __init__(self, cell, resolution=None, gpts=None):
        super().__init__(cell=cell, resolution=resolution, gpts=gpts)
        self.grid = ti.field(dtype=ti.f32, shape=tuple(int(x) for x in self.gpts))

    def to_numpy(self):
        """Return the voxel values as a NumPy array."""
        return self.grid.to_numpy()

    def set_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _set_sphere_offsets(self.grid, center_idx, offsets, float(value))

    def add_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _add_sphere_offsets(self.grid, center_idx, offsets, float(value))

    def mul_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _mul_sphere_offsets(self.grid, center_idx, offsets, float(factor))

    def div_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _div_sphere_offsets(self.grid, center_idx, offsets, float(factor))

    def add_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            for center_idx in center_indices[radii == radius]:
                _add_sphere_offsets(self.grid, center_idx, offsets, float(value))

    def set_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            for center_idx in center_indices[radii == radius]:
                _set_sphere_offsets(self.grid, center_idx, offsets, float(value))

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        _clamp_grid(self.grid, float(min_val), float(max_val))

    def synchronize(self):
        """Synchronize Taichi kernels."""
        ti.sync()


__all__ = ["VoxelGridTaichi"]
