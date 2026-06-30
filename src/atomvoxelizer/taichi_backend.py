import os
from pathlib import Path

import numpy as np

from .voxelgrid import VoxelGrid, _cached_sphere_offsets, _cached_sphere_offsets_and_distances

_default_cache = Path.home() / ".cache"
if not os.access(_default_cache, os.W_OK):
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("TI_CACHE_HOME", "/tmp/taichi-cache")

try:
    import taichi as ti
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "VoxelGridTaichi requires the optional dependency Taichi. Install Taichi directly, "
        "for example with `pip install taichi` or your environment manager of choice."
    ) from exc


def _arch_name(arch):
    if isinstance(arch, (list, tuple)):
        return "gpu"
    return getattr(arch, "name", str(arch)).lower()


def _current_arch():
    try:
        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is None:
            return None
        return ti.lang.impl.current_cfg().arch
    except Exception:
        return None


def _init_taichi_once(arch):
    current_arch = _current_arch()
    if current_arch is None:
        ti.init(arch=arch, offline_cache=False, enable_fallback=False)
        return

    if isinstance(arch, (list, tuple)):
        if current_arch not in arch:
            raise RuntimeError(
                "Taichi is already initialized with "
                f"arch={_arch_name(current_arch)}; cannot create arch={_arch_name(arch)} grid "
                "in the same Python process."
            )
        return

    if current_arch != arch:
        raise RuntimeError(
            "Taichi is already initialized with "
            f"arch={_arch_name(current_arch)}; cannot create arch={_arch_name(arch)} grid "
            "in the same Python process."
        )


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
def _set_sphere_distance_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    distances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] = distances[n] * scale


@ti.kernel
def _add_sphere_distance_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    distances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] += distances[n] * scale


@ti.kernel
def _mul_sphere_distance_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    distances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] *= distances[n] * scale


@ti.kernel
def _div_sphere_distance_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    distances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        grid[x, y, z] /= distances[n] * scale


@ti.kernel
def _min_sphere_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    value: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        if value < grid[x, y, z]:
            grid[x, y, z] = value


@ti.kernel
def _min_sphere_distance_offsets(
    grid: ti.template(),
    center_idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=2),
    distances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % grid.shape[0]
        y = (center_idx[1] + offsets[n, 1]) % grid.shape[1]
        z = (center_idx[2] + offsets[n, 2]) % grid.shape[2]
        value = distances[n] * scale
        if value < grid[x, y, z]:
            grid[x, y, z] = value


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
    taichi_arch = ti.cpu

    def __init__(self, cell, resolution=None, gpts=None, arch=None):
        self.taichi_arch = self.taichi_arch if arch is None else arch
        _init_taichi_once(self.taichi_arch)
        super().__init__(cell=cell, resolution=resolution, gpts=gpts)
        self.grid = ti.field(dtype=ti.f32, shape=tuple(int(x) for x in self.gpts))

    def to_numpy(self):
        """Return the voxel values as a NumPy array."""
        return self.grid.to_numpy()

    def _offsets_for_mask(self, radius, mask):
        self._validate_mask(mask)
        if mask == "constant":
            return self._sphere_offsets(radius), None
        return self._sphere_offsets_and_distances(radius)

    def set_sphere(self, center, radius, value=1, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        if mask == "constant":
            _set_sphere_offsets(self.grid, center_idx, offsets, float(value))
        else:
            _set_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def add_sphere(self, center, radius, value=1, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        if mask == "constant":
            _add_sphere_offsets(self.grid, center_idx, offsets, float(value))
        else:
            _add_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def mul_sphere(self, center, radius, factor=2, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        if mask == "constant":
            _mul_sphere_offsets(self.grid, center_idx, offsets, float(factor))
        else:
            _mul_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(factor))

    def div_sphere(self, center, radius, factor=2, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        if mask == "constant":
            _div_sphere_offsets(self.grid, center_idx, offsets, float(factor))
        else:
            _div_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(factor))

    def min_sphere(self, center, radius, value=1, mask="distance"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        if mask == "constant":
            _min_sphere_offsets(self.grid, center_idx, offsets, float(value))
        else:
            _min_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def _grouped_offsets(self, radius, mask):
        if mask == "constant":
            return _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))), None
        return _cached_sphere_offsets_and_distances(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))

    def add_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            for center_idx in center_indices[radii == radius]:
                if mask == "constant":
                    _add_sphere_offsets(self.grid, center_idx, offsets, float(value))
                else:
                    _add_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def set_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            for center_idx in center_indices[radii == radius]:
                if mask == "constant":
                    _set_sphere_offsets(self.grid, center_idx, offsets, float(value))
                else:
                    _set_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def mul_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            for center_idx in center_indices[radii == radius]:
                if mask == "constant":
                    _mul_sphere_offsets(self.grid, center_idx, offsets, float(factor))
                else:
                    _mul_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(factor))

    def div_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            for center_idx in center_indices[radii == radius]:
                if mask == "constant":
                    _div_sphere_offsets(self.grid, center_idx, offsets, float(factor))
                else:
                    _div_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(factor))

    def min_spheres(self, centers, radii, value=1, mask="distance"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            for center_idx in center_indices[radii == radius]:
                if mask == "constant":
                    _min_sphere_offsets(self.grid, center_idx, offsets, float(value))
                else:
                    _min_sphere_distance_offsets(self.grid, center_idx, offsets, distances, float(value))

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        _clamp_grid(self.grid, float(min_val), float(max_val))

    def synchronize(self):
        """Synchronize Taichi kernels."""
        ti.sync()


class VoxelGridTaichiGPU(VoxelGridTaichi):
    """Voxel grid with Taichi GPU kernels.

    This class requests Taichi's GPU arch list. It requires a GPU-supported
    Taichi installation and a visible GPU device.
    """

    backend_name = "taichi-gpu"
    taichi_arch = ti.gpu


__all__ = ["VoxelGridTaichi", "VoxelGridTaichiGPU"]
