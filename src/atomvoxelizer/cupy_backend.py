from __future__ import annotations

import numpy as np

from .voxelgrid import VoxelGrid, _cached_sphere_offsets, _cached_sphere_offsets_and_distances

try:
    from .numba_backend import VoxelGridNumba as _BaseVoxelGrid
except ImportError:  # pragma: no cover - depends on optional dependency
    _BaseVoxelGrid = VoxelGrid

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "VoxelGridCuPy requires the optional dependency CuPy. Install the CuPy package "
        "that matches your CUDA environment, for example `pip install cupy-cuda12x`."
    ) from exc


class VoxelGridCuPy(_BaseVoxelGrid):
    """CuPy-backed voxel grid.

    The class keeps the same public API as :class:`atomvoxelizer.VoxelGrid`, uses
    the Numba backend as its base when available, stores ``grid`` as a CuPy array,
    and overrides the mutating sphere operations.
    """

    @property
    def backend_name(self):
        return "cupy"

    def __init__(self, cell, resolution=None, gpts=None):
        super().__init__(cell=cell, resolution=resolution, gpts=gpts)
        self.grid = cp.asarray(self.grid)

    def to_numpy(self):
        """Return the voxel values as a NumPy array."""
        return cp.asnumpy(self.grid)

    def _sphere_indices(self, center, radius):
        center_frac = np.asarray(center, dtype=np.float64) @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)
        offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
        indices = (offsets + center_idx) % self.gpts
        return tuple(cp.asarray(indices[:, axis]) for axis in range(3))

    def _sphere_indices_and_values(self, center, radius, value, mask):
        self._validate_mask(mask)
        center_idx = self._center_index(center)
        if mask == "constant":
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            values = value
        else:
            offsets, distances = _cached_sphere_offsets_and_distances(
                float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))
            )
            values = cp.asarray(distances) * value
        indices = (offsets + center_idx) % self.gpts
        return tuple(cp.asarray(indices[:, axis]) for axis in range(3)), values

    def set_sphere(self, center, radius, value=1, mask="constant"):
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        self.grid[indices] = values

    def add_sphere(self, center, radius, value=1, mask="constant"):
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        cp.add.at(self.grid, indices, values)

    def mul_sphere(self, center, radius, factor=2, mask="constant"):
        indices, values = self._sphere_indices_and_values(center, radius, factor, mask)
        self.grid[indices] *= values

    def div_sphere(self, center, radius, factor=2, mask="constant"):
        indices, values = self._sphere_indices_and_values(center, radius, factor, mask)
        self.grid[indices] /= values

    def min_sphere(self, center, radius, value=1, mask="distance"):
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        cp.minimum.at(self.grid, indices, values)

    def add_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.add_sphere(center, float(radius), value=value, mask=mask)

    def set_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.set_sphere(center, float(radius), value=value, mask=mask)

    def mul_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.mul_sphere(center, float(radius), factor=factor, mask=mask)

    def div_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.div_sphere(center, float(radius), factor=factor, mask=mask)

    def min_spheres(self, centers, radii, value=1, mask="distance"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.min_sphere(center, float(radius), value=value, mask=mask)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        self.grid = cp.clip(self.grid, min_val, max_val)

    def synchronize(self):
        """Synchronize the current CuPy device."""
        cp.cuda.Stream.null.synchronize()
