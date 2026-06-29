from __future__ import annotations

import numpy as np

from .voxelgrid import VoxelGrid, _cached_sphere_offsets

try:
    from .numba_backend import VoxelGridNumba as _BaseVoxelGrid
except ImportError:  # pragma: no cover - depends on optional dependency
    _BaseVoxelGrid = VoxelGrid

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "VoxelGridCuPy requires CuPy. Install it with `pip install AtomVoxelizer[cupy]`."
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

    def set_sphere(self, center, radius, value=1):
        self.grid[self._sphere_indices(center, radius)] = value

    def add_sphere(self, center, radius, value=1):
        cp.add.at(self.grid, self._sphere_indices(center, radius), value)

    def mul_sphere(self, center, radius, factor=2):
        indices = self._sphere_indices(center, radius)
        self.grid[indices] *= factor

    def div_sphere(self, center, radius, factor=2):
        indices = self._sphere_indices(center, radius)
        self.grid[indices] /= factor

    def add_spheres(self, centers, radii, value=1):
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape (N,)")

        for center, radius in zip(centers, radii):
            self.add_sphere(center, float(radius), value=value)

    def set_spheres(self, centers, radii, value=1):
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape (N,)")

        for center, radius in zip(centers, radii):
            self.set_sphere(center, float(radius), value=value)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        self.grid = cp.clip(self.grid, min_val, max_val)

    def synchronize(self):
        """Synchronize the current CuPy device."""
        cp.cuda.Stream.null.synchronize()
