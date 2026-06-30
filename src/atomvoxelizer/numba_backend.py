from __future__ import annotations

import numpy as np
from numba import njit

from .voxelgrid import VoxelGrid, _cached_sphere_offsets


@njit
def _set_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] = value


@njit
def _add_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] += value


@njit
def _mul_sphere_offsets_flat(grid, center_idx, offsets, factor, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] *= factor


@njit
def _div_sphere_offsets_flat(grid, center_idx, offsets, divisor, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] /= divisor


@njit
def _set_many_sphere_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] = value


@njit
def _add_many_sphere_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] += value


class VoxelGridNumba(VoxelGrid):
    """Voxel grid with Numba-compiled mutation kernels."""

    backend_name = "numba"

    def _flat_grid_and_shape(self):
        nx, ny, nz = self.grid.shape
        return self.grid.ravel(), int(nx), int(ny), int(nz)

    def set_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        _set_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz)

    def add_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        _add_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz)

    def mul_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        _mul_sphere_offsets_flat(grid, center_idx, offsets, factor, nx, ny, nz)

    def div_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        _div_sphere_offsets_flat(grid, center_idx, offsets, factor, nx, ny, nz)

    def add_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _add_many_sphere_offsets_flat(grid, subset, offsets, value, nx, ny, nz)

    def set_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _set_many_sphere_offsets_flat(grid, subset, offsets, value, nx, ny, nz)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        np.clip(self.grid, min_val, max_val, out=self.grid)


__all__ = ["VoxelGridNumba"]
