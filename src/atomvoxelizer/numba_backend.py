from __future__ import annotations

import numpy as np
from numba import njit, prange

from .voxelgrid import VoxelGrid, _cached_sphere_offsets


@njit
def _set_sphere_offsets(grid, center_idx, offsets, value):
    nx, ny, nz = grid.shape
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[x, y, z] = value


@njit
def _add_sphere_offsets(grid, center_idx, offsets, value):
    nx, ny, nz = grid.shape
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[x, y, z] += value


@njit
def _mul_sphere_offsets(grid, center_idx, offsets, factor):
    nx, ny, nz = grid.shape
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[x, y, z] *= factor


@njit
def _div_sphere_offsets(grid, center_idx, offsets, divisor):
    nx, ny, nz = grid.shape
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[x, y, z] /= divisor


@njit
def _set_many_sphere_offsets(grid, center_indices, offsets, value):
    nx, ny, nz = grid.shape
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[x, y, z] = value


@njit
def _add_many_sphere_offsets(grid, center_indices, offsets, value):
    nx, ny, nz = grid.shape
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[x, y, z] += value


@njit(parallel=True)
def _clamp_grid(grid, min_val, max_val):
    nx, ny, nz = grid.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                v = grid[i, j, k]
                if v < min_val:
                    grid[i, j, k] = min_val
                elif v > max_val:
                    grid[i, j, k] = max_val


class VoxelGridNumba(VoxelGrid):
    """Voxel grid with Numba-compiled mutation kernels."""

    backend_name = "numba"

    def set_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _set_sphere_offsets(self.grid, center_idx, offsets, value)

    def add_sphere(self, center, radius, value=1):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _add_sphere_offsets(self.grid, center_idx, offsets, value)

    def mul_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _mul_sphere_offsets(self.grid, center_idx, offsets, factor)

    def div_sphere(self, center, radius, factor=2):
        center_idx = self._center_index(center)
        offsets = self._sphere_offsets(radius)
        _div_sphere_offsets(self.grid, center_idx, offsets, factor)

    def add_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _add_many_sphere_offsets(self.grid, subset, offsets, value)

    def set_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _set_many_sphere_offsets(self.grid, subset, offsets, value)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        _clamp_grid(self.grid, min_val, max_val)


__all__ = ["VoxelGridNumba"]
