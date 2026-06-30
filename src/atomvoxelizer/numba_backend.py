from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "VoxelGridNumba requires the optional dependency Numba. Install Numba directly, "
        "for example with `pip install numba` or your environment manager of choice."
    ) from exc

from .voxelgrid import VoxelGrid, _cached_sphere_offsets, _cached_sphere_offsets_and_distances


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
def _set_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, scale, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] = distances[n] * scale


@njit
def _add_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, scale, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] += distances[n] * scale


@njit
def _mul_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, scale, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] *= distances[n] * scale


@njit
def _div_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, scale, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        grid[(x * ny + y) * nz + z] /= distances[n] * scale


@njit
def _min_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        if value < grid[flat]:
            grid[flat] = value


@njit
def _min_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, scale, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        value = distances[n] * scale
        if value < grid[flat]:
            grid[flat] = value


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


@njit
def _mul_many_sphere_offsets_flat(grid, center_indices, offsets, factor, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] *= factor


@njit
def _div_many_sphere_offsets_flat(grid, center_indices, offsets, divisor, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] /= divisor


@njit
def _min_many_sphere_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            flat = (x * ny + y) * nz + z
            if value < grid[flat]:
                grid[flat] = value


@njit
def _set_many_sphere_distance_offsets_flat(grid, center_indices, offsets, distances, scale, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] = distances[n] * scale


@njit
def _add_many_sphere_distance_offsets_flat(grid, center_indices, offsets, distances, scale, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] += distances[n] * scale


@njit
def _mul_many_sphere_distance_offsets_flat(grid, center_indices, offsets, distances, scale, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] *= distances[n] * scale


@njit
def _div_many_sphere_distance_offsets_flat(grid, center_indices, offsets, distances, scale, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] /= distances[n] * scale


@njit
def _min_many_sphere_distance_offsets_flat(grid, center_indices, offsets, distances, scale, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            flat = (x * ny + y) * nz + z
            value = distances[n] * scale
            if value < grid[flat]:
                grid[flat] = value


class VoxelGridNumba(VoxelGrid):
    """Voxel grid with Numba-compiled mutation kernels."""

    backend_name = "numba"

    def _flat_grid_and_shape(self):
        nx, ny, nz = self.grid.shape
        return self.grid.ravel(), int(nx), int(ny), int(nz)

    def _offsets_for_mask(self, radius, mask):
        self._validate_mask(mask)
        if mask == "constant":
            return self._sphere_offsets(radius), None
        return self._sphere_offsets_and_distances(radius)

    def set_sphere(self, center, radius, value=1, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            _set_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz)
        else:
            _set_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, value, nx, ny, nz)

    def add_sphere(self, center, radius, value=1, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            _add_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz)
        else:
            _add_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, value, nx, ny, nz)

    def mul_sphere(self, center, radius, factor=2, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            _mul_sphere_offsets_flat(grid, center_idx, offsets, factor, nx, ny, nz)
        else:
            _mul_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, factor, nx, ny, nz)

    def div_sphere(self, center, radius, factor=2, mask="constant"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            _div_sphere_offsets_flat(grid, center_idx, offsets, factor, nx, ny, nz)
        else:
            _div_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, factor, nx, ny, nz)

    def min_sphere(self, center, radius, value=1, mask="distance"):
        center_idx = self._center_index(center)
        offsets, distances = self._offsets_for_mask(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            _min_sphere_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz)
        else:
            _min_sphere_distance_offsets_flat(grid, center_idx, offsets, distances, value, nx, ny, nz)

    def _grouped_offsets(self, radius, mask):
        if mask == "constant":
            return _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))), None
        return _cached_sphere_offsets_and_distances(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))

    def add_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            subset = center_indices[radii == radius]
            if mask == "constant":
                _add_many_sphere_offsets_flat(grid, subset, offsets, value, nx, ny, nz)
            else:
                _add_many_sphere_distance_offsets_flat(grid, subset, offsets, distances, value, nx, ny, nz)

    def set_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            subset = center_indices[radii == radius]
            if mask == "constant":
                _set_many_sphere_offsets_flat(grid, subset, offsets, value, nx, ny, nz)
            else:
                _set_many_sphere_distance_offsets_flat(grid, subset, offsets, distances, value, nx, ny, nz)

    def mul_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            subset = center_indices[radii == radius]
            if mask == "constant":
                _mul_many_sphere_offsets_flat(grid, subset, offsets, factor, nx, ny, nz)
            else:
                _mul_many_sphere_distance_offsets_flat(grid, subset, offsets, distances, factor, nx, ny, nz)

    def div_spheres(self, centers, radii, factor=2, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            subset = center_indices[radii == radius]
            if mask == "constant":
                _div_many_sphere_offsets_flat(grid, subset, offsets, factor, nx, ny, nz)
            else:
                _div_many_sphere_distance_offsets_flat(grid, subset, offsets, distances, factor, nx, ny, nz)

    def min_spheres(self, centers, radii, value=1, mask="distance"):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            offsets, distances = self._grouped_offsets(radius, mask)
            subset = center_indices[radii == radius]
            if mask == "constant":
                _min_many_sphere_offsets_flat(grid, subset, offsets, value, nx, ny, nz)
            else:
                _min_many_sphere_distance_offsets_flat(grid, subset, offsets, distances, value, nx, ny, nz)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        np.clip(self.grid, min_val, max_val, out=self.grid)


__all__ = ["VoxelGridNumba"]
