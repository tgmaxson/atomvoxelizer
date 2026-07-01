from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "FieldVoxelGridNumba requires the optional dependency Numba. Install Numba directly, "
        "for example with `pip install numba` or your environment manager of choice."
    ) from exc

from .vectorgrid import FieldVoxelGrid, _cached_sphere_offsets_and_vectors


@njit
def _set_field_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(value.shape[0]):
            grid[flat, c] = value[c]


@njit
def _add_field_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(value.shape[0]):
            grid[flat, c] += value[c]


@njit
def _mul_field_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(value.shape[0]):
            grid[flat, c] *= value[c]


@njit
def _div_field_offsets_flat(grid, center_idx, offsets, value, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(value.shape[0]):
            grid[flat, c] /= value[c]


@njit
def _set_field_values_flat(grid, center_idx, offsets, values, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(values.shape[1]):
            grid[flat, c] = values[n, c]


@njit
def _add_field_values_flat(grid, center_idx, offsets, values, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(values.shape[1]):
            grid[flat, c] += values[n, c]


@njit
def _mul_field_values_flat(grid, center_idx, offsets, values, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(values.shape[1]):
            grid[flat, c] *= values[n, c]


@njit
def _div_field_values_flat(grid, center_idx, offsets, values, nx, ny, nz):
    for n in range(offsets.shape[0]):
        x = (center_idx[0] + offsets[n, 0]) % nx
        y = (center_idx[1] + offsets[n, 1]) % ny
        z = (center_idx[2] + offsets[n, 2]) % nz
        flat = (x * ny + y) * nz + z
        for c in range(values.shape[1]):
            grid[flat, c] /= values[n, c]


@njit
def _set_many_field_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _set_field_offsets_flat(grid, center_indices[center], offsets, value, nx, ny, nz)


@njit
def _add_many_field_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _add_field_offsets_flat(grid, center_indices[center], offsets, value, nx, ny, nz)


@njit
def _mul_many_field_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _mul_field_offsets_flat(grid, center_indices[center], offsets, value, nx, ny, nz)


@njit
def _div_many_field_offsets_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _div_field_offsets_flat(grid, center_indices[center], offsets, value, nx, ny, nz)


@njit
def _set_many_field_values_flat(grid, center_indices, offsets, values, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _set_field_values_flat(grid, center_indices[center], offsets, values, nx, ny, nz)


@njit
def _add_many_field_values_flat(grid, center_indices, offsets, values, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _add_field_values_flat(grid, center_indices[center], offsets, values, nx, ny, nz)


@njit
def _mul_many_field_values_flat(grid, center_indices, offsets, values, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _mul_field_values_flat(grid, center_indices[center], offsets, values, nx, ny, nz)


@njit
def _div_many_field_values_flat(grid, center_indices, offsets, values, nx, ny, nz):
    for center in range(center_indices.shape[0]):
        _div_field_values_flat(grid, center_indices[center], offsets, values, nx, ny, nz)


class FieldVoxelGridNumba(FieldVoxelGrid):
    """Field voxel grid with Numba-compiled sphere mutation kernels."""

    backend_name = "numba-field"

    def _flat_grid_and_shape(self):
        nx, ny, nz = self.gpts
        return self.grid.reshape((int(nx) * int(ny) * int(nz), -1)), int(nx), int(ny), int(nz)

    def _flat_value(self, value):
        return np.ascontiguousarray(self._validate_value(value).reshape(-1), dtype=self.dtype)

    def _values_for_normal_mask(self, radius, value):
        offsets, normal_vectors = _cached_sphere_offsets_and_vectors(
            float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))
        )
        values = np.ascontiguousarray(normal_vectors.astype(self.dtype, copy=False) * self.dtype.type(value))
        return offsets, values

    def _constant_offsets(self, radius):
        return self._sphere_offsets(radius)

    def _apply_sphere(self, center, radius, value, mask, constant_kernel, values_kernel):
        self._validate_field_mask(mask)
        center_idx = self._center_index(center)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        if mask == "constant":
            offsets = self._constant_offsets(radius)
            constant_kernel(grid, center_idx, offsets, self._flat_value(value), nx, ny, nz)
        else:
            if self.value_shape != (3,):
                raise ValueError('normal mask requires value_shape=(3,)')
            offsets, values = self._values_for_normal_mask(radius, value)
            values_kernel(grid, center_idx, offsets, values, nx, ny, nz)

    def set_sphere(self, center, radius, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_sphere(center, radius, value, mask, _set_field_offsets_flat, _set_field_values_flat)

    def add_sphere(self, center, radius, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_sphere(center, radius, value, mask, _add_field_offsets_flat, _add_field_values_flat)

    def mul_sphere(self, center, radius, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_sphere(center, radius, factor, mask, _mul_field_offsets_flat, _mul_field_values_flat)

    def div_sphere(self, center, radius, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_sphere(center, radius, factor, mask, _div_field_offsets_flat, _div_field_values_flat)

    def _apply_spheres(self, centers, radii, value, mask, constant_kernel, values_kernel):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_field_mask(mask)
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for radius in np.unique(radii):
            subset = center_indices[radii == radius]
            if mask == "constant":
                offsets = self._constant_offsets(radius)
                constant_kernel(grid, subset, offsets, self._flat_value(value), nx, ny, nz)
            else:
                if self.value_shape != (3,):
                    raise ValueError('normal mask requires value_shape=(3,)')
                offsets, values = self._values_for_normal_mask(radius, value)
                values_kernel(grid, subset, offsets, values, nx, ny, nz)

    def set_spheres(self, centers, radii, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_spheres(centers, radii, value, mask, _set_many_field_offsets_flat, _set_many_field_values_flat)

    def add_spheres(self, centers, radii, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_spheres(centers, radii, value, mask, _add_many_field_offsets_flat, _add_many_field_values_flat)

    def mul_spheres(self, centers, radii, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_spheres(centers, radii, factor, mask, _mul_many_field_offsets_flat, _mul_many_field_values_flat)

    def div_spheres(self, centers, radii, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        self._apply_spheres(centers, radii, factor, mask, _div_many_field_offsets_flat, _div_many_field_values_flat)


class VectorVoxelGridNumba(FieldVoxelGridNumba):
    """Convenience Numba field grid with a three-component vector value."""

    def __init__(self, cell, resolution=None, gpts=None, dtype=np.float32, components=3, value_shape=None):
        if value_shape is None:
            value_shape = (int(components),)
        super().__init__(cell=cell, resolution=resolution, gpts=gpts, dtype=dtype, value_shape=value_shape)


__all__ = ["FieldVoxelGridNumba", "VectorVoxelGridNumba"]
