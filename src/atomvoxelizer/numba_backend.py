from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
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
        flat = (x * ny + y) * nz + z
        divisor = distances[n] * scale
        if divisor == 0.0:
            if grid[flat] == 0.0:
                grid[flat] = np.nan
            elif grid[flat] > 0.0:
                grid[flat] = np.inf
            else:
                grid[flat] = -np.inf
        else:
            grid[flat] /= divisor


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


@njit(parallel=True)
def _add_many_sphere_offsets_cell_colored_flat(
    grid,
    center_indices,
    cell_starts,
    cell_counts,
    color_cells,
    offsets,
    value,
    nx,
    ny,
    nz,
):
    for color_cell_index in prange(color_cells.shape[0]):
        cell_id = color_cells[color_cell_index]
        atom_start = cell_starts[cell_id]
        atom_stop = atom_start + cell_counts[cell_id]
        for atom in range(atom_start, atom_stop):
            cx = center_indices[atom, 0]
            cy = center_indices[atom, 1]
            cz = center_indices[atom, 2]
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
            flat = (x * ny + y) * nz + z
            divisor = distances[n] * scale
            if divisor == 0.0:
                if grid[flat] == 0.0:
                    grid[flat] = np.nan
                elif grid[flat] > 0.0:
                    grid[flat] = np.inf
                else:
                    grid[flat] = -np.inf
            else:
                grid[flat] /= divisor


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


@njit
def _apply_operation_sequence_flat(
    grid,
    center_indices,
    center_operation_ids,
    group_ids,
    offsets,
    distances,
    uses_distances,
    starts,
    counts,
    operation_codes,
    scales,
    nx,
    ny,
    nz,
):
    for c in range(center_indices.shape[0]):
        operation_id = center_operation_ids[c]
        operation = operation_codes[operation_id]
        scale = scales[operation_id]
        group = group_ids[c]
        start = starts[group]
        stop = start + counts[group]
        use_distances = uses_distances[group] != 0

        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(start, stop):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            flat = (x * ny + y) * nz + z
            value = distances[n] * scale if use_distances else scale

            if operation == 0:
                grid[flat] = value
            elif operation == 1:
                grid[flat] += value
            elif operation == 2:
                grid[flat] *= value
            elif operation == 3:
                grid[flat] /= value
            elif operation == 4:
                if value < grid[flat]:
                    grid[flat] = value


_OPERATION_CODES = {
    "set": 0,
    "add": 1,
    "mul": 2,
    "div": 3,
    "min": 4,
}


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
        self._check_ordered_grid("min_sphere")
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

    def add_spheres_cell_colored(self, centers, radii, value=1):
        """Add constant spheres with a 27-color parallel CPU schedule.

        This experimental path partitions atoms into spatial cells and processes
        cells with the same color in parallel. Cells of one color are separated
        by at least two other cells in each periodic direction, so their stencils
        cannot overlap when the coloring-cell width is at least the largest
        radius in the batch. Atoms within a cell are processed sequentially to
        avoid write races without atomics.

        Mixed radii are supported by building the coloring schedule from
        ``max(radii)`` and grouping atoms by unique radius for the stencil
        update. The method only implements constant-mask addition.
        """
        centers, radii = self._validate_spheres(centers, radii)
        if centers.shape[0] == 0:
            return
        unique_radii = np.unique(radii)
        coloring_radius = float(np.max(unique_radii))
        for radius in unique_radii:
            selected = radii == radius
            plan = self.prepare_cell_coloring(centers[selected], float(radius), coloring_radius=coloring_radius)
            self._run_cell_colored_add(plan, value=value)

    def prepare_cell_coloring(self, centers, radius, coloring_radius=None):
        """Prepare a 27-color plan for repeated same-radius Numba add updates."""
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if coloring_radius is None:
            coloring_radius = radius
        coloring_radius = float(coloring_radius)
        if coloring_radius < radius:
            raise ValueError("coloring_radius must be at least radius")

        lengths = np.linalg.norm(self.cell, axis=1)
        max_cells = np.floor(lengths / coloring_radius).astype(int)
        color_cells = np.maximum(1, max_cells // 3)
        cell_counts_3d = np.maximum(3, 3 * color_cells)
        cell_widths = lengths / cell_counts_3d
        if np.any(cell_widths < coloring_radius):
            raise ValueError("cell coloring requires coloring-cell widths >= stencil radius")

        frac = centers @ self.cell_inv
        frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
        atom_cells_3d = np.floor(frac_wrapped * cell_counts_3d).astype(np.int32)
        atom_cells_3d = np.minimum(atom_cells_3d, cell_counts_3d - 1)
        atom_cell_ids = (
            (atom_cells_3d[:, 0] * cell_counts_3d[1] + atom_cells_3d[:, 1]) * cell_counts_3d[2]
            + atom_cells_3d[:, 2]
        ).astype(np.int32)
        n_cells = int(np.prod(cell_counts_3d))

        order = np.argsort(atom_cell_ids, kind="stable")
        sorted_cell_ids = atom_cell_ids[order]
        sorted_centers = self.positions_to_indices(centers)[order]

        cell_starts = np.zeros(n_cells, dtype=np.int32)
        cell_counts = np.zeros(n_cells, dtype=np.int32)
        unique_cells, starts, counts = np.unique(sorted_cell_ids, return_index=True, return_counts=True)
        cell_starts[unique_cells] = starts.astype(np.int32)
        cell_counts[unique_cells] = counts.astype(np.int32)

        cell_ids = np.arange(n_cells, dtype=np.int32)
        ix = cell_ids // (cell_counts_3d[1] * cell_counts_3d[2])
        iy = (cell_ids // cell_counts_3d[2]) % cell_counts_3d[1]
        iz = cell_ids % cell_counts_3d[2]
        colors = (ix % 3) * 9 + (iy % 3) * 3 + (iz % 3)
        color_to_cells = [
            np.ascontiguousarray(cell_ids[(colors == color) & (cell_counts > 0)], dtype=np.int32)
            for color in range(27)
        ]
        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))

        return {
            "radius": radius,
            "coloring_radius": coloring_radius,
            "cell_counts_3d": cell_counts_3d.astype(np.int32),
            "cell_widths": cell_widths,
            "center_indices": np.ascontiguousarray(sorted_centers, dtype=np.int32),
            "cell_starts": np.ascontiguousarray(cell_starts, dtype=np.int32),
            "cell_counts": np.ascontiguousarray(cell_counts, dtype=np.int32),
            "color_to_cells": color_to_cells,
            "offsets": np.ascontiguousarray(offsets, dtype=np.int32),
        }

    def apply_prepared_cell_coloring(self, plan, value=1):
        """Apply a prepared same-radius 27-color Numba add plan."""
        self._run_cell_colored_add(plan, value=value)

    def _run_cell_colored_add(self, plan, value=1):
        grid, nx, ny, nz = self._flat_grid_and_shape()
        for color_cells in plan["color_to_cells"]:
            if color_cells.shape[0] == 0:
                continue
            _add_many_sphere_offsets_cell_colored_flat(
                grid,
                plan["center_indices"],
                plan["cell_starts"],
                plan["cell_counts"],
                color_cells,
                plan["offsets"],
                value,
                nx,
                ny,
                nz,
            )

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
        self._check_ordered_grid("min_spheres")
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

    def apply_sphere_operations(self, operations):
        """Apply an ordered sequence of batched sphere operations in one Numba call."""
        self._check_ordered_grid("apply_sphere_operations")
        packed = self._pack_sphere_operations(operations)
        if packed is None:
            return

        grid, nx, ny, nz = self._flat_grid_and_shape()
        _apply_operation_sequence_flat(grid, *packed, nx, ny, nz)

    def _pack_sphere_operations(self, operations):
        if operations is None:
            raise ValueError("operations must be an iterable of operation mappings")

        center_parts = []
        center_operation_parts = []
        center_group_parts = []
        operation_codes = []
        scales = []
        offset_parts = []
        distance_parts = []
        uses_distances = []
        starts = []
        counts = []
        group_lookup = {}
        offset_start = 0

        for spec in operations:
            if not isinstance(spec, dict):
                raise TypeError("each operation must be a mapping")
            operation = spec.get("operation")
            if operation not in _OPERATION_CODES:
                raise ValueError("operation must be one of 'set', 'add', 'mul', 'div', or 'min'")
            mask = spec.get("mask", "distance" if operation == "min" else "constant")
            self._validate_mask(mask)

            centers, radii = self._validate_spheres(spec.get("centers"), spec.get("radii"))
            if centers.shape[0] == 0:
                continue

            operation_index = len(operation_codes)
            scale = self._operation_scale(spec, operation)
            center_indices = self.positions_to_indices(centers)
            group_ids = np.empty(centers.shape[0], dtype=np.int32)

            for radius in np.unique(radii):
                key = (mask, float(radius))
                group_id = group_lookup.get(key)
                if group_id is None:
                    offsets, distances = self._grouped_offsets(radius, mask)
                    group_id = len(starts)
                    group_lookup[key] = group_id
                    offset_parts.append(np.ascontiguousarray(offsets, dtype=np.int32))
                    if distances is None:
                        distance_parts.append(np.zeros(offsets.shape[0], dtype=np.float32))
                        uses_distances.append(0)
                    else:
                        distance_parts.append(np.ascontiguousarray(distances, dtype=np.float32))
                        uses_distances.append(1)
                    starts.append(offset_start)
                    counts.append(offsets.shape[0])
                    offset_start += offsets.shape[0]

                group_ids[radii == radius] = group_id

            center_parts.append(np.ascontiguousarray(center_indices, dtype=np.int32))
            center_operation_parts.append(np.full(centers.shape[0], operation_index, dtype=np.int32))
            center_group_parts.append(group_ids)
            operation_codes.append(_OPERATION_CODES[operation])
            scales.append(scale)

        if not center_parts:
            return None

        return (
            np.ascontiguousarray(np.vstack(center_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(center_operation_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(center_group_parts), dtype=np.int32),
            np.ascontiguousarray(np.vstack(offset_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(distance_parts), dtype=np.float32),
            np.ascontiguousarray(uses_distances, dtype=np.uint8),
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(operation_codes, dtype=np.int32),
            np.ascontiguousarray(scales, dtype=np.float64),
        )

    @staticmethod
    def _operation_scale(spec, operation):
        if operation in {"mul", "div"}:
            return float(spec.get("factor", spec.get("value", 2.0)))
        return float(spec.get("value", 1.0))

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        self._check_ordered_grid("clamp_grid")
        np.clip(self.grid, min_val, max_val, out=self.grid)


__all__ = ["VoxelGridNumba"]
