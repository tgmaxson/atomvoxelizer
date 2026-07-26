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


_CELL_COLOR_ADD_KERNEL = r"""
extern "C" __global__
void add_colored_cells(
    float* grid,
    const int* centers,
    const int* cell_starts,
    const int* cell_counts,
    const int* color_cells,
    const int n_color_cells,
    const int* offsets,
    const int n_offsets,
    const int nx,
    const int ny,
    const int nz,
    const float value
) {
    int color_cell_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (color_cell_index >= n_color_cells) {
        return;
    }

    int cell_id = color_cells[color_cell_index];
    int atom_start = cell_starts[cell_id];
    int atom_stop = atom_start + cell_counts[cell_id];

    for (int atom = atom_start; atom < atom_stop; ++atom) {
        int cx = centers[3 * atom + 0];
        int cy = centers[3 * atom + 1];
        int cz = centers[3 * atom + 2];

        for (int n = 0; n < n_offsets; ++n) {
            int x = (cx + offsets[3 * n + 0]) % nx;
            int y = (cy + offsets[3 * n + 1]) % ny;
            int z = (cz + offsets[3 * n + 2]) % nz;
            if (x < 0) x += nx;
            if (y < 0) y += ny;
            if (z < 0) z += nz;
            grid[(x * ny + y) * nz + z] += value;
        }
    }
}
"""


class VoxelGridCuPy(_BaseVoxelGrid):
    """CuPy-backed voxel grid.

    The class keeps the same public API as :class:`atomvoxelizer.VoxelGrid`, uses
    the Numba backend as its base when available, stores ``grid`` as a CuPy array,
    and overrides the mutating sphere operations.
    """

    @property
    def backend_name(self):
        return "cupy"

    def __init__(self, cell, resolution=None, gpts=None, dtype=np.float32):
        super().__init__(cell=cell, resolution=resolution, gpts=gpts, dtype=dtype)
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
        self._check_ordered_grid("min_sphere")
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        cp.minimum.at(self.grid, indices, values)

    def add_spheres(self, centers, radii, value=1, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.add_sphere(center, float(radius), value=value, mask=mask)

    def add_spheres_cell_colored(self, centers, radii, value=1):
        """Add constant spheres with a 27-color GPU cell scheme.

        This experimental path processes spatial cells of one color in
        parallel. Cells with the same color are separated by at least two other
        cells in each direction, so their stencils cannot overlap when the
        coloring-cell width is at least the largest radius in the batch. Atoms
        within a cell are processed sequentially inside one CUDA thread to avoid
        races without atomics.

        The method currently supports constant-mask ``float32`` grids. Atoms are
        grouped by unique radius so each kernel call can reuse one compact
        stencil while the coloring schedule is built from ``max(radii)``.
        """
        if self.grid.dtype != cp.dtype(cp.float32):
            raise TypeError("add_spheres_cell_colored currently requires a float32 CuPy grid")
        centers, radii = self._validate_spheres(centers, radii)
        if centers.shape[0] == 0:
            return
        unique_radii = np.unique(radii)
        coloring_radius = float(np.max(unique_radii))
        for radius in unique_radii:
            selected = radii == radius
            plan = self.prepare_cell_coloring(centers[selected], float(radius), coloring_radius=coloring_radius)
            self._run_cell_colored_add(plan, value=value)

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
        self._check_ordered_grid("min_spheres")
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.min_sphere(center, float(radius), value=value, mask=mask)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        self._check_ordered_grid("clamp_grid")
        self.grid = cp.clip(self.grid, min_val, max_val)

    def synchronize(self):
        """Synchronize the current CuPy device."""
        cp.cuda.Stream.null.synchronize()

    def prepare_cell_coloring(self, centers, radius, coloring_radius=None):
        """Prepare a 27-color plan for repeated same-radius GPU add updates.

        ``radius`` controls the stencil applied to each atom in this plan.
        ``coloring_radius`` controls the spatial-cell width used to avoid write
        races and should be at least as large as the maximum radius in a mixed
        batch. When omitted, it defaults to ``radius``.
        """
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
        color_to_cells = [cell_ids[(colors == color) & (cell_counts > 0)].astype(np.int32) for color in range(27)]

        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))
        return {
            "radius": radius,
            "coloring_radius": coloring_radius,
            "cell_counts_3d": cell_counts_3d.astype(np.int32),
            "cell_widths": cell_widths,
            "center_indices": cp.asarray(sorted_centers.ravel(), dtype=cp.int32),
            "cell_starts": cp.asarray(cell_starts, dtype=cp.int32),
            "cell_counts": cp.asarray(cell_counts, dtype=cp.int32),
            "color_to_cells": [cp.asarray(cells, dtype=cp.int32) for cells in color_to_cells],
            "offsets": cp.asarray(offsets.ravel(), dtype=cp.int32),
            "offset_count": int(offsets.shape[0]),
            "kernel": cp.RawKernel(_CELL_COLOR_ADD_KERNEL, "add_colored_cells"),
        }

    def apply_prepared_cell_coloring(self, plan, value=1):
        """Apply a prepared same-radius 27-color add plan."""
        self._run_cell_colored_add(plan, value=value)

    def _run_cell_colored_add(self, plan, value=1):
        nx, ny, nz = [int(v) for v in self.gpts]
        threads = 128
        for color_cells in plan["color_to_cells"]:
            n_color_cells = int(color_cells.size)
            if n_color_cells == 0:
                continue
            blocks = (int((n_color_cells + threads - 1) // threads),)
            plan["kernel"](
                blocks,
                (threads,),
                (
                    self.grid.ravel(),
                    plan["center_indices"],
                    plan["cell_starts"],
                    plan["cell_counts"],
                    color_cells,
                    n_color_cells,
                    plan["offsets"],
                    plan["offset_count"],
                    nx,
                    ny,
                    nz,
                    np.float32(value),
                ),
            )
        self.synchronize()
