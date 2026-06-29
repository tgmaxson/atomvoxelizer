from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=50)
def _cached_sphere_mask(radius, gpts, cell):
    nx, ny, nz = gpts
    ix, iy, iz = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij",
    )
    frac_coords = (np.stack([ix, iy, iz], axis=-1) + 0.5) / gpts
    center_frac = np.array([0.5, 0.5, 0.5])
    disp_frac = frac_coords - center_frac
    disp_frac -= np.round(disp_frac)
    disp_mic = disp_frac @ cell
    dist2 = np.sum(disp_mic**2, axis=-1)
    return dist2 <= radius**2


@lru_cache(maxsize=200)
def _cached_sphere_offsets(radius, gpts, cell):
    gpts_arr = np.array(gpts, dtype=np.int32)
    cell_arr = np.array(cell, dtype=np.float64)
    lengths = np.linalg.norm(cell_arr, axis=1)
    max_offsets = np.ceil(radius / lengths * gpts_arr).astype(np.int32)

    offsets = []
    for dx in range(-max_offsets[0], max_offsets[0] + 1):
        for dy in range(-max_offsets[1], max_offsets[1] + 1):
            for dz in range(-max_offsets[2], max_offsets[2] + 1):
                disp_frac = np.array([dx, dy, dz], dtype=np.float64) / gpts_arr
                disp = disp_frac @ cell_arr
                if np.dot(disp, disp) <= radius**2:
                    offsets.append((dx, dy, dz))

    return np.array(offsets, dtype=np.int32)


class VoxelGrid:
    """Periodic voxel grid implemented with NumPy only."""

    backend_name = "numpy"

    def __init__(self, cell, resolution=None, gpts=None):
        self.cell = np.array(cell, dtype=np.float64)
        self.cell_inv = np.linalg.inv(self.cell)

        if resolution is None and gpts is None:
            raise ValueError("Either resolution or gpts must be specified")
        if resolution is not None and gpts is not None:
            raise ValueError("Only one of resolution or gpts can be specified")

        lengths = np.linalg.norm(self.cell, axis=1)
        if resolution is not None:
            self.gpts = np.ceil(lengths / resolution).astype(int)
        else:
            self.gpts = np.array(gpts, dtype=int)
        self.resolution = lengths / self.gpts

        self.grid = np.zeros(tuple(self.gpts), dtype=np.float32)

    def to_numpy(self):
        """Return the voxel values as a NumPy array."""
        return self.grid

    def position_to_index(self, r):
        """Convert real-space position to voxel index using periodic wrapping."""
        frac = np.asarray(r, dtype=np.float64) @ self.cell_inv
        frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
        idx = np.floor(frac_wrapped * self.gpts).astype(int)
        return tuple(idx)

    def index_to_position(self, i, j, k):
        """Convert a grid index to the real-space voxel center."""
        frac = (np.array([i, j, k]) + 0.5) / self.gpts
        return frac @ self.cell

    def _center_index(self, center):
        center_frac = np.asarray(center, dtype=np.float64) @ self.cell_inv % 1.0
        return np.floor(center_frac * self.gpts).astype(np.int32)

    def _offset_indices(self, center_idx, offsets):
        indices = (offsets + center_idx) % self.gpts
        return tuple(indices[:, axis] for axis in range(3))

    def _sphere_offsets(self, radius):
        return _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))

    def _sphere_indices(self, center, radius):
        return self._offset_indices(self._center_index(center), self._sphere_offsets(radius))

    def set_sphere(self, center, radius, value=1):
        self.grid[self._sphere_indices(center, radius)] = value

    def add_sphere(self, center, radius, value=1):
        np.add.at(self.grid, self._sphere_indices(center, radius), value)

    def mul_sphere(self, center, radius, factor=2):
        self.grid[self._sphere_indices(center, radius)] *= factor

    def div_sphere(self, center, radius, factor=2):
        self.grid[self._sphere_indices(center, radius)] /= factor

    def positions_to_indices(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        frac = positions @ self.cell_inv
        frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
        return np.floor(frac_wrapped * self.gpts).astype(np.int32)

    def _validate_spheres(self, centers, radii):
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape (N,)")
        return centers, radii

    def add_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.add_sphere(center, radius, value=value)

    def set_spheres(self, centers, radii, value=1):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.set_sphere(center, radius, value=value)

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        np.clip(self.grid, min_val, max_val, out=self.grid)

    def sample_voxels_in_range(self, min_val=0.0, max_val=1.0, min_dist=0.0, return_indices=False, seed=None):
        """
        Yield voxel positions or indices whose values lie in [min_val, max_val].

        When returning real-space positions, ``min_dist`` enforces a minimum
        Euclidean separation in Angstrom between yielded samples.
        """
        rng = np.random.default_rng(seed)
        grid = self.to_numpy()
        mask = (grid >= min_val) & (grid <= max_val)
        candidates = np.argwhere(mask)

        if candidates.shape[0] == 0:
            raise ValueError("No voxels in specified value range.")
        if return_indices and min_dist > 0:
            raise ValueError("min_dist only supported when return_indices=False")

        positions = candidates if return_indices else np.array([self.index_to_position(*idx) for idx in candidates])
        selected = []
        indices = rng.permutation(len(positions))
        min_dist2 = min_dist**2

        for i in indices:
            pos = positions[i]
            if min_dist > 0 and selected:
                d2 = np.sum((np.array(selected) - pos) ** 2, axis=1)
                if np.any(d2 < min_dist2):
                    continue
            selected.append(pos)
            yield tuple(candidates[i]) if return_indices else pos

    def plot_3D(self, threshold=0.1, s=5, draw_cell=True):
        """Plot voxels with values above ``threshold`` in real space."""
        import matplotlib.pyplot as plt

        nx, ny, nz = self.gpts
        ix, iy, iz = np.meshgrid(
            np.arange(nx) + 0.5,
            np.arange(ny) + 0.5,
            np.arange(nz) + 0.5,
            indexing="ij",
        )

        frac_coords = np.stack([ix / nx, iy / ny, iz / nz], axis=-1)
        real_coords = frac_coords @ self.cell
        grid = self.to_numpy()
        mask = grid > threshold
        xyz = real_coords[mask]
        values = grid[mask]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap="viridis", s=s)
        fig.colorbar(p, ax=ax, label="Voxel value")

        if draw_cell:
            corners_frac = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ]
            )
            corners = corners_frac @ self.cell
            edges = [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (1, 5),
                (2, 4),
                (2, 6),
                (3, 5),
                (3, 6),
                (4, 7),
                (5, 7),
                (6, 7),
            ]
            for i, j in edges:
                ax.plot(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    color="black",
                )

        all_coords = np.concatenate([xyz, corners]) if draw_cell else xyz
        xlim = [all_coords[:, 0].min(), all_coords[:, 0].max()]
        ylim = [all_coords[:, 1].min(), all_coords[:, 1].max()]
        zlim = [all_coords[:, 2].min(), all_coords[:, 2].max()]
        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2.0
        mid_x, mid_y, mid_z = np.mean(xlim), np.mean(ylim), np.mean(zlim)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        plt.show()

    def plot_2D(self, axis="z", index=None, position=None, threshold=0.1, draw_cell=True, real_space=True):
        """Plot a 2D slice of the voxel grid along ``axis``."""
        import matplotlib.pyplot as plt

        ax_map = {"x": 0, "y": 1, "z": 2}
        if axis not in ax_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        ax_idx = ax_map[axis]

        if index is not None and position is not None:
            raise ValueError("Specify either `index` or `position`, not both")
        if position is not None:
            index = self.position_to_index(np.eye(3)[ax_idx] * position)[ax_idx]
        if index is None:
            index = self.gpts[ax_idx] // 2

        shape = self.grid.shape
        if not (0 <= index < shape[ax_idx]):
            raise IndexError(f"{axis}-index {index} out of bounds (0 to {shape[ax_idx] - 1})")

        axes = [0, 1, 2]
        axes.remove(ax_idx)
        ax1, ax2 = axes

        slicers = [slice(None)] * 3
        slicers[ax_idx] = index
        slice_grid = self.to_numpy()[tuple(slicers)]

        n1, n2 = self.gpts[ax1], self.gpts[ax2]
        if real_space:
            i1 = (np.arange(n1) + 0.5) / n1
            i2 = (np.arange(n2) + 0.5) / n2
            coords = np.meshgrid(i1, i2, indexing="ij")
            frac_coords = np.stack(coords, axis=-1)
            xy = frac_coords @ self.cell[[ax1, ax2], :]
            xvals, yvals = xy[..., 0], xy[..., 1]
        else:
            xvals, yvals = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

        mask = slice_grid > threshold
        fig, ax = plt.subplots()
        sc = ax.scatter(xvals[mask], yvals[mask], c=slice_grid[mask], cmap="viridis", s=10)
        fig.colorbar(sc, ax=ax, label="Voxel value")

        if draw_cell and real_space:
            corners_frac = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
            corners_real = corners_frac @ self.cell[[ax1, ax2], :]
            ax.plot(corners_real[:, 0], corners_real[:, 1], "k--", lw=1)

        ax.set_xlabel(f'{["x", "y", "z"][ax1]}' + (" [Angstrom]" if real_space else " (voxel)"))
        ax.set_ylabel(f'{["x", "y", "z"][ax2]}' + (" [Angstrom]" if real_space else " (voxel)"))
        ax.set_title(f"{axis.upper()} Slice at index {index}")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"VoxelGrid\n{self.cell} Cell\n{self.resolution} Resolution\n{self.gpts} gpts"


VoxelGridNumPy = VoxelGrid


__all__ = ["VoxelGrid", "VoxelGridNumPy", "_cached_sphere_mask", "_cached_sphere_offsets"]
