from functools import lru_cache
import numpy as np
from numba import njit, prange, cuda


@lru_cache(maxsize=50)
def _cached_sphere_mask(radius, gpts, cell):
    #gpts = np.array(gpts)
    #cell = np.array(cell)
    nx, ny, nz = gpts
    ix, iy, iz = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing='ij'
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


@cuda.jit
def _add_sphere_mask_cuda(grid, mask, cx, cy, cz, value, ox, oy, oz):
    i, j, k = cuda.grid(3)
    mx, my, mz = mask.shape
    nx, ny, nz = grid.shape

    if i >= mx or j >= my or k >= mz:
        return

    if mask[i, j, k]:
        x = (cx + i - ox) % nx
        y = (cy + j - oy) % ny
        z = (cz + k - oz) % nz
        cuda.atomic.add(grid, (x, y, z), value)


class VoxelGrid(object):
    def __init__(self, cell, resolution=None, gpts=None):
        self.cell = np.array(cell)
        self.cell_inv = np.linalg.inv(self.cell)

        if not resolution and not gpts:
            raise ValueError("Either resolution or gpts must be specified")
        if resolution and gpts:
            raise ValueError("Only one of resolution or gpts can be specified")
        
        # If given resolution, calculate gpts then recalculate real resolution
        # If given gpts, use it directly and calculate resolution
        if resolution is not None:
            self.resolution = resolution
            lengths = np.linalg.norm(self.cell, axis=1)
            self.gpts = np.ceil(lengths / resolution).astype(int)
            self.resolution = lengths / self.gpts
        else:
            self.gpts = np.array(gpts, dtype=int)
            lengths = np.linalg.norm(self.cell, axis=1)
            self.resolution = lengths / self.gpts

        self.grid = np.zeros((self.gpts[0], self.gpts[1], self.gpts[2]), dtype=np.float32)


    def position_to_index(self, r):
        """Convert real-space position to voxel index using real-space wrapping."""
        # Map to fractional coords (can be outside [0,1))
        frac = r @ self.cell_inv

        # Convert back to real space in [0,1) using the cell — ensures wrapped real-space image
        r_wrapped = self.cell @ (frac % 1.0)

        # Convert wrapped position to fractional coords again
        frac_wrapped = self.cell_inv @ r_wrapped
        frac_wrapped = np.clip(frac_wrapped, 0.0, np.nextafter(1.0, 0.0))

        idx = np.floor(frac_wrapped * self.gpts).astype(int)
        return tuple(idx)


    def index_to_position(self, i, j, k):
        """Convert grid index to real-space position at center of voxel."""
        frac = (np.array([i, j, k]) + 0.5) / self.gpts
        r = frac @ self.cell
        return r


    def plot_3D(self, threshold=0.1, s=5, draw_cell=True):
        """
        Plot the VoxelGrid in real space using a scatter plot.
        Only voxels with value > threshold are plotted.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Create mesh of fractional coordinates
        nx, ny, nz = self.gpts
        ix, iy, iz = np.meshgrid(
            np.arange(nx) + 0.5,
            np.arange(ny) + 0.5,
            np.arange(nz) + 0.5,
            indexing='ij'
        )

        frac_coords = np.stack([ix / nx, iy / ny, iz / nz], axis=-1)  # (nx, ny, nz, 3)
        # Convert fractional coords to Cartesian using row-wise cell vectors
        real_coords = frac_coords @ self.cell  # (nx, ny, nz, 3)

        # Mask
        mask = self.grid > threshold
        xyz = real_coords[mask]
        values = self.grid[mask]

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=values, cmap='viridis', s=s)
        fig.colorbar(p, ax=ax, label='Voxel value')

        # Draw cell if requested
        if draw_cell:
            # Cell corners in fractional coords
            corners_frac = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ])
            corners = corners_frac @ self.cell  # shape (8, 3)

            # Define the 12 edges
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7)
            ]
            for i, j in edges:
                xs = [corners[i, 0], corners[j, 0]]
                ys = [corners[i, 1], corners[j, 1]]
                zs = [corners[i, 2], corners[j, 2]]
                ax.plot(xs, ys, zs, color='black')

        # Set equal aspect ratio
        all_coords = np.concatenate([xyz, corners]) if draw_cell else xyz
        xlim = [all_coords[:, 0].min(), all_coords[:, 0].max()]
        ylim = [all_coords[:, 1].min(), all_coords[:, 1].max()]
        zlim = [all_coords[:, 2].min(), all_coords[:, 2].max()]

        max_range = max(
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            zlim[1] - zlim[0]
        ) / 2.0

        mid_x = np.mean(xlim)
        mid_y = np.mean(ylim)
        mid_z = np.mean(zlim)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.tight_layout()
        plt.show()
        

    def plot_2D(self, axis='z', index=None, position=None, threshold=0.1, draw_cell=True, real_space=True):
        """
        Plot a 2D slice of the VoxelGrid along a given axis.

        Parameters:
        - axis: 'x', 'y', or 'z' (which axis to slice along)
        - index: int, voxel index to slice at (mutually exclusive with position)
        - position: float, real-space coordinate along that axis (Å)
        - threshold: float, only show voxels with value > threshold
        - draw_cell: bool, whether to overlay the 2D projection of the unit cell (only in real_space mode)
        - real_space: bool, whether to plot in real-space coordinates (default True).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        ax_map = {'x': 0, 'y': 1, 'z': 2}
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

        # Slice axes
        axes = [0, 1, 2]
        axes.remove(ax_idx)
        ax1, ax2 = axes

        # Extract 2D slice
        slicers = [slice(None)] * 3
        slicers[ax_idx] = index
        slice_grid = self.grid[tuple(slicers)]

        n1, n2 = self.gpts[ax1], self.gpts[ax2]

        if real_space:
            i1 = (np.arange(n1) + 0.5) / n1
            i2 = (np.arange(n2) + 0.5) / n2
            coords = np.meshgrid(i1, i2, indexing='ij')
            frac_coords = np.stack(coords, axis=-1)
            xy = frac_coords @ self.cell[[ax1, ax2], :]
            xvals, yvals = xy[..., 0], xy[..., 1]
        else:
            xvals, yvals = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')

        mask = slice_grid > threshold

        fig, ax = plt.subplots()
        sc = ax.scatter(xvals[mask], yvals[mask], c=slice_grid[mask], cmap='viridis', s=10)
        fig.colorbar(sc, ax=ax, label='Voxel value')

        if draw_cell and real_space:
            corners_frac = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0]
            ])
            corners_real = corners_frac @ self.cell[[ax1, ax2], :]
            ax.plot(corners_real[:, 0], corners_real[:, 1], 'k--', lw=1)

        ax.set_xlabel(f'{["x", "y", "z"][ax1]}' + (' [Å]' if real_space else ' (voxel)'))
        ax.set_ylabel(f'{["x", "y", "z"][ax2]}' + (' [Å]' if real_space else ' (voxel)'))
        ax.set_title(f'{axis.upper()} Slice at index {index}')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()


    def set_sphere(self, center, radius, value=1):
        center_frac = center @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)
        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))
        _set_sphere_offsets(self.grid, center_idx, offsets, value)


    def add_sphere(self, center, radius, value=1):
        center_frac = center @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)
        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))
        _add_sphere_offsets(self.grid, center_idx, offsets, value)


    def positions_to_indices(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        frac = positions @ self.cell_inv
        frac_wrapped = frac % 1.0
        frac_wrapped = np.clip(frac_wrapped, 0.0, np.nextafter(1.0, 0.0))
        return np.floor(frac_wrapped * self.gpts).astype(np.int32)


    def add_spheres(self, centers, radii, value=1):
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape (N,)")

        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _add_many_sphere_offsets(self.grid, subset, offsets, value)


    def set_spheres(self, centers, radii, value=1):
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape (N,)")

        center_indices = self.positions_to_indices(centers)
        for radius in np.unique(radii):
            offsets = _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))
            subset = center_indices[radii == radius]
            _set_many_sphere_offsets(self.grid, subset, offsets, value)


    """
    def add_sphere_gpu(self, center, radius, value=1.0):
        center_frac = center @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)

        mask = _cached_sphere_mask(radius, tuple(self.gpts), tuple(map(tuple, self.cell))).astype(np.uint8)
        ox, oy, oz = np.array(mask.shape) // 2

        # Transfer data to GPU
        d_grid = cuda.to_device(self.grid.astype(np.float32))
        d_mask = cuda.to_device(mask)

        threadsperblock = (8, 8, 8)
        blockspergrid = tuple(
            int((mask.shape[i] + threadsperblock[i] - 1) // threadsperblock[i])
            for i in range(3)
        )

        # Launch
        cx, cy, cz = int(center_idx[0]), int(center_idx[1]), int(center_idx[2])
        ox, oy, oz = int(mask.shape[0] // 2), int(mask.shape[1] // 2), int(mask.shape[2] // 2)
        value = float(value)
        _add_sphere_mask_cuda[blockspergrid, threadsperblock](
            d_grid, d_mask, cx, cy, cz, value, ox, oy, oz
        )

        # Copy result back
        self.grid[:] = d_grid.copy_to_host()
    """

    def mul_sphere(self, center, radius, factor=2):
        center_frac = center @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)
        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))
        _mul_sphere_offsets(self.grid, center_idx, offsets, factor)


    def div_sphere(self, center, radius, factor=2):
        center_frac = center @ self.cell_inv % 1.0
        center_idx = np.floor(center_frac * self.gpts).astype(np.int32)
        offsets = _cached_sphere_offsets(radius, tuple(self.gpts), tuple(map(tuple, self.cell)))
        _div_sphere_offsets(self.grid, center_idx, offsets, factor)


    def clamp_grid(self, min_val=0.0, max_val=1.0):
        _clamp_grid(self.grid, min_val, max_val)


    def sample_voxels_in_range(self, min_val=0.0, max_val=1.0, min_dist=0.0, return_indices=False, seed=None):
        """
        Generator that yields voxel positions (real-space or index) whose values lie in [min_val, max_val],
        skipping voxels within `min_dist` Å of previously returned ones (real-space only).

        Parameters:
        - min_val, max_val: value range
        - min_dist: minimum real-space distance (Å) between returned samples
        - return_indices: if True, yields (i,j,k) index tuples instead of positions
        - seed: optional RNG seed

        Yields:
        - Real-space 3D position (np.array of shape (3,)) or index (i, j, k)
        """
        rng = np.random.default_rng(seed)
        mask = (self.grid >= min_val) & (self.grid <= max_val)
        candidates = np.argwhere(mask)

        if candidates.shape[0] == 0:
            raise ValueError("No voxels in specified value range.")

        if return_indices and min_dist > 0:
            raise ValueError("min_dist only supported when return_indices=False")

        if not return_indices:
            positions = np.array([self.index_to_position(*idx) for idx in candidates])
        else:
            positions = candidates

        selected = []
        indices = rng.permutation(len(positions))
        min_dist2 = min_dist**2

        for i in indices:
            pos = positions[i]
            if min_dist > 0 and selected:
                d2 = np.sum((np.array(selected) - pos)**2, axis=1)
                if np.any(d2 < min_dist2):
                    continue

            selected.append(pos)
            yield tuple(candidates[i]) if return_indices else pos


    def __repr__(self):
        return "VoxelGrid\n{} Cell\n{} Resolution\n{} gpts".format(self.cell, self.resolution, self.gpts)
