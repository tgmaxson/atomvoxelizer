from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelRegion:
    """Summary of one connected voxel region."""

    label: int
    voxel_count: int
    volume: float
    surface_area: float


class VoxelGridAnalysis:
    """Analyze connected voxel volumes and their surfaces."""

    def __init__(self, voxel_grid):
        self.voxel_grid = voxel_grid

    @property
    def grid(self):
        return self.voxel_grid.to_numpy()

    @property
    def cell(self):
        return self.voxel_grid.cell

    @property
    def gpts(self):
        return self.voxel_grid.gpts

    @property
    def voxel_volume(self):
        return abs(float(np.linalg.det(self.cell))) / float(np.prod(self.gpts))

    def mask(self, min_value=None, max_value=None, threshold=None, above=True):
        """Build a boolean mask from voxel values."""
        if threshold is not None and (min_value is not None or max_value is not None):
            raise ValueError("Specify either threshold or min_value/max_value, not both")

        grid = self.grid
        if threshold is not None:
            return grid > threshold if above else grid < threshold

        selected = np.ones(grid.shape, dtype=bool)
        if min_value is not None:
            selected &= grid >= min_value
        if max_value is not None:
            selected &= grid <= max_value
        return selected

    def connected_components(self, selected, connectivity=1, periodic=True):
        """Label connected components in a boolean mask."""
        try:
            from skimage.measure import label
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "VoxelGridAnalysis requires scikit-image. Install it with "
                "`pip install AtomVoxelizer[analysis]`."
            ) from exc

        selected = np.asarray(selected, dtype=bool)
        labels = label(selected, connectivity=connectivity)
        if periodic:
            labels = self._merge_periodic_labels(labels, selected)
        return labels, int(labels.max())

    def region_volume(self, selected):
        """Return the volume represented by a boolean mask."""
        return int(np.count_nonzero(selected)) * self.voxel_volume

    def surface_area(self, selected, periodic=True):
        """Estimate the surface area of a boolean region with marching cubes."""
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "VoxelGridAnalysis requires scikit-image. Install it with "
                "`pip install AtomVoxelizer[analysis]`."
            ) from exc

        selected = np.asarray(selected, dtype=bool)
        if not np.any(selected):
            return 0.0
        if periodic and np.all(selected):
            return 0.0

        if periodic:
            values = np.tile(selected.astype(np.float32), (3, 3, 3))
            vertices, faces, _normals, _values = marching_cubes(values, level=0.5)
            central_offset = self.gpts
            centroids = vertices[faces].mean(axis=1)
            in_central_cell = np.all((centroids >= central_offset) & (centroids < 2 * central_offset), axis=1)
            faces = faces[in_central_cell]
            vertices = vertices - central_offset
        else:
            values = np.pad(selected.astype(np.float32), 1, mode="constant", constant_values=0.0)
            vertices, faces, _normals, _values = marching_cubes(values, level=0.5)
            vertices = vertices - 1.0

        if faces.size == 0:
            return 0.0

        real_vertices = self._index_vertices_to_real(vertices)
        return self._mesh_surface_area(real_vertices, faces)

    def mesh_at_value(self, level, periodic=True, clip_periodic=True):
        """Return a marching-cubes mesh for a scalar voxel value.

        ``level`` is interpreted in the units stored in the voxel grid. For a
        distance-mask grid this traces the surface at a fixed distance from the
        nearest atom. Vertices are returned in real-space coordinates and faces
        index into that vertex array. Periodic meshes are clipped to the primary
        cell by default so boundary-crossing triangles are cut at the cell edge
        instead of being wrapped across the cell.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "VoxelGridAnalysis requires scikit-image. Install it with "
                "`pip install scikit-image`."
            ) from exc

        level = float(level)
        grid = self._finite_grid_for_level(level)
        finite = grid[np.isfinite(grid)]
        if finite.size == 0 or level < finite.min() or level > finite.max():
            raise ValueError("level must lie within the finite voxel value range")

        if periodic:
            values = np.tile(grid.astype(np.float32), (3, 3, 3))
            vertices, faces, _normals, _values = marching_cubes(values, level=level)
            central_offset = self.gpts
            centroids = vertices[faces].mean(axis=1)
            in_central_cell = np.all((centroids >= central_offset) & (centroids < 2 * central_offset), axis=1)
            faces = faces[in_central_cell]
            vertices = vertices - central_offset
            if clip_periodic:
                vertices, faces = self._clip_mesh_to_index_cell(vertices, faces)
        else:
            vertices, faces, _normals, _values = marching_cubes(grid.astype(np.float32), level=level)

        if faces.size == 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

        used = np.unique(faces)
        remap = np.full(vertices.shape[0], -1, dtype=int)
        remap[used] = np.arange(used.shape[0])
        real_vertices = self._index_vertices_to_real(vertices[used])
        return real_vertices, remap[faces]

    def surface_area_at_value(self, level, periodic=True, clip_periodic=True):
        """Estimate the area of a scalar isosurface at ``level``."""
        vertices, faces = self.mesh_at_value(level=level, periodic=periodic, clip_periodic=clip_periodic)
        if faces.size == 0:
            return 0.0
        return self._mesh_surface_area(vertices, faces)

    def analyze_regions(
        self,
        min_value=None,
        max_value=None,
        threshold=None,
        above=True,
        connectivity=1,
        periodic=True,
    ):
        """Return volume and marching-cubes area for each connected region."""
        selected = self.mask(min_value=min_value, max_value=max_value, threshold=threshold, above=above)
        labels, label_count = self.connected_components(selected, connectivity=connectivity, periodic=periodic)

        regions = []
        for label_id in range(1, label_count + 1):
            region_mask = labels == label_id
            voxel_count = int(np.count_nonzero(region_mask))
            regions.append(
                VoxelRegion(
                    label=label_id,
                    voxel_count=voxel_count,
                    volume=voxel_count * self.voxel_volume,
                    surface_area=self.surface_area(region_mask, periodic=periodic),
                )
            )
        return regions

    @staticmethod
    def volume_angstrom3_to_cm3_per_g(volume_angstrom3, mass_amu):
        """Convert a cell/supercell volume from Angstrom^3 to cm^3/g."""
        if mass_amu <= 0:
            raise ValueError("mass_amu must be positive")
        mass_g = float(mass_amu) * 1.66053906660e-24
        return float(volume_angstrom3) * 1.0e-24 / mass_g

    @staticmethod
    def area_angstrom2_to_m2_per_g(area_angstrom2, mass_amu):
        """Convert a cell/supercell area from Angstrom^2 to m^2/g."""
        if mass_amu <= 0:
            raise ValueError("mass_amu must be positive")
        mass_g = float(mass_amu) * 1.66053906660e-24
        return float(area_angstrom2) * 1.0e-20 / mass_g

    def _index_vertices_to_real(self, vertices):
        frac = vertices / self.gpts
        return frac @ self.cell

    def _finite_grid_for_level(self, level):
        grid = np.asarray(self.grid, dtype=np.float32)
        if np.all(np.isfinite(grid)):
            return grid

        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return grid

        high = max(float(finite.max()), float(level)) + 1.0
        low = min(float(finite.min()), float(level)) - 1.0
        clean = grid.copy()
        clean[np.isposinf(clean)] = high
        clean[np.isneginf(clean)] = low
        return clean

    def _clip_mesh_to_index_cell(self, vertices, faces):
        clipped_vertices = []
        clipped_faces = []
        bounds = [(0.0, float(n)) for n in self.gpts]

        for face in faces:
            polygon = [vertices[int(vertex_index)].astype(float) for vertex_index in face]
            for axis, (low, high) in enumerate(bounds):
                polygon = self._clip_polygon_axis(polygon, axis, low, keep_greater=True)
                polygon = self._clip_polygon_axis(polygon, axis, high, keep_greater=False)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue

            start = len(clipped_vertices)
            clipped_vertices.extend(polygon)
            for index in range(1, len(polygon) - 1):
                clipped_faces.append((start, start + index, start + index + 1))

        if not clipped_faces:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        return np.asarray(clipped_vertices, dtype=float), np.asarray(clipped_faces, dtype=int)

    @staticmethod
    def _clip_polygon_axis(polygon, axis, boundary, keep_greater):
        if not polygon:
            return []

        clipped = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary

        for current in polygon:
            current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if current_inside != previous_inside:
                delta = current[axis] - previous[axis]
                if delta != 0.0:
                    t = (boundary - previous[axis]) / delta
                    clipped.append(previous + t * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside

        return clipped

    @staticmethod
    def _merge_periodic_labels(labels, selected):
        label_count = int(labels.max())
        if label_count == 0:
            return labels

        parent = np.arange(label_count + 1)

        def find(label_id):
            while parent[label_id] != label_id:
                parent[label_id] = parent[parent[label_id]]
                label_id = parent[label_id]
            return label_id

        def union(a, b):
            if a == 0 or b == 0:
                return
            root_a = find(int(a))
            root_b = find(int(b))
            if root_a != root_b:
                parent[root_b] = root_a

        for axis in range(labels.ndim):
            first_labels = np.take(labels, 0, axis=axis)
            last_labels = np.take(labels, -1, axis=axis)
            first_selected = np.take(selected, 0, axis=axis)
            last_selected = np.take(selected, -1, axis=axis)
            for a, b in zip(first_labels[first_selected & last_selected], last_labels[first_selected & last_selected]):
                union(a, b)

        root_to_new_label = {}
        next_label = 1
        merged = np.zeros_like(labels)
        for label_id in range(1, label_count + 1):
            root = find(label_id)
            if root not in root_to_new_label:
                root_to_new_label[root] = next_label
                next_label += 1
            merged[labels == label_id] = root_to_new_label[root]

        return merged

    @staticmethod
    def _mesh_surface_area(vertices, faces):
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    @staticmethod
    def mesh_surface_area(vertices, faces):
        """Return the area of a triangular mesh."""
        return VoxelGridAnalysis._mesh_surface_area(np.asarray(vertices), np.asarray(faces, dtype=int))


__all__ = ["VoxelGridAnalysis", "VoxelRegion"]
