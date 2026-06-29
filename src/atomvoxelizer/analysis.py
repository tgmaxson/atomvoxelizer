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


__all__ = ["VoxelGridAnalysis", "VoxelRegion"]
