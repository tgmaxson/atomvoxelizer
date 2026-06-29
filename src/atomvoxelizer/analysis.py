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
        grid = self.grid
        if threshold is not None:
            return grid > threshold if above else grid < threshold

        selected = np.ones(grid.shape, dtype=bool)
        if min_value is not None:
            selected &= grid >= min_value
        if max_value is not None:
            selected &= grid <= max_value
        return selected

    def connected_components(self, selected, connectivity=1):
        """Label connected components in a boolean mask."""
        try:
            from skimage.measure import label
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "VoxelGridAnalysis requires scikit-image. Install it with "
                "`pip install AtomVoxelizer[analysis]`."
            ) from exc

        labels = label(np.asarray(selected, dtype=bool), connectivity=connectivity)
        return labels, int(labels.max())

    def region_volume(self, selected):
        """Return the volume represented by a boolean mask."""
        return int(np.count_nonzero(selected)) * self.voxel_volume

    def surface_area(self, selected):
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

        padded = np.pad(selected.astype(np.float32), 1, mode="constant", constant_values=0.0)
        vertices, faces, _normals, _values = marching_cubes(padded, level=0.5)
        vertices = vertices - 1.0
        real_vertices = self._index_vertices_to_real(vertices)
        return self._mesh_surface_area(real_vertices, faces)

    def analyze_regions(
        self,
        min_value=None,
        max_value=None,
        threshold=None,
        above=True,
        connectivity=1,
    ):
        """Return volume and marching-cubes area for each connected region."""
        selected = self.mask(min_value=min_value, max_value=max_value, threshold=threshold, above=above)
        labels, label_count = self.connected_components(selected, connectivity=connectivity)

        regions = []
        for label_id in range(1, label_count + 1):
            region_mask = labels == label_id
            voxel_count = int(np.count_nonzero(region_mask))
            regions.append(
                VoxelRegion(
                    label=label_id,
                    voxel_count=voxel_count,
                    volume=voxel_count * self.voxel_volume,
                    surface_area=self.surface_area(region_mask),
                )
            )
        return regions

    @staticmethod
    def volume_angstrom3_to_cm3_per_g(volume_angstrom3, mass_amu):
        """Convert a cell/supercell volume from Angstrom^3 to cm^3/g."""
        mass_g = float(mass_amu) * 1.66053906660e-24
        return float(volume_angstrom3) * 1.0e-24 / mass_g

    @staticmethod
    def area_angstrom2_to_m2_per_g(area_angstrom2, mass_amu):
        """Convert a cell/supercell area from Angstrom^2 to m^2/g."""
        mass_g = float(mass_amu) * 1.66053906660e-24
        return float(area_angstrom2) * 1.0e-20 / mass_g

    def _index_vertices_to_real(self, vertices):
        frac = vertices / self.gpts
        return frac @ self.cell

    @staticmethod
    def _mesh_surface_area(vertices, faces):
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())


__all__ = ["VoxelGridAnalysis", "VoxelRegion"]
