import importlib.util

import numpy as np
import pytest

from atomvoxelizer import VoxelGrid, VoxelGridAnalysis


pytestmark = pytest.mark.skipif(importlib.util.find_spec("skimage") is None, reason="scikit-image is not installed")


def test_region_volume_uses_cell_volume_per_voxel():
    grid = VoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[1:3, 1:3, 1:3] = 1.0

    analysis = VoxelGridAnalysis(grid)

    assert analysis.voxel_volume == pytest.approx(1.0)
    assert analysis.region_volume(grid.grid > 0.0) == pytest.approx(8.0)


def test_analyze_regions_finds_connected_components_and_surface_area():
    grid = VoxelGrid(np.eye(3) * 6.0, gpts=(6, 6, 6))
    grid.grid[1:3, 1:3, 1:3] = 1.0
    grid.grid[4, 4, 4] = 1.0

    regions = VoxelGridAnalysis(grid).analyze_regions(threshold=0.5)

    assert [region.voxel_count for region in regions] == [8, 1]
    assert [region.volume for region in regions] == pytest.approx([8.0, 1.0])
    assert all(region.surface_area > 0.0 for region in regions)


def test_experimental_unit_conversions():
    analysis = VoxelGridAnalysis(VoxelGrid(np.eye(3), gpts=(1, 1, 1)))
    mass_amu = 1.0 / 1.66053906660

    assert analysis.volume_angstrom3_to_cm3_per_g(1.0, mass_amu) == pytest.approx(1.0)
    assert analysis.area_angstrom2_to_m2_per_g(1.0, mass_amu) == pytest.approx(1.0e4)
