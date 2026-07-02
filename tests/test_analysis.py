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

    with pytest.raises(ValueError, match="mass_amu"):
        analysis.volume_angstrom3_to_cm3_per_g(1.0, 0.0)

    with pytest.raises(ValueError, match="mass_amu"):
        analysis.area_angstrom2_to_m2_per_g(1.0, -1.0)


def test_mask_rejects_mixed_threshold_and_range():
    analysis = VoxelGridAnalysis(VoxelGrid(np.eye(3), gpts=(1, 1, 1)))

    with pytest.raises(ValueError, match="threshold"):
        analysis.mask(threshold=0.5, min_value=0.0)


def test_connected_components_merge_periodic_boundaries():
    grid = VoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[0, 1, 1] = 1.0
    grid.grid[-1, 1, 1] = 1.0
    analysis = VoxelGridAnalysis(grid)

    _labels, periodic_count = analysis.connected_components(grid.grid > 0.0, periodic=True)
    _labels, nonperiodic_count = analysis.connected_components(grid.grid > 0.0, periodic=False)

    assert periodic_count == 1
    assert nonperiodic_count == 2


def test_periodic_surface_area_removes_cell_boundary_surface_for_full_mask():
    grid = VoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4))
    selected = np.ones(grid.grid.shape, dtype=bool)
    analysis = VoxelGridAnalysis(grid)

    assert analysis.surface_area(selected, periodic=True) == pytest.approx(0.0)
    assert analysis.surface_area(selected, periodic=False) > 0.0


def test_voxel_face_surface_area_counts_exposed_faces():
    grid = VoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4))
    selected = np.zeros(grid.grid.shape, dtype=bool)
    selected[1:3, 1:3, 1:3] = True
    analysis = VoxelGridAnalysis(grid)

    assert analysis.surface_area_voxel_faces(selected, periodic=True) == pytest.approx(24.0)
    assert analysis.surface_area_voxel_faces(selected, periodic=False) == pytest.approx(24.0)


def test_analyze_regions_supports_voxel_face_surface_method():
    grid = VoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[1:3, 1:3, 1:3] = 1.0

    regions = VoxelGridAnalysis(grid).analyze_regions(threshold=0.5, surface_method="voxel-faces")

    assert len(regions) == 1
    assert regions[0].surface_area == pytest.approx(24.0)

    with pytest.raises(ValueError, match="surface_method"):
        VoxelGridAnalysis(grid).analyze_regions(threshold=0.5, surface_method="bad")


def test_probe_accessible_mask_inflates_atom_radii_without_mutating_grid_by_default():
    grid = VoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5), dtype=np.float32)
    grid.grid.fill(7.0)
    analysis = VoxelGridAnalysis(grid)
    positions = np.array([[2.5, 2.5, 2.5]])
    radii = np.array([0.1])

    small_probe = analysis.probe_accessible_mask(positions, radii, probe_radius=0.0)
    large_probe = analysis.probe_accessible_mask(positions, radii, probe_radius=1.01)

    assert np.count_nonzero(large_probe) < np.count_nonzero(small_probe)
    assert small_probe[2, 2, 2] == np.False_
    assert large_probe[3, 2, 2] == np.False_
    np.testing.assert_allclose(grid.grid, np.full((5, 5, 5), 7.0))


def test_probe_accessible_analysis_reports_regions_and_can_write_grid():
    grid = VoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5), dtype=np.float32)
    analysis = VoxelGridAnalysis(grid)
    positions = np.array([[2.5, 2.5, 2.5]])
    radii = np.array([1.01])

    result = analysis.analyze_probe_accessibility(
        positions,
        radii,
        probe_radius=0.0,
        surface_method="voxel-faces",
        write_grid=True,
    )

    assert result.probe_radius == pytest.approx(0.0)
    assert result.accessible_voxel_count == np.count_nonzero(result.accessible_mask)
    assert result.accessible_volume == pytest.approx(result.accessible_voxel_count * analysis.voxel_volume)
    assert result.accessible_surface_area > 0.0
    assert len(result.regions) == 1
    np.testing.assert_allclose(grid.grid, result.accessible_mask.astype(grid.dtype))


def test_probe_accessible_analysis_rejects_invalid_inputs():
    analysis = VoxelGridAnalysis(VoxelGrid(np.eye(3), gpts=(2, 2, 2)))

    with pytest.raises(ValueError, match="positions"):
        analysis.probe_accessible_mask([0.0, 0.0, 0.0], [1.0], probe_radius=1.0)
    with pytest.raises(ValueError, match="radii"):
        analysis.probe_accessible_mask([[0.0, 0.0, 0.0]], [1.0, 2.0], probe_radius=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        analysis.probe_accessible_mask([[0.0, 0.0, 0.0]], [-1.0], probe_radius=1.0)
    with pytest.raises(ValueError, match="probe_radius"):
        analysis.probe_accessible_mask([[0.0, 0.0, 0.0]], [1.0], probe_radius=-1.0)
