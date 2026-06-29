import numpy as np
import pytest

from atomvoxelizer.voxelgrid import VoxelGridNumPy


def test_requires_exactly_one_grid_size_argument():
    cell = np.eye(3)

    with pytest.raises(ValueError, match="Either resolution or gpts"):
        VoxelGridNumPy(cell)

    with pytest.raises(ValueError, match="Only one"):
        VoxelGridNumPy(cell, resolution=0.5, gpts=(2, 2, 2))


def test_resolution_and_gpts_initialization():
    grid = VoxelGridNumPy(np.diag([2.0, 3.0, 4.0]), resolution=1.5)
    np.testing.assert_array_equal(grid.gpts, np.array([2, 2, 3]))
    np.testing.assert_allclose(grid.resolution, np.array([1.0, 1.5, 4.0 / 3.0]))
    assert grid.grid.shape == (2, 2, 3)

    direct = VoxelGridNumPy(np.diag([2.0, 3.0, 4.0]), gpts=(2, 3, 4))
    np.testing.assert_allclose(direct.resolution, np.array([1.0, 1.0, 1.0]))


def test_positions_wrap_periodically():
    grid = VoxelGridNumPy(np.eye(3) * 4.0, gpts=(4, 4, 4))

    assert grid.position_to_index(np.array([0.1, 0.1, 0.1])) == (0, 0, 0)
    assert grid.position_to_index(np.array([4.1, -0.1, 8.1])) == (0, 3, 0)
    np.testing.assert_allclose(grid.index_to_position(0, 1, 2), np.array([0.5, 1.5, 2.5]))


def test_single_sphere_mutations_and_clamping():
    grid = VoxelGridNumPy(np.eye(3) * 5.0, gpts=(5, 5, 5))
    center = np.array([2.5, 2.5, 2.5])

    grid.add_sphere(center, radius=1.01, value=2.0)
    assert np.count_nonzero(grid.grid == 2.0) == 7

    grid.set_sphere(center, radius=0.1, value=-3.0)
    assert grid.grid[2, 2, 2] == -3.0

    grid.mul_sphere(center, radius=0.1, factor=2.0)
    assert grid.grid[2, 2, 2] == -6.0

    grid.div_sphere(center, radius=0.1, factor=3.0)
    assert grid.grid[2, 2, 2] == -2.0

    grid.clamp_grid(min_val=-1.0, max_val=1.0)
    assert grid.grid.min() >= -1.0
    assert grid.grid.max() <= 1.0


def test_batched_spheres_match_repeated_single_spheres():
    cell = np.eye(3) * 6.0
    centers = np.array(
        [
            [1.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
            [5.8, 5.8, 5.8],
        ]
    )
    radii = np.array([1.01, 0.51, 1.01])

    batched = VoxelGridNumPy(cell, gpts=(6, 6, 6))
    repeated = VoxelGridNumPy(cell, gpts=(6, 6, 6))

    batched.add_spheres(centers, radii, value=1.25)
    for center, radius in zip(centers, radii):
        repeated.add_sphere(center, radius, value=1.25)
    np.testing.assert_allclose(batched.grid, repeated.grid)

    batched.set_spheres(centers, radii, value=-1.0)
    for center, radius in zip(centers, radii):
        repeated.set_sphere(center, radius, value=-1.0)
    np.testing.assert_allclose(batched.grid, repeated.grid)


def test_sample_voxels_in_range_is_seeded_and_validates():
    grid = VoxelGridNumPy(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[0, 0, 0] = 1.0
    grid.grid[1, 1, 1] = 1.0

    first = list(grid.sample_voxels_in_range(1.0, 1.0, return_indices=True, seed=7))
    second = list(grid.sample_voxels_in_range(1.0, 1.0, return_indices=True, seed=7))
    assert first == second
    assert set(first) == {(0, 0, 0), (1, 1, 1)}

    with pytest.raises(ValueError, match="No voxels"):
        list(grid.sample_voxels_in_range(2.0, 3.0))

    with pytest.raises(ValueError, match="min_dist"):
        list(grid.sample_voxels_in_range(1.0, 1.0, return_indices=True, min_dist=1.0))

