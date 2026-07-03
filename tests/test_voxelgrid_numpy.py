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
    assert grid.grid.dtype == np.dtype(np.float32)

    direct = VoxelGridNumPy(np.diag([2.0, 3.0, 4.0]), gpts=(2, 3, 4))
    np.testing.assert_allclose(direct.resolution, np.array([1.0, 1.0, 1.0]))


def test_grid_dtype_can_be_selected_for_integer_and_float_grids():
    int_grid = VoxelGridNumPy(np.eye(3) * 5.0, gpts=(5, 5, 5), dtype=np.int16)
    center = np.array([2.5, 2.5, 2.5])
    assert int_grid.grid.dtype == np.dtype(np.int16)

    int_grid.add_sphere(center, radius=1.01, value=2)
    assert int_grid.grid[2, 2, 2] == 2
    int_grid.set_sphere(center, radius=0.1, value=7)
    int_grid.mul_sphere(center, radius=0.1, factor=2)
    assert int_grid.grid[2, 2, 2] == 14
    int_grid.grid.fill(9)
    int_grid.min_sphere(center, radius=0.1, value=3, mask="constant")
    assert int_grid.grid[2, 2, 2] == 3
    int_grid.clamp_grid(0, 4)
    assert int_grid.grid.max() == 4

    float_grid = VoxelGridNumPy(np.eye(3) * 5.0, gpts=(5, 5, 5), dtype=np.float64)
    assert float_grid.grid.dtype == np.dtype(np.float64)
    float_grid.min_sphere(center, radius=1.01, value=1.0, mask="distance")
    assert float_grid.grid.dtype == np.dtype(np.float64)


def test_complex_grid_supports_arithmetic_but_not_ordered_operations():
    grid = VoxelGridNumPy(np.eye(3) * 5.0, gpts=(5, 5, 5), dtype=np.complex128)
    center = np.array([2.5, 2.5, 2.5])
    assert grid.grid.dtype == np.dtype(np.complex128)

    grid.set_sphere(center, radius=0.1, value=1.0 + 2.0j)
    grid.add_sphere(center, radius=0.1, value=2.0 - 1.0j)
    grid.mul_sphere(center, radius=0.1, factor=2.0)
    grid.div_sphere(center, radius=0.1, factor=2.0)
    assert grid.grid[2, 2, 2] == pytest.approx(3.0 + 1.0j)

    with pytest.raises(TypeError, match="complex"):
        grid.min_sphere(center, radius=0.1)
    with pytest.raises(TypeError, match="complex"):
        grid.clamp_grid(0.0, 1.0)
    with pytest.raises(TypeError, match="complex"):
        list(grid.sample_voxels_in_range(0.0, 1.0))


def test_grid_dtype_must_be_numeric():
    with pytest.raises(TypeError, match="numeric"):
        VoxelGridNumPy(np.eye(3), gpts=(2, 2, 2), dtype=object)


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


def test_batched_arithmetic_operations_match_repeated_single_spheres():
    cell = np.eye(3) * 5.0
    centers = np.array([[1.2, 1.3, 1.4], [3.8, 3.7, 3.6]])
    radii = np.array([1.1, 0.9])

    for operation, keyword, value, fill in [
        ("mul_spheres", "factor", 2.0, 3.0),
        ("div_spheres", "factor", 2.0, 8.0),
        ("min_spheres", "value", 1.0, 5.0),
    ]:
        batched = VoxelGridNumPy(cell, gpts=(5, 5, 5))
        repeated = VoxelGridNumPy(cell, gpts=(5, 5, 5))
        batched.grid.fill(fill)
        repeated.grid.fill(fill)

        getattr(batched, operation)(centers, radii, mask="constant", **{keyword: value})
        single_operation = operation.replace("_spheres", "_sphere")
        for center, radius in zip(centers, radii):
            getattr(repeated, single_operation)(center, radius, mask="constant", **{keyword: value})

        np.testing.assert_allclose(batched.grid, repeated.grid)


def test_distance_mask_arithmetic_has_expected_center_and_neighbor_values():
    grid = VoxelGridNumPy(np.eye(3) * 5.0, gpts=(5, 5, 5))
    center = np.array([2.5, 2.5, 2.5])

    grid.set_sphere(center, radius=1.01, value=2.0, mask="distance")
    assert grid.grid[2, 2, 2] == pytest.approx(0.0)
    assert grid.grid[3, 2, 2] == pytest.approx(2.0)

    grid.grid.fill(1.0)
    grid.add_sphere(center, radius=1.01, value=0.5, mask="distance")
    assert grid.grid[2, 2, 2] == pytest.approx(1.0)
    assert grid.grid[3, 2, 2] == pytest.approx(1.5)

    grid.grid.fill(2.0)
    grid.mul_sphere(center, radius=1.01, factor=0.5, mask="distance")
    assert grid.grid[3, 2, 2] == pytest.approx(1.0)


def test_sample_voxels_in_range_min_dist_filters_positions():
    grid = VoxelGridNumPy(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[0, 0, 0] = 1.0
    grid.grid[0, 0, 1] = 1.0

    samples = list(grid.sample_voxels_in_range(1.0, 1.0, min_dist=1.1, seed=4))
    assert len(samples) == 1
    assert samples[0].shape == (3,)


def test_plot_2d_validates_slice_arguments_and_can_render(monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    grid = VoxelGridNumPy(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[1, 2, 3] = 1.0
    monkeypatch.setattr(plt, "show", lambda: None)

    grid.plot_2D(axis="z", index=3, threshold=0.5, real_space=False, draw_cell=False)
    grid.plot_2D(axis="x", position=1.5, threshold=0.5)

    with pytest.raises(ValueError, match="Axis"):
        grid.plot_2D(axis="bad")
    with pytest.raises(ValueError, match="either"):
        grid.plot_2D(index=1, position=1.0)
    with pytest.raises(IndexError, match="out of bounds"):
        grid.plot_2D(axis="z", index=99)


def test_plot_3d_can_render_with_and_without_cell(monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    grid = VoxelGridNumPy(np.eye(3) * 4.0, gpts=(4, 4, 4))
    grid.grid[1, 1, 1] = 1.0
    grid.grid[2, 2, 2] = 2.0
    monkeypatch.setattr(plt, "show", lambda: None)

    grid.plot_3D(threshold=0.5, draw_cell=True)
    grid.plot_3D(threshold=0.5, draw_cell=False)
