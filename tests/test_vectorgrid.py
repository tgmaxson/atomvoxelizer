import numpy as np
import pytest

import atomvoxelizer
from atomvoxelizer import FieldVoxelGrid, VectorVoxelGrid


def test_field_grid_supports_scalar_value_shape():
    grid = FieldVoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4), value_shape=())
    assert grid.grid.shape == (4, 4, 4)

    grid.set_sphere([1.5, 1.5, 1.5], radius=0.1, value=2.0)
    grid.add_sphere([1.5, 1.5, 1.5], radius=0.1, value=3.0)
    grid.mul_sphere([1.5, 1.5, 1.5], radius=0.1, factor=2.0)
    grid.div_sphere([1.5, 1.5, 1.5], radius=0.1, factor=5.0)

    np.testing.assert_allclose(grid.grid[1, 1, 1], 2.0)
    np.testing.assert_allclose(grid.scalar_values()[1, 1, 1], 2.0)


def test_field_grid_supports_length_one_scalar_vectors():
    grid = FieldVoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4), value_shape=(1,))
    assert grid.grid.shape == (4, 4, 4, 1)

    grid.set_sphere([1.5, 1.5, 1.5], radius=0.1, value=2.0)
    grid.add_sphere([1.5, 1.5, 1.5], radius=0.1, value=[3.0])
    np.testing.assert_allclose(grid.grid[1, 1, 1], [5.0])
    np.testing.assert_allclose(grid.scalar_values()[1, 1, 1], 5.0)


def test_field_grid_supports_matrix_values():
    grid = FieldVoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4), value_shape=(2, 2))
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    grid.set_sphere([1.5, 1.5, 1.5], radius=0.1, value=matrix)
    grid.add_sphere([1.5, 1.5, 1.5], radius=0.1, value=np.eye(2))
    np.testing.assert_allclose(grid.grid[1, 1, 1], matrix + np.eye(2))

    normalized = grid.normalize_values(inplace=False)
    assert normalized.shape == grid.grid.shape
    assert np.linalg.norm(normalized[1, 1, 1]) == pytest.approx(1.0)


def test_vector_grid_initialization_and_constant_mask():
    grid = VectorVoxelGrid(np.eye(3) * 4.0, gpts=(4, 4, 4), dtype=np.float64)
    assert grid.grid.shape == (4, 4, 4, 3)
    assert grid.grid.dtype == np.dtype(np.float64)

    grid.set_sphere([1.5, 1.5, 1.5], radius=0.1, value=[1.0, 2.0, 3.0])
    np.testing.assert_allclose(grid.grid[1, 1, 1], [1.0, 2.0, 3.0])

    grid.add_sphere([1.5, 1.5, 1.5], radius=0.1, value=[0.5, -1.0, 2.0])
    np.testing.assert_allclose(grid.grid[1, 1, 1], [1.5, 1.0, 5.0])


def test_normal_mask_points_away_from_atom_center():
    grid = VectorVoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))
    grid.set_sphere([2.5, 2.5, 2.5], radius=1.01, mask="normal")

    np.testing.assert_allclose(grid.grid[2, 2, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[3, 2, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[1, 2, 2], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[2, 3, 2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(grid.grid[2, 2, 1], [0.0, 0.0, -1.0])


def test_normal_mask_uses_periodic_wrapping():
    grid = VectorVoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))
    grid.set_sphere([0.5, 0.5, 0.5], radius=1.01, mask="normal")

    np.testing.assert_allclose(grid.grid[4, 0, 0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[1, 0, 0], [1.0, 0.0, 0.0])


def test_add_spheres_normal_mask_can_cancel_and_normalize_safely():
    grid = VectorVoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))
    centers = np.array([[1.5, 2.5, 2.5], [3.5, 2.5, 2.5]])
    radii = np.array([1.01, 1.01])
    grid.add_spheres(centers, radii, mask="normal")

    np.testing.assert_allclose(grid.grid[2, 2, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[0, 2, 2], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.grid[4, 2, 2], [1.0, 0.0, 0.0])

    normalized = grid.normalize_vectors(inplace=False)
    np.testing.assert_allclose(normalized[2, 2, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[0, 2, 2], [-1.0, 0.0, 0.0])

    grid.grid[0, 0, 0] = [2.0, 0.0, 0.0]
    returned = grid.normalize_vectors()
    assert returned is grid
    np.testing.assert_allclose(grid.grid[0, 0, 0], [1.0, 0.0, 0.0])


def test_quiver_slice_data_prepares_2d_vectors_without_plotting():
    grid = VectorVoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))
    grid.set_sphere([2.5, 2.5, 2.5], radius=1.01, mask="normal")

    data = grid.quiver_slice_data(axis="z", index=2, min_norm=0.1, normalize=True)
    assert data["axis"] == "z"
    assert data["index"] == 2
    assert data["axes"] == (0, 1)
    assert len(data["x"]) == 4
    assert len(data["u"]) == len(data["x"])
    assert set(np.round(data["norm"], 6)) == {1.0}


def test_quiver_3d_data_samples_and_normalizes_vectors_without_plotting():
    grid = VectorVoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))
    grid.set_sphere([2.5, 2.5, 2.5], radius=1.01, mask="normal")

    data = grid.quiver_3d_data(stride=1, min_norm=0.1, normalize=True)
    assert len(data["x"]) == 6
    norms = np.linalg.norm(np.column_stack([data["u"], data["v"], data["w"]]), axis=1)
    np.testing.assert_allclose(norms, np.ones(6))


def test_field_grid_rejects_unsupported_shapes_and_masks():
    with pytest.raises(TypeError, match="floating"):
        FieldVoxelGrid(np.eye(3), gpts=(2, 2, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="value_shape"):
        FieldVoxelGrid(np.eye(3), gpts=(2, 2, 2), value_shape=(3, 0))

    grid = FieldVoxelGrid(np.eye(3), gpts=(2, 2, 2), value_shape=(2,))
    with pytest.raises(ValueError, match="value_shape"):
        grid.scalar_values()
    with pytest.raises(ValueError, match="shape"):
        grid.set_sphere([0.5, 0.5, 0.5], radius=0.1, value=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="normal"):
        grid.add_sphere([0.5, 0.5, 0.5], radius=0.1, mask="normal")
    with pytest.raises(ValueError, match="mask"):
        grid.add_sphere([0.5, 0.5, 0.5], radius=0.1, mask="distance")
    with pytest.raises(ValueError, match="value_shape=\\(3,\\)"):
        grid.quiver_3d_data()
    with pytest.raises(ValueError, match="axis"):
        VectorVoxelGrid(np.eye(3), gpts=(2, 2, 2)).quiver_slice_data(axis="bad")
    with pytest.raises(NotImplementedError, match="min_sphere"):
        grid.min_sphere([0.5, 0.5, 0.5], radius=0.1)
    with pytest.raises(NotImplementedError, match="clamp_grid"):
        grid.clamp_grid()
    with pytest.raises(NotImplementedError, match="plot_2D"):
        grid.plot_2D()
    with pytest.raises(NotImplementedError, match="plot_3D"):
        grid.plot_3D()


def test_accelerated_field_grid_names_raise_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        atomvoxelizer.VectorVoxelGridCuPy
    with pytest.raises(NotImplementedError, match="not implemented"):
        atomvoxelizer.FieldVoxelGridTaichi


def test_numba_field_grid_matches_numpy_for_scalar_constant_mask():
    pytest.importorskip("numba")
    from atomvoxelizer import FieldVoxelGridNumba

    cell = np.eye(3) * 4.0
    kwargs = dict(cell=cell, gpts=(4, 4, 4), value_shape=(), dtype=np.float64)
    numpy_grid = FieldVoxelGrid(**kwargs)
    numba_grid = FieldVoxelGridNumba(**kwargs)

    numpy_grid.add_sphere([1.2, 1.3, 1.4], radius=1.1, value=2.5)
    numba_grid.add_sphere([1.2, 1.3, 1.4], radius=1.1, value=2.5)

    np.testing.assert_allclose(numba_grid.grid, numpy_grid.grid)


def test_numba_field_grid_matches_numpy_for_matrix_batch_mask():
    pytest.importorskip("numba")
    from atomvoxelizer import FieldVoxelGridNumba

    cell = np.eye(3) * 5.0
    kwargs = dict(cell=cell, gpts=(5, 5, 5), value_shape=(2, 2), dtype=np.float64)
    numpy_grid = FieldVoxelGrid(**kwargs)
    numba_grid = FieldVoxelGridNumba(**kwargs)
    centers = np.array([[1.2, 1.3, 1.4], [3.2, 3.3, 3.4]])
    radii = np.array([1.1, 1.1])
    value = np.array([[1.0, 2.0], [3.0, 4.0]])

    numpy_grid.set_spheres(centers, radii, value=value)
    numpy_grid.mul_spheres(centers, radii, factor=value)
    numba_grid.set_spheres(centers, radii, value=value)
    numba_grid.mul_spheres(centers, radii, factor=value)

    np.testing.assert_allclose(numba_grid.grid, numpy_grid.grid)


def test_numba_vector_grid_matches_numpy_for_normal_mask():
    pytest.importorskip("numba")
    from atomvoxelizer import VectorVoxelGridNumba

    cell = np.eye(3) * 5.0
    kwargs = dict(cell=cell, gpts=(5, 5, 5), dtype=np.float64)
    numpy_grid = VectorVoxelGrid(**kwargs)
    numba_grid = VectorVoxelGridNumba(**kwargs)
    centers = np.array([[1.2, 1.3, 1.4], [3.2, 3.3, 3.4]])
    radii = np.array([1.1, 1.1])

    numpy_grid.add_spheres(centers, radii, mask="normal")
    numba_grid.add_spheres(centers, radii, mask="normal")

    np.testing.assert_allclose(numba_grid.grid, numpy_grid.grid)
