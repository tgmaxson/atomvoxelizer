import importlib.util

import numpy as np
import pytest

from atomvoxelizer import VoxelGrid, VoxelGridAnalysis
from atomvoxelizer.voxelgrid import VoxelGridNumPy


def _fill_grid(grid, value):
    if hasattr(grid.grid, "fill"):
        grid.grid.fill(value)
    else:
        grid.grid[...] = value


def build_distance_grid(cls):
    grid = cls(np.eye(3) * 6.0, gpts=(6, 6, 6))
    _fill_grid(grid, np.inf)
    centers = np.array([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5]])
    radii = np.array([2.1, 2.1])
    grid.min_spheres(centers, radii, mask="distance")
    return grid.to_numpy()


def test_distance_mask_stores_real_space_distances():
    grid = VoxelGrid(np.eye(3) * 5.0, gpts=(5, 5, 5))

    grid.set_sphere([2.0, 2.0, 2.0], radius=1.01, mask="distance")

    assert grid.grid[2, 2, 2] == pytest.approx(0.0)
    assert grid.grid[3, 2, 2] == pytest.approx(1.0)
    assert grid.grid[2, 3, 2] == pytest.approx(1.0)
    assert grid.grid[2, 2, 3] == pytest.approx(1.0)


def test_min_spheres_distance_mask_keeps_nearest_atom_distance():
    values = build_distance_grid(VoxelGrid)

    assert values[1, 1, 1] == pytest.approx(0.0)
    assert values[4, 1, 1] == pytest.approx(0.0)
    assert values[2, 1, 1] == pytest.approx(1.0)


def test_invalid_mask_is_rejected():
    grid = VoxelGrid(np.eye(3), gpts=(3, 3, 3))

    with pytest.raises(ValueError, match="mask"):
        grid.add_sphere([0.5, 0.5, 0.5], radius=0.5, mask="nearest")


def test_numba_distance_mask_matches_numpy_when_installed():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is not installed")

    from atomvoxelizer.numba_backend import VoxelGridNumba

    np.testing.assert_allclose(build_distance_grid(VoxelGridNumba), build_distance_grid(VoxelGridNumPy))


def test_taichi_distance_mask_matches_numpy_when_installed():
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")

    from atomvoxelizer import VoxelGridTaichi

    np.testing.assert_allclose(build_distance_grid(VoxelGridTaichi), build_distance_grid(VoxelGridNumPy))


def test_cupy_distance_mask_matches_numpy_when_installed():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")

    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but no CUDA device is available: {exc}")

    from atomvoxelizer import VoxelGridCuPy

    np.testing.assert_allclose(build_distance_grid(VoxelGridCuPy), build_distance_grid(VoxelGridNumPy))


@pytest.mark.skipif(importlib.util.find_spec("skimage") is None, reason="scikit-image is not installed")
def test_analysis_mesh_at_value_traces_distance_surface():
    grid = VoxelGrid(np.eye(3) * 8.0, gpts=(16, 16, 16))
    grid.grid.fill(np.inf)
    grid.min_sphere([4.0, 4.0, 4.0], radius=3.0, mask="distance")

    analysis = VoxelGridAnalysis(grid)
    vertices, faces = analysis.mesh_at_value(1.5, periodic=True)

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 0
    assert np.all(vertices >= -1.0e-8)
    assert np.all(vertices <= 8.0 + 1.0e-8)
    assert analysis.mesh_surface_area(vertices, faces) > 0.0
