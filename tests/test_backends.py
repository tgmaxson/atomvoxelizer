import importlib.util

import numpy as np
import pytest

from atomvoxelizer import VoxelGrid
from atomvoxelizer.voxelgrid import VoxelGridNumPy


def build_reference_grid(cls):
    grid = cls(np.eye(3) * 6.0, gpts=(6, 6, 6))
    centers = np.array(
        [
            [1.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
            [5.8, 5.8, 5.8],
        ]
    )
    radii = np.array([1.01, 0.51, 1.01])
    grid.add_spheres(centers, radii, value=1.0)
    grid.set_sphere(np.array([1.5, 1.5, 1.5]), 0.1, value=-2.0)
    grid.mul_sphere(np.array([4.5, 1.5, 1.5]), 0.1, factor=3.0)
    grid.div_sphere(np.array([4.5, 1.5, 1.5]), 0.1, factor=2.0)
    grid.clamp_grid(-1.0, 2.0)
    return grid.to_numpy()


def test_default_voxelgrid_matches_numpy_backend():
    assert VoxelGrid.backend_name == "numpy"
    np.testing.assert_allclose(build_reference_grid(VoxelGrid), build_reference_grid(VoxelGridNumPy))


def test_numba_backend_matches_numpy_backend_when_installed():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is not installed")

    from atomvoxelizer.numba_backend import VoxelGridNumba

    assert VoxelGridNumba.backend_name == "numba"
    np.testing.assert_allclose(build_reference_grid(VoxelGridNumba), build_reference_grid(VoxelGridNumPy))


def test_cupy_backend_matches_numpy_backend_when_installed():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")

    from atomvoxelizer import VoxelGridCuPy

    cupy_grid = build_reference_grid(VoxelGridCuPy)
    np.testing.assert_allclose(cupy_grid, build_reference_grid(VoxelGridNumPy))


def test_taichi_cpu_backend_matches_numpy_backend_when_installed():
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")

    from atomvoxelizer import VoxelGridTaichi

    assert VoxelGridTaichi.backend_name == "taichi-cpu"
    np.testing.assert_allclose(build_reference_grid(VoxelGridTaichi), build_reference_grid(VoxelGridNumPy))
