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


def test_numba_backend_preserves_selected_dtype_when_installed():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is not installed")

    from atomvoxelizer.numba_backend import VoxelGridNumba

    grid = VoxelGridNumba(np.eye(3) * 4.0, gpts=(4, 4, 4), dtype=np.int32)
    grid.add_sphere(np.array([1.5, 1.5, 1.5]), radius=0.1, value=3)
    assert grid.grid.dtype == np.dtype(np.int32)
    assert grid.grid[1, 1, 1] == 3


def test_cpp_backend_matches_numpy_backend_when_built():
    if importlib.util.find_spec("atomvoxelizer._cpp_backend") is None:
        pytest.skip("AtomVoxelizer C++ extension is not built")

    from atomvoxelizer import VoxelGridCpp

    assert VoxelGridCpp.backend_name == "cpp"
    np.testing.assert_allclose(build_reference_grid(VoxelGridCpp), build_reference_grid(VoxelGridNumPy))


def test_cpp_backend_distance_masks_match_numpy_when_built():
    if importlib.util.find_spec("atomvoxelizer._cpp_backend") is None:
        pytest.skip("AtomVoxelizer C++ extension is not built")

    from atomvoxelizer import VoxelGridCpp

    cell = np.eye(3) * 5.0
    centers = np.array([[1.5, 1.5, 1.5], [3.5, 3.5, 3.5]])
    radii = np.array([1.01, 1.01])
    cpp_grid = VoxelGridCpp(cell, gpts=(5, 5, 5))
    numpy_grid = VoxelGridNumPy(cell, gpts=(5, 5, 5))

    cpp_grid.grid.fill(np.inf)
    numpy_grid.grid.fill(np.inf)
    cpp_grid.min_spheres(centers, radii, value=2.0, mask="distance")
    numpy_grid.min_spheres(centers, radii, value=2.0, mask="distance")
    np.testing.assert_allclose(cpp_grid.to_numpy(), numpy_grid.to_numpy())


def test_cpp_backend_preserves_selected_dtype_when_built():
    if importlib.util.find_spec("atomvoxelizer._cpp_backend") is None:
        pytest.skip("AtomVoxelizer C++ extension is not built")

    from atomvoxelizer import VoxelGridCpp

    grid = VoxelGridCpp(np.eye(3) * 4.0, gpts=(4, 4, 4), dtype=np.int32)
    grid.add_sphere(np.array([1.5, 1.5, 1.5]), radius=0.1, value=3)
    assert grid.grid.dtype == np.dtype(np.int32)
    assert grid.grid[1, 1, 1] == 3

    with pytest.raises(TypeError, match="complex"):
        VoxelGridCpp(np.eye(3), gpts=(2, 2, 2), dtype=np.complex64)


def test_cpp_backend_applies_ordered_sphere_operation_sequence_when_built():
    if importlib.util.find_spec("atomvoxelizer._cpp_backend") is None:
        pytest.skip("AtomVoxelizer C++ extension is not built")

    from atomvoxelizer import VoxelGridCpp

    cell = np.eye(3) * 6.0
    centers = np.array([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5], [5.8, 5.8, 5.8]])
    outer = np.array([1.2, 0.8, 1.2])
    inner = np.array([0.6, 0.4, 0.6])

    cpp_grid = VoxelGridCpp(cell, gpts=(6, 6, 6))
    cpp_grid.apply_sphere_operations(
        [
            {"operation": "add", "centers": centers, "radii": outer, "value": 1.0},
            {"operation": "set", "centers": centers, "radii": inner, "value": -1.0},
            {"operation": "add", "centers": centers[:2], "radii": inner[:2], "value": 0.5, "mask": "distance"},
        ]
    )

    numpy_grid = VoxelGridNumPy(cell, gpts=(6, 6, 6))
    numpy_grid.add_spheres(centers, outer, value=1.0)
    numpy_grid.set_spheres(centers, inner, value=-1.0)
    numpy_grid.add_spheres(centers[:2], inner[:2], value=0.5, mask="distance")
    np.testing.assert_allclose(cpp_grid.to_numpy(), numpy_grid.to_numpy())


def test_cupy_backend_matches_numpy_backend_when_installed():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")

    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but no CUDA device is available: {exc}")

    from atomvoxelizer import VoxelGridCuPy

    cupy_grid = build_reference_grid(VoxelGridCuPy)
    np.testing.assert_allclose(cupy_grid, build_reference_grid(VoxelGridNumPy))


def test_cupy_cell_colored_add_matches_numpy_when_installed():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")

    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but no CUDA device is available: {exc}")

    from atomvoxelizer import VoxelGridCuPy

    cell = np.eye(3) * 8.0
    centers = np.array(
        [
            [1.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
            [7.6, 7.6, 7.6],
            [2.5, 5.5, 4.5],
        ]
    )
    radii = np.array([1.01, 1.51, 1.01, 1.51])
    numpy_grid = VoxelGridNumPy(cell, gpts=(8, 8, 8))
    numpy_grid.add_spheres(centers, radii, value=1.0)

    cupy_grid = VoxelGridCuPy(cell, gpts=(8, 8, 8))
    cupy_grid.add_spheres_cell_colored(centers, radii, value=1.0)
    np.testing.assert_allclose(cupy_grid.to_numpy(), numpy_grid.to_numpy())


def test_taichi_cpu_backend_matches_numpy_backend_when_installed():
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")

    from atomvoxelizer import VoxelGridTaichi

    assert VoxelGridTaichi.backend_name == "taichi-cpu"
    np.testing.assert_allclose(build_reference_grid(VoxelGridTaichi), build_reference_grid(VoxelGridNumPy))
