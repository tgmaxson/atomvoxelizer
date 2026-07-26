import importlib.util

import numpy as np
import pytest

from atomvoxelizer import VoxelGrid


pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba is not installed")


def make_matching_grids(dtype=np.float64):
    from atomvoxelizer.numba_backend import VoxelGridNumba

    cell = np.array(
        [
            [6.0, 0.2, 0.0],
            [0.0, 5.5, 0.1],
            [0.0, 0.0, 5.0],
        ]
    )
    kwargs = dict(cell=cell, gpts=(6, 5, 5), dtype=dtype)
    return VoxelGrid(**kwargs), VoxelGridNumba(**kwargs)


@pytest.mark.parametrize("mask", ["constant", "distance"])
@pytest.mark.parametrize("operation", ["set_sphere", "add_sphere", "mul_sphere", "div_sphere", "min_sphere"])
def test_numba_single_sphere_operations_match_numpy(operation, mask):
    numpy_grid, numba_grid = make_matching_grids()
    center = np.array([5.8, 0.3, 4.9])
    radius = 1.35

    if operation in {"mul_sphere", "div_sphere"}:
        numpy_grid.grid.fill(3.0)
        numba_grid.grid.fill(3.0)
        kwargs = {"factor": 1.5}
    else:
        if operation == "min_sphere":
            numpy_grid.grid.fill(10.0)
            numba_grid.grid.fill(10.0)
        kwargs = {"value": 1.5}

    if operation == "div_sphere" and mask == "distance":
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            getattr(numpy_grid, operation)(center, radius, mask=mask, **kwargs)
    else:
        getattr(numpy_grid, operation)(center, radius, mask=mask, **kwargs)
    getattr(numba_grid, operation)(center, radius, mask=mask, **kwargs)

    np.testing.assert_allclose(numba_grid.grid, numpy_grid.grid)


@pytest.mark.parametrize("mask", ["constant", "distance"])
@pytest.mark.parametrize("operation", ["set_spheres", "add_spheres", "mul_spheres", "div_spheres", "min_spheres"])
def test_numba_batched_sphere_operations_match_numpy(operation, mask):
    numpy_grid, numba_grid = make_matching_grids()
    centers = np.array(
        [
            [1.1, 1.2, 1.3],
            [4.9, 4.8, 4.7],
            [5.8, 0.3, 4.9],
        ]
    )
    radii = np.array([1.35, 0.95, 1.35])

    if operation in {"mul_spheres", "div_spheres"}:
        numpy_grid.grid.fill(3.0)
        numba_grid.grid.fill(3.0)
        kwargs = {"factor": 1.5}
    else:
        if operation == "min_spheres":
            numpy_grid.grid.fill(10.0)
            numba_grid.grid.fill(10.0)
        kwargs = {"value": 1.5}

    if operation == "div_spheres" and mask == "distance":
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            getattr(numpy_grid, operation)(centers, radii, mask=mask, **kwargs)
    else:
        getattr(numpy_grid, operation)(centers, radii, mask=mask, **kwargs)
    getattr(numba_grid, operation)(centers, radii, mask=mask, **kwargs)

    np.testing.assert_allclose(numba_grid.grid, numpy_grid.grid)


def test_numba_backend_preserves_complex_arithmetic_and_ordered_rejections():
    from atomvoxelizer.numba_backend import VoxelGridNumba

    grid = VoxelGridNumba(np.eye(3) * 4.0, gpts=(4, 4, 4), dtype=np.complex128)
    center = np.array([1.5, 1.5, 1.5])

    grid.set_sphere(center, radius=0.1, value=1.0 + 2.0j)
    grid.add_sphere(center, radius=0.1, value=2.0 - 1.0j)
    grid.mul_sphere(center, radius=0.1, factor=2.0)
    grid.div_sphere(center, radius=0.1, factor=2.0)
    assert grid.grid[1, 1, 1] == pytest.approx(3.0 + 1.0j)

    with pytest.raises(TypeError, match="complex"):
        grid.min_sphere(center, radius=0.1)
    with pytest.raises(TypeError, match="complex"):
        grid.min_spheres(np.array([center]), np.array([0.1]))
    with pytest.raises(TypeError, match="complex"):
        grid.clamp_grid()


def test_numba_backend_validates_masks_and_sphere_inputs():
    from atomvoxelizer.numba_backend import VoxelGridNumba

    grid = VoxelGridNumba(np.eye(3) * 4.0, gpts=(4, 4, 4))
    with pytest.raises(ValueError, match="mask"):
        grid.add_sphere([1.0, 1.0, 1.0], radius=0.5, mask="bad")
    with pytest.raises(ValueError, match="centers"):
        grid.add_spheres(np.array([1.0, 1.0, 1.0]), np.array([0.5]))
    with pytest.raises(ValueError, match="radii"):
        grid.add_spheres(np.array([[1.0, 1.0, 1.0]]), np.array([0.5, 0.6]))


def test_numba_backend_applies_ordered_sphere_operation_sequence():
    from atomvoxelizer.numba_backend import VoxelGridNumba

    cell = np.eye(3) * 6.0
    centers = np.array([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5], [5.8, 5.8, 5.8]])
    outer = np.array([1.2, 0.8, 1.2])
    inner = np.array([0.6, 0.4, 0.6])

    numba_grid = VoxelGridNumba(cell, gpts=(6, 6, 6))
    numba_grid.apply_sphere_operations(
        [
            {"operation": "add", "centers": centers, "radii": outer, "value": 1.0},
            {"operation": "set", "centers": centers, "radii": inner, "value": -1.0},
            {"operation": "add", "centers": centers[:2], "radii": inner[:2], "value": 0.5, "mask": "distance"},
        ]
    )

    numpy_grid = VoxelGrid(cell, gpts=(6, 6, 6))
    numpy_grid.add_spheres(centers, outer, value=1.0)
    numpy_grid.set_spheres(centers, inner, value=-1.0)
    numpy_grid.add_spheres(centers[:2], inner[:2], value=0.5, mask="distance")
    np.testing.assert_allclose(numba_grid.to_numpy(), numpy_grid.to_numpy())
