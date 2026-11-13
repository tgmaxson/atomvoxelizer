"""Generic radial masks are geometry primitives, independent of LJ."""

from itertools import product
import tracemalloc

import numpy as np
import pytest

from atomvoxelizer import VoxelGrid, lj_insertion_grid, lj_pair_energy, uff_parameters


@pytest.mark.parametrize("radius", [2.0, 7.0])
def test_callable_mask_matches_cartesian_periodic_image_sum(radius):
    cell = np.array([[8.0, 0.0, 0.0], [7.0, 3.0, 0.0], [1.0, 0.5, 7.0]])
    center = np.array([0.13, 0.92, 0.27]) @ cell
    grid = VoxelGrid(cell, gpts=(5, 4, 6), dtype=np.float64)

    def profile(r):
        return np.exp(-r * r / 3)

    grid.add_sphere(center, radius, mask=profile, chunk_size=17)
    expected = np.zeros_like(grid.grid)
    for image in product(range(-5, 6), repeat=3):
        distance = np.linalg.norm(
            grid.voxel_centers() - center - np.array(image) @ cell, axis=-1
        )
        selected = distance <= radius
        expected[selected] += profile(distance[selected])
    np.testing.assert_allclose(grid.grid, expected, atol=1e-13, rtol=1e-12)
    original = grid.grid.copy()
    grid.add_sphere(center, radius, value=-1, mask=profile, chunk_size=17)
    np.testing.assert_allclose(grid.grid, 0, atol=1e-15)
    grid.add_sphere(center + 2 * cell[0] - cell[1], radius, mask=profile)
    np.testing.assert_allclose(grid.grid, original, atol=1e-13, rtol=1e-12)


def test_callback_chunks_only_cover_local_sphere_and_scale_values():
    grid = VoxelGrid(np.eye(3) * 64, gpts=(64, 64, 64), dtype=np.float64)
    batches = []

    def mask(r):
        batches.append(len(r))
        return r * r

    grid.add_sphere([20.2, 21.3, 22.4], 3, value=-2, mask=mask, chunk_size=11)
    assert batches and max(batches) <= 11 and sum(batches) < 200
    for indices, distances in grid.iter_sphere_samples([20.2, 21.3, 22.4], 3):
        np.testing.assert_allclose(grid.grid[indices], -2 * distances**2)
    assert np.count_nonzero(grid.grid) == sum(batches)


def test_in_place_mask_does_not_allocate_a_box_sized_contribution():
    grid = VoxelGrid(np.eye(3) * 64, gpts=(64, 64, 64), dtype=np.float64)
    pair = uff_parameters("Ar")
    mask = lambda r: lj_pair_energy(r, pair, 3, shift=True, energy_cap=10)
    tracemalloc.start()
    grid.add_sphere([20.2, 21.3, 22.4], 3, mask=mask)
    _, in_place_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    contribution = lj_insertion_grid(
        grid.cell,
        [[20.2, 21.3, 22.4]],
        ["Ar"],
        gpts=grid.gpts,
        cutoff=3,
        shift=True,
        energy_cap=10,
    )
    _, dense_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.testing.assert_allclose(grid.grid, contribution.grid, atol=1e-12, rtol=1e-12)
    assert in_place_peak < grid.grid.nbytes / 4
    assert dense_peak >= grid.grid.nbytes


@pytest.mark.parametrize(
    "kwargs", [{"center": [np.nan, 0, 0]}, {"radius": -1}, {"chunk_size": 0}]
)
def test_invalid_radial_geometry(kwargs):
    grid = VoxelGrid(np.eye(3) * 5, gpts=(5, 5, 5))
    arguments = dict(center=[1, 2, 3], radius=1, mask=lambda r: 1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        grid.add_sphere(**arguments)
    assert not grid.grid.any()
