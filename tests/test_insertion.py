import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atomvoxelizer import (
    LJParameters,
    UFF_ELEMENTS,
    insertion_probabilities,
    lj_insertion_grid,
    uff_parameters,
)


def test_uff_units_and_reference_values():
    assert len(UFF_ELEMENTS) == 31
    for symbol, distance, depth in [
        ("Ar", 3.868, 0.185),
        ("Al", 4.499, 0.505),
        ("Na", 2.983, 0.030),
    ]:
        p = uff_parameters(symbol)
        assert p.sigma * 2 ** (1 / 6) == pytest.approx(distance)
        assert p.epsilon == pytest.approx(depth * 0.0433641153087705)
    with pytest.raises(KeyError, match="custom"):
        uff_parameters("Unknown")


@pytest.mark.parametrize("mixing", ["geometric", "lorentz-berthelot"])
def test_direct_pair_sum_off_grid_wrapping_and_chunking(mixing):
    cell = np.diag([12.0, 14.0, 16.0])
    positions = np.array([[0.23, 6.2, 9.3], [11.8, 4.1, 3.6]])
    parameters = {"host": LJParameters(2.0, 0.02), "guest": LJParameters(3.0, 0.08)}
    grid = lj_insertion_grid(
        cell,
        positions,
        ["host"] * 2,
        "guest",
        gpts=(6, 7, 8),
        cutoff=5.0,
        parameters=parameters,
        mixing=mixing,
        chunk_size=13,
    )
    sigma = np.sqrt(6.0) if mixing == "geometric" else 2.5
    expected = np.zeros_like(grid.grid)
    for point in np.ndindex(*grid.gpts):
        center = ((np.array(point) + 0.5) / grid.gpts) @ cell
        for atom in positions:
            delta = center - atom
            delta -= np.rint(delta / np.diag(cell)) * np.diag(cell)
            r = np.linalg.norm(delta)
            if r < 5.0:
                expected[point] += 4 * 0.04 * ((sigma / r) ** 12 - (sigma / r) ** 6)
    np.testing.assert_allclose(grid.grid, expected, rtol=1e-11, atol=1e-13)
    shifted = lj_insertion_grid(
        cell,
        positions + cell[0] - 2 * cell[1],
        ["host"] * 2,
        "guest",
        gpts=(6, 7, 8),
        cutoff=5.0,
        parameters=parameters,
        mixing=mixing,
    )
    np.testing.assert_allclose(shifted.grid, expected, rtol=1e-11, atol=1e-13)


def test_minimum_overlap_cutoff_and_zero_epsilon():
    # Voxel centers at .5, 1.5, ...; host is exactly at the first center.
    params = {"X": LJParameters(2 / 2 ** (1 / 6), 0.1)}
    grid = lj_insertion_grid(
        np.eye(3) * 10,
        [[0.5, 0.5, 0.5]],
        ["X"],
        "X",
        gpts=(10, 10, 10),
        cutoff=3,
        parameters=params,
    )
    assert np.isposinf(grid.grid[0, 0, 0])
    assert grid.grid[2, 0, 0] == pytest.approx(-0.1)
    assert grid.grid[3, 0, 0] == 0
    assert grid.grid[8, 0, 0] == pytest.approx(-0.1)
    params["X"] = LJParameters(2, 0)
    grid = lj_insertion_grid(
        np.eye(3) * 10,
        [[0.5, 0.5, 0.5]],
        ["X"],
        "X",
        gpts=(10, 10, 10),
        parameters=params,
    )
    assert np.all(grid.grid == 0)


def test_rotated_orthogonal_cell():
    rotation = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]])
    cell = np.diag([10.0, 12.0, 14.0])
    atoms = np.array([[1.3, 2.6, 4.1]])
    a = lj_insertion_grid(cell, atoms, ["C"], gpts=(5, 6, 7))
    b = lj_insertion_grid(cell @ rotation, atoms @ rotation, ["C"], gpts=(5, 6, 7))
    np.testing.assert_allclose(a.grid, b.grid, rtol=1e-12, atol=1e-12)


def test_probabilities_stability_ratios_support_and_exclusion():
    energy = np.array([-1000.0, -999.99, 1000.0, np.inf])
    p = insertion_probabilities(energy, 300.0, uniform_fraction=0)
    assert p.sum() == pytest.approx(1)
    assert p[1] / p[0] == pytest.approx(np.exp(-0.01 / (8.617333262145e-5 * 300)))
    p = insertion_probabilities(energy, 300.0)
    assert np.all(p[:3] > 0)
    assert p[3] == 0
    np.testing.assert_allclose(p, insertion_probabilities(energy + 400, 300))
    np.testing.assert_allclose(insertion_probabilities(np.zeros((2, 3)), 300), 1 / 6)
    assert np.all(
        lj_insertion_grid(np.eye(3) * 10, np.empty((0, 3)), [], gpts=(2, 2, 2)).grid
        == 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoff": 6},
        {"cutoff": np.nan},
        {"chunk_size": 0},
        {"gpts": (2, 0, 2)},
        {"gpts": (2, 1.5, 2)},
        {"mixing": "invalid"},
        {"resolution": -1, "gpts": None},
        {"cell": [[10, 1, 0], [0, 10, 0], [0, 0, 10]]},
        {"positions": [[np.nan, 0, 0]]},
    ],
)
def test_invalid_grids(kwargs):
    arguments = dict(
        cell=np.eye(3) * 10, positions=[[0, 0, 0]], symbols=["C"], gpts=(2, 2, 2)
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        lj_insertion_grid(**arguments)


@pytest.mark.parametrize(
    "energy,temperature,fraction",
    [
        ([np.inf], 300, 0.05),
        ([np.nan], 300, 0.05),
        ([-np.inf], 300, 0.05),
        ([0], 0, 0.05),
        ([0], np.nan, 0.05),
        ([0], 300, 1.1),
    ],
)
def test_invalid_probabilities(energy, temperature, fraction):
    with pytest.raises(ValueError):
        insertion_probabilities(energy, temperature, uniform_fraction=fraction)


@pytest.mark.parametrize("sigma,epsilon", [(0, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_parameters(sigma, epsilon):
    with pytest.raises(ValueError):
        LJParameters(sigma, epsilon)


def test_offline_example(tmp_path):
    script = Path(__file__).parents[1] / "examples/insertion/uff_argon.py"
    output = tmp_path / "bias.npz"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--resolution",
            "2",
            "--samples",
            "20",
            "--output",
            str(output),
            "--plot",
            str(tmp_path / "slice.png"),
        ],
        check=True,
    )
    with np.load(output) as data:
        assert data["probability"].sum() == pytest.approx(1)
        assert data["samples_angstrom"].shape == (20, 3)
        assert np.all((data["samples_angstrom"] >= 0) & (data["samples_angstrom"] < 16))
        indices = (data["samples_angstrom"] / 16 * data["gpts"]).astype(int)
        expected = data["probability"][tuple(indices.T)] / (16**3 / 8**3)
        np.testing.assert_allclose(data["proposal_density_per_angstrom3"], expected)
