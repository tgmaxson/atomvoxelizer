"""Check the sampling example's density correction independently of its target."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atomvoxelizer import lj_insertion_grid

SCRIPT = Path(__file__).parents[1] / "benchmarks/benchmark_insertion_sampling.py"
spec = importlib.util.spec_from_file_location("sampling_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_continuous_evaluator_matches_voxel_centers():
    cell, atoms = benchmark.carbon_cage()
    grid = lj_insertion_grid(cell, atoms, ["C"] * len(atoms), resolution=1, cutoff=8)
    direct = benchmark.direct_energy(
        grid.voxel_centers().reshape(-1, 3), cell, atoms, 8
    )
    np.testing.assert_allclose(direct.reshape(grid.grid.shape), grid.grid, rtol=1e-12)


def test_density_correction_integrates_constant_target_exactly():
    # Two unequal-probability voxels in a non-unit-volume cell.  Without the
    # V*q correction this expectation (and the physical units) would be wrong.
    cell = np.diag([4.0, 3.0, 2.0])
    probability = np.array([0.8, 0.2])
    points = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    density = probability / (np.linalg.det(cell) / 2)
    contribution, _ = benchmark.importance_contributions(
        points, density, cell, [], 300, 1
    )
    assert np.sum(probability * contribution) == pytest.approx(1)
    np.testing.assert_allclose(contribution, 1 / (2 * probability))


def test_draws_return_correct_piecewise_density():
    probability = np.array([0.8, 0.2]).reshape(2, 1, 1)
    cell = np.diag([4.0, 3.0, 2.0])
    points, density = benchmark.draw_proposals(
        np.random.default_rng(0), probability, cell, 2000
    )
    assert np.all((points >= 0) & (points < np.diag(cell)))
    indices = (points / np.diag(cell) * probability.shape).astype(int)
    expected = probability[tuple(indices.T)] / (np.linalg.det(cell) / 2)
    np.testing.assert_allclose(density, expected)
    # Checks selection uses p, rather than uniform voxels with a p-labelled q.
    assert np.mean(indices[:, 0] == 0) == pytest.approx(0.8, abs=0.04)


def test_benchmark_cli_and_corrected_estimates(tmp_path):
    output = tmp_path / "results.json"
    plot = tmp_path / "results.png"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--resolutions",
            "1",
            "--temperatures",
            "77",
            "--sample-counts",
            "256",
            "1024",
            "--repeats",
            "16",
            "--reference-resolutions",
            "0.5",
            "0.25",
            "--output",
            str(output),
            "--plot",
            str(plot),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert plot.stat().st_size > 0
    entry = result["temperatures"][0]
    reference = entry["quadrature"][-1]["value"]
    quadrature_change = abs(reference - entry["quadrature"][0]["value"])
    for method in entry["methods"]:
        row = method["estimates"][-1]
        assert (
            abs(row["estimate"] - reference)
            < 6 * row["standard_error"] + quadrature_change
        )
        assert method["variance_reduction"] > 0
