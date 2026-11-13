import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from atomvoxelizer import (
    LJParameters,
    lj_insertion_grid,
    lj_pair_energy,
    uff_parameters,
)

SCRIPT = Path(__file__).parents[1] / "examples/insertion/argon_gcmc.py"
spec = importlib.util.spec_from_file_location("argon_gcmc", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_cap_zero_distance_and_strict_continuous_cutoff():
    pair = uff_parameters("Ar")
    cutoff = 8.0
    distances = np.array(
        [0.0, 1e-300, 1.0, 3.868, 7.0, np.nextafter(cutoff, 0), cutoff, 9.0, np.inf]
    )
    energy = lj_pair_energy(distances, pair, cutoff, shift=True, energy_cap=10.0)
    assert np.all(np.isfinite(energy))
    np.testing.assert_array_equal(energy[:3], 10.0)
    np.testing.assert_array_equal(energy[-3:], 0.0)
    assert abs(energy[5]) < 1e-16
    unbounded = lj_pair_energy(distances[3:6], pair, cutoff, shift=True)
    np.testing.assert_array_equal(energy[3:6], unbounded)
    assert energy[3] < 0 and energy[4] < 0
    assert float(lj_pair_energy(0, pair, cutoff, shift=True, energy_cap=10)) == 10
    assert float(lj_pair_energy(cutoff, pair, cutoff, shift=True, energy_cap=10)) == 0


def test_pair_shift_is_analytic_and_cap_is_per_pair():
    pair = LJParameters(2.0, 0.1)
    r, cutoff = 3.0, 5.0
    raw = lambda x: 4 * 0.1 * ((2 / x) ** 12 - (2 / x) ** 6)
    assert lj_pair_energy(r, pair, cutoff, shift=True) == pytest.approx(
        raw(r) - raw(cutoff)
    )
    grid = lj_insertion_grid(
        np.eye(3) * 10,
        [[0.5, 0.5, 0.5]] * 2,
        ["Ar"] * 2,
        gpts=(10, 10, 10),
        cutoff=4,
        shift=True,
        energy_cap=10,
    )
    assert grid.grid[0, 0, 0] == 20  # No clipping of the accumulated field.
    assert np.all(np.isfinite(grid.grid))
    assert grid.grid[4, 0, 0] == 0.0  # Exactly on the 4 A cutoff.
    assert grid.grid[5, 0, 0] == 0.0  # Outside both periodic cutoff spheres.
    zero = lj_pair_energy([0, 1, 2], LJParameters(1, 0), 2, shift=True, energy_cap=10)
    np.testing.assert_array_equal(zero, 0)


@pytest.mark.parametrize(
    "options",
    [
        {"energy_cap": 0},
        {"energy_cap": np.inf},
        {"energy_cap": np.nan},
        {"cutoff": 0},
        {"cutoff": np.nan},
    ],
)
def test_pair_invalid_options(options):
    args = dict(cutoff=8.0, energy_cap=10.0)
    args.update(options)
    with pytest.raises(ValueError):
        lj_pair_energy([1], uff_parameters("Ar"), **args)


@pytest.mark.parametrize("distance", [-1, np.nan])
def test_pair_invalid_distance(distance):
    with pytest.raises(ValueError):
        lj_pair_energy(distance, uff_parameters("Ar"), 8)


def test_shifted_grid_matches_independent_pair_evaluation():
    mc = module.ArgonMC(length=12, resolution=1, cutoff=5)
    points = np.array([[0.3, 2.4, 4.5], [11.8, 9.7, 3.1]])
    grid = mc.field(points)
    centers = mc.grid.voxel_centers()
    expected = np.zeros_like(grid)
    for position in points:
        delta = centers - position
        delta -= 12 * np.rint(delta / 12)
        distances = np.linalg.norm(delta, axis=-1)
        inside = (distances > 0) & (distances < 5)
        sigma, epsilon = mc.pair.sigma, mc.pair.epsilon
        offset = 4 * epsilon * ((sigma / 5) ** 12 - (sigma / 5) ** 6)
        energy = (
            4
            * epsilon
            * ((sigma / distances[inside]) ** 12 - (sigma / distances[inside]) ** 6)
            - offset
        )
        expected[inside] += np.minimum(energy, 10)
    np.testing.assert_allclose(grid, expected, rtol=1e-11, atol=1e-13)


def test_iterative_mc_matches_full_rebuild_without_resets():
    mc = module.ArgonMC(length=12, resolution=1, cutoff=5, activity=0.003, seed=19)
    for step in range(1, 501):
        previous_grid = mc.grid.grid.copy()
        previous_positions = mc.positions.copy()
        accepted = mc.step()
        if not accepted:
            np.testing.assert_array_equal(mc.grid.grid, previous_grid)
            np.testing.assert_array_equal(mc.positions, previous_positions)
        if step % 10 == 0:
            audit = mc.audit(step)
            assert audit["max_abs_ev"] < 2e-11
            assert audit["low_energy_rms_ev"] < 1e-12
            assert audit["probability_l1"] < 1e-10
    assert mc.accepts["insert"] > 10
    assert mc.accepts["delete"] > 10
    assert np.all(np.isfinite(mc.grid.grid))


def test_repeated_add_remove_retains_weak_attractions():
    mc = module.ArgonMC(length=12, resolution=1, cutoff=5)
    baseline = mc.field(np.array([[1.23, 2.34, 3.45]]))
    grid = baseline.copy()
    rng = np.random.default_rng(12)
    for _ in range(1000):
        addition = mc.field(rng.random((1, 3)) * 12)
        grid += addition
        grid -= addition
    assert np.max(np.abs(grid - baseline)) < 2e-12
    weak = (baseline < 0) & (baseline > -0.001)
    assert weak.any()
    np.testing.assert_allclose(grid[weak], baseline[weak], atol=2e-12, rtol=0)


def test_ideal_gas_birth_death_ratios_obey_poisson_balance():
    # For U=0 the field stays zero and proposals are uniform. The stationary
    # number distribution must satisfy P(N+1)/P(N) = z*V/(N+1).
    mc = module.ArgonMC(length=10, resolution=5, cutoff=4, pair=LJParameters(1, 0))
    volume = mc.length**3
    for n in range(6):
        q = mc.density(mc.probability(), np.array([1.0, 2.0, 3.0]))
        birth_ratio = mc.activity / ((n + 1) * q)
        death_ratio = (n + 1) * q / mc.activity
        assert min(1, birth_ratio) / min(1, death_ratio) == pytest.approx(
            mc.activity * volume / (n + 1)
        )
    # Exercise the actual kernel against the analytically known ideal gas.
    counts = []
    for step in range(16000):
        mc.step()
        if step > 1000:
            counts.append(len(mc.positions))
    assert np.mean(counts) == pytest.approx(mc.activity * volume, abs=0.12)
    np.testing.assert_array_equal(mc.grid.grid, 0)


def test_example_report_contains_drift_and_frozen_cutoff_comparison(tmp_path):
    report, output, plot = [
        tmp_path / name for name in ["run.json", "run.npz", "run.png"]
    ]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--steps",
            "100",
            "--audit-every",
            "10",
            "--resolution",
            "2",
            "--output",
            str(output),
            "--report",
            str(report),
            "--plot",
            str(plot),
        ],
        check=True,
    )
    data = json.loads(report.read_text())
    assert len(data["drift"]) == 11
    assert data["drift"][0]["atoms"] == 0
    assert max(row["max_abs_ev"] for row in data["drift"]) < 2e-11
    assert all(row["energy_at_cutoff_ev"] == 0 for row in data["cutoff_comparison"])
    assert data["cutoff_comparison"][-1]["rms_ev"] == 0
    with np.load(output) as saved:
        np.testing.assert_allclose(
            saved["incremental_grid_ev"], saved["rebuilt_grid_ev"], atol=2e-11
        )
    assert plot.stat().st_size > 0


def test_deletion_uses_reverse_density_without_deleted_atom():
    class DeleteRNG:
        def __init__(self):
            self.values = iter([0.9, 0.2])

        def random(self):
            return next(self.values)

        def integers(self, n):
            return 0

    mc = module.ArgonMC(length=12, resolution=1, cutoff=5, activity=0.001)
    mc.positions = np.array([[1.5, 2.5, 3.5]])
    mc.grid.grid[:] = mc.field(mc.positions)
    # The reverse insertion is uniform in the empty box: alpha=1/(z*V)=.579.
    # Including this atom's own 10 eV core would leave only the uniform floor
    # there and incorrectly reject the prescribed random number 0.2.
    mc.rng = DeleteRNG()
    assert mc.step()
    assert len(mc.positions) == 0
    np.testing.assert_array_equal(mc.grid.grid, 0)


@pytest.mark.parametrize("forward_ratio", [0.4, 2.5])
def test_interacting_forward_reverse_kernel_acceptance_thresholds(forward_ratio):
    """Probe both sides of the actual kernel's MH acceptance thresholds."""
    base = np.array([[1.5, 2.5, 3.5], [8.0, 9.0, 9.0]])
    position = np.array([5.25, 2.75, 3.25])
    reference = module.ArgonMC(length=12, resolution=1, cutoff=5)
    reference.positions = base.copy()
    reference.grid.grid[:] = reference.field(base)
    probability = reference.probability()
    q = reference.density(probability, position)
    delta = reference.interaction_energy(position, base)
    assert delta < 0  # Interacting, attractive insertion, not the ideal gas.
    beta = 1 / (module.BOLTZMANN_EV_K * reference.temperature)
    activity = forward_ratio * (len(base) + 1) * q * np.exp(beta * delta)
    alphas = [min(1, forward_ratio), min(1, 1 / forward_ratio)]
    assert alphas[0] / alphas[1] == pytest.approx(forward_ratio)

    class PrescribedRNG:
        def __init__(self, insertion, draw):
            self.draws = iter([0.1 if insertion else 0.9, draw])

        def random(self, size=None):
            if size is not None:
                return position - np.floor(position)  # Unit grid spacing.
            return next(self.draws)

        def choice(self, size, p):
            return np.ravel_multi_index(
                tuple(np.floor(position).astype(int)), (12, 12, 12)
            )

        def integers(self, n):
            return n - 1  # Delete the inserted atom, leaving precisely base.

    for insertion, alpha in zip([True, False], alphas):
        cases = [(alpha * (1 - 1e-5), True)]
        if alpha < 1:
            cases.append((alpha * (1 + 1e-5), False))
        for draw, accepted in cases:
            mc = module.ArgonMC(length=12, resolution=1, cutoff=5, activity=activity)
            mc.positions = base.copy() if insertion else np.vstack([base, position])
            mc.grid.grid[:] = mc.field(mc.positions)
            mc.rng = PrescribedRNG(insertion, draw)
            assert mc.step() == accepted
