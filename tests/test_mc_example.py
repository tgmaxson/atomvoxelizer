import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_mc_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "mc" / "orb_v3_co_mcmd.py"
    spec = importlib.util.spec_from_file_location("orb_v3_co_mcmd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimal_mc_emt_score_moves_atoms_without_accepting_everything():
    ase = pytest.importorskip("ase")
    example = load_mc_example()
    atoms = ase.Atoms(
        "Pt8",
        positions=np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 7.0],
                [5.0, 7.0, 5.0],
                [7.0, 5.0, 5.0],
                [7.0, 7.0, 5.0],
                [7.0, 5.0, 7.0],
                [5.0, 7.0, 7.0],
                [7.0, 7.0, 7.0],
            ]
        ),
        cell=np.eye(3) * 12.0,
    )
    trial_sites = np.array(
        [
            [6.0, 6.0, 8.0],
            [6.0, 8.0, 6.0],
            [8.0, 6.0, 6.0],
            [4.0, 6.0, 6.0],
            [6.0, 4.0, 6.0],
            [6.0, 6.0, 4.0],
        ]
    )
    initial_positions = atoms.positions.copy()
    score = example.make_emt_score()

    trajectory = example.run_minimal_mc(
        atoms,
        trial_sites,
        steps=100,
        temperature=50000.0,
        max_displacement=0.35,
        local_trial_radius=3.0,
        relax=False,
        seed=11,
        score_fn=score,
    )
    accepted = int(trajectory[:, 4].sum())
    displacement = np.linalg.norm(atoms.positions - initial_positions, axis=1)

    assert 0 < accepted < len(trajectory)
    assert displacement.max() > 0.0


def test_minimal_mc_can_write_ase_trajectory(tmp_path):
    ase = pytest.importorskip("ase")
    from ase.io import read

    example = load_mc_example()
    atoms = ase.Atoms(
        "Pt4",
        positions=np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 7.0],
                [5.0, 7.0, 5.0],
                [7.0, 5.0, 5.0],
            ]
        ),
        cell=np.eye(3) * 12.0,
    )
    trial_sites = np.array([[6.0, 6.0, 8.0], [6.0, 8.0, 6.0], [8.0, 6.0, 6.0]])

    trajectory, frames = example.run_minimal_mc(
        atoms,
        trial_sites,
        steps=5,
        temperature=1500.0,
        max_displacement=0.35,
        relax=False,
        seed=2,
        return_frames=True,
    )
    output = example.write_mc_trajectory(frames, tmp_path / "mc.traj")
    loaded = read(output, index=":")

    assert trajectory.shape == (5, 5)
    assert len(frames) == 6
    assert len(loaded) == 6
    assert loaded[0].info["mc_step"] == -1
    assert loaded[-1].info["mc_step"] == 4
    assert "mc_accepted" in loaded[-1].info


def test_initial_final_state_plot_is_written(tmp_path):
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    initial = ase.Atoms(
        "Pt2",
        positions=np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
        cell=np.eye(3) * 12.0,
    )
    final = initial.copy()
    final.positions[1] += np.array([0.2, 0.1, 0.0])

    output = example.plot_initial_final_states(initial, final, tmp_path / "states.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_minimal_mc_can_relax_trial_states():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms(
        "Pt4",
        positions=np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 7.0],
                [5.0, 7.0, 5.0],
                [7.0, 5.0, 5.0],
            ]
        ),
        cell=np.eye(3) * 12.0,
    )
    trial_sites = np.array([[6.0, 6.0, 8.0], [6.0, 8.0, 6.0], [8.0, 6.0, 6.0]])

    trajectory = example.run_minimal_mc(
        atoms,
        trial_sites,
        steps=2,
        temperature=1500.0,
        max_displacement=0.35,
        relax=True,
        relax_steps=2,
        calculator=example.make_emt_calculator(),
    )

    assert trajectory.shape == (2, 5)
    assert np.isfinite(trajectory[:, 2]).all()


def test_co_adsorption_mcmd_runs_md_after_accept_and_reject():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms(
        "Pt4",
        positions=np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 7.0],
                [5.0, 7.0, 5.0],
                [7.0, 5.0, 5.0],
            ]
        ),
        cell=np.eye(3) * 12.0,
    )
    trial_sites = np.array([[6.0, 6.0, 8.4], [6.0, 8.4, 6.0]])
    md_calls = []

    def md_runner(atoms, calculator, temperature, steps, timestep_fs, friction):
        md_calls.append((len(atoms), steps))
        return example.potential_energy(atoms, calculator)

    final_atoms, trajectory = example.run_co_adsorption_mcmd(
        atoms,
        trial_sites,
        calculator=example.make_emt_calculator(),
        steps=4,
        temperature=300.0,
        co_chemical_potential=-100.0,
        md_steps=50,
        seed=3,
        md_runner=md_runner,
    )

    assert trajectory.shape == (4, 9)
    assert len(md_calls) == 4
    assert all(call[1] == 50 for call in md_calls)
    assert int(trajectory[:, 2].sum()) < len(trajectory)
    assert len(final_atoms) >= len(atoms)


def test_co_adsorption_helpers_count_and_remove_adsorbates():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms("Pt", positions=[[5.0, 5.0, 5.0]], cell=np.eye(3) * 12.0)
    site = np.array([5.0, 5.0, 7.0])

    adsorbed = example.add_co_adsorbate(atoms, site, bond_length=1.15)
    assert list(adsorbed.symbols) == ["Pt", "C", "O"]
    assert example.surface_atom_count(adsorbed, cn_threshold=11) == 1
    assert example.coverage_fraction(adsorbed, surface_count=1) == pytest.approx(1.0)

    removed = example.remove_co_adsorbate(adsorbed, example.co_carbon_indices(adsorbed)[0])
    assert list(removed.symbols) == ["Pt"]


def test_surface_atom_count_uses_substrate_coordination_numbers():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms(
        "Pt8",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.8],
                [0.0, 2.8, 0.0],
                [0.0, 2.8, 2.8],
                [2.8, 0.0, 0.0],
                [2.8, 0.0, 2.8],
                [2.8, 2.8, 0.0],
                [2.8, 2.8, 2.8],
            ]
        ),
        cell=np.eye(3) * 12.0,
    )

    np.testing.assert_array_equal(example.substrate_coordination_numbers(atoms, cutoff=3.0), np.full(8, 3))
    assert example.surface_atom_count(atoms, cn_threshold=11, cutoff=3.0) == 8


def test_co_coverage_uses_count_after_md_drift_from_voxel_site():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms("Pt", positions=[[5.0, 5.0, 5.0]], cell=np.eye(3) * 12.0)
    site = np.array([5.0, 5.0, 7.0])
    adsorbed = example.add_co_adsorbate(atoms, site, bond_length=1.15)
    adsorbed.positions[1] += np.array([3.0, 0.0, 0.0])

    occupied, assignments = example.site_occupancy(adsorbed, np.array([site]), cutoff=0.2)

    assert not occupied.any()
    assert assignments == {}
    assert example.coverage_fraction(adsorbed, surface_count=1, cutoff=0.2) == pytest.approx(1.0)


def test_co_mcmd_can_desorb_when_adsorbate_is_not_assigned_to_site():
    ase = pytest.importorskip("ase")

    example = load_mc_example()
    atoms = ase.Atoms("Pt", positions=[[5.0, 5.0, 5.0]], cell=np.eye(3) * 12.0)
    site = np.array([5.0, 5.0, 7.0])
    adsorbed = example.add_co_adsorbate(atoms, site, bond_length=1.15)
    adsorbed.positions[1] += np.array([3.0, 0.0, 0.0])

    def md_runner(atoms, calculator, temperature, steps, timestep_fs, friction):
        return example.potential_energy(atoms, calculator)

    final_atoms, trajectory = example.run_co_adsorption_mcmd(
        adsorbed,
        np.array([site]),
        calculator=example.make_emt_calculator(),
        steps=1,
        temperature=300.0,
        co_chemical_potential=-100.0,
        adsorption_probability=0.0,
        md_steps=1,
        md_runner=md_runner,
    )

    assert trajectory[0, 1] == -1.0
    assert trajectory[0, 3] == 0.0
    assert example.co_carbon_indices(final_atoms).size == 0
