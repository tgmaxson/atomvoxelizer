import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_mc_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "mc" / "orb_v3_wulff_mc.py"
    spec = importlib.util.spec_from_file_location("orb_v3_wulff_mc", path)
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
