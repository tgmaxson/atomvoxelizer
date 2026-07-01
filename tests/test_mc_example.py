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


def test_minimal_mc_geometric_score_moves_atoms_without_accepting_everything():
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
    score = example.make_geometric_score(initial_positions)

    trajectory = example.run_minimal_mc(
        atoms,
        trial_sites,
        steps=100,
        temperature=0.05,
        max_displacement=0.35,
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
        temperature=0.05,
        max_displacement=0.35,
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
