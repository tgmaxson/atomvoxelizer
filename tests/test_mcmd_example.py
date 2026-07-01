import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_mcmd_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "mcmd" / "orb_v3_wulff_mc.py"
    spec = importlib.util.spec_from_file_location("orb_v3_wulff_mc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimal_mc_geometric_score_moves_atoms_without_accepting_everything():
    ase = pytest.importorskip("ase")
    example = load_mcmd_example()
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
