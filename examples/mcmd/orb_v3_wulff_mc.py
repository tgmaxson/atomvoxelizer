from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from atomvoxelizer import VoxelGrid


def build_wulff_nanoparticle(symbol="Pt", natoms=201, lattice_constant=3.92):
    """Build a finite fcc Wulff particle with WulffPack."""
    try:
        from ase.build import bulk
        from wulffpack import SingleCrystal
    except ImportError as exc:
        raise SystemExit(
            "This example requires ASE and WulffPack. Install them in your environment, "
            "for example with `pip install ase wulffpack`."
        ) from exc

    primitive = bulk(symbol, "fcc", a=lattice_constant)
    surface_energies = {
        (1, 1, 1): 1.00,
        (1, 0, 0): 1.12,
        (1, 1, 0): 1.25,
    }
    return SingleCrystal(surface_energies, primitive_structure=primitive, natoms=natoms).atoms


def put_cluster_in_voxel_cell(atoms, padding=6.0):
    """Return a copy of a finite cluster translated into a padded cubic cell."""
    atoms = atoms.copy()
    positions = atoms.positions
    span = positions.max(axis=0) - positions.min(axis=0)
    cell_length = float(span.max() + 2.0 * padding)
    atoms.positions = positions - positions.min(axis=0) + padding
    atoms.set_cell(np.eye(3) * cell_length)
    atoms.set_pbc(False)
    return atoms


def covalent_radii_for_atoms(atoms):
    from ase.data import covalent_radii

    return covalent_radii[atoms.numbers]


def build_coordination_surface_grid(atoms, resolution=0.35, shell_scale=1.4, core_scale=1.1):
    """Build a voxel coordination surface around a finite nanoparticle.

    The first pass adds one count for every atom whose larger shell covers a
    voxel. The second pass erases the smaller atomic cores. Sampling values near
    three then targets low-coordination surface regions without placing trials
    inside atomic cores.
    """
    radii = covalent_radii_for_atoms(atoms)
    grid = VoxelGrid(atoms.cell.array, resolution=resolution, dtype=np.float32)
    grid.add_spheres(atoms.positions, shell_scale * radii, value=1.0)
    grid.set_spheres(atoms.positions, core_scale * radii, value=0.0)
    return grid


def sample_surface_trial_sites(grid, min_count=2.5, max_count=3.5, max_sites=500, min_dist=0.6, seed=7):
    """Sample candidate MC trial destinations from the coordination surface."""
    sites = []
    for position in grid.sample_voxels_in_range(min_count, max_count, min_dist=min_dist, seed=seed):
        sites.append(np.asarray(position, dtype=float))
        if len(sites) >= max_sites:
            break
    if not sites:
        raise RuntimeError("No trial sites were found. Try a coarser resolution or wider count range.")
    return np.array(sites)


def outer_atom_indices(atoms, fraction=0.35):
    """Pick likely surface atoms from their distance to the nanoparticle center."""
    center = atoms.positions.mean(axis=0)
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    cutoff = np.quantile(distances, 1.0 - fraction)
    return np.flatnonzero(distances >= cutoff)


def geometric_score(atoms):
    """Cheap placeholder score for the tutorial dry run.

    ORB-V3 should be used for a physical MC run. This score only keeps the
    example deterministic and runnable without downloading model weights.
    """
    center = atoms.positions.mean(axis=0)
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    return float(np.var(distances))


def make_orb_v3_score(device="cpu"):
    """Create an ORB-V3 energy scorer when ORB models are installed.

    The exact output object can vary between ORB releases, so the helper accepts
    common energy-key names. If your local ORB version returns a different
    object, adapt only this function; the voxel trial-site workflow is unchanged.
    """
    try:
        import torch
        from orb_models.forcefield import pretrained
    except ImportError as exc:
        raise SystemExit(
            "ORB-V3 scoring requires orb-models and torch. Install those packages "
            "in the environment where you run this example."
        ) from exc

    model, adapter = pretrained.orb_v3_conservative_inf_omat(device=device)
    model.eval()

    def score(atoms):
        graph = adapter.from_ase_atoms(atoms)
        with torch.no_grad():
            output = model.predict(graph)
        if isinstance(output, dict):
            for key in ("energy", "total_energy", "predicted_energy", "graph_energy"):
                if key in output:
                    value = output[key]
                    return float(value.detach().cpu().reshape(-1)[0])
        for key in ("energy", "total_energy", "predicted_energy", "graph_energy"):
            if hasattr(output, key):
                value = getattr(output, key)
                return float(value.detach().cpu().reshape(-1)[0])
        raise RuntimeError(f"Could not find an energy field in ORB-V3 output of type {type(output)!r}")

    return score


def run_minimal_mc(atoms, trial_sites, steps=50, temperature=600.0, max_displacement=0.35, seed=11, score_fn=None):
    """Run a tiny Metropolis loop using voxel-sampled trial directions."""
    rng = np.random.default_rng(seed)
    score_fn = geometric_score if score_fn is None else score_fn
    movable = outer_atom_indices(atoms)
    current_score = score_fn(atoms)
    accepted = 0
    trajectory = []
    beta = 1.0 / max(temperature, 1.0)

    for step in range(steps):
        atom_index = int(rng.choice(movable))
        target = trial_sites[int(rng.integers(len(trial_sites)))]
        old_position = atoms.positions[atom_index].copy()
        direction = target - old_position
        distance = np.linalg.norm(direction)
        if distance > max_displacement:
            direction *= max_displacement / distance
        atoms.positions[atom_index] = old_position + direction

        trial_score = score_fn(atoms)
        delta = trial_score - current_score
        accept = delta <= 0.0 or rng.random() < math.exp(-beta * delta)
        if accept:
            current_score = trial_score
            accepted += 1
        else:
            atoms.positions[atom_index] = old_position

        trajectory.append((step, atom_index, current_score, accepted / (step + 1)))

    return np.array(trajectory, dtype=float)


def plot_trial_sites(atoms, trial_sites, output):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*atoms.positions.T, s=18, c="#8a8a8a", alpha=0.85, label="Wulff atoms")
    ax.scatter(*trial_sites.T, s=8, c="#2ca02c", alpha=0.45, label="Voxel trial sites")
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_zlabel("z [A]")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    return output


def main():
    parser = argparse.ArgumentParser(description="Voxel-guided MC trial moves for a Wulff nanoparticle.")
    parser.add_argument("--symbol", default="Pt")
    parser.add_argument("--natoms", type=int, default=201)
    parser.add_argument("--resolution", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--score", choices=("geometric", "orb-v3"), default="geometric")
    parser.add_argument("--device", default="cpu", help="Device passed to ORB-V3 when --score orb-v3 is used.")
    parser.add_argument("--plot", default=None, help="Optional path for a 3D plot of atoms and trial sites.")
    args = parser.parse_args()

    atoms = put_cluster_in_voxel_cell(build_wulff_nanoparticle(args.symbol, args.natoms))
    grid = build_coordination_surface_grid(atoms, resolution=args.resolution)
    trial_sites = sample_surface_trial_sites(grid, seed=args.seed)
    score_fn = make_orb_v3_score(args.device) if args.score == "orb-v3" else geometric_score
    trajectory = run_minimal_mc(atoms, trial_sites, steps=args.steps, seed=args.seed, score_fn=score_fn)

    print(f"atoms: {len(atoms)}")
    print(f"grid points: {tuple(int(x) for x in grid.gpts)}")
    print(f"trial sites: {len(trial_sites)}")
    print(f"final score: {trajectory[-1, 2]:.6f}")
    print(f"acceptance ratio: {trajectory[-1, 3]:.3f}")
    if args.plot:
        print(f"plot: {plot_trial_sites(atoms, trial_sites, args.plot)}")


if __name__ == "__main__":
    main()
