from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from atomvoxelizer import VoxelGrid


def build_wulff_nanoparticle(symbol="Pt", natoms=201, lattice_constant=3.92, shape="cube"):
    """Build a finite fcc WulffPack particle."""
    try:
        from ase.build import bulk
        from wulffpack import SingleCrystal
    except ImportError as exc:
        raise SystemExit(
            "This example requires ASE and WulffPack. Install them in your environment, "
            "for example with `pip install ase wulffpack`."
        ) from exc

    primitive = bulk(symbol, "fcc", a=lattice_constant)
    if shape == "cube":
        surface_energies = {(1, 0, 0): 1.0}
    elif shape == "wulff":
        surface_energies = {
            (1, 1, 1): 1.00,
            (1, 0, 0): 1.12,
            (1, 1, 0): 1.25,
        }
    else:
        raise ValueError("shape must be 'cube' or 'wulff'")
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


def radial_variance(atoms):
    """Return the variance of atom distances from the nanoparticle center."""
    center = atoms.positions.mean(axis=0)
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    return float(np.var(distances))


def make_emt_score():
    """Create an ASE EMT potential-energy scorer."""
    try:
        from ase.calculators.emt import EMT
    except ImportError as exc:
        raise SystemExit("EMT scoring requires ASE. Install ASE before running this example.") from exc

    def score(atoms):
        if atoms.calc is None or not isinstance(atoms.calc, EMT):
            atoms.calc = EMT()
        return float(atoms.get_potential_energy())

    return score


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


def run_minimal_mc(
    atoms,
    trial_sites,
    steps=50,
    temperature=0.05,
    max_displacement=0.35,
    seed=11,
    score_fn=None,
    return_frames=False,
):
    """Run a tiny Metropolis loop using voxel-sampled trial directions."""
    rng = np.random.default_rng(seed)
    score_fn = make_emt_score() if score_fn is None else score_fn
    movable = outer_atom_indices(atoms)
    current_score = score_fn(atoms)
    accepted = 0
    trajectory = []
    frames = []
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / temperature
    if return_frames:
        initial = atoms.copy()
        initial.info.update(
            {
                "mc_step": -1,
                "mc_atom_index": -1,
                "mc_accepted": True,
                "mc_score": current_score,
                "mc_acceptance_ratio": 0.0,
                "mc_radial_variance": radial_variance(atoms),
            }
        )
        frames.append(initial)

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

        acceptance_ratio = accepted / (step + 1)
        trajectory.append((step, atom_index, current_score, acceptance_ratio, float(accept)))
        if return_frames:
            frame = atoms.copy()
            frame.info.update(
                {
                    "mc_step": step,
                    "mc_atom_index": atom_index,
                    "mc_accepted": bool(accept),
                    "mc_score": current_score,
                    "mc_acceptance_ratio": acceptance_ratio,
                    "mc_radial_variance": radial_variance(atoms),
                }
            )
            frames.append(frame)

    trajectory = np.array(trajectory, dtype=float)
    if return_frames:
        return trajectory, frames
    return trajectory


def write_mc_trajectory(frames, output):
    """Write MC frames to an ASE-readable trajectory file."""
    from ase.io import write

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, frames)
    return output


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
    parser.add_argument("--shape", choices=("cube", "wulff"), default="cube")
    parser.add_argument("--resolution", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--score", choices=("emt", "orb-v3"), default="emt")
    parser.add_argument("--device", default="cpu", help="Device passed to ORB-V3 when --score orb-v3 is used.")
    parser.add_argument("--plot", default=None, help="Optional path for a 3D plot of atoms and trial sites.")
    parser.add_argument(
        "--trajectory",
        default=str(Path(__file__).with_name("orb_v3_wulff_mc.traj")),
        help="ASE trajectory output path. Use an empty string to disable writing frames.",
    )
    args = parser.parse_args()

    atoms = put_cluster_in_voxel_cell(build_wulff_nanoparticle(args.symbol, args.natoms, shape=args.shape))
    initial_positions = atoms.positions.copy()
    initial_radial_variance = radial_variance(atoms)
    grid = build_coordination_surface_grid(atoms, resolution=args.resolution)
    trial_sites = sample_surface_trial_sites(grid, seed=args.seed)
    if args.score == "orb-v3":
        score_fn = make_orb_v3_score(args.device)
    else:
        score_fn = make_emt_score()
    mc_result = run_minimal_mc(
        atoms,
        trial_sites,
        steps=args.steps,
        temperature=args.temperature,
        max_displacement=args.max_displacement,
        seed=args.seed,
        score_fn=score_fn,
        return_frames=bool(args.trajectory),
    )
    if args.trajectory:
        trajectory, frames = mc_result
    else:
        trajectory = mc_result
        frames = None
    displacements = np.linalg.norm(atoms.positions - initial_positions, axis=1)

    print(f"atoms: {len(atoms)}")
    print(f"grid points: {tuple(int(x) for x in grid.gpts)}")
    print(f"trial sites: {len(trial_sites)}")
    print(f"accepted moves: {int(trajectory[:, 4].sum())}/{len(trajectory)}")
    print(f"initial radial variance: {initial_radial_variance:.6f}")
    print(f"final radial variance: {radial_variance(atoms):.6f}")
    print(f"final score: {trajectory[-1, 2]:.6f}")
    print(f"acceptance ratio: {trajectory[-1, 3]:.3f}")
    print(f"mean displacement: {displacements.mean():.4f} A")
    print(f"max displacement: {displacements.max():.4f} A")
    if args.trajectory:
        print(f"trajectory: {write_mc_trajectory(frames, args.trajectory)}")
    if args.plot:
        print(f"plot: {plot_trial_sites(atoms, trial_sites, args.plot)}")


if __name__ == "__main__":
    main()
