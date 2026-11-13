"""Build an Ar insertion bias in an offline carbon cage or a periodic ASE file."""

import argparse
from pathlib import Path

import numpy as np

from atomvoxelizer import lj_insertion_grid, insertion_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structure",
        nargs="?",
        help="Optional orthorhombic periodic structure (ASE format)",
    )
    parser.add_argument("--probe", default="Ar")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("argon_insertion.npz"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be non-negative")
    if args.structure:
        from ase.io import read

        atoms = read(args.structure)
        if not np.all(atoms.pbc):
            parser.error("the input structure must be periodic in all three directions")
        cell, positions, symbols = (
            atoms.cell.array,
            atoms.positions,
            atoms.get_chemical_symbols(),
        )
    else:
        # Eight carbon sites surround an open cavity; no downloads required.
        cell = np.eye(3) * 16
        positions = (
            np.array(np.meshgrid([4.0, 12.0], [4.0, 12.0], [4.0, 12.0], indexing="ij"))
            .reshape(3, -1)
            .T
        )
        symbols = ["C"] * len(positions)
    grid = lj_insertion_grid(
        cell,
        positions,
        symbols,
        probe=args.probe,
        resolution=args.resolution,
        cutoff=args.cutoff,
    )
    probability = insertion_probabilities(grid.grid, args.temperature)
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(probability.size, size=args.samples, p=probability.ravel())
    indices = np.column_stack(np.unravel_index(selected, grid.gpts))
    # Uniform jitter makes a continuous proposal with known density q(r).
    samples = ((indices + rng.random((args.samples, 3))) / grid.gpts) @ cell
    voxel_volume = abs(np.linalg.det(cell)) / probability.size
    proposal_density = probability.ravel()[selected] / voxel_volume
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        energy_ev=grid.grid,
        probability=probability,
        cell=cell,
        gpts=grid.gpts,
        temperature_k=args.temperature,
        probe=args.probe,
        cutoff_angstrom=(
            min(np.linalg.norm(cell, axis=1)) / 2
            if args.cutoff is None
            else args.cutoff
        ),
        uniform_fraction=0.05,
        samples_angstrom=samples,
        proposal_density_per_angstrom3=proposal_density,
    )
    print(
        f"Saved {tuple(grid.gpts)} grid and {args.samples} proposals to {args.output}"
    )
    print(f"Minimum insertion energy: {grid.grid.min():.6f} eV")
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        # Show the layer nearest a host site so repulsive cores are visible.
        mid = (
            int(np.floor((positions[0] @ grid.cell_inv)[2] * grid.gpts[2]))
            % grid.gpts[2]
            if len(positions)
            else grid.gpts[2] // 2
        )
        image = axes[0].imshow(
            np.clip(grid.grid[:, :, mid].T, -0.05, 0.05),
            origin="lower",
            cmap="coolwarm",
            vmin=-0.05,
            vmax=0.05,
        )
        fig.colorbar(image, ax=axes[0], label="Insertion energy (eV; clipped)")
        image = axes[1].imshow(probability[:, :, mid].T, origin="lower", cmap="magma")
        fig.colorbar(image, ax=axes[1], label="Voxel probability")
        for ax in axes:
            ax.set(xlabel="Voxel x index", ylabel="Voxel y index")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
