"""Trace a constant-distance surface around a Wulff nanoparticle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase.cluster import wulff_construction
from ase.data import covalent_radii

from atomvoxelizer import VoxelGrid, VoxelGridAnalysis


def build_wulff_cluster(symbol="Pt", size=147, vacuum=6.0):
    """Build a centered Wulff cluster in an orthorhombic periodic box."""
    atoms = wulff_construction(
        symbol,
        surfaces=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        energies=[1.0, 1.1, 0.9],
        size=size,
        structure="fcc",
        rounding="above",
    )
    positions = atoms.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0)
    cell_lengths = span + 2.0 * vacuum
    atoms.set_cell(np.diag(cell_lengths))
    atoms.set_pbc(True)
    atoms.translate(-positions.min(axis=0) + vacuum)
    return atoms


def distance_surface(atoms, resolution=0.35, cutoff=4.0, distance=2.0):
    """Return a nearest-atom distance grid and its isosurface mesh."""
    grid = VoxelGrid(atoms.cell.array, resolution=resolution)
    grid.grid.fill(np.inf)

    atom_cutoffs = np.full(len(atoms), float(cutoff), dtype=float)
    grid.min_spheres(atoms.get_positions(), atom_cutoffs, mask="distance")

    analysis = VoxelGridAnalysis(grid)
    vertices, faces = analysis.mesh_at_value(distance, periodic=True)
    area = analysis.mesh_surface_area(vertices, faces)
    return grid, vertices, faces, area


def save_mesh_npz(path, atoms, grid, vertices, faces, distance, area):
    np.savez_compressed(
        path,
        vertices=vertices,
        faces=faces,
        atoms=atoms.get_positions(),
        cell=atoms.cell.array,
        gpts=grid.gpts,
        distance=float(distance),
        surface_area=float(area),
    )


def plot_mesh(atoms, vertices, faces, path=None, show=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        faces,
        vertices[:, 2],
        color="#5f9ed1",
        alpha=0.45,
        linewidth=0.0,
    )
    positions = atoms.get_positions()
    radii = [covalent_radii[atom.number] for atom in atoms]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=np.asarray(radii) * 20.0, color="#2b2b2b")
    ax.set_xlabel("x [Angstrom]")
    ax.set_ylabel("y [Angstrom]")
    ax.set_zlabel("z [Angstrom]")
    ax.set_box_aspect(np.ptp(vertices, axis=0))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="Pt", help="Element symbol for the Wulff cluster.")
    parser.add_argument("--size", type=int, default=147, help="Approximate number of atoms in the cluster.")
    parser.add_argument("--resolution", type=float, default=0.35, help="Voxel resolution in Angstrom.")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Distance-mask cutoff radius in Angstrom.")
    parser.add_argument("--distance", type=float, default=2.0, help="Distance isosurface level in Angstrom.")
    parser.add_argument("--vacuum", type=float, default=6.0, help="Vacuum padding around the cluster in Angstrom.")
    parser.add_argument("--output", type=Path, default=Path("wulff_distance_surface.npz"), help="Output mesh archive.")
    parser.add_argument("--plot", type=Path, help="Optional PNG path for a mesh preview.")
    parser.add_argument("--show", action="store_true", help="Show an interactive Matplotlib 3D mesh plot.")
    args = parser.parse_args()

    atoms = build_wulff_cluster(args.symbol, args.size, args.vacuum)
    grid, vertices, faces, area = distance_surface(
        atoms,
        resolution=args.resolution,
        cutoff=args.cutoff,
        distance=args.distance,
    )
    save_mesh_npz(args.output, atoms, grid, vertices, faces, args.distance, area)

    if args.plot or args.show:
        plot_mesh(atoms, vertices, faces, path=args.plot, show=args.show)

    print(f"atoms: {len(atoms)}")
    print(f"grid: {tuple(int(n) for n in grid.gpts)}")
    print(f"vertices: {len(vertices)}")
    print(f"faces: {len(faces)}")
    print(f"surface_area_A2: {area:.6f}")
    print(f"mesh: {args.output}")


if __name__ == "__main__":
    main()
