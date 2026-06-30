"""Trace a periodic distance surface above a Pt(211) slab."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase.build import surface
from ase.data import covalent_radii

from atomvoxelizer import VoxelGrid, VoxelGridAnalysis


def build_pt211_slab(layers=6, vacuum=8.0, repeat=(2, 2, 1)):
    """Build a periodic Pt(211) slab with vacuum along the surface normal."""
    atoms = surface("Pt", (2, 1, 1), layers=layers, vacuum=vacuum, periodic=True)
    atoms = atoms.repeat(repeat)
    atoms.set_pbc((True, True, True))
    return atoms


def distance_surface(atoms, resolution=0.35, cutoff=3.5, distance=1.8):
    """Return a nearest-atom distance grid and a periodic isosurface mesh."""
    grid = VoxelGrid(atoms.cell.array, resolution=resolution)
    grid.grid.fill(np.inf)
    radii = np.full(len(atoms), float(cutoff), dtype=float)
    grid.min_spheres(atoms.get_positions(), radii, mask="distance")

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

    fig = plt.figure(figsize=(8, 6))
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
    sizes = np.array([covalent_radii[atom.number] for atom in atoms]) * 22.0
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=sizes, color="#2b2b2b")

    cell = atoms.cell.array
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    ) @ cell
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]:
        ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], [corners[i, 2], corners[j, 2]], "k-", lw=0.7)

    ax.set_xlabel("x [Angstrom]")
    ax.set_ylabel("y [Angstrom]")
    ax.set_zlabel("z [Angstrom]")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect(np.ptp(corners, axis=0))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=6, help="Number of Pt(211) slab layers.")
    parser.add_argument("--repeat", type=int, nargs=3, default=(2, 2, 1), help="Slab repeat along cell vectors.")
    parser.add_argument("--vacuum", type=float, default=8.0, help="Vacuum padding along the surface normal.")
    parser.add_argument("--resolution", type=float, default=0.35, help="Voxel resolution in Angstrom.")
    parser.add_argument("--cutoff", type=float, default=3.5, help="Distance-mask cutoff radius in Angstrom.")
    parser.add_argument("--distance", type=float, default=1.8, help="Distance isosurface level in Angstrom.")
    parser.add_argument("--output", type=Path, default=Path("pt211_distance_surface.npz"), help="Output mesh archive.")
    parser.add_argument("--plot", type=Path, help="Optional PNG path for a mesh preview.")
    parser.add_argument("--show", action="store_true", help="Show an interactive Matplotlib 3D mesh plot.")
    args = parser.parse_args()

    atoms = build_pt211_slab(layers=args.layers, vacuum=args.vacuum, repeat=tuple(args.repeat))
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
