from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.io import read
from matplotlib.colors import TwoSlopeNorm

from atomvoxelizer import VoxelGrid


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "source" / "_static" / "zeolite_voxel_slice.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a documentation slice through a zeolite voxel grid.")
    parser.add_argument(
        "--framework",
        default="BEA",
        help="Framework CIF name in examples/zeolite, without the .cif suffix.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Voxel spacing target in Angstrom.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="y",
        help="Slice-normal axis.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Slice index along the slice-normal axis. Defaults to the middle slice.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated PNG.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    return parser.parse_args()


def build_framework_grid(framework: str, resolution: float) -> VoxelGrid:
    atoms = read(EXAMPLE_DIR / f"{framework.upper()}.cif")
    grid = VoxelGrid(cell=atoms.cell.array, resolution=resolution, dtype=np.float32)

    positions = atoms.get_positions()
    shell_radii = np.array([covalent_radii[number] * 1.5 for number in atoms.numbers], dtype=np.float64)
    core_radii = np.array([covalent_radii[number] * 0.9 for number in atoms.numbers], dtype=np.float64)

    grid.add_spheres(positions, shell_radii, value=1.0)
    grid.set_spheres(positions, core_radii, value=-1.0)
    return grid


def slice_grid(grid: VoxelGrid, axis: str, index: int | None) -> tuple[np.ndarray, tuple[float, float, float, float], str, str, int]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if index is None:
        index = int(grid.gpts[axis_index] // 2)
    if index < 0 or index >= grid.gpts[axis_index]:
        raise ValueError(f"slice index must be in [0, {grid.gpts[axis_index] - 1}]")

    lengths = np.linalg.norm(grid.cell, axis=1)
    if axis == "x":
        data = grid.grid[index, :, :]
        extent = (0.0, lengths[1], 0.0, lengths[2])
        x_label, y_label = "y [A]", "z [A]"
    elif axis == "y":
        data = grid.grid[:, index, :]
        extent = (0.0, lengths[0], 0.0, lengths[2])
        x_label, y_label = "x [A]", "z [A]"
    else:
        data = grid.grid[:, :, index]
        extent = (0.0, lengths[0], 0.0, lengths[1])
        x_label, y_label = "x [A]", "y [A]"

    return data.T, extent, x_label, y_label, index


def plot_slice(framework: str, grid: VoxelGrid, axis: str, index: int | None, output: Path, show: bool) -> None:
    data, extent, x_label, y_label, index = slice_grid(grid, axis, index)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=max(1.0, float(np.max(data))))

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(
        f"{framework.upper()} voxel slice, {axis} index {index}\n"
        f"grid {tuple(int(value) for value in grid.gpts)}, resolution {np.mean(grid.resolution):.3f} A"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("voxel value")
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["core", "void", "shell"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    grid = build_framework_grid(args.framework, args.resolution)
    plot_slice(args.framework, grid, args.axis, args.index, args.output, args.show)


if __name__ == "__main__":
    main()
