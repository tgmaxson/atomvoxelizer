from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read
from matplotlib.colors import TwoSlopeNorm

from atomvoxelizer import VoxelGrid


EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and benchmark zeolite voxel grids.")
    parser.add_argument(
        "framework",
        nargs="?",
        default="BEA",
        help="IZA framework code, for example BEA or MWW",
    )
    return parser.parse_args()


def load_atoms(framework: str) -> tuple[Atoms, str]:
    framework = framework.upper()
    cif_path = EXAMPLE_DIR / f"{framework}.cif"
    try:
        return read(cif_path), str(cif_path)  # type: ignore[return-value]
    except FileNotFoundError:
        import requests

        url = f"https://europe.iza-structure.org/IZA-SC/cif/{framework}.cif"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(cif_path, "wb") as handle:
            handle.write(response.content)

        return read(cif_path), str(cif_path)  # type: ignore[return-value]


def build_voxel_grid(atoms: Atoms, resolution: float) -> tuple[VoxelGrid, float]:
    print(f"\nBuilding voxel grid at resolution {resolution:.2f} A")
    voxel_grid = VoxelGrid(cell=atoms.cell.array, resolution=resolution)
    print(f"Grid points: {tuple(voxel_grid.gpts)}")
    centers = atoms.get_positions()
    shell_radii = np.array([covalent_radii[atom.number] * 1.5 for atom in atoms], dtype=np.float64)
    core_radii = np.array([covalent_radii[atom.number] * 0.9 for atom in atoms], dtype=np.float64)

    start = time.perf_counter()

    print(f"Tracing coordination shell (+1) for {len(centers)} atoms")
    voxel_grid.add_spheres(centers, shell_radii, value=1.0)

    print(f"Blocking atomic cores (-1) for {len(centers)} atoms")
    voxel_grid.set_spheres(centers, core_radii, value=-1.0)

    elapsed = time.perf_counter() - start
    print(f"Finished resolution {resolution:.2f} A in {elapsed:.3f} s")
    return voxel_grid, elapsed


def warm_up_numba(cell: np.ndarray) -> None:
    dummy = VoxelGrid(cell=cell, resolution=2.0)
    center = np.zeros(3, dtype=float)
    dummy.add_sphere(center, 0.5, value=1.0)
    dummy.set_sphere(center, 0.25, value=-1.0)


def plot_middle_xz_slice(
    ax: plt.Axes,
    voxel_grid: VoxelGrid,
    resolution: float,
    elapsed: float,
) -> None:
    y_index = voxel_grid.gpts[1] // 2
    xz_slice = voxel_grid.grid[:, y_index, :]

    cell_lengths = np.linalg.norm(voxel_grid.cell, axis=1)
    extent = [0.0, cell_lengths[0], 0.0, cell_lengths[2]]

    vmax = max(1.0, float(np.max(xz_slice)))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=vmax)

    image = ax.imshow(
        xz_slice.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
    )
    image.format_cursor_data = lambda data: f"{float(data):.3g}" if np.isfinite(data) else "nan"

    ax.set_title(
        f"res = {resolution:.2f} A\n"
        f"grid = {tuple(voxel_grid.gpts)}\n"
        f"time = {elapsed:.3f} s"
    )
    ax.set_xlabel("x [A]")
    ax.set_ylabel("z [A]")
    plt.colorbar(image, ax=ax, label="voxel value")


def plot_supercell_scaling(ax: plt.Axes, scaling_results: list[dict[str, object]]) -> None:
    atom_counts = [result["atom_count"] for result in scaling_results]
    times = [result["elapsed"] for result in scaling_results]
    times_per_atom = [result["time_per_atom_ms"] for result in scaling_results]
    labels = [result["label"] for result in scaling_results]

    line_time = ax.plot(atom_counts, times, marker="o", color="tab:blue", label="total time [s]")
    ax.set_xlabel("atom count")
    ax.set_ylabel("total voxelization time [s]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    twin = ax.twinx()
    line_per_atom = twin.plot(
        atom_counts,
        times_per_atom,
        marker="s",
        color="tab:red",
        label="time per atom [ms]",
    )
    twin.set_ylabel("time per atom [ms]", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")

    for atom_count, elapsed, per_atom, label in zip(atom_counts, times, times_per_atom, labels):
        ax.annotate(label, (atom_count, elapsed), textcoords="offset points", xytext=(0, 8), ha="center")
        twin.annotate(f"{per_atom:.2f}", (atom_count, per_atom), textcoords="offset points", xytext=(0, -14), ha="center")

    lines = line_time + line_per_atom
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right")
    framework = scaling_results[0]["framework"]
    ax.set_title(f"{framework} supercell scaling at 0.10 A")


def main(framework: str | None = None) -> None:
    if framework is None:
        args = parse_args()
        framework = args.framework

    framework = framework.upper()
    atoms, cif_path = load_atoms(framework)
    resolutions = [0.5, 0.1, 0.05]
    resolution_3d = 0.2
    supercell_factors = [1, 2, 3, 4]
    supercell_resolution = 0.1

    print(f"Loaded {len(atoms)} atoms from {cif_path}")
    print("Warming up numba kernels...")
    warm_up_numba(atoms.cell.array)
    print("Warm-up complete")

    results: list[tuple[float, VoxelGrid, float]] = []
    for resolution in resolutions:
        voxel_grid, elapsed = build_voxel_grid(atoms, resolution)
        results.append((resolution, voxel_grid, elapsed))
        positive_sites = int(np.count_nonzero(voxel_grid.grid > 0.0))
        blocked_sites = int(np.count_nonzero(voxel_grid.grid < 0.0))
        gas_sites = int(np.count_nonzero(voxel_grid.grid == 0.0))
        print(
            f"resolution={resolution:.2f} A, "
            f"gpts={tuple(voxel_grid.gpts)}, "
            f"time={elapsed:.3f} s, "
            f"positive={positive_sites}, "
            f"blocked={blocked_sites}, "
            f"gas={gas_sites}"
        )

    print(f"\nBuilding 3D visualization grid at {resolution_3d:.2f} A...")
    voxel_grid_3d, elapsed_3d = build_voxel_grid(atoms, resolution_3d)
    print(
        f"3D plot grid: resolution={resolution_3d:.2f} A, "
        f"gpts={tuple(voxel_grid_3d.gpts)}, "
        f"time={elapsed_3d:.3f} s"
    )

    print("\nPreparing middle XZ slice plots...")
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (resolution, voxel_grid, elapsed) in zip(axes, results):
        plot_middle_xz_slice(ax, voxel_grid, resolution, elapsed)

    fig.suptitle(f"{framework} middle XZ slice (slice normal: y)")

    scaling_results: list[dict[str, object]] = []
    print(f"\nBenchmarking supercells at {supercell_resolution:.2f} A...")
    for factor in supercell_factors:
        supercell = atoms.repeat((factor, factor, factor))
        label = f"{factor}x{factor}x{factor}"
        voxel_grid, elapsed = build_voxel_grid(supercell, supercell_resolution)
        atom_count = len(supercell)
        time_per_atom_ms = elapsed * 1000.0 / atom_count
        scaling_results.append(
            {
                "factor": float(factor),
                "atom_count": float(atom_count),
                "elapsed": float(elapsed),
                "time_per_atom_ms": float(time_per_atom_ms),
                "label": label,
                "framework": framework,
            }
        )
        print(
            f"{label}: atoms={atom_count}, "
            f"gpts={tuple(voxel_grid.gpts)}, "
            f"time={elapsed:.3f} s, "
            f"time_per_atom={time_per_atom_ms:.3f} ms"
        )

    scaling_fig, scaling_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_supercell_scaling(scaling_ax, scaling_results)

    print(f"\nPreparing 3D plot at {resolution_3d:.2f} A...")
    voxel_grid_3d.plot_3D(threshold=0.1, s=4, draw_cell=True)

    plt.show()


if __name__ == "__main__":
    main()
