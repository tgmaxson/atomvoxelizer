from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ase.build import fcc111
from ase.cluster import wulff_construction
from ase.data import covalent_radii
from ase.io import read
from numba import njit


ROOT = Path(__file__).resolve().parents[1]


@njit
def add_many_c_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(x * ny + y) * nz + z] += value


@njit
def add_many_c_3d(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[x, y, z] += value


@njit
def add_many_f_flat(grid, center_indices, offsets, value, nx, ny, nz):
    for c in range(center_indices.shape[0]):
        cx = center_indices[c, 0]
        cy = center_indices[c, 1]
        cz = center_indices[c, 2]
        for n in range(offsets.shape[0]):
            x = (cx + offsets[n, 0]) % nx
            y = (cy + offsets[n, 1]) % ny
            z = (cz + offsets[n, 2]) % nz
            grid[(z * ny + y) * nx + x] += value


def sphere_offsets(radius, gpts, cell):
    gpts_arr = np.array(gpts, dtype=np.int32)
    cell_arr = np.array(cell, dtype=np.float64)
    lengths = np.linalg.norm(cell_arr, axis=1)
    max_offsets = np.ceil(radius / lengths * gpts_arr).astype(np.int32)

    offsets = []
    for dx in range(-max_offsets[0], max_offsets[0] + 1):
        for dy in range(-max_offsets[1], max_offsets[1] + 1):
            for dz in range(-max_offsets[2], max_offsets[2] + 1):
                disp_frac = np.array([dx, dy, dz], dtype=np.float64) / gpts_arr
                disp = disp_frac @ cell_arr
                if np.dot(disp, disp) <= radius * radius:
                    offsets.append((dx, dy, dz))

    return np.array(offsets, dtype=np.int32)


def positions_to_indices(positions, cell, gpts):
    frac = positions @ np.linalg.inv(cell)
    frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
    return np.floor(frac_wrapped * gpts).astype(np.int32)


def center_cluster(atoms, padding):
    atoms = atoms.copy()
    positions = atoms.get_positions()
    positions -= positions.min(axis=0)
    positions += padding
    atoms.positions = positions
    atoms.set_cell(np.eye(3) * float(positions.max() + padding))
    atoms.set_pbc(False)
    return atoms


def make_atoms(workload):
    if workload == "zeolite":
        return read(ROOT / "examples" / "zeolite" / "BEA.cif").repeat((2, 2, 4))
    if workload == "nanoparticle":
        atoms = wulff_construction(
            "Pt",
            surfaces=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
            energies=[1.0, 1.08, 0.92],
            size=2869,
            structure="fcc",
            latticeconstant=3.92,
            rounding="closest",
        )
        return center_cluster(atoms, padding=5.0)
    if workload == "surface":
        atoms = fcc111("Pt", size=(22, 22, 6), vacuum=6.0, a=3.92)
        atoms.center(vacuum=6.0, axis=2)
        return atoms
    raise ValueError(f"Unknown workload: {workload}")


def make_workload(workload, resolution, radius_scale):
    atoms = make_atoms(workload)
    cell = atoms.cell.array
    centers = atoms.get_positions()
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    radius = float(np.median(radii))
    gpts = np.ceil(np.linalg.norm(cell, axis=1) / resolution).astype(np.int32)
    center_indices = positions_to_indices(centers, cell, gpts)
    offsets = sphere_offsets(radius, gpts, cell)
    return atoms, center_indices, offsets, gpts


def sorted_offsets(offsets, gpts, order):
    nx, ny, nz = [int(value) for value in gpts]
    if order == "c-flat":
        keys = (offsets[:, 0] * ny + offsets[:, 1]) * nz + offsets[:, 2]
    elif order == "f-flat":
        keys = (offsets[:, 2] * ny + offsets[:, 1]) * nx + offsets[:, 0]
    elif order == "z-fast":
        return offsets[np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0]))].copy()
    elif order == "x-fast":
        return offsets[np.lexsort((offsets[:, 0], offsets[:, 1], offsets[:, 2]))].copy()
    else:
        raise ValueError(f"Unknown offset order: {order}")
    return offsets[np.argsort(keys)].copy()


def compile_kernels():
    dummy_grid = np.zeros(8, dtype=np.float32)
    dummy_grid_3d = np.zeros((2, 2, 2), dtype=np.float32)
    dummy_centers = np.zeros((1, 3), dtype=np.int32)
    dummy_offsets = np.zeros((1, 3), dtype=np.int32)
    add_many_c_flat(dummy_grid, dummy_centers, dummy_offsets, 1.0, 2, 2, 2)
    add_many_f_flat(dummy_grid, dummy_centers, dummy_offsets, 1.0, 2, 2, 2)
    add_many_c_3d(dummy_grid_3d, dummy_centers, dummy_offsets, 1.0, 2, 2, 2)


def time_call(func, repeats):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return np.array(times, dtype=float), value


def benchmark_variant(variant, center_indices, offsets, gpts, repeats):
    nx, ny, nz = [int(value) for value in gpts]

    if variant == "c-flat":
        ordered_offsets = offsets

        def run():
            grid = np.zeros(nx * ny * nz, dtype=np.float32)
            add_many_c_flat(grid, center_indices, ordered_offsets, 1.0, nx, ny, nz)
            return grid

    elif variant == "c-flat-sorted":
        ordered_offsets = sorted_offsets(offsets, gpts, "c-flat")

        def run():
            grid = np.zeros(nx * ny * nz, dtype=np.float32)
            add_many_c_flat(grid, center_indices, ordered_offsets, 1.0, nx, ny, nz)
            return grid

    elif variant == "c-3d":
        ordered_offsets = offsets

        def run():
            grid = np.zeros((nx, ny, nz), dtype=np.float32)
            add_many_c_3d(grid, center_indices, ordered_offsets, 1.0, nx, ny, nz)
            return grid.ravel()

    elif variant == "f-flat":
        ordered_offsets = offsets

        def run():
            grid = np.zeros(nx * ny * nz, dtype=np.float32)
            add_many_f_flat(grid, center_indices, ordered_offsets, 1.0, nx, ny, nz)
            return grid

    elif variant == "f-flat-sorted":
        ordered_offsets = sorted_offsets(offsets, gpts, "f-flat")

        def run():
            grid = np.zeros(nx * ny * nz, dtype=np.float32)
            add_many_f_flat(grid, center_indices, ordered_offsets, 1.0, nx, ny, nz)
            return grid

    else:
        raise ValueError(f"Unknown variant: {variant}")

    times, values = time_call(run, repeats)
    return times, values


def format_float(value, precision=5):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = ["workload", "variant", "atoms", "gpts", "offsets", "best [s]", "mean [s]", "std [s]", "sum"]
    table = [
        [
            row["workload"],
            row["variant"],
            str(row["atoms"]),
            row["gpts"],
            str(row["offsets"]),
            format_float(row["best_s"]),
            format_float(row["mean_s"]),
            format_float(row["std_s"]),
            format_float(row["sum"], precision=1),
        ]
        for row in rows
    ]
    widths = [max(len(str(value)) for value in [header] + [line[i] for line in table]) for i, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for line in table:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(line)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Numba voxel-grid memory/index order variants.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["zeolite", "nanoparticle", "surface"],
        default=["zeolite", "nanoparticle", "surface"],
    )
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--radius-scale", type=float, default=1.4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["c-flat", "c-flat-sorted", "c-3d", "f-flat", "f-flat-sorted"],
        default=["c-flat", "c-flat-sorted", "c-3d", "f-flat", "f-flat-sorted"],
    )
    args = parser.parse_args()

    compile_kernels()
    rows = []
    for workload in args.workloads:
        atoms, center_indices, offsets, gpts = make_workload(workload, args.resolution, args.radius_scale)
        reference_sum = None
        for variant in args.variants:
            times, values = benchmark_variant(variant, center_indices, offsets, gpts, args.repeats)
            value_sum = float(values.sum())
            if reference_sum is None:
                reference_sum = value_sum
            elif not np.isclose(value_sum, reference_sum):
                raise RuntimeError(f"{variant} produced a different voxel sum for {workload}")
            rows.append(
                {
                    "workload": workload,
                    "variant": variant,
                    "atoms": len(atoms),
                    "gpts": "x".join(str(int(value)) for value in gpts),
                    "offsets": offsets.shape[0],
                    "best_s": float(times.min()),
                    "mean_s": float(times.mean()),
                    "std_s": float(times.std()),
                    "sum": value_sum,
                }
            )

    print_table(rows)


if __name__ == "__main__":
    main()
