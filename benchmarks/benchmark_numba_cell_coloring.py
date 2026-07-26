from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from ase.build import fcc111
from ase.cluster import wulff_construction
from ase.data import covalent_radii
from ase.io import read
from numba import get_num_threads, set_num_threads

from atomvoxelizer import VoxelGridNumba


ROOT = Path(__file__).resolve().parents[1]


def make_atoms_workload(atoms, radius_scale):
    centers = atoms.get_positions()
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    return atoms.cell.array, centers, radii


def center_cluster(atoms, padding):
    atoms = atoms.copy()
    positions = atoms.get_positions()
    positions -= positions.min(axis=0)
    positions += padding
    atoms.positions = positions
    atoms.set_cell(np.eye(3) * float(positions.max() + padding))
    atoms.set_pbc(False)
    return atoms


def make_zeolite_workloads(framework, radius_scale):
    atoms = read(ROOT / "examples" / "zeolite" / f"{framework.upper()}.cif")
    scale_factors = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (2, 2, 4)]
    workloads = []
    for scale in scale_factors:
        scaled = atoms.repeat(scale)
        cell, centers, radii = make_atoms_workload(scaled, radius_scale)
        workloads.append(("zeolite", "x".join(str(value) for value in scale), cell, centers, radii))
    return workloads


def make_nanoparticle_workloads(radius_scale, padding):
    workloads = []
    for size in [55, 147, 309, 561, 923, 1415, 2057, 2869]:
        atoms = wulff_construction(
            "Pt",
            surfaces=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
            energies=[1.0, 1.08, 0.92],
            size=size,
            structure="fcc",
            latticeconstant=3.92,
            rounding="closest",
        )
        atoms = center_cluster(atoms, padding=padding)
        cell, centers, radii = make_atoms_workload(atoms, radius_scale)
        workloads.append(("nanoparticle", str(size), cell, centers, radii))
    return workloads


def make_surface_workloads(radius_scale):
    workloads = []
    for size in [(4, 4, 3), (6, 6, 4), (8, 8, 5), (12, 12, 5), (16, 16, 6), (22, 22, 6)]:
        atoms = fcc111("Pt", size=size, vacuum=6.0, a=3.92)
        atoms.center(vacuum=6.0, axis=2)
        atoms.set_pbc([True, True, False])
        cell, centers, radii = make_atoms_workload(atoms, radius_scale)
        workloads.append(("surface", "x".join(str(value) for value in size), cell, centers, radii))
    return workloads


def make_workloads(names, framework, radius_scale, padding):
    workloads = []
    for name in names:
        if name == "zeolite":
            workloads.extend(make_zeolite_workloads(framework, radius_scale))
        elif name == "nanoparticle":
            workloads.extend(make_nanoparticle_workloads(radius_scale, padding))
        elif name == "surface":
            workloads.extend(make_surface_workloads(radius_scale))
        else:
            raise ValueError(f"Unknown workload: {name}")
    return workloads


def build_plans(grid, centers, radii):
    unique_radii = np.unique(radii)
    coloring_radius = float(np.max(unique_radii))
    plans = []
    for radius in unique_radii:
        selected = radii == radius
        plans.append(grid.prepare_cell_coloring(centers[selected], float(radius), coloring_radius=coloring_radius))
    return plans


def apply_plans(grid, plans, value):
    for plan in plans:
        grid.apply_prepared_cell_coloring(plan, value=value)


def time_call(func, repeats):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return np.asarray(times, dtype=float), value


def row(workload, scale, method, atoms, gpts, times, diff=0.0, note=""):
    return {
        "workload": workload,
        "scale": scale,
        "method": method,
        "threads": int(get_num_threads()),
        "atoms": int(atoms),
        "gpts": "x".join(str(int(value)) for value in gpts),
        "best_s": float(times.min()),
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "max_diff": float(diff),
        "note": note,
    }


def format_float(value, precision=5):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = [
        "workload",
        "scale",
        "method",
        "threads",
        "atoms",
        "gpts",
        "best [s]",
        "mean [s]",
        "std [s]",
        "max |diff|",
        "note",
    ]
    table = [
        [
            item["workload"],
            item["scale"],
            item["method"],
            str(item["threads"]),
            str(item["atoms"]),
            item["gpts"],
            format_float(item["best_s"]),
            format_float(item["mean_s"]),
            format_float(item["std_s"]),
            format_float(item["max_diff"], precision=3),
            item["note"],
        ]
        for item in rows
    ]
    widths = [max(len(str(value)) for value in [header] + [line[i] for line in table]) for i, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for line in table:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(line)))


def write_csv(rows, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "workload",
                "scale",
                "method",
                "threads",
                "atoms",
                "gpts",
                "best_s",
                "mean_s",
                "std_s",
                "max_diff",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    colors = {
        "Numba serial": "#2f9e44",
        "Numba 27-color including plan": "#d97706",
        "Numba 27-color prepared apply": "#7c3aed",
        "Numba 27-color plan only": "#6b7280",
    }
    markers = {"zeolite": "o", "nanoparticle": "s", "surface": "^"}
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=220)
    for method, color in colors.items():
        for workload, marker in markers.items():
            points = [
                item
                for item in rows
                if item["method"] == method and item["workload"] == workload and np.isfinite(item["best_s"])
            ]
            if not points:
                continue
            points.sort(key=lambda item: item["atoms"])
            ax.plot(
                [item["atoms"] for item in points],
                [item["best_s"] for item in points],
                color=color,
                marker=marker,
                linewidth=1.6,
                markersize=4.3,
                label=f"{method}, {workload}",
            )

    finite_times = [item["best_s"] for item in rows if np.isfinite(item["best_s"])]
    ax.set_yscale("log")
    if finite_times:
        ax.set_ylim(min(finite_times) * 0.5, max(finite_times) * 3.0)
    ax.set_xlabel("Atoms")
    ax.set_ylabel("Best wall time [s]")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.18)

    method_handles = [
        plt.Line2D([0], [0], color=color, lw=2.0, label=method.replace("Numba ", ""))
        for method, color in colors.items()
    ]
    system_handles = [
        plt.Line2D([0], [0], color="#4a4a4a", marker=marker, linestyle="None", label=workload.capitalize())
        for workload, marker in markers.items()
    ]
    first = ax.legend(handles=method_handles, fontsize=7.2, loc="upper left", frameon=True, title="Path")
    ax.add_artist(first)
    ax.legend(handles=system_handles, fontsize=7.2, loc="lower right", frameon=True, title="System")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)


def plot_thread_rows(rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    colors = {
        "Numba serial": "#2f9e44",
        "Numba 27-color including plan": "#d97706",
        "Numba 27-color prepared apply": "#7c3aed",
        "Numba 27-color plan only": "#6b7280",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=220)
    for method, color in colors.items():
        points = [item for item in rows if item["method"] == method and np.isfinite(item["best_s"])]
        if not points:
            continue
        points.sort(key=lambda item: item["threads"])
        ax.plot(
            [item["threads"] for item in points],
            [item["best_s"] for item in points],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.3,
            label=method.replace("Numba ", ""),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted({item["threads"] for item in rows}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    finite_times = [item["best_s"] for item in rows if np.isfinite(item["best_s"])]
    if finite_times:
        ax.set_ylim(min(finite_times) * 0.6, max(finite_times) * 2.5)
    ax.set_xlabel("Numba threads")
    ax.set_ylabel("Best wall time [s]")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.18)
    ax.legend(fontsize=7.2, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)


def warm_kernels():
    cell = np.eye(3) * 8.0
    centers = np.array([[1.5, 1.5, 1.5], [4.5, 4.5, 4.5]], dtype=np.float64)
    radii = np.array([1.0, 1.0], dtype=np.float64)
    grid = VoxelGridNumba(cell, gpts=(8, 8, 8), dtype=np.float32)
    grid.add_spheres(centers, radii, value=1.0)
    grid = VoxelGridNumba(cell, gpts=(8, 8, 8), dtype=np.float32)
    grid.add_spheres_cell_colored(centers, radii, value=1.0)


def benchmark_workload(workload, scale, cell, centers, radii, resolution, repeats):
    reference = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    serial_times, serial_result = time_call(
        lambda: _run_serial(cell, resolution, centers, radii),
        repeats,
    )
    reference_values = serial_result.to_numpy().copy()

    colored_times, colored_result = time_call(
        lambda: _run_colored(cell, resolution, centers, radii),
        repeats,
    )
    colored_diff = float(np.max(np.abs(colored_result.to_numpy() - reference_values)))

    plan_grid = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    plan_times, plans = time_call(lambda: build_plans(plan_grid, centers, radii), repeats)
    plan_note = f"groups={len(plans)}, threads={get_num_threads()}"

    prepared_grid = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    prepared_plans = build_plans(prepared_grid, centers, radii)
    prepared_times, prepared_result = time_call(
        lambda: _run_prepared(cell, resolution, prepared_plans),
        repeats,
    )
    prepared_diff = float(np.max(np.abs(prepared_result.to_numpy() - reference_values)))

    gpts = reference.gpts
    atoms = centers.shape[0]
    return [
        row(workload, scale, "Numba serial", atoms, gpts, serial_times),
        row(workload, scale, "Numba 27-color including plan", atoms, gpts, colored_times, colored_diff, plan_note),
        row(workload, scale, "Numba 27-color plan only", atoms, gpts, plan_times, 0.0, plan_note),
        row(workload, scale, "Numba 27-color prepared apply", atoms, gpts, prepared_times, prepared_diff, plan_note),
    ]


def _run_serial(cell, resolution, centers, radii):
    grid = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    grid.add_spheres(centers, radii, value=1.0)
    return grid


def _run_colored(cell, resolution, centers, radii):
    grid = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    grid.add_spheres_cell_colored(centers, radii, value=1.0)
    return grid


def _run_prepared(cell, resolution, plans):
    grid = VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    apply_plans(grid, plans, value=1.0)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Benchmark the opt-in Numba 27-color cell-coloring add path.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["zeolite", "nanoparticle", "surface"],
        default=["zeolite", "nanoparticle", "surface"],
    )
    parser.add_argument("--resolution", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius-scale", type=float, default=1.4)
    parser.add_argument("--framework", default="BEA")
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=None, help="Optional Numba thread count.")
    parser.add_argument(
        "--thread-scaling",
        action="store_true",
        help="Benchmark one workload while sweeping Numba thread counts.",
    )
    parser.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 28])
    parser.add_argument(
        "--thread-workload",
        choices=["zeolite", "nanoparticle", "surface"],
        default="nanoparticle",
        help="Workload used by --thread-scaling.",
    )
    parser.add_argument("--plot", default=None, help="Output path for a log-scale timing plot.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    if args.threads is not None:
        set_num_threads(args.threads)
    warm_kernels()

    rows = []
    if args.thread_scaling:
        workloads = make_workloads([args.thread_workload], args.framework, args.radius_scale, args.padding)
        workload = workloads[-1]
        for threads in args.thread_counts:
            set_num_threads(threads)
            rows.extend(benchmark_workload(*workload, resolution=args.resolution, repeats=args.repeats))
    else:
        for workload in make_workloads(args.workloads, args.framework, args.radius_scale, args.padding):
            rows.extend(benchmark_workload(*workload, resolution=args.resolution, repeats=args.repeats))

    print(f"Numba threads: {get_num_threads()}")
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\ncsv: {args.csv}")
    if args.plot:
        if args.thread_scaling:
            plot_thread_rows(rows, args.plot)
        else:
            plot_rows(rows, args.plot)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
