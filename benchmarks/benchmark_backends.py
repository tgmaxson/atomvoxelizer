from __future__ import annotations

import argparse
import csv
import importlib
import os
import time
from pathlib import Path

import numpy as np
from ase.build import fcc111
from ase.cluster import wulff_construction
from ase.data import covalent_radii
from ase.io import read

from atomvoxelizer import VoxelGrid


ROOT = Path(__file__).resolve().parents[1]


def make_atoms_workload(atoms, radius_scale):
    centers = atoms.get_positions()
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    return atoms.cell.array, centers, radii


def make_zeolite_workloads(framework, radius_scale):
    atoms = read(ROOT / "examples" / "zeolite" / f"{framework.upper()}.cif")
    scale_factors = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (2, 2, 4),
        (2, 2, 5),
        (2, 2, 6),
        (2, 3, 5),
    ]
    workloads = []
    for scale in scale_factors:
        scaled = atoms.repeat(scale)
        cell, centers, radii = make_atoms_workload(scaled, radius_scale)
        workloads.append(
            {
                "workload": "zeolite",
                "scale": "x".join(str(value) for value in scale),
                "cell": cell,
                "centers": centers,
                "radii": radii,
            }
        )
    return workloads


def center_cluster(atoms, padding):
    atoms = atoms.copy()
    positions = atoms.get_positions()
    positions -= positions.min(axis=0)
    positions += padding
    lengths = positions.max(axis=0) + padding
    cell_length = max(float(lengths.max()), 2.0 * padding)
    atoms.positions = positions
    atoms.set_cell(np.eye(3) * cell_length)
    atoms.set_pbc(False)
    return atoms


def make_nanoparticle_workloads(radius_scale, padding):
    target_sizes = [55, 147, 309, 561, 923, 1415, 2057, 2869, 3400, 3800, 5000, 5700]
    workloads = []
    for size in target_sizes:
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
        workloads.append(
            {
                "workload": "nanoparticle",
                "scale": str(size),
                "cell": cell,
                "centers": centers,
                "radii": radii,
            }
        )
    return workloads


def make_surface_workloads(radius_scale):
    sizes = [
        (4, 4, 3),
        (6, 6, 4),
        (8, 8, 5),
        (12, 12, 5),
        (16, 16, 6),
        (20, 20, 6),
        (22, 22, 6),
        (24, 24, 6),
        (25, 25, 6),
        (28, 28, 6),
        (31, 31, 6),
    ]
    workloads = []
    for size in sizes:
        atoms = fcc111("Pt", size=size, vacuum=6.0, a=3.92)
        atoms.center(vacuum=6.0, axis=2)
        atoms.set_pbc([True, True, False])
        cell, centers, radii = make_atoms_workload(atoms, radius_scale)
        workloads.append(
            {
                "workload": "surface",
                "scale": "x".join(str(value) for value in size),
                "cell": cell,
                "centers": centers,
                "radii": radii,
            }
        )
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


def voxel_centers(cell, gpts):
    nx, ny, nz = [int(value) for value in gpts]
    ix, iy, iz = np.meshgrid(
        np.arange(nx) + 0.5,
        np.arange(ny) + 0.5,
        np.arange(nz) + 0.5,
        indexing="ij",
    )
    frac = np.stack([ix / nx, iy / ny, iz / nz], axis=-1)
    return frac @ np.asarray(cell, dtype=np.float64)


def direct_atom_grid_scan(cell, gpts, centers, radii):
    """Simple all-pairs atom-grid mask used as a baseline."""
    cell = np.asarray(cell, dtype=np.float64)
    cell_inv = np.linalg.inv(cell)
    points = voxel_centers(cell, gpts)
    mask = np.zeros(tuple(int(value) for value in gpts), dtype=np.float32)
    for center, radius in zip(centers, radii):
        disp = points - center
        disp_frac = disp @ cell_inv
        disp_frac -= np.round(disp_frac)
        disp_mic = disp_frac @ cell
        dist2 = np.sum(disp_mic * disp_mic, axis=-1)
        mask[dist2 <= radius * radius] = 1.0
    return mask


def make_numpy_mask(cell, resolution, centers, radii):
    grid = VoxelGrid(cell=cell, resolution=resolution, dtype=np.float32)
    grid.set_spheres(centers, radii, value=1.0)
    return grid.to_numpy(), grid.gpts


def make_numba_mask(cell, resolution, centers, radii):
    module = importlib.import_module("atomvoxelizer.numba_backend")
    grid = module.VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    grid.set_spheres(centers, radii, value=1.0)
    return grid.to_numpy(), grid.gpts


def make_numba_cell_colored_mask(cell, resolution, centers, radii):
    module = importlib.import_module("atomvoxelizer.numba_backend")
    grid = module.VoxelGridNumba(cell=cell, resolution=resolution, dtype=np.float32)
    grid.add_spheres_cell_colored(centers, radii, value=1.0)
    grid.clamp_grid(0.0, 1.0)
    return grid.to_numpy(), grid.gpts


def make_cpp_mask(cell, resolution, centers, radii):
    module = importlib.import_module("atomvoxelizer.cpp_backend")
    grid = module.VoxelGridCpp(cell=cell, resolution=resolution, dtype=np.float32)
    grid.set_spheres(centers, radii, value=1.0)
    return grid.to_numpy(), grid.gpts


def prepare_cupy_plans(cell, resolution, centers, radii):
    module = importlib.import_module("atomvoxelizer.cupy_backend")
    plan_grid = module.VoxelGridCuPy(cell=cell, resolution=resolution, dtype=np.float32)
    unique_radii = np.unique(radii)
    coloring_radius = float(np.max(unique_radii))
    plans = []
    for radius in unique_radii:
        selected = radii == radius
        plans.append(
            plan_grid.prepare_cell_coloring(
                centers[selected],
                float(radius),
                coloring_radius=coloring_radius,
            )
        )
    return plans, plan_grid.gpts


def make_cupy_mask(cell, resolution, centers, radii, plans=None):
    module = importlib.import_module("atomvoxelizer.cupy_backend")
    if plans is None:
        plans, _ = prepare_cupy_plans(cell, resolution, centers, radii)
    grid = module.VoxelGridCuPy(cell=cell, resolution=resolution, dtype=np.float32)
    for plan in plans:
        grid.apply_prepared_cell_coloring(plan, value=1.0)
    grid.clamp_grid(0.0, 1.0)
    return grid.to_numpy(), grid.gpts


def time_call(func, repeats):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return np.array(times, dtype=float), value


def warm_numba():
    try:
        cell = np.eye(3) * 8.0
        centers = np.array([[4.0, 4.0, 4.0]], dtype=float)
        radii = np.array([1.0], dtype=float)
        make_numba_mask(cell, 1.0, centers, radii)
        make_numba_cell_colored_mask(cell, 1.0, centers, radii)
    except Exception:
        pass


def warm_cupy():
    try:
        cell = np.eye(3) * 8.0
        centers = np.array([[4.0, 4.0, 4.0]], dtype=float)
        radii = np.array([1.0], dtype=float)
        make_cupy_mask(cell, 1.0, centers, radii)
    except Exception:
        pass


def set_numba_threads(count):
    if count is None:
        return
    module = importlib.import_module("numba")
    module.set_num_threads(int(count))


def benchmark_one(workload, methods, resolution, repeats, include_direct):
    cell = workload["cell"]
    centers = workload["centers"]
    radii = workload["radii"]
    rows = []

    numpy_times, (numpy_values, gpts) = time_call(
        lambda: make_numpy_mask(cell, resolution, centers, radii),
        repeats=repeats,
    )
    rows.append(
        row_from_timing(workload, "VoxelGrid NumPy", numpy_times, gpts)
    )

    if "numba" in methods:
        try:
            numba_times, (numba_values, numba_gpts) = time_call(
                lambda: make_numba_mask(cell, resolution, centers, radii),
                repeats=repeats,
            )
            rows.append(
                row_from_timing(
                    workload,
                    "VoxelGrid Numba",
                    numba_times,
                    numba_gpts,
                    numba_values,
                )
            )
        except Exception as exc:
            rows.append(missing_row(workload, "VoxelGrid Numba", gpts, str(exc)))

    if "numba-cell-colored" in methods:
        try:
            colored_times, (colored_values, colored_gpts) = time_call(
                lambda: make_numba_cell_colored_mask(cell, resolution, centers, radii),
                repeats=repeats,
            )
            rows.append(
                row_from_timing(
                    workload,
                    "VoxelGrid Numba 27-color",
                    colored_times,
                    colored_gpts,
                    colored_values,
                )
            )
        except Exception as exc:
            rows.append(missing_row(workload, "VoxelGrid Numba 27-color", gpts, str(exc)))

    if "cpp" in methods:
        try:
            cpp_times, (cpp_values, cpp_gpts) = time_call(
                lambda: make_cpp_mask(cell, resolution, centers, radii),
                repeats=repeats,
            )
            rows.append(
                row_from_timing(
                    workload,
                    "VoxelGrid C++",
                    cpp_times,
                    cpp_gpts,
                    cpp_values,
                )
            )
        except Exception as exc:
            rows.append(missing_row(workload, "VoxelGrid C++", gpts, str(exc)))

    if "cupy" in methods:
        try:
            cupy_plans, cupy_gpts = prepare_cupy_plans(
                cell,
                resolution,
                centers,
                radii,
            )
            cupy_times, (cupy_values, cupy_gpts) = time_call(
                lambda: make_cupy_mask(cell, resolution, centers, radii, plans=cupy_plans),
                repeats=repeats,
            )
            rows.append(
                row_from_timing(
                    workload,
                    "VoxelGrid CuPy",
                    cupy_times,
                    cupy_gpts,
                    cupy_values,
                )
            )
        except Exception as exc:
            rows.append(missing_row(workload, "VoxelGrid CuPy", gpts, str(exc)))

    if include_direct:
        direct_times, direct_values = time_call(
            lambda: direct_atom_grid_scan(cell, gpts, centers, radii),
            repeats=1,
        )
        rows.append(
            row_from_timing(
                workload,
                "Direct atom-grid scan",
                direct_times,
                gpts,
                direct_values,
            )
        )
    return rows


def row_from_timing(workload, method, times, gpts, values=None):
    return {
        "workload": workload["workload"],
        "scale": workload["scale"],
        "method": method,
        "atoms": int(len(workload["centers"])),
        "gpts": "x".join(str(int(value)) for value in gpts),
        "best_s": float(times.min()),
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "note": "",
    }


def missing_row(workload, method, gpts, note):
    return {
        "workload": workload["workload"],
        "scale": workload["scale"],
        "method": method,
        "atoms": int(len(workload["centers"])),
        "gpts": "x".join(str(int(value)) for value in gpts),
        "best_s": np.nan,
        "mean_s": np.nan,
        "std_s": np.nan,
        "note": note,
    }


def format_float(value, precision=4):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = ["workload", "scale", "method", "atoms", "gpts", "best [s]", "mean [s]", "std [s]", "note"]
    table_rows = [
        [
            row["workload"],
            row["scale"],
            row["method"],
            str(row["atoms"]),
            row["gpts"],
            format_float(row["best_s"]),
            format_float(row["mean_s"]),
            format_float(row["std_s"]),
            row["note"],
        ]
        for row in rows
    ]
    widths = [
        max(len(str(value)) for value in [header] + [table_row[index] for table_row in table_rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for table_row in table_rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(table_row)))


def write_csv(rows, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["workload", "scale", "method", "atoms", "gpts", "best_s", "mean_s", "std_s", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(input_path):
    with open(input_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["atoms"] = int(row["atoms"])
        for key in ("best_s", "mean_s", "std_s"):
            row[key] = float(row[key])
    return rows


def plot_rows(rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    colors = {
        "Direct atom-grid scan": "#b23a48",
        "VoxelGrid NumPy": "#2a6fbb",
        "VoxelGrid Numba": "#2f9e44",
        "VoxelGrid Numba 27-color": "#d97706",
        "VoxelGrid C++": "#7c4dff",
        "VoxelGrid CuPy": "#0891b2",
    }
    plot_methods = [
        "Direct atom-grid scan",
        "VoxelGrid NumPy",
        "VoxelGrid Numba",
        "VoxelGrid Numba 27-color",
        "VoxelGrid C++",
        "VoxelGrid CuPy",
    ]
    markers = {
        "zeolite": "o",
        "nanoparticle": "s",
        "surface": "^",
    }
    workloads = [name for name in ["zeolite", "nanoparticle", "surface"] if any(row["workload"] == name for row in rows)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)

    present_methods = [method for method in plot_methods if any(row["method"] == method for row in rows)]
    for method in plot_methods:
        for workload in workloads:
            points = [
                row
                for row in rows
                if row["workload"] == workload and row["method"] == method and np.isfinite(row["best_s"])
            ]
            if not points:
                continue
            points.sort(key=lambda row: row["atoms"])
            ax.plot(
                [row["atoms"] for row in points],
                [row["best_s"] for row in points],
                marker=markers[workload],
                color=colors[method],
                label=f"{method}, {workload}",
                linewidth=1.8,
                markersize=6.0,
            )
    ax.set_yscale("log")
    finite_times = [
        row["best_s"]
        for row in rows
        if row["method"] in present_methods and np.isfinite(row["best_s"])
    ]
    if finite_times:
        ax.set_ylim(min(finite_times) * 0.15, max(finite_times) * 4.0)
    ax.set_xlabel("Atoms")
    ax.set_ylabel("Best wall time [s]")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.18)

    method_handles = [
        plt.Line2D([0], [0], color=color, lw=2.0, label=method.replace("VoxelGrid ", ""))
        for method, color in colors.items()
        if method in present_methods
    ]
    system_handles = [
        plt.Line2D([0], [0], color="#4a4a4a", marker=marker, linestyle="None", label=workload.capitalize())
        for workload, marker in markers.items()
        if workload in workloads
    ]
    first = ax.legend(handles=method_handles, fontsize=7.4, loc="upper left", frameon=True, title="Backend")
    ax.add_artist(first)
    ax.legend(handles=system_handles, fontsize=7.4, loc="lower right", frameon=True, title="System")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark direct atom-grid mask generation against AtomVoxelizer CPU and GPU backends."
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["zeolite", "nanoparticle", "surface"],
        default=["zeolite", "nanoparticle", "surface"],
    )
    parser.add_argument("--resolution", type=float, default=1.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius-scale", type=float, default=0.85)
    parser.add_argument("--framework", default="BEA")
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--plot", default=None, help="Output path for a log-scale timing plot.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--no-direct", action="store_true", help="Skip the direct atom-grid scan baseline.")
    parser.add_argument("--no-numba", action="store_true", help="Skip the optional Numba backend.")
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=24,
        help="Numba thread count for the Numba and 27-color paths (default: 24).",
    )
    parser.add_argument(
        "--no-numba-cell-colored",
        action="store_true",
        help="Skip the opt-in Numba 27-color cell-colored add path.",
    )
    parser.add_argument("--no-cpp", action="store_true", help="Skip the optional C++ backend.")
    parser.add_argument(
        "--cupy",
        action="store_true",
        help="Include the CuPy GPU backend (requires a visible CUDA device).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append only workload scales not already present in --csv.",
    )
    args = parser.parse_args()

    if args.append and not args.csv:
        parser.error("--append requires --csv")

    methods = ["numpy"]
    set_numba_threads(args.numba_threads)
    if not args.no_numba:
        methods.append("numba")
        warm_numba()
    if not args.no_numba_cell_colored:
        methods.append("numba-cell-colored")
        warm_numba()
    if not args.no_cpp:
        methods.append("cpp")
    if args.cupy:
        methods.append("cupy")
        warm_cupy()

    workloads = make_workloads(args.workloads, args.framework, args.radius_scale, args.padding)
    rows = read_csv_rows(args.csv) if args.append and Path(args.csv).exists() else []
    existing_workloads = {(row["workload"], row["scale"]) for row in rows}
    if args.csv and not args.append:
        write_csv(rows, args.csv)
    for index, workload in enumerate(workloads, start=1):
        workload_key = (workload["workload"], workload["scale"])
        if workload_key in existing_workloads:
            continue
        rows.extend(
            benchmark_one(
                workload,
                methods=methods,
                resolution=args.resolution,
                repeats=args.repeats,
                include_direct=not args.no_direct,
            )
        )
        existing_workloads.add(workload_key)
        if args.csv:
            write_csv(rows, args.csv)
        print(
            f"completed {index}/{len(workloads)}: {workload['workload']} {workload['scale']}",
            flush=True,
        )

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\ncsv: {args.csv}")
    if args.plot:
        plot_rows(rows, args.plot)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
