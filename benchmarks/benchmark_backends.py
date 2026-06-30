from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

import numpy as np
from ase.cluster import wulff_construction
from ase.data import covalent_radii
from ase.io import read

from atomvoxelizer import VoxelGrid


ROOT = Path(__file__).resolve().parents[1]


def load_backend(name):
    if name == "numpy":
        return VoxelGrid
    if name == "numba":
        module = importlib.import_module("atomvoxelizer.numba_backend")
        return module.VoxelGridNumba
    if name == "taichi":
        module = importlib.import_module("atomvoxelizer.taichi_backend")
        return module.VoxelGridTaichi
    if name in {"taichi-gpu", "taichi_gpu"}:
        module = importlib.import_module("atomvoxelizer.taichi_backend")
        return module.VoxelGridTaichiGPU
    if name == "cupy":
        module = importlib.import_module("atomvoxelizer.cupy_backend")
        return module.VoxelGridCuPy
    raise ValueError(f"Unknown backend: {name}")


def make_synthetic_workload(cell_length, atom_count, seed):
    rng = np.random.default_rng(seed)
    centers = rng.random((atom_count, 3)) * cell_length
    radii = rng.choice(np.array([0.55, 0.75, 1.0], dtype=np.float64), size=atom_count)
    return np.eye(3) * cell_length, centers, radii


def make_zeolite_workload(framework, radius_scale):
    atoms = read(ROOT / "examples" / f"{framework.upper()}.cif")
    return make_atoms_workload(atoms, radius_scale)


def make_atoms_workload(atoms, radius_scale):
    centers = atoms.get_positions()
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    return atoms.cell.array, centers, radii


def make_wulff_workload(size, radius_scale, padding):
    atoms = wulff_construction(
        "Pt",
        surfaces=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        energies=[1.0, 1.08, 0.92],
        size=size,
        structure="fcc",
        latticeconstant=3.92,
        rounding="closest",
    )
    positions = atoms.get_positions()
    positions -= positions.min(axis=0)
    positions += padding
    lengths = positions.max(axis=0) + padding
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    return np.diag(lengths), positions, radii


def make_workload(args):
    if args.workload == "synthetic":
        return make_synthetic_workload(args.cell_length, args.atoms, args.seed)
    if args.workload == "zeolite":
        return make_zeolite_workload(args.framework, args.radius_scale)
    if args.workload == "wulff":
        return make_wulff_workload(args.wulff_size, args.radius_scale, args.padding)
    raise ValueError(f"Unknown workload: {args.workload}")


def run_once(cls, cell, resolution, centers, radii):
    grid = cls(cell=cell, resolution=resolution)
    grid.add_spheres(centers, radii, value=1.0)
    grid.set_spheres(centers, radii * 0.5, value=-1.0)
    grid.clamp_grid(-1.0, 1.0)
    synchronize = getattr(grid, "synchronize", None)
    if synchronize is not None:
        synchronize()
    return grid


def time_backend(cls, cell, resolution, centers, radii, repeats):
    run_once(cls, cell, resolution, centers[: min(2, len(centers))], radii[: min(2, len(radii))])
    times = []
    grid = None
    for _ in range(repeats):
        start = time.perf_counter()
        grid = run_once(cls, cell, resolution, centers, radii)
        times.append(time.perf_counter() - start)
    return np.array(times), grid


def consistency_summary(values, reference):
    occupied = int(np.count_nonzero(values))
    total = float(values.sum())
    if reference is None:
        return occupied, total, 0.0
    return occupied, total, float(np.max(np.abs(values - reference)))


def benchmark_workload(workload, backend_names, repeats):
    cell, centers, radii = workload["cell"], workload["centers"], workload["radii"]
    rows = []
    reference = None
    for backend in backend_names:
        try:
            cls = load_backend(backend)
            times, grid = time_backend(cls, cell, workload["resolution"], centers, radii, repeats)
        except Exception as exc:
            rows.append(
                {
                    "workload": workload["name"],
                    "scale": workload.get("scale", "-"),
                    "backend": backend,
                    "backend_name": "missing",
                    "atoms": len(centers),
                    "gpts": "-",
                    "best_s": np.nan,
                    "mean_s": np.nan,
                    "std_s": np.nan,
                    "occupied": 0,
                    "value_sum": np.nan,
                    "max_abs_diff": np.nan,
                    "note": str(exc),
                }
            )
            continue

        values = grid.to_numpy()
        if reference is None:
            reference = values
        occupied, total, max_abs_diff = consistency_summary(values, reference)
        rows.append(
            {
                "workload": workload["name"],
                "scale": workload.get("scale", "-"),
                "backend": backend,
                "backend_name": grid.backend_name,
                "atoms": len(centers),
                "gpts": "x".join(str(int(x)) for x in grid.gpts),
                "best_s": float(times.min()),
                "mean_s": float(times.mean()),
                "std_s": float(times.std()),
                "occupied": occupied,
                "value_sum": total,
                "max_abs_diff": max_abs_diff,
                "note": "",
            }
        )
    return rows


def format_float(value, precision=4):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = [
        "workload",
        "scale",
        "backend",
        "atoms",
        "gpts",
        "best [s]",
        "mean [s]",
        "std [s]",
        "max |diff|",
        "note",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["workload"],
                row["scale"],
                f"{row['backend']} ({row['backend_name']})",
                str(row["atoms"]),
                row["gpts"],
                format_float(row["best_s"]),
                format_float(row["mean_s"]),
                format_float(row["std_s"]),
                format_float(row["max_abs_diff"], precision=3),
                row["note"],
            ]
        )

    widths = [
        max(len(str(value)) for value in [header] + [table_row[i] for table_row in table_rows])
        for i, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for table_row in table_rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(table_row)))


def zeolite_scaling_workloads(framework, resolution, radius_scale):
    atoms = read(ROOT / "examples" / f"{framework.upper()}.cif")
    scale_factors = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]
    workloads = []
    for scale in scale_factors:
        scaled_atoms = atoms.repeat(scale)
        cell, centers, radii = make_atoms_workload(scaled_atoms, radius_scale)
        workloads.append(
            {
                "name": f"{framework.upper()} zeolite",
                "scale": "x".join(str(value) for value in scale),
                "cell": cell,
                "centers": centers,
                "radii": radii,
                "resolution": resolution,
            }
        )
    return workloads


def plot_scaling(rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    backends = []
    for row in rows:
        if row["backend_name"] != "missing" and row["backend"] not in backends:
            backends.append(row["backend"])

    for backend in backends:
        backend_rows = [row for row in rows if row["backend"] == backend and row["backend_name"] != "missing"]
        backend_rows.sort(key=lambda row: row["atoms"])
        ax.plot(
            [row["atoms"] for row in backend_rows],
            [row["best_s"] for row in backend_rows],
            marker="o",
            label=backend,
        )

    ax.set_xlabel("atom count")
    ax.set_ylabel("best time [s]")
    ax.set_title("Zeolite voxelization scaling")
    ax.legend()
    fig.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AtomVoxelizer backends.")
    parser.add_argument("--backends", nargs="+", default=["numpy", "numba", "taichi", "cupy"])
    parser.add_argument("--workload", choices=["synthetic", "zeolite", "wulff"], default="zeolite")
    parser.add_argument("--zeolite-scaling", action="store_true")
    parser.add_argument("--plot", default=None, help="Output path for benchmark plot.")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--framework", default="BEA")
    parser.add_argument("--wulff-size", type=int, default=1000)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--cell-length", type=float, default=32.0)
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.zeolite_scaling:
        workloads = zeolite_scaling_workloads(args.framework, args.resolution, args.radius_scale)
    else:
        cell, centers, radii = make_workload(args)
        workloads = [
            {
                "name": args.workload,
                "cell": cell,
                "centers": centers,
                "radii": radii,
                "resolution": args.resolution,
            }
        ]

    rows = []
    for workload in workloads:
        rows.extend(benchmark_workload(workload, args.backends, args.repeats))

    print_table(rows)
    if args.plot:
        plot_scaling(rows, args.plot)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
