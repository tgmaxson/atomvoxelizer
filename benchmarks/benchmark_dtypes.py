from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
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
    raise ValueError(f"Unknown dtype benchmark backend: {name}")


def dtype_from_name(name):
    try:
        return np.dtype(name)
    except TypeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown dtype: {name}") from exc


def zeolite_workload(framework, radius_scale):
    atoms = read(ROOT / "examples" / "zeolite" / f"{framework.upper()}.cif")
    centers = atoms.get_positions()
    radii = np.array([covalent_radii[atom.number] * radius_scale for atom in atoms], dtype=np.float64)
    return atoms.cell.array, centers, radii


def run_once(cls, cell, resolution, centers, radii, dtype):
    grid = cls(cell=cell, resolution=resolution, dtype=dtype)
    grid.add_spheres(centers, radii, value=1)
    grid.set_spheres(centers, radii * 0.5, value=0)
    return grid


def time_dtype(cls, cell, resolution, centers, radii, dtype, repeats):
    run_once(cls, cell, resolution, centers[: min(2, len(centers))], radii[: min(2, len(radii))], dtype)
    times = []
    grid = None
    for _ in range(repeats):
        start = time.perf_counter()
        grid = run_once(cls, cell, resolution, centers, radii, dtype)
        times.append(time.perf_counter() - start)
    return np.array(times), grid


def format_float(value, precision=4):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = ["backend", "dtype", "atoms", "gpts", "grid MiB", "best [s]", "mean [s]", "std [s]", "note"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["backend"],
                row["dtype"],
                str(row["atoms"]),
                row["gpts"],
                format_float(row["grid_mib"], precision=2),
                format_float(row["best_s"]),
                format_float(row["mean_s"]),
                format_float(row["std_s"]),
                row["note"],
            ]
        )
    widths = [
        max(len(str(value)) for value in [header] + [table_row[i] for table_row in table_rows])
        for i, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for table_row in table_rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(table_row)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark AtomVoxelizer grid dtype choices.")
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    parser.add_argument("--framework", default="BEA")
    parser.add_argument("--resolution", type=float, default=0.35)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=dtype_from_name,
        default=[np.dtype(name) for name in ["int16", "int32", "float32", "float64", "complex64", "complex128"]],
    )
    args = parser.parse_args()

    cell, centers, radii = zeolite_workload(args.framework, args.radius_scale)
    rows = []
    for dtype in args.dtypes:
        try:
            cls = load_backend(args.backend)
            times, grid = time_dtype(cls, cell, args.resolution, centers, radii, dtype, args.repeats)
            values = grid.to_numpy()
            rows.append(
                {
                    "backend": args.backend,
                    "dtype": dtype.name,
                    "atoms": len(centers),
                    "gpts": "x".join(str(int(x)) for x in grid.gpts),
                    "grid_mib": values.nbytes / 1024**2,
                    "best_s": float(times.min()),
                    "mean_s": float(times.mean()),
                    "std_s": float(times.std()),
                    "note": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "backend": args.backend,
                    "dtype": dtype.name,
                    "atoms": len(centers),
                    "gpts": "-",
                    "grid_mib": np.nan,
                    "best_s": np.nan,
                    "mean_s": np.nan,
                    "std_s": np.nan,
                    "note": str(exc),
                }
            )
    print_table(rows)


if __name__ == "__main__":
    main()
