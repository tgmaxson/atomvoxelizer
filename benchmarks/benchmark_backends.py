from __future__ import annotations

import argparse
import importlib
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark AtomVoxelizer backends.")
    parser.add_argument("--backends", nargs="+", default=["numpy", "numba", "taichi", "cupy"])
    parser.add_argument("--workload", choices=["synthetic", "zeolite", "wulff"], default="zeolite")
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

    cell, centers, radii = make_workload(args)
    reference = None

    print(
        "workload,backend,backend_name,atoms,gpts,best_s,mean_s,std_s,"
        "occupied_voxels,value_sum,max_abs_diff"
    )
    for backend in args.backends:
        try:
            cls = load_backend(backend)
            times, grid = time_backend(cls, cell, args.resolution, centers, radii, args.repeats)
        except ImportError as exc:
            print(f"{args.workload},{backend},missing,{len(centers)},(),nan,nan,nan,0,nan,nan # {exc}")
            continue

        values = grid.to_numpy()
        if reference is None:
            reference = values
        occupied, total, max_abs_diff = consistency_summary(values, reference)
        print(
            f"{args.workload},{backend},{grid.backend_name},{len(centers)},"
            f"{tuple(int(x) for x in grid.gpts)},{times.min():.6f},{times.mean():.6f},"
            f"{times.std():.6f},{occupied},{total:.6f},{max_abs_diff:.6g}"
        )


if __name__ == "__main__":
    main()

