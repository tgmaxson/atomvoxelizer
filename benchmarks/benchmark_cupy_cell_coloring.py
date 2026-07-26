from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
from ase.cluster import wulff_construction
from ase.data import covalent_radii

from atomvoxelizer import VoxelGrid
from atomvoxelizer.voxelgrid import _cached_sphere_offsets


ROOT = Path(__file__).resolve().parents[1]


CELL_COLOR_KERNEL = r"""
extern "C" __global__
void add_colored_cells(
    float* grid,
    const int* centers,
    const int* cell_starts,
    const int* cell_counts,
    const int* color_cells,
    const int n_color_cells,
    const int* offsets,
    const int n_offsets,
    const int nx,
    const int ny,
    const int nz,
    const float value
) {
    int color_cell_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (color_cell_index >= n_color_cells) {
        return;
    }

    int cell_id = color_cells[color_cell_index];
    int atom_start = cell_starts[cell_id];
    int atom_stop = atom_start + cell_counts[cell_id];

    for (int atom = atom_start; atom < atom_stop; ++atom) {
        int cx = centers[3 * atom + 0];
        int cy = centers[3 * atom + 1];
        int cz = centers[3 * atom + 2];

        for (int n = 0; n < n_offsets; ++n) {
            int x = (cx + offsets[3 * n + 0]) % nx;
            int y = (cy + offsets[3 * n + 1]) % ny;
            int z = (cz + offsets[3 * n + 2]) % nz;
            if (x < 0) x += nx;
            if (y < 0) y += ny;
            if (z < 0) z += nz;
            grid[(x * ny + y) * nz + z] += value;
        }
    }
}
"""


def require_cupy():
    if importlib.util.find_spec("cupy") is None:
        raise RuntimeError("CuPy is not installed")
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    return cp


def optional_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return torch


def make_nanoparticle(size, padding):
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
    atoms.positions = positions
    atoms.set_cell(np.eye(3) * float(positions.max() + padding))
    atoms.set_pbc(False)
    return atoms


def positions_to_indices(positions, cell, gpts):
    frac = positions @ np.linalg.inv(cell)
    frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
    return np.floor(frac_wrapped * gpts).astype(np.int32)


def build_cell_coloring_plan(centers, cell, gpts, radius):
    lengths = np.linalg.norm(cell, axis=1)
    max_cells = np.floor(lengths / radius).astype(int)
    color_cells = np.maximum(1, max_cells // 3)
    cell_counts_3d = np.maximum(3, 3 * color_cells)
    cell_widths = lengths / cell_counts_3d
    if np.any(cell_widths < radius):
        raise ValueError("cell coloring requires coloring-cell widths >= stencil radius")

    frac = centers @ np.linalg.inv(cell)
    frac_wrapped = np.clip(frac % 1.0, 0.0, np.nextafter(1.0, 0.0))
    atom_cells_3d = np.floor(frac_wrapped * cell_counts_3d).astype(np.int32)
    atom_cells_3d = np.minimum(atom_cells_3d, cell_counts_3d - 1)
    atom_cell_ids = (
        (atom_cells_3d[:, 0] * cell_counts_3d[1] + atom_cells_3d[:, 1]) * cell_counts_3d[2]
        + atom_cells_3d[:, 2]
    ).astype(np.int32)
    n_cells = int(np.prod(cell_counts_3d))

    order = np.argsort(atom_cell_ids, kind="stable")
    sorted_cell_ids = atom_cell_ids[order]
    sorted_centers = positions_to_indices(centers, cell, gpts)[order]

    cell_starts = np.zeros(n_cells, dtype=np.int32)
    cell_counts = np.zeros(n_cells, dtype=np.int32)
    unique_cells, starts, counts = np.unique(sorted_cell_ids, return_index=True, return_counts=True)
    cell_starts[unique_cells] = starts.astype(np.int32)
    cell_counts[unique_cells] = counts.astype(np.int32)

    color_to_cells = []
    cell_ids = np.arange(n_cells, dtype=np.int32)
    ix = cell_ids // (cell_counts_3d[1] * cell_counts_3d[2])
    iy = (cell_ids // cell_counts_3d[2]) % cell_counts_3d[1]
    iz = cell_ids % cell_counts_3d[2]
    colors = (ix % 3) * 9 + (iy % 3) * 3 + (iz % 3)
    for color in range(27):
        cells = cell_ids[(colors == color) & (cell_counts > 0)]
        color_to_cells.append(cells.astype(np.int32, copy=True))

    return {
        "center_indices": sorted_centers.astype(np.int32, copy=False),
        "cell_starts": cell_starts,
        "cell_counts": cell_counts,
        "color_to_cells": color_to_cells,
        "cell_counts_3d": cell_counts_3d.astype(np.int32),
        "cell_widths": cell_widths,
    }


def cupy_baseline(cp, cell, gpts, centers, radius, value):
    from atomvoxelizer import VoxelGridCuPy

    grid = VoxelGridCuPy(cell=cell, gpts=gpts, dtype=np.float32)
    radii = np.full(len(centers), radius, dtype=np.float64)
    grid.add_spheres(centers, radii, value=value)
    grid.synchronize()
    return grid.to_numpy()


def cupy_cell_coloring(cp, cell, gpts, centers, radius, value):
    from atomvoxelizer import VoxelGridCuPy

    grid = VoxelGridCuPy(cell=cell, gpts=gpts, dtype=np.float32)
    plan = grid.prepare_cell_coloring(centers, radius)
    grid.apply_prepared_cell_coloring(plan, value=value)
    return grid.to_numpy(), plan


def prepare_cupy_cell_coloring(cp, cell, gpts, centers, radius):
    from atomvoxelizer import VoxelGridCuPy

    grid = VoxelGridCuPy(cell=cell, gpts=gpts, dtype=np.float32)
    plan = grid.prepare_cell_coloring(centers, radius)
    grid.apply_prepared_cell_coloring(plan, value=0.0)
    return grid, plan


def run_prepared_cupy_cell_coloring(cp, prepared, gpts, value):
    grid, plan = prepared
    grid.grid.fill(0.0)
    grid.apply_prepared_cell_coloring(plan, value=value)
    return grid.to_numpy()


def torch_baseline(torch, cell, gpts, centers, radius, value):
    device = torch.device("cuda")
    offsets = _cached_sphere_offsets(float(radius), tuple(gpts), tuple(map(tuple, cell)))
    center_indices = positions_to_indices(centers, cell, gpts)
    centers_gpu = torch.as_tensor(center_indices, dtype=torch.int64, device=device)
    offsets_gpu = torch.as_tensor(offsets, dtype=torch.int64, device=device)
    grid = torch.zeros(int(np.prod(gpts)), dtype=torch.float32, device=device)
    nx, ny, nz = [int(v) for v in gpts]
    values = torch.full((offsets.shape[0],), float(value), dtype=torch.float32, device=device)
    for center in centers_gpu:
        idx = center[None, :] + offsets_gpu
        x = torch.remainder(idx[:, 0], nx)
        y = torch.remainder(idx[:, 1], ny)
        z = torch.remainder(idx[:, 2], nz)
        flat = (x * ny + y) * nz + z
        grid.index_put_((flat,), values, accumulate=True)
    torch.cuda.synchronize()
    return grid.reshape(tuple(int(v) for v in gpts)).cpu().numpy()


def torch_cell_coloring(torch, cell, gpts, centers, radius, value):
    device = torch.device("cuda")
    plan = build_cell_coloring_plan(centers, cell, gpts, radius)
    offsets = _cached_sphere_offsets(float(radius), tuple(gpts), tuple(map(tuple, cell)))
    centers_gpu = torch.as_tensor(plan["center_indices"], dtype=torch.int64, device=device)
    offsets_gpu = torch.as_tensor(offsets, dtype=torch.int64, device=device)
    grid = torch.zeros(int(np.prod(gpts)), dtype=torch.float32, device=device)
    nx, ny, nz = [int(v) for v in gpts]
    values = torch.full((offsets.shape[0],), float(value), dtype=torch.float32, device=device)

    for color_cells in plan["color_to_cells"]:
        for cell_id in color_cells:
            start = int(plan["cell_starts"][cell_id])
            stop = start + int(plan["cell_counts"][cell_id])
            for atom in range(start, stop):
                idx = centers_gpu[atom][None, :] + offsets_gpu
                x = torch.remainder(idx[:, 0], nx)
                y = torch.remainder(idx[:, 1], ny)
                z = torch.remainder(idx[:, 2], nz)
                flat = (x * ny + y) * nz + z
                grid.index_put_((flat,), values, accumulate=True)

    torch.cuda.synchronize()
    return grid.reshape(tuple(int(v) for v in gpts)).cpu().numpy(), plan


def numpy_reference(cell, gpts, centers, radius, value):
    grid = VoxelGrid(cell=cell, gpts=gpts, dtype=np.float32)
    radii = np.full(len(centers), radius, dtype=np.float64)
    grid.add_spheres(centers, radii, value=value)
    return grid.to_numpy()


def numba_reference(cell, gpts, centers, radius, value):
    from atomvoxelizer import VoxelGridNumba

    grid = VoxelGridNumba(cell=cell, gpts=gpts, dtype=np.float32)
    radii = np.full(len(centers), radius, dtype=np.float64)
    grid.add_spheres(centers, radii, value=value)
    return grid.to_numpy()


def time_call(func, repeats):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return np.asarray(times, dtype=float), value


def format_float(value):
    return f"{value:.5f}" if np.isfinite(value) else "n/a"


def append_row(rows, method, atoms, gpts, radius, times, diff, note, device):
    rows.append(
        {
            "method": method,
            "atoms": int(atoms),
            "gpts": "x".join(str(int(v)) for v in gpts),
            "radius": float(radius),
            "best_s": float(np.nanmin(times)),
            "mean_s": float(np.nanmean(times)),
            "max_diff": float(diff) if np.isfinite(diff) else np.nan,
            "note": note,
            "device": device,
        }
    )


def write_csv(rows, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "atoms", "gpts", "radius", "best_s", "mean_s", "max_diff", "note", "device"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    colors = {
        "CPU Numba add_spheres": "#2f9e44",
        "CuPy add.at baseline": "#2a6fbb",
        "CuPy 27-color cell coloring": "#d97706",
        "CuPy 27-color prepared": "#7c3aed",
        "Torch index_put baseline": "#6b7280",
        "Torch 27-color cell coloring": "#111827",
    }
    markers = {
        "CPU Numba add_spheres": "o",
        "CuPy add.at baseline": "s",
        "CuPy 27-color cell coloring": "^",
        "CuPy 27-color prepared": "D",
        "Torch index_put baseline": "v",
        "Torch 27-color cell coloring": "P",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=220)
    for method, color in colors.items():
        points = [row for row in rows if row["method"] == method and np.isfinite(row["best_s"])]
        if not points:
            continue
        points.sort(key=lambda row: row["atoms"])
        ax.plot(
            [row["atoms"] for row in points],
            [row["best_s"] for row in points],
            marker=markers[method],
            color=color,
            linewidth=1.8,
            markersize=4.5,
            label=method.replace(" add_spheres", "").replace(" baseline", ""),
        )
    finite_times = [row["best_s"] for row in rows if np.isfinite(row["best_s"])]
    ax.set_yscale("log")
    if finite_times:
        ax.set_ylim(min(finite_times) * 0.55, max(finite_times) * 2.5)
    ax.set_xlabel("Pt atoms")
    ax.set_ylabel("Best wall time [s]")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.18)
    ax.legend(fontsize=7.0, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)


def main():
    parser = argparse.ArgumentParser(description="Experimental CuPy 27-color cell-coloring benchmark.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[309, 923, 2869])
    parser.add_argument("--resolution", type=float, default=0.6)
    parser.add_argument("--radius-scale", type=float, default=1.4)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--plot", default=None, help="Optional log-scale timing plot path.")
    args = parser.parse_args()

    cp = require_cupy()
    torch = optional_torch()
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"CuPy device: {device_name}")
    if torch is None:
        print("Torch CUDA: unavailable")
    else:
        print(f"Torch device: {torch.cuda.get_device_name(0)}")
    print(
        "method                          atoms  gpts         radius  best [s]  mean [s]  "
        "max |diff|  note"
    )
    print(
        "------------------------------  -----  -----------  ------  --------  --------  "
        "----------  ----"
    )

    rows = []
    for size in args.sizes:
        atoms = make_nanoparticle(size, args.padding)
        cell = atoms.cell.array
        centers = atoms.get_positions()
        radius = float(np.median([covalent_radii[atom.number] for atom in atoms]) * args.radius_scale)
        gpts = np.ceil(np.linalg.norm(cell, axis=1) / args.resolution).astype(np.int32)
        reference = numpy_reference(cell, gpts, centers, radius, value=1.0)

        numba_times, numba_result = time_call(
            lambda: numba_reference(cell, gpts, centers, radius, value=1.0),
            args.repeats,
        )
        numba_diff = float(np.max(np.abs(numba_result - reference)))
        append_row(
            rows,
            "CPU Numba add_spheres",
            len(atoms),
            gpts,
            radius,
            numba_times,
            numba_diff,
            "",
            "CPU",
        )
        print(
            f"CPU Numba add_spheres           {len(atoms):5d}  "
            f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
            f"{format_float(numba_times.min())}  {format_float(numba_times.mean())}  "
            f"{numba_diff:10.3g}  "
        )

        base_times, baseline = time_call(
            lambda: cupy_baseline(cp, cell, gpts, centers, radius, value=1.0),
            args.repeats,
        )
        base_diff = float(np.max(np.abs(baseline - reference)))
        append_row(
            rows,
            "CuPy add.at baseline",
            len(atoms),
            gpts,
            radius,
            base_times,
            base_diff,
            "",
            device_name,
        )
        print(
            f"CuPy add.at baseline            {len(atoms):5d}  "
            f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
            f"{format_float(base_times.min())}  {format_float(base_times.mean())}  "
            f"{base_diff:10.3g}  "
        )

        try:
            color_times, color_result = time_call(
                lambda: cupy_cell_coloring(cp, cell, gpts, centers, radius, value=1.0),
                args.repeats,
            )
            colored, plan = color_result
            diff = float(np.max(np.abs(colored - reference)))
            note = f"cells={tuple(int(v) for v in plan['cell_counts_3d'])}"
        except Exception as exc:
            color_times = np.array([np.nan])
            diff = np.nan
            note = repr(exc)
        append_row(
            rows,
            "CuPy 27-color cell coloring",
            len(atoms),
            gpts,
            radius,
            color_times,
            diff,
            note,
            device_name,
        )

        print(
            f"CuPy 27-color cell coloring     {len(atoms):5d}  "
            f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
            f"{format_float(color_times.min())}  {format_float(color_times.mean())}  "
            f"{diff:10.3g}  {note}"
        )

        try:
            prepared = prepare_cupy_cell_coloring(cp, cell, gpts, centers, radius)
            prepared_times, prepared_result = time_call(
                lambda: run_prepared_cupy_cell_coloring(cp, prepared, gpts, value=1.0),
                args.repeats,
            )
            prepared_diff = float(np.max(np.abs(prepared_result - reference)))
            note = f"prepared cells={tuple(int(v) for v in prepared[1]['cell_counts_3d'])}"
        except Exception as exc:
            prepared_times = np.array([np.nan])
            prepared_diff = np.nan
            note = repr(exc)
        append_row(
            rows,
            "CuPy 27-color prepared",
            len(atoms),
            gpts,
            radius,
            prepared_times,
            prepared_diff,
            note,
            device_name,
        )

        print(
            f"CuPy 27-color prepared          {len(atoms):5d}  "
            f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
            f"{format_float(prepared_times.min())}  {format_float(prepared_times.mean())}  "
            f"{prepared_diff:10.3g}  {note}"
        )

        if torch is not None:
            try:
                torch_times, torch_result = time_call(
                    lambda: torch_baseline(torch, cell, gpts, centers, radius, value=1.0),
                    args.repeats,
                )
                torch_diff = float(np.max(np.abs(torch_result - reference)))
                note = "index_put per atom"
            except Exception as exc:
                torch_times = np.array([np.nan])
                torch_diff = np.nan
                note = repr(exc)
            append_row(
                rows,
                "Torch index_put baseline",
                len(atoms),
                gpts,
                radius,
                torch_times,
                torch_diff,
                note,
                torch.cuda.get_device_name(0),
            )

            print(
                f"Torch index_put baseline        {len(atoms):5d}  "
                f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
                f"{format_float(torch_times.min())}  {format_float(torch_times.mean())}  "
                f"{torch_diff:10.3g}  {note}"
            )

            try:
                torch_color_times, torch_color_result = time_call(
                    lambda: torch_cell_coloring(torch, cell, gpts, centers, radius, value=1.0),
                    args.repeats,
                )
                torch_colored, torch_plan = torch_color_result
                torch_color_diff = float(np.max(np.abs(torch_colored - reference)))
                note = f"sequential cells={tuple(int(v) for v in torch_plan['cell_counts_3d'])}"
            except Exception as exc:
                torch_color_times = np.array([np.nan])
                torch_color_diff = np.nan
                note = repr(exc)
            append_row(
                rows,
                "Torch 27-color cell coloring",
                len(atoms),
                gpts,
                radius,
                torch_color_times,
                torch_color_diff,
                note,
                torch.cuda.get_device_name(0),
            )

            print(
                f"Torch 27-color cell coloring    {len(atoms):5d}  "
                f"{'x'.join(str(int(v)) for v in gpts):11s}  {radius:6.3f}  "
                f"{format_float(torch_color_times.min())}  {format_float(torch_color_times.mean())}  "
                f"{torch_color_diff:10.3g}  {note}"
            )

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\ncsv: {args.csv}")
    if args.plot:
        plot_rows(rows, args.plot)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
