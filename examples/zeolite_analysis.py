from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read

from atomvoxelizer import VoxelGrid, VoxelGridAnalysis


EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate zeolite pore volume and surface area from a voxel grid.")
    parser.add_argument("framework", nargs="?", default="BEA")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument(
        "--convergence",
        type=float,
        nargs="+",
        help="Run a resolution-convergence study for the given resolutions in Angstrom.",
    )
    parser.add_argument("--core-scale", type=float, default=0.9)
    parser.add_argument("--plot", default=None, help="Output path for convergence plot.")
    return parser.parse_args()


def analyze_zeolite(atoms, resolution, core_scale):
    grid = VoxelGrid(cell=atoms.cell.array, resolution=resolution)
    centers = atoms.get_positions()
    core_radii = np.array([covalent_radii[atom.number] * core_scale for atom in atoms], dtype=np.float64)
    grid.set_spheres(centers, core_radii, value=1.0)

    analysis = VoxelGridAnalysis(grid)
    pore_regions = analysis.analyze_regions(max_value=0.0)
    pore_volume_a3 = sum(region.volume for region in pore_regions)
    pore_area_a2 = sum(region.surface_area for region in pore_regions)
    mass_amu = float(np.sum(atoms.get_masses()))
    return {
        "resolution": float(resolution),
        "regions": len(pore_regions),
        "gpts": tuple(int(value) for value in grid.gpts),
        "pore_volume_a3": pore_volume_a3,
        "pore_volume_cm3_g": analysis.volume_angstrom3_to_cm3_per_g(pore_volume_a3, mass_amu),
        "surface_area_a2": pore_area_a2,
        "surface_area_m2_g": analysis.area_angstrom2_to_m2_per_g(pore_area_a2, mass_amu),
    }


def plot_convergence(results, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    resolutions = np.array([result["resolution"] for result in results])
    pore_volumes = np.array([result["pore_volume_cm3_g"] for result in results])
    surface_areas = np.array([result["surface_area_m2_g"] for result in results])

    order = np.argsort(resolutions)
    resolutions = resolutions[order]
    pore_volumes = pore_volumes[order]
    surface_areas = surface_areas[order]

    fig, ax_volume = plt.subplots(figsize=(7, 5), constrained_layout=True)
    volume_line = ax_volume.plot(resolutions, pore_volumes, marker="o", color="tab:blue", label="pore volume")
    ax_volume.set_xlabel("grid resolution [A]")
    ax_volume.set_ylabel("pore volume [cm^3/g]", color="tab:blue")
    ax_volume.tick_params(axis="y", labelcolor="tab:blue")
    ax_volume.invert_xaxis()

    ax_area = ax_volume.twinx()
    area_line = ax_area.plot(resolutions, surface_areas, marker="s", color="tab:red", label="surface area")
    ax_area.set_ylabel("surface area [m^2/g]", color="tab:red")
    ax_area.tick_params(axis="y", labelcolor="tab:red")

    lines = volume_line + area_line
    ax_volume.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.savefig(output_path, dpi=200)


def print_result(result):
    print(f"resolution={result['resolution']:.3f} A")
    print(f"gpts={result['gpts']}")
    print(f"regions={result['regions']}")
    print(f"pore_volume={result['pore_volume_a3']:.3f} A^3")
    print(f"pore_volume={result['pore_volume_cm3_g']:.4f} cm^3/g")
    print(f"internal_surface_area={result['surface_area_a2']:.3f} A^2")
    print(f"internal_surface_area={result['surface_area_m2_g']:.1f} m^2/g")


def main():
    args = parse_args()
    atoms = read(EXAMPLE_DIR / f"{args.framework.upper()}.cif")

    print(f"framework={args.framework.upper()}")
    print(f"atoms={len(atoms)}")
    if args.convergence:
        results = [analyze_zeolite(atoms, resolution, args.core_scale) for resolution in args.convergence]
        print("resolution_A,gpts,regions,pore_volume_cm3_g,surface_area_m2_g")
        for result in results:
            print(
                f"{result['resolution']:.4f},{result['gpts']},{result['regions']},"
                f"{result['pore_volume_cm3_g']:.6f},{result['surface_area_m2_g']:.6f}"
            )
        if args.plot:
            plot_convergence(results, args.plot)
            print(f"plot={args.plot}")
    else:
        print_result(analyze_zeolite(atoms, args.resolution, args.core_scale))


if __name__ == "__main__":
    main()
