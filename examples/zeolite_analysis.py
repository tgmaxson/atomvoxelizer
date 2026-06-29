from __future__ import annotations

import argparse
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
    parser.add_argument("--core-scale", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    atoms = read(EXAMPLE_DIR / f"{args.framework.upper()}.cif")

    grid = VoxelGrid(cell=atoms.cell.array, resolution=args.resolution)
    centers = atoms.get_positions()
    core_radii = np.array([covalent_radii[atom.number] * args.core_scale for atom in atoms], dtype=np.float64)
    grid.set_spheres(centers, core_radii, value=1.0)

    analysis = VoxelGridAnalysis(grid)
    pore_regions = analysis.analyze_regions(max_value=0.0)
    pore_volume_a3 = sum(region.volume for region in pore_regions)
    pore_area_a2 = sum(region.surface_area for region in pore_regions)
    mass_amu = float(np.sum(atoms.get_masses()))

    print(f"framework={args.framework.upper()}")
    print(f"atoms={len(atoms)}")
    print(f"resolution={args.resolution:.3f} A")
    print(f"regions={len(pore_regions)}")
    print(f"pore_volume={pore_volume_a3:.3f} A^3")
    print(f"pore_volume={analysis.volume_angstrom3_to_cm3_per_g(pore_volume_a3, mass_amu):.4f} cm^3/g")
    print(f"internal_surface_area={pore_area_a2:.3f} A^2")
    print(f"internal_surface_area={analysis.area_angstrom2_to_m2_per_g(pore_area_a2, mass_amu):.1f} m^2/g")


if __name__ == "__main__":
    main()

