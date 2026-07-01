from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atomvoxelizer import VectorVoxelGrid


def build_normal_mask_grid():
    cell = np.eye(3) * 8.0
    centers = np.array(
        [
            [3.0, 4.0, 4.0],
            [5.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    radii = np.array([2.1, 2.1], dtype=float)

    grid = VectorVoxelGrid(cell=cell, resolution=0.5)
    grid.add_spheres(centers, radii, mask="normal")
    raw = grid.grid.copy()
    normalized = grid.normalize_vectors(inplace=False)
    return grid, centers, raw, normalized


def plot_normal_mask_process(output):
    import matplotlib.pyplot as plt

    grid, centers, raw, normalized = build_normal_mask_grid()
    z_index = grid.position_to_index([0.0, 0.0, 4.0])[2]
    grid.grid = raw
    raw_data = grid.quiver_slice_data(axis="z", index=z_index, stride=1, min_norm=0.1, normalize=False)
    grid.grid = normalized
    normalized_data = grid.quiver_slice_data(axis="z", index=z_index, stride=1, min_norm=0.1, normalize=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, data, title in [
        (axes[0], raw_data, "Summed normal masks"),
        (axes[1], normalized_data, "Normalized direction field"),
    ]:
        q = ax.quiver(
            data["x"],
            data["y"],
            data["u"],
            data["v"],
            data["norm"],
            angles="xy",
            scale_units="xy",
            scale=3.0,
            cmap="viridis",
        )
        ax.scatter(centers[:, 0], centers[:, 1], c="#d62728", s=80, edgecolors="black", zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(1.0, 7.0)
        ax.set_ylim(1.0, 7.0)
        ax.set_xlabel("x [A]")
        ax.set_ylabel("y [A]")
        ax.set_title(title)
        fig.colorbar(q, ax=ax, label="|vector|")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot FieldVoxelGrid normal-mask construction.")
    parser.add_argument("--output", default="docs/source/_static/field_normal_mask.png")
    args = parser.parse_args()
    output = plot_normal_mask_process(args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
