"""Compare continuous uniform and UFF-grid importance sampling in a carbon cage.

Run from the repository root; only NumPy and Matplotlib are required.
The target is Q = (1/V) integral exp(-E(r)/(kT)) dr for a single Ar probe.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from atomvoxelizer import (
    VoxelGrid,
    LJParameters,
    insertion_probabilities,
    lj_pair_energy,
    uff_parameters,
)
from atomvoxelizer.insertion import BOLTZMANN_EV_K


def carbon_cage():
    """Return the same offline 16 A carbon cage used by uff_argon.py."""
    cell = np.eye(3) * 16
    atoms = (
        np.array(np.meshgrid([4.0, 12.0], [4.0, 12.0], [4.0, 12.0], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    return cell, atoms


def carbon_field(cell, atoms, resolution, cutoff):
    grid = VoxelGrid(cell, resolution=resolution, dtype=np.float64)
    host, guest = uff_parameters("C"), uff_parameters("Ar")
    pair = LJParameters(
        np.sqrt(host.sigma * guest.sigma), np.sqrt(host.epsilon * guest.epsilon)
    )
    for atom in atoms:
        grid.add_sphere(atom, cutoff, mask=lambda r: lj_pair_energy(r, pair, cutoff))
    return grid


def direct_energy(points, cell, atoms, cutoff):
    """Evaluate continuous Ar-C energies independently of the voxel field."""
    host, guest = uff_parameters("C"), uff_parameters("Ar")
    sigma = np.sqrt(host.sigma * guest.sigma)
    epsilon = np.sqrt(host.epsilon * guest.epsilon)
    inverse = np.linalg.inv(cell)
    result = np.zeros(len(points))
    for atom in atoms:
        delta = (points - atom) @ inverse
        delta -= np.rint(delta)
        r2 = np.sum((delta @ cell) ** 2, axis=1)
        active = (r2 > 0) & (r2 < cutoff**2)
        with np.errstate(over="ignore"):
            ratio6 = (sigma**2 / r2[active]) ** 3
            result[active] += 4 * epsilon * ratio6 * (ratio6 - 1)
        result[r2 == 0] = np.inf
    return result


def importance_contributions(points, density, cell, atoms, temperature, cutoff):
    """Return unbiased contributions exp(-beta E)/(V q) and actual energies."""
    energy = direct_energy(points, cell, atoms, cutoff)
    weight = np.exp(-energy / (BOLTZMANN_EV_K * temperature))
    return weight / (abs(np.linalg.det(cell)) * density), energy


def draw_proposals(rng, probability, cell, count):
    """Sample voxels and then uniformly within each; return positions and q."""
    flat = rng.choice(probability.size, size=count, p=probability.ravel())
    indices = np.column_stack(np.unravel_index(flat, probability.shape))
    points = ((indices + rng.random((count, 3))) / probability.shape) @ cell
    voxel_volume = abs(np.linalg.det(cell)) / probability.size
    return points, probability.ravel()[flat] / voxel_volume


def summarize(contributions, energies, counts, reference):
    """Summarize independent repeat means at each evaluation budget."""
    cumulative = np.cumsum(contributions, axis=1)
    rows = []
    for count in counts:
        estimates = cumulative[:, count - 1] / count
        rows.append(
            dict(
                samples=count,
                estimate=float(estimates.mean()),
                repeat_sd=float(estimates.std(ddof=1)),
                standard_error=float(estimates.std(ddof=1) / np.sqrt(len(estimates))),
                rmse=float(np.sqrt(np.mean((estimates - reference) ** 2))),
            )
        )
    # A fixed, physically interpretable favorable-energy threshold.
    favorable = energies < -0.008
    return dict(estimates=rows, favorable_fraction=float(favorable.mean()))


def run_benchmark(
    resolutions, temperatures, counts, repeats, seed, reference_resolutions
):
    cell, atoms = carbon_cage()
    cutoff = 8.0
    result = dict(
        seed=seed,
        repeats=repeats,
        sample_counts=counts,
        host="8-site carbon cage, 16 A orthogonal periodic cell",
        probe="Ar",
        cutoff_angstrom=cutoff,
        full_cell_uniform_fraction=0.05,
        numpy_version=np.__version__,
        favorable_threshold_ev=-0.008,
        temperatures=[],
    )
    references = []
    for resolution in reference_resolutions:
        grid = carbon_field(cell, atoms, resolution, cutoff)
        references.append((resolution, grid.grid))
    grids = []
    for resolution in resolutions:
        start = perf_counter()
        grid = carbon_field(cell, atoms, resolution, cutoff)
        grids.append((resolution, grid, perf_counter() - start))
    streams = iter(
        np.random.SeedSequence(seed).spawn(len(temperatures) * (len(grids) + 1))
    )
    for temperature in temperatures:
        reference_values = [
            float(np.exp(-energy / (BOLTZMANN_EV_K * temperature)).mean())
            for _, energy in references
        ]
        reference = reference_values[-1]
        entry = dict(
            temperature_k=temperature,
            quadrature=[
                dict(resolution_angstrom=r, value=v)
                for (r, _), v in zip(references, reference_values)
            ],
            methods=[],
        )
        for resolution, grid, build_seconds in [(None, None, 0.0)] + grids:
            rng = np.random.default_rng(next(streams))
            start = perf_counter()
            n = repeats * max(counts)
            if grid is None:
                points = rng.random((n, 3)) @ cell
                density = np.full(n, 1 / abs(np.linalg.det(cell)))
            else:
                # Mix uniform mass over ALL voxels, including infinite-energy
                # centers: continuous off-center positions still need support.
                probability = (
                    0.95
                    * insertion_probabilities(
                        grid.grid, temperature, uniform_fraction=0
                    )
                    + 0.05 / grid.grid.size
                )
                points, density = draw_proposals(rng, probability, cell, n)
            contribution, energy = importance_contributions(
                points, density, cell, atoms, temperature, cutoff
            )
            seconds = perf_counter() - start
            metrics = summarize(
                contribution.reshape(repeats, -1), energy, counts, reference
            )
            entry["methods"].append(
                dict(
                    method="uniform" if grid is None else f"grid {resolution:g} A",
                    resolution_angstrom=resolution,
                    build_seconds=build_seconds,
                    sampling_and_energy_seconds=seconds,
                    **metrics,
                )
            )
        uniform_sd = entry["methods"][0]["estimates"][-1]["repeat_sd"]
        for method in entry["methods"]:
            method["variance_reduction"] = (
                uniform_sd / method["estimates"][-1]["repeat_sd"]
            ) ** 2
        result["temperatures"].append(entry)
    return result


def plot_results(result, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(result["temperatures"]),
        figsize=(11, 4),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, entry in zip(axes[0], result["temperatures"]):
        for method in entry["methods"]:
            rows = method["estimates"]
            ax.loglog(
                [r["samples"] for r in rows],
                [r["rmse"] for r in rows],
                "o-",
                label=method["method"],
            )
        ax.set(
            xlabel="Continuous energy evaluations per repeat",
            ylabel="RMSE of insertion factor Q",
            title=f'{entry["temperature_k"]:g} K; {result["repeats"]} repeats',
        )
        ax.grid(alpha=0.25)
        ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", type=float, nargs="+", default=[1.0, 0.5, 0.25]
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[298.15, 77.0])
    parser.add_argument(
        "--sample-counts", type=int, nargs="+", default=[256, 1024, 4096]
    )
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument(
        "--reference-resolutions", type=float, nargs="+", default=[0.25, 0.125]
    )
    parser.add_argument("--output", type=Path, default=Path("insertion_sampling.json"))
    parser.add_argument("--plot", type=Path, default=Path("insertion_sampling.png"))
    args = parser.parse_args()
    if args.repeats < 2 or any(n < 1 for n in args.sample_counts):
        parser.error("use at least two repeats and positive sample counts")
    if any(
        not np.isfinite(x) or x <= 0
        for x in args.resolutions + args.temperatures + args.reference_resolutions
    ):
        parser.error("resolutions and temperatures must be finite and positive")
    result = run_benchmark(
        args.resolutions,
        args.temperatures,
        args.sample_counts,
        args.repeats,
        args.seed,
        args.reference_resolutions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot_results(result, args.plot)
    for entry in result["temperatures"]:
        print(
            f'{entry["temperature_k"]:g} K; quadrature Q={entry["quadrature"][-1]["value"]:.6f}'
        )
        for method in entry["methods"]:
            row = method["estimates"][-1]
            print(
                f'  {method["method"]:12s} Q={row["estimate"]:.6f} +/- {row["standard_error"]:.6f}; '
                f'variance gain={method["variance_reduction"]:.2f}x; favorable={100*method["favorable_fraction"]:.2f}%'
            )


if __name__ == "__main__":
    main()
