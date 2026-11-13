"""Ar birth/death MC from an empty box; audit incremental grid drift without resets."""

import argparse
import json
from pathlib import Path

import numpy as np

from atomvoxelizer import (
    insertion_probabilities,
    lj_insertion_grid,
    lj_pair_energy,
    uff_parameters,
)
from atomvoxelizer.insertion import BOLTZMANN_EV_K


class ArgonMC:
    """Cubic periodic GCMC with equally likely insertion and deletion attempts.

    Activity z=exp(beta*mu)/Lambda^3 is supplied in Angstrom^-3. The simulation
    uses the same capped, shifted potential as the grid, evaluated at actual
    particle positions. Cutoff <= L/2 permits a single minimum image per pair.
    """

    def __init__(
        self,
        length=20.0,
        resolution=0.5,
        cutoff=8.0,
        temperature=150.0,
        activity=0.001,
        energy_cap=10.0,
        seed=42,
        pair=None,
    ):
        if not np.isfinite(length) or length <= 0:
            raise ValueError("length must be finite and positive")
        if not np.isfinite(cutoff) or not 0 < cutoff <= length / 2:
            raise ValueError("MC cutoff must be positive and <= half the box length")
        if (
            not np.isfinite(temperature)
            or temperature <= 0
            or not np.isfinite(activity)
            or activity <= 0
        ):
            raise ValueError("temperature and activity must be finite and positive")
        if not np.isfinite(energy_cap) or energy_cap <= 0:
            raise ValueError("energy_cap must be finite and positive")
        self.cell = np.eye(3) * length
        self.length, self.cutoff = length, cutoff
        self.temperature, self.activity = temperature, activity
        self.energy_cap = energy_cap
        self.pair = uff_parameters("Ar") if pair is None else pair
        self.parameters = {"Ar": self.pair}
        self.rng = np.random.default_rng(seed)
        self.positions = np.empty((0, 3))
        self.grid = lj_insertion_grid(
            self.cell,
            self.positions,
            [],
            resolution=resolution,
            cutoff=cutoff,
            parameters=self.parameters,
            shift=True,
            energy_cap=energy_cap,
        )
        self.attempts = {"insert": 0, "delete": 0}
        self.accepts = {"insert": 0, "delete": 0}

    def field(self, positions, cutoff=None):
        return lj_insertion_grid(
            self.cell,
            positions,
            ["Ar"] * len(positions),
            gpts=self.grid.gpts,
            cutoff=self.cutoff if cutoff is None else cutoff,
            parameters=self.parameters,
            shift=True,
            energy_cap=self.energy_cap,
        ).grid

    def probability(self, grid=None):
        return insertion_probabilities(
            self.grid.grid if grid is None else grid, self.temperature
        )

    def density(self, probability, position):
        index = (
            np.floor((position / self.length) * self.grid.gpts).astype(int)
            % self.grid.gpts
        )
        return probability[tuple(index)] / (self.length**3 / probability.size)

    def interaction_energy(self, position, others):
        delta = position - others
        delta -= self.length * np.rint(delta / self.length)
        return float(
            lj_pair_energy(
                np.linalg.norm(delta, axis=1),
                self.pair,
                self.cutoff,
                shift=True,
                energy_cap=self.energy_cap,
            ).sum()
        )

    def step(self):
        n = len(self.positions)
        beta = 1 / (BOLTZMANN_EV_K * self.temperature)
        if self.rng.random() < 0.5:
            move = "insert"
            self.attempts[move] += 1
            probability = self.probability()
            selected = self.rng.choice(probability.size, p=probability.ravel())
            index = np.array(np.unravel_index(selected, probability.shape))
            position = (index + self.rng.random(3)) / self.grid.gpts * self.length
            q = self.density(probability, position)
            delta = self.interaction_energy(position, self.positions)
            log_ratio = np.log(self.activity) - np.log(n + 1) - np.log(q) - beta * delta
            if np.log(self.rng.random()) < min(0.0, log_ratio):
                self.grid.grid += self.field(position[None, :])
                self.positions = np.vstack([self.positions, position])
                self.accepts[move] += 1
                return True
        else:
            move = "delete"
            self.attempts[move] += 1
            if n == 0:
                return False  # Keep the 1/2 move probabilities even at N=0.
            chosen = self.rng.integers(n)
            position = self.positions[chosen]
            remaining = np.delete(self.positions, chosen, axis=0)
            removed_field = self.field(position[None, :])
            candidate_grid = self.grid.grid - removed_field
            # Reverse insertion is proposed in the state WITHOUT this atom.
            q = self.density(self.probability(candidate_grid), position)
            removed_energy = self.interaction_energy(position, remaining)
            log_ratio = (
                np.log(n) + np.log(q) - np.log(self.activity) + beta * removed_energy
            )
            if np.log(self.rng.random()) < min(0.0, log_ratio):
                self.grid.grid -= removed_field
                self.positions = remaining
                self.accepts[move] += 1
                return True
        return False

    def audit(self, step):
        regenerated = self.field(self.positions)
        difference = self.grid.grid - regenerated
        low = regenerated < 0.1  # Include weak attractive and near-zero values.
        return dict(
            step=step,
            atoms=len(self.positions),
            updates=sum(self.accepts.values()),
            max_abs_ev=float(np.max(np.abs(difference))),
            rms_ev=float(np.sqrt(np.mean(difference**2))),
            low_energy_rms_ev=(
                float(np.sqrt(np.mean(difference[low] ** 2))) if low.any() else None
            ),
            probability_l1=float(
                np.abs(self.probability() - self.probability(regenerated)).sum()
            ),
        )


def cutoff_comparison(mc, cutoffs):
    """Compare one frozen snapshot; changing cutoff does not change the atoms."""
    if any(not np.isfinite(c) or not 0 < c <= mc.length / 2 for c in cutoffs):
        raise ValueError(
            "comparison cutoffs must be positive and <= half the box length"
        )
    reference = mc.field(mc.positions, max(cutoffs))
    low = reference < 0.1
    rows = []
    for cutoff in cutoffs:
        grid = mc.field(mc.positions, cutoff)
        difference = grid - reference
        rows.append(
            dict(
                cutoff_angstrom=cutoff,
                max_abs_ev=float(np.max(np.abs(difference))),
                rms_ev=float(np.sqrt(np.mean(difference**2))),
                low_energy_rms_ev=(
                    float(np.sqrt(np.mean(difference[low] ** 2))) if low.any() else None
                ),
                energy_at_cutoff_ev=float(
                    lj_pair_energy(
                        cutoff, mc.pair, cutoff, shift=True, energy_cap=mc.energy_cap
                    )
                ),
            )
        )
    return rows


def plot_results(mc, history, comparison, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    steps = [row["step"] for row in history]
    axes[0, 0].plot(steps, [row["atoms"] for row in history])
    axes[0, 0].set(xlabel="MC attempts", ylabel="Ar atoms")
    for key, label in [
        ("max_abs_ev", "Maximum"),
        ("rms_ev", "RMS"),
        ("low_energy_rms_ev", "RMS where E < 0.1 eV"),
    ]:
        axes[0, 1].plot(steps, [row[key] for row in history], label=label)
    axes[0, 1].set(xlabel="MC attempts", ylabel="Incremental − rebuilt grid error (eV)")
    # Put the exponent on every tick; an offset such as 1e-14 is easy to miss.
    axes[0, 1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.1e}" if value else "0")
    )
    maximum_error = max(row["max_abs_ev"] for row in history)
    axes[0, 1].set_title(f"Maximum audited error: {maximum_error:.2e} eV", fontsize=10)
    axes[0, 1].legend(fontsize=8)
    distances = np.linspace(
        3.0, max(row["cutoff_angstrom"] for row in comparison) + 0.5, 1000
    )
    for row in comparison:
        cutoff = row["cutoff_angstrom"]
        axes[1, 0].plot(
            distances,
            lj_pair_energy(
                distances, mc.pair, cutoff, shift=True, energy_cap=mc.energy_cap
            ),
            label=f"{cutoff:g} Å",
        )
    axes[1, 0].set(
        xlabel="Ar–Ar distance (Å)", ylabel="Pair energy (eV)", ylim=(-0.009, 0.015)
    )
    axes[1, 0].legend(title="Cutoff", fontsize=8)
    axes[1, 1].plot(
        [r["cutoff_angstrom"] for r in comparison],
        [r["low_energy_rms_ev"] for r in comparison],
        "o-",
    )
    axes[1, 1].set(
        xlabel="Cutoff (Å)", ylabel="Low-energy grid RMS vs longest cutoff (eV)"
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--audit-every", type=int, default=25)
    parser.add_argument("--length", type=float, default=20.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument(
        "--compare-cutoffs", nargs="+", type=float, default=[4.0, 6.0, 8.0, 10.0]
    )
    parser.add_argument("--temperature", type=float, default=150.0)
    parser.add_argument("--activity", type=float, default=0.001)
    parser.add_argument("--energy-cap", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("argon_gcmc.npz"))
    parser.add_argument("--report", type=Path, default=Path("argon_gcmc.json"))
    parser.add_argument("--plot", type=Path, default=Path("argon_gcmc.png"))
    args = parser.parse_args()
    if args.steps < 0 or args.audit_every <= 0:
        parser.error("steps must be non-negative and audit-every positive")
    mc = ArgonMC(
        length=args.length,
        resolution=args.resolution,
        cutoff=args.cutoff,
        temperature=args.temperature,
        activity=args.activity,
        energy_cap=args.energy_cap,
        seed=args.seed,
    )
    if any(
        not np.isfinite(c) or not 0 < c <= mc.length / 2 for c in args.compare_cutoffs
    ):
        parser.error("comparison cutoffs must be positive and <= half the box length")
    history = [mc.audit(0)]
    for step in range(1, args.steps + 1):
        mc.step()
        if step % args.audit_every == 0 or step == args.steps:
            history.append(mc.audit(step))
    comparison = cutoff_comparison(mc, args.compare_cutoffs)
    report = dict(
        parameters={
            key: value
            for key, value in vars(args).items()
            if not isinstance(value, Path)
        },
        attempts=mc.attempts,
        accepts=mc.accepts,
        drift=history,
        cutoff_comparison=comparison,
    )
    for path in [args.output, args.report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions_angstrom=mc.positions,
        cell=mc.cell,
        incremental_grid_ev=mc.grid.grid,
        rebuilt_grid_ev=mc.field(mc.positions),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    plot_results(mc, history, comparison, args.plot)
    print(f"Accepted moves: {mc.accepts}; final N={len(mc.positions)}")
    print(f'Maximum grid drift: {max(row["max_abs_ev"] for row in history):.3g} eV')
    print(
        f'Maximum probability L1 drift: {max(row["probability_l1"] for row in history):.3g}'
    )
    print("Cutoff comparison:", json.dumps(comparison))


if __name__ == "__main__":
    main()
