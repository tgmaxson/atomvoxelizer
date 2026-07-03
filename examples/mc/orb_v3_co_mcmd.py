from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from atomvoxelizer import VoxelGrid


KB_EV_PER_K = 8.617333262145e-5
REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_SITE_FIGURE = REPO_ROOT / "docs" / "source" / "_static" / "quickstart_co_mcmd_sites.png"
QUICKSTART_FINAL_FIGURE = REPO_ROOT / "docs" / "source" / "_static" / "quickstart_co_mcmd_final.png"


def build_wulff_nanoparticle(symbol="Pt", natoms=201, lattice_constant=3.92, shape="cube"):
    """Build a finite fcc WulffPack particle."""
    try:
        from ase.build import bulk
        from wulffpack import SingleCrystal
    except ImportError as exc:
        raise SystemExit(
            "This example requires ASE and WulffPack. Install them in your environment, "
            "for example with `pip install ase wulffpack`."
        ) from exc

    primitive = bulk(symbol, "fcc", a=lattice_constant)
    if shape == "cube":
        surface_energies = {(1, 0, 0): 1.0}
    elif shape == "wulff":
        surface_energies = {
            (1, 1, 1): 1.00,
            (1, 0, 0): 1.12,
            (1, 1, 0): 1.25,
        }
    else:
        raise ValueError("shape must be 'cube' or 'wulff'")
    return SingleCrystal(surface_energies, primitive_structure=primitive, natoms=natoms).atoms


def put_cluster_in_voxel_cell(atoms, padding=20.0):
    """Return a copy of a finite cluster translated into a padded cubic cell."""
    atoms = atoms.copy()
    positions = atoms.positions
    span = positions.max(axis=0) - positions.min(axis=0)
    cell_length = float(span.max() + 2.0 * padding)
    atoms.positions = positions - positions.min(axis=0) + padding
    atoms.set_cell(np.eye(3) * cell_length)
    atoms.set_pbc(False)
    return atoms


def covalent_radii_for_atoms(atoms):
    from ase.data import covalent_radii

    return covalent_radii[atoms.numbers]


def build_coordination_surface_grid(atoms, resolution=0.35, shell_scale=1.4, core_scale=1.1):
    """Build a voxel coordination surface around a finite nanoparticle.

    The first pass adds one count for every atom whose larger shell covers a
    voxel. The second pass erases the smaller atomic cores. Sampling values near
    three then targets low-coordination surface regions without placing trials
    inside atomic cores.
    """
    radii = covalent_radii_for_atoms(atoms)
    grid = VoxelGrid(atoms.cell.array, resolution=resolution, dtype=np.float32)
    grid.add_spheres(atoms.positions, shell_scale * radii, value=1.0)
    grid.set_spheres(atoms.positions, core_scale * radii, value=0.0)
    return grid


def sample_surface_trial_sites(grid, min_count=0.5, max_count=100.0, max_sites=500, min_dist=0.6, seed=7):
    """Sample candidate CO adsorption sites from the coordination surface."""
    sites = []
    for position in grid.sample_voxels_in_range(min_count, max_count, min_dist=min_dist, seed=seed):
        sites.append(np.asarray(position, dtype=float))
        if len(sites) >= max_sites:
            break
    if not sites:
        raise RuntimeError("No trial sites were found. Try a coarser resolution or wider count range.")
    return np.array(sites)


def outer_atom_indices(atoms, fraction=0.35):
    """Pick likely surface atoms from their distance to the nanoparticle center."""
    center = atoms.positions.mean(axis=0)
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    cutoff = np.quantile(distances, 1.0 - fraction)
    return np.flatnonzero(distances >= cutoff)


def radial_variance(atoms):
    """Return the variance of atom distances from the nanoparticle center."""
    center = atoms.positions.mean(axis=0)
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    return float(np.var(distances))


def make_emt_calculator():
    """Create an ASE EMT calculator."""
    try:
        from ase.calculators.emt import EMT
    except ImportError as exc:
        raise SystemExit("EMT scoring requires ASE. Install ASE before running this example.") from exc

    return EMT()


def make_calculator_score(calculator):
    """Create a potential-energy scorer from an ASE calculator."""

    def score(atoms):
        atoms.calc = calculator
        return float(atoms.get_potential_energy())

    return score


def make_emt_score():
    """Create an ASE EMT potential-energy scorer."""
    return make_calculator_score(make_emt_calculator())


def make_orb_v3_calculator(device="cpu", model_size="inf", max_num_neighbors=None):
    """Create an ORB-V3 ASE calculator when ORB models are installed."""
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError as exc:
        raise SystemExit(
            "ORB-V3 scoring requires orb-models and torch. Install those packages "
            "in the environment where you run this example."
        ) from exc

    if model_size not in {"20", "inf"}:
        raise ValueError("model_size must be '20' or 'inf'")
    if model_size == "20":
        if max_num_neighbors is None:
            max_num_neighbors = 20
        model, adapter = pretrained.orb_v3_conservative_20_omat(device=device)
    else:
        model, adapter = pretrained.orb_v3_conservative_inf_omat(device=device)
    model.eval()
    return ORBCalculator(model, adapter, device=device, max_num_neighbors=max_num_neighbors)


def make_orb_v3_score(device="cpu"):
    """Create an ORB-V3 ASE-calculator energy scorer when ORB models are installed."""
    return make_calculator_score(make_orb_v3_calculator(device=device))


def relax_atoms(atoms, calculator, fmax=0.05, steps=50):
    """Relax atoms with an ASE optimizer and return the relaxed potential energy."""
    from ase.optimize import FIRE

    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())


def relax_new_adsorbate(atoms, calculator, adsorbate_indices, fmax=0.05, steps=50):
    """Relax selected adsorbate atoms while keeping all other atoms fixed."""
    if steps <= 0:
        return potential_energy(atoms, calculator)
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE

    adsorbate_indices = {int(index) for index in adsorbate_indices}
    fixed_indices = [index for index in range(len(atoms)) if index not in adsorbate_indices]
    old_constraint = atoms.constraints
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=int(steps))
    atoms.set_constraint(old_constraint)
    return float(atoms.get_potential_energy())


def run_minimal_mc(
    atoms,
    trial_sites,
    steps=50,
    temperature=900.0,
    max_displacement=0.35,
    local_trial_radius=3.0,
    relax=True,
    relax_fmax=0.05,
    relax_steps=50,
    seed=11,
    score_fn=None,
    calculator=None,
    return_frames=False,
    progress_every=0,
):
    """Run a Metropolis loop using voxel-sampled trial directions."""
    rng = np.random.default_rng(seed)
    if calculator is None and score_fn is None:
        calculator = make_emt_calculator()
    if score_fn is None:
        score_fn = make_calculator_score(calculator)
    if relax and calculator is None:
        raise ValueError("relax=True requires an ASE calculator")

    movable = outer_atom_indices(atoms)
    accepted = 0
    trajectory = []
    frames = []
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / (KB_EV_PER_K * temperature)

    if relax:
        current_score = relax_atoms(atoms, calculator, fmax=relax_fmax, steps=relax_steps)
    else:
        current_score = score_fn(atoms)

    if return_frames:
        initial = atoms.copy()
        initial.info.update(
            {
                "mc_step": -1,
                "mc_atom_index": -1,
                "mc_accepted": True,
                "mc_score": current_score,
                "mc_acceptance_ratio": 0.0,
                "mc_radial_variance": radial_variance(atoms),
            }
        )
        frames.append(initial)

    for step in range(steps):
        atom_index = int(rng.choice(movable))
        old_positions = atoms.positions.copy()
        old_position = atoms.positions[atom_index].copy()
        if local_trial_radius > 0.0:
            trial_distances = np.linalg.norm(trial_sites - old_position, axis=1)
            local_sites = trial_sites[trial_distances <= local_trial_radius]
            if len(local_sites) == 0:
                local_sites = trial_sites
        else:
            local_sites = trial_sites
        target = local_sites[int(rng.integers(len(local_sites)))]
        direction = target - old_position
        distance = np.linalg.norm(direction)
        if distance > max_displacement:
            direction *= max_displacement / distance
        atoms.positions[atom_index] = old_position + direction

        if relax:
            trial_score = relax_atoms(atoms, calculator, fmax=relax_fmax, steps=relax_steps)
        else:
            trial_score = score_fn(atoms)
        delta = trial_score - current_score
        accept = delta <= 0.0 or rng.random() < math.exp(-beta * delta)
        if accept:
            current_score = trial_score
            accepted += 1
        else:
            atoms.positions = old_positions

        acceptance_ratio = accepted / (step + 1)
        trajectory.append((step, atom_index, current_score, acceptance_ratio, float(accept)))
        if progress_every > 0 and ((step + 1) % progress_every == 0 or step == steps - 1):
            print(
                f"step {step + 1}/{steps}: "
                f"accepted={accepted}, "
                f"acceptance={acceptance_ratio:.3f}, "
                f"score={current_score:.6f}, "
                f"radial_variance={radial_variance(atoms):.6f}",
                flush=True,
            )
        if return_frames:
            frame = atoms.copy()
            frame.info.update(
                {
                    "mc_step": step,
                    "mc_atom_index": atom_index,
                    "mc_accepted": bool(accept),
                    "mc_score": current_score,
                    "mc_acceptance_ratio": acceptance_ratio,
                    "mc_radial_variance": radial_variance(atoms),
                }
            )
            frames.append(frame)

    trajectory = np.array(trajectory, dtype=float)
    if return_frames:
        return trajectory, frames
    return trajectory


def write_mc_trajectory(frames, output):
    """Write MC frames to an ASE-readable trajectory file."""
    from ase.io import write

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, frames)
    return output


def plot_trial_sites(atoms, trial_sites, output):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*atoms.positions.T, s=18, c="#8a8a8a", alpha=0.85, label="Wulff atoms")
    ax.scatter(*trial_sites.T, s=8, c="#2ca02c", alpha=0.45, label="CO adsorption sites")
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_zlabel("z [A]")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    return output


def plot_final_state(atoms, output):
    """Render the final ASE structure."""
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    plot_atoms(atoms, ax, rotation="12x,18y,0z", radii=0.78, show_unit_cell=0)
    ax.set_title("Final MCMD state")
    ax.set_axis_off()
    ax.set_aspect("equal")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_initial_final_states(initial_atoms, final_atoms, output):
    """Backward-compatible wrapper that now renders only the final state."""
    return plot_final_state(final_atoms, output)


def quickstart_figure_paths():
    """Return the documentation figure paths generated by ``--quickstart-figures``."""
    return QUICKSTART_SITE_FIGURE, QUICKSTART_FINAL_FIGURE


def substrate_indices(atoms):
    """Return indices of nanoparticle atoms, excluding adsorbed CO atoms."""
    return np.array([index for index, atom in enumerate(atoms) if atom.symbol not in {"C", "O"}], dtype=int)


def substrate_atoms(atoms):
    """Return a copy containing only nanoparticle atoms."""
    substrate = atoms[substrate_indices(atoms)]
    substrate.set_cell(atoms.cell)
    substrate.set_pbc(atoms.pbc)
    return substrate


def co_carbon_indices(atoms):
    """Return carbon indices used to count adsorbed CO molecules."""
    return np.array([index for index, atom in enumerate(atoms) if atom.symbol == "C"], dtype=int)


def sample_current_adsorption_sites(
    atoms,
    resolution=0.35,
    shell_scale=1.4,
    core_scale=1.1,
    min_count=0.5,
    max_count=100.0,
    max_sites=500,
    min_dist=0.6,
    seed=7,
):
    """Rebuild the voxel surface grid and sample adsorption sites for the current nanoparticle."""
    nanoparticle = substrate_atoms(atoms)
    grid = build_coordination_surface_grid(
        nanoparticle,
        resolution=resolution,
        shell_scale=shell_scale,
        core_scale=core_scale,
    )
    return sample_surface_trial_sites(
        grid,
        min_count=min_count,
        max_count=max_count,
        max_sites=max_sites,
        min_dist=min_dist,
        seed=seed,
    )


def site_occupancy(atoms, trial_sites, cutoff=2.5):
    """Return adsorption sites blocked by nearby CO carbon atoms."""
    trial_sites = np.asarray(trial_sites, dtype=float)
    occupied = np.zeros(len(trial_sites), dtype=bool)
    assignments = {}
    if len(trial_sites) == 0:
        return occupied, assignments

    for carbon_index in co_carbon_indices(atoms):
        distances = np.linalg.norm(trial_sites - atoms.positions[carbon_index], axis=1)
        site_index = int(np.argmin(distances))
        if distances[site_index] <= cutoff and not occupied[site_index]:
            occupied[site_index] = True
            assignments[site_index] = int(carbon_index)
    return occupied, assignments


def substrate_coordination_numbers(atoms, cutoff=None):
    """Return coordination numbers for nanoparticle atoms, excluding C/O."""
    substrate = substrate_indices(atoms)
    if len(substrate) == 0:
        return np.array([], dtype=int)
    positions = atoms.positions[substrate]
    if cutoff is None:
        from ase.data import covalent_radii

        radii = covalent_radii[atoms.numbers[substrate]]
        cutoff = 1.25 * 2.0 * float(np.median(radii))
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return np.count_nonzero((distances > 0.0) & (distances <= float(cutoff)), axis=1).astype(int)


def surface_atom_count(atoms, cn_threshold=11, cutoff=None):
    """Return the number of nanoparticle surface atoms with CN < ``cn_threshold``."""
    coordination = substrate_coordination_numbers(atoms, cutoff=cutoff)
    return int(np.count_nonzero(coordination < int(cn_threshold)))


def coverage_fraction(atoms, surface_count, cutoff=None):
    """Return fractional CO coverage as ``N_CO / N_surface_atoms``."""
    del cutoff
    surface_count = int(surface_count)
    if surface_count <= 0:
        return 0.0
    return float(len(co_carbon_indices(atoms)) / surface_count)


def add_co_adsorbate(atoms, site, bond_length=1.15, height=0.0):
    """Return a copy with one CO molecule placed with C at the selected site."""
    from ase import Atoms

    trial = atoms.copy()
    substrate = substrate_indices(trial)
    center = trial.positions[substrate].mean(axis=0) if len(substrate) else trial.positions.mean(axis=0)
    normal = np.asarray(site, dtype=float) - center
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm
    carbon_position = np.asarray(site, dtype=float) + height * normal
    oxygen_position = carbon_position + bond_length * normal
    trial += Atoms("CO", positions=[carbon_position, oxygen_position])
    trial.set_cell(atoms.cell)
    trial.set_pbc(atoms.pbc)
    return trial


def remove_co_adsorbate(atoms, carbon_index):
    """Return a copy with the CO molecule containing ``carbon_index`` removed."""
    trial = atoms.copy()
    carbon_position = trial.positions[int(carbon_index)]
    oxygen_indices = [index for index, atom in enumerate(trial) if atom.symbol == "O"]
    if not oxygen_indices:
        raise ValueError("No oxygen atom found for CO desorption")
    distances = np.linalg.norm(trial.positions[oxygen_indices] - carbon_position, axis=1)
    oxygen_index = int(oxygen_indices[int(np.argmin(distances))])
    for index in sorted([int(carbon_index), oxygen_index], reverse=True):
        del trial[index]
    return trial


def potential_energy(atoms, calculator):
    """Evaluate potential energy with an ASE calculator."""
    atoms.calc = calculator
    return float(atoms.get_potential_energy())


def run_md_segment(atoms, calculator, temperature=500.0, steps=50, timestep_fs=1.0, friction=0.02):
    """Run a short Langevin MD segment and return the final potential energy."""
    atoms.calc = calculator
    if steps <= 0:
        return potential_energy(atoms, calculator)
    from ase import units
    from ase.md.langevin import Langevin

    dynamics = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature, friction=friction)
    dynamics.run(int(steps))
    return potential_energy(atoms, calculator)


def run_co_adsorption_mcmd(
    atoms,
    trial_sites,
    calculator,
    steps=100,
    temperature=500.0,
    co_chemical_potential=-1.0,
    target_coverage=0.5,
    max_coverage=1.0,
    chemical_potential_step=0.005,
    chemical_potential_min=-5.0,
    chemical_potential_max=5.0,
    md_steps=50,
    md_timestep_fs=1.0,
    md_friction=0.02,
    adsorption_probability=0.5,
    rebuild_sites_each_step=True,
    grid_resolution=0.35,
    shell_scale=1.4,
    core_scale=1.1,
    min_count=0.5,
    max_count=100.0,
    max_sites=500,
    site_min_dist=0.6,
    site_block_radius=2.5,
    surface_cn_threshold=11,
    surface_cn_cutoff=None,
    co_bond_length=1.15,
    optimize_added_co=True,
    co_opt_fmax=0.05,
    co_opt_steps=50,
    seed=11,
    return_frames=False,
    progress_every=10,
    md_runner=None,
):
    """Run simple grand-canonical CO adsorption/desorption MCMD.

    Adsorption attempts add one CO molecule at an empty voxel-sampled surface
    site. By default the voxel grid and site list are rebuilt from the current
    nanoparticle geometry every MC cycle, excluding CO atoms. Existing CO blocks
    adsorption attempts within ``site_block_radius``. Desorption attempts remove
    one adsorbed CO. Coverage is ``N_CO`` divided by the number of nanoparticle
    atoms with CN < 11 by default, and adsorption is capped at
    ``max_coverage`` times that surface-atom count. The Metropolis score is
    ``E - mu_CO * N_CO``. After every MC decision, accepted or rejected, the
    accepted state is propagated by a short MD segment.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    if max_coverage <= 0.0:
        raise ValueError("max_coverage must be positive")
    if target_coverage > max_coverage:
        raise ValueError("target_coverage cannot exceed max_coverage")
    if chemical_potential_min > chemical_potential_max:
        raise ValueError("chemical_potential_min cannot exceed chemical_potential_max")
    trial_sites = np.asarray(trial_sites, dtype=float)
    if trial_sites.ndim != 2 or trial_sites.shape[1] != 3 or len(trial_sites) == 0:
        raise ValueError("trial_sites must have shape (N, 3)")

    rng = np.random.default_rng(seed)
    md_runner = run_md_segment if md_runner is None else md_runner
    beta = 1.0 / (KB_EV_PER_K * temperature)
    current = atoms.copy()
    current_energy = potential_energy(current, calculator)
    mu = float(co_chemical_potential)
    accepted = 0
    records = []
    frames = []

    for step in range(int(steps)):
        if rebuild_sites_each_step:
            trial_sites = sample_current_adsorption_sites(
                current,
                resolution=grid_resolution,
                shell_scale=shell_scale,
                core_scale=core_scale,
                min_count=min_count,
                max_count=max_count,
                max_sites=max_sites,
                min_dist=site_min_dist,
                seed=seed + step,
            )
        occupied, assignments = site_occupancy(current, trial_sites, cutoff=site_block_radius)
        empty_sites = np.flatnonzero(~occupied)
        occupied_sites = np.flatnonzero(occupied)
        current_carbon_indices = co_carbon_indices(current)
        current_surface_count = surface_atom_count(
            current,
            cn_threshold=surface_cn_threshold,
            cutoff=surface_cn_cutoff,
        )
        max_co = max(0, int(math.floor(max_coverage * current_surface_count)))
        can_adsorb = len(empty_sites) > 0 and len(current_carbon_indices) < max_co
        can_desorb = len(current_carbon_indices) > 0

        if not can_adsorb and not can_desorb:
            raise RuntimeError("No adsorption or desorption moves are available")
        if not can_adsorb:
            action = "desorb"
        elif not can_desorb:
            action = "adsorb"
        else:
            action = "adsorb" if rng.random() < adsorption_probability else "desorb"

        old_n_co = len(current_carbon_indices)
        old_grand = current_energy - mu * old_n_co
        if action == "adsorb":
            site_index = int(empty_sites[int(rng.integers(len(empty_sites)))])
            trial = add_co_adsorbate(current, trial_sites[site_index], bond_length=co_bond_length)
            if optimize_added_co:
                relax_new_adsorbate(
                    trial,
                    calculator,
                    adsorbate_indices=[len(trial) - 2, len(trial) - 1],
                    fmax=co_opt_fmax,
                    steps=co_opt_steps,
                )
        else:
            if len(occupied_sites) > 0:
                site_index = int(occupied_sites[int(rng.integers(len(occupied_sites)))])
                carbon_index = assignments[site_index]
            else:
                site_index = -1
                carbon_index = int(current_carbon_indices[int(rng.integers(len(current_carbon_indices)))])
            trial = remove_co_adsorbate(current, carbon_index)

        trial_energy = potential_energy(trial, calculator)
        trial_n_co = len(co_carbon_indices(trial))
        trial_grand = trial_energy - mu * trial_n_co
        delta = trial_grand - old_grand
        accept = delta <= 0.0 or rng.random() < math.exp(-beta * delta)
        if accept:
            current = trial
            current_energy = trial_energy
            accepted += 1

        current_energy = md_runner(
            current,
            calculator,
            temperature=temperature,
            steps=md_steps,
            timestep_fs=md_timestep_fs,
            friction=md_friction,
        )
        current_n_co = len(co_carbon_indices(current))
        current_surface_count = surface_atom_count(
            current,
            cn_threshold=surface_cn_threshold,
            cutoff=surface_cn_cutoff,
        )
        coverage = coverage_fraction(current, current_surface_count)
        mu += float(chemical_potential_step) * (float(target_coverage) - coverage)
        mu = float(np.clip(mu, chemical_potential_min, chemical_potential_max))
        current_grand = current_energy - mu * current_n_co
        acceptance_ratio = accepted / (step + 1)
        action_code = 1.0 if action == "adsorb" else -1.0
        records.append(
            (
                float(step),
                action_code,
                float(accept),
                float(current_n_co),
                coverage,
                current_energy,
                current_grand,
                mu,
                acceptance_ratio,
            )
        )

        if progress_every > 0 and ((step + 1) % progress_every == 0 or step == steps - 1):
            print(
                f"step {step + 1}/{steps}: action={action}, accepted={accepted}, "
                f"acceptance={acceptance_ratio:.3f}, n_CO={current_n_co}, "
                f"surface_atoms={current_surface_count}, coverage={coverage:.3f}, "
                f"mu_CO={mu:.3f} eV, energy={current_energy:.6f}",
                flush=True,
            )
        if return_frames:
            frame = current.copy()
            frame.info.update(
                {
                    "mcmd_step": step,
                    "mcmd_action": action,
                    "mcmd_accepted": bool(accept),
                    "mcmd_n_co": current_n_co,
                    "mcmd_surface_atoms": current_surface_count,
                    "mcmd_coverage": coverage,
                    "mcmd_energy": current_energy,
                    "mcmd_mu_co": mu,
                    "mcmd_acceptance_ratio": acceptance_ratio,
                }
            )
            frames.append(frame)

    trajectory = np.array(records, dtype=float)
    if return_frames:
        return current, trajectory, frames
    return current, trajectory


def main():
    parser = argparse.ArgumentParser(description="Voxel-guided CO adsorption/desorption MCMD on a small Wulff nanoparticle.")
    parser.add_argument("--symbol", default="Pt")
    parser.add_argument("--natoms", type=int, default=55, help="Target nanoparticle atom count; WulffPack returns a shell-compatible count.")
    parser.add_argument("--shape", choices=("cube", "wulff"), default="cube")
    parser.add_argument("--padding", type=float, default=20.0, help="Vacuum padding around the finite nanoparticle in Angstrom.")
    parser.add_argument("--resolution", type=float, default=0.35)
    parser.add_argument("--shell-scale", type=float, default=1.4)
    parser.add_argument("--core-scale", type=float, default=1.1)
    parser.add_argument("--min-count", type=float, default=0.5, help="Minimum voxel shell count for adsorption sites.")
    parser.add_argument("--max-count", type=float, default=100.0, help="Maximum voxel shell count for adsorption sites.")
    parser.add_argument("--max-sites", type=int, default=500)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=500.0,
        help="Metropolis and MD temperature in Kelvin.",
    )
    parser.add_argument("--mu-co", type=float, default=-1.0, help="Initial CO chemical potential in eV.")
    parser.add_argument("--target-coverage", type=float, default=0.5, help="Target CO coverage relative to nanoparticle surface atoms.")
    parser.add_argument("--max-coverage", type=float, default=1.0, help="Maximum allowed CO coverage relative to nanoparticle surface atoms.")
    parser.add_argument("--mu-step", type=float, default=0.005, help="Feedback step used to adjust the CO chemical potential.")
    parser.add_argument("--mu-min", type=float, default=-5.0, help="Lower clamp for the adaptive CO chemical potential in eV.")
    parser.add_argument("--mu-max", type=float, default=5.0, help="Upper clamp for the adaptive CO chemical potential in eV.")
    parser.add_argument("--md-steps", type=int, default=50, help="MD steps run after every MC decision.")
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-friction", type=float, default=0.02)
    parser.add_argument("--site-min-dist", type=float, default=0.6, help="Minimum distance between sampled adsorption sites.")
    parser.add_argument("--site-block-radius", type=float, default=2.5, help="CO carbon atoms block adsorption sites within this radius.")
    parser.add_argument("--no-rebuild-sites", action="store_true", help="Reuse the initial adsorption-site sample instead of rebuilding it each MC cycle.")
    parser.add_argument("--surface-cn-threshold", type=int, default=11, help="Surface atoms have nanoparticle CN below this value.")
    parser.add_argument(
        "--surface-cn-cutoff",
        type=float,
        default=None,
        help="Distance cutoff for nanoparticle coordination numbers. Default uses covalent radii.",
    )
    parser.add_argument("--co-bond-length", type=float, default=1.15)
    parser.add_argument("--no-initial-opt", action="store_true", help="Skip optimization of the clean nanoparticle before site generation.")
    parser.add_argument("--initial-opt-fmax", type=float, default=0.05, help="Initial nanoparticle optimizer force threshold in eV/A.")
    parser.add_argument("--initial-opt-steps", type=int, default=100, help="Maximum optimizer steps for the clean nanoparticle.")
    parser.add_argument("--no-co-opt", action="store_true", help="Skip CO-only optimization after adsorption insertion.")
    parser.add_argument("--co-opt-fmax", type=float, default=0.05, help="CO-only optimizer force threshold in eV/A.")
    parser.add_argument("--co-opt-steps", type=int, default=50, help="Maximum optimizer steps for newly inserted CO.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--progress-every", type=int, default=10, help="Print MCMD progress every N steps. Use 0 to disable.")
    parser.add_argument("--calculator", choices=("orb-v3", "emt"), default="orb-v3")
    parser.add_argument("--device", default="cpu", help="Device passed to ORB-V3.")
    parser.add_argument("--orb-model-size", choices=("inf", "20"), default="inf", help="Conservative ORB-V3 model size.")
    parser.add_argument("--orb-neighbors", type=int, default=None, help="Maximum ORB neighbor count. Defaults to 20 for --orb-model-size 20.")
    parser.add_argument("--plot", default=None, help="Optional path for a 3D plot of atoms and trial sites.")
    parser.add_argument("--state-plot", default=None, help="Optional path for an ASE final-state plot.")
    parser.add_argument(
        "--quickstart-figures",
        action="store_true",
        help="Write the site and final-state figures used by the quickstart documentation.",
    )
    parser.add_argument(
        "--trajectory",
        default=str(Path(__file__).with_name("orb_v3_co_mcmd.traj")),
        help="ASE trajectory output path. Use an empty string to disable writing frames.",
    )
    args = parser.parse_args()
    if args.quickstart_figures:
        site_figure, final_figure = quickstart_figure_paths()
        if args.plot is None:
            args.plot = str(site_figure)
        if args.state_plot is None:
            args.state_plot = str(final_figure)

    atoms = put_cluster_in_voxel_cell(build_wulff_nanoparticle(args.symbol, args.natoms, shape=args.shape), padding=args.padding)
    if args.calculator == "orb-v3":
        calculator = make_orb_v3_calculator(
            args.device,
            model_size=args.orb_model_size,
            max_num_neighbors=args.orb_neighbors,
        )
    else:
        calculator = make_emt_calculator()
    if not args.no_initial_opt and args.initial_opt_steps > 0:
        print(
            f"optimizing initial nanoparticle: fmax={args.initial_opt_fmax}, "
            f"steps={args.initial_opt_steps}",
            flush=True,
        )
        relax_atoms(atoms, calculator, fmax=args.initial_opt_fmax, steps=args.initial_opt_steps)
    initial_atoms = atoms.copy()
    initial_positions = atoms.positions.copy()
    initial_radial_variance = radial_variance(atoms)
    grid = build_coordination_surface_grid(
        atoms,
        resolution=args.resolution,
        shell_scale=args.shell_scale,
        core_scale=args.core_scale,
    )
    trial_sites = sample_surface_trial_sites(
        grid,
        min_count=args.min_count,
        max_count=args.max_count,
        max_sites=args.max_sites,
        min_dist=args.site_min_dist,
        seed=args.seed,
    )
    mcmd_result = run_co_adsorption_mcmd(
        atoms,
        trial_sites,
        calculator,
        steps=args.steps,
        temperature=args.temperature,
        co_chemical_potential=args.mu_co,
        target_coverage=args.target_coverage,
        max_coverage=args.max_coverage,
        chemical_potential_step=args.mu_step,
        chemical_potential_min=args.mu_min,
        chemical_potential_max=args.mu_max,
        md_steps=args.md_steps,
        md_timestep_fs=args.md_timestep_fs,
        md_friction=args.md_friction,
        rebuild_sites_each_step=not args.no_rebuild_sites,
        grid_resolution=args.resolution,
        shell_scale=args.shell_scale,
        core_scale=args.core_scale,
        min_count=args.min_count,
        max_count=args.max_count,
        max_sites=args.max_sites,
        site_min_dist=args.site_min_dist,
        site_block_radius=args.site_block_radius,
        surface_cn_threshold=args.surface_cn_threshold,
        surface_cn_cutoff=args.surface_cn_cutoff,
        co_bond_length=args.co_bond_length,
        optimize_added_co=not args.no_co_opt,
        co_opt_fmax=args.co_opt_fmax,
        co_opt_steps=args.co_opt_steps,
        seed=args.seed,
        return_frames=bool(args.trajectory),
        progress_every=args.progress_every,
    )
    if args.trajectory:
        atoms, trajectory, frames = mcmd_result
    else:
        atoms, trajectory = mcmd_result
        frames = None
    final_substrate = substrate_indices(atoms)
    final_surface_atoms = surface_atom_count(
        atoms,
        cn_threshold=args.surface_cn_threshold,
        cutoff=args.surface_cn_cutoff,
    )
    displacements = np.linalg.norm(atoms.positions[final_substrate] - initial_positions[: len(final_substrate)], axis=1)

    print(f"atoms: {len(atoms)}")
    print(f"substrate atoms: {len(final_substrate)}")
    print(f"surface atoms (CN < {args.surface_cn_threshold}): {final_surface_atoms}")
    print(f"CO molecules: {len(co_carbon_indices(atoms))}")
    print(f"grid points: {tuple(int(x) for x in grid.gpts)}")
    print(f"trial sites: {len(trial_sites)}")
    print(f"accepted moves: {int(trajectory[:, 2].sum())}/{len(trajectory)}")
    print(f"initial radial variance: {initial_radial_variance:.6f}")
    print(f"final radial variance: {radial_variance(atoms):.6f}")
    print(f"final coverage: {trajectory[-1, 4]:.3f}")
    print(f"final energy: {trajectory[-1, 5]:.6f}")
    print(f"final mu_CO: {trajectory[-1, 7]:.3f} eV")
    print(f"acceptance ratio: {trajectory[-1, 8]:.3f}")
    print(f"mean substrate displacement: {displacements.mean():.4f} A")
    print(f"max substrate displacement: {displacements.max():.4f} A")
    if args.trajectory:
        print(f"trajectory: {write_mc_trajectory(frames, args.trajectory)}")
    if args.plot:
        print(f"plot: {plot_trial_sites(initial_atoms, trial_sites, args.plot)}")
    if args.state_plot:
        print(f"state plot: {plot_final_state(atoms, args.state_plot)}")


if __name__ == "__main__":
    main()
