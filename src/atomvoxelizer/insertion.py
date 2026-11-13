"""Single-site Lennard-Jones insertion energies (eV, Angstrom, Kelvin).

UFF data: Rappe et al., JACS 114 (1992), doi:10.1021/ja00051a040,
Table I. Only the van der Waals term is implemented, with geometric mixing.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .voxelgrid import VoxelGrid

BOLTZMANN_EV_K = 8.617333262145e-5
KCAL_MOL_TO_EV = 0.0433641153087705

# Element: (minimum-energy separation x1 / Angstrom, well depth / kcal mol-1).
# These vdW values are shared by the UFF coordination types of each element.
# This is a documented subset, not an automatic UFF atom-typing implementation.
_UFF = {
    "H": (2.886, 0.044),
    "He": (2.362, 0.056),
    "Li": (2.451, 0.025),
    "B": (4.083, 0.180),
    "C": (3.851, 0.105),
    "N": (3.660, 0.069),
    "O": (3.500, 0.060),
    "F": (3.364, 0.050),
    "Ne": (3.243, 0.042),
    "Na": (2.983, 0.030),
    "Mg": (3.021, 0.111),
    "Al": (4.499, 0.505),
    "Si": (4.295, 0.402),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "Cl": (3.947, 0.227),
    "Ar": (3.868, 0.185),
    "K": (3.812, 0.035),
    "Ca": (3.399, 0.238),
    "Ti": (3.175, 0.017),
    "Fe": (2.912, 0.013),
    "Co": (2.872, 0.014),
    "Ni": (2.834, 0.015),
    "Cu": (3.495, 0.005),
    "Zn": (2.763, 0.124),
    "Br": (4.189, 0.251),
    "Kr": (4.141, 0.220),
    "Zr": (3.124, 0.069),
    "Ag": (3.148, 0.036),
    "I": (4.500, 0.339),
    "Xe": (4.404, 0.332),
}
UFF_ELEMENTS = tuple(_UFF)


@dataclass(frozen=True)
class LJParameters:
    """Parameters for ``4 epsilon [(sigma/r)^12 - (sigma/r)^6]`` in A/eV."""

    sigma: float
    epsilon: float

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")


def uff_parameters(element):
    """Return UFF vdW parameters for an element symbol in :data:`UFF_ELEMENTS`.

    Convert the tabulated minimum distance to sigma and kcal/mol to eV.
    Unknown elements raise KeyError; custom parameters can be supplied to
    :func:`lj_insertion_grid` as a mapping of labels to :class:`LJParameters`.
    """
    try:
        distance, depth = _UFF[element]
    except KeyError as exc:
        raise KeyError(
            f"No bundled UFF data for {element!r}; supply custom parameters"
        ) from exc
    return LJParameters(distance / 2 ** (1 / 6), depth * KCAL_MOL_TO_EV)


def lj_insertion_grid(
    cell,
    positions,
    symbols,
    probe="Ar",
    *,
    resolution=None,
    gpts=None,
    cutoff=None,
    parameters=None,
    mixing="geometric",
    chunk_size=65536,
):
    """Return a float64 VoxelGrid of single-atom insertion energies in eV.

    Evaluate atom-centered LJ spheres at actual voxel centers without snapping
    atoms. Any nonsingular cell with row lattice vectors is supported and is
    periodic in all directions. All periodic images strictly inside ``cutoff``
    contribute, including multiple images when the sphere spans the cell.
    The potential is truncated, unshifted, without tail corrections. The default
    cutoff is half the shortest perpendicular cell height. Positive infinity
    marks an exact atom overlap.

    ``parameters=None`` selects the bundled UFF subset. A custom mapping must
    contain LJParameters for every framework label and the probe. Geometric
    sigma/epsilon mixing follows UFF; ``mixing='lorentz-berthelot'`` uses an
    arithmetic sigma mean instead. Work arrays scale with ``chunk_size``;
    only the fractional bounding box of each atom's cutoff sphere is visited.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("cell must be a finite (3, 3) matrix")
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cell must be nonsingular") from exc
    # A sphere projects onto fractional axis i with radius rc*|inv(cell)[:,i]|.
    # These reciprocal-vector bounds remain valid for skew cells.
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    default_cutoff = 0.5 / reciprocal_norms.max()
    cutoff = default_cutoff if cutoff is None else float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if mixing not in ("geometric", "lorentz-berthelot"):
        raise ValueError("mixing must be 'geometric' or 'lorentz-berthelot'")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    positions = np.asarray(positions, dtype=float)
    symbols = list(symbols)
    if positions.shape != (len(symbols), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite with shape (len(symbols), 3)")
    if resolution is not None and (not np.isfinite(resolution) or resolution <= 0):
        raise ValueError("resolution must be finite and positive")
    if gpts is not None:
        counts = np.asarray(gpts)
        if (
            counts.shape != (3,)
            or not np.all(np.isfinite(counts))
            or np.any(counts <= 0)
            or np.any(counts != counts.astype(int))
        ):
            raise ValueError("gpts must contain three positive integers")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of labels to LJParameters")
    lookup = uff_parameters if parameters is None else parameters.__getitem__
    pairs = {}
    guest = lookup(probe)
    for symbol in set(symbols):
        host = lookup(symbol)
        sigma = (
            np.sqrt(host.sigma) * np.sqrt(guest.sigma)
            if mixing == "geometric"
            else (host.sigma + guest.sigma) / 2
        )
        pairs[symbol] = LJParameters(
            sigma, np.sqrt(host.epsilon) * np.sqrt(guest.epsilon)
        )
    grid = VoxelGrid(cell, resolution=resolution, gpts=gpts, dtype=np.float64)
    flat = grid.grid.ravel()
    fractional_unwrapped = positions @ inverse
    translations = np.floor(fractional_unwrapped)
    fractional_atoms = fractional_unwrapped - translations
    wrapped_positions = positions - translations @ cell
    extent = cutoff * reciprocal_norms
    for atom, cartesian_atom, symbol in zip(
        fractional_atoms, wrapped_positions, symbols
    ):
        pair = pairs[symbol]
        if pair.epsilon == 0:
            continue
        lower = np.ceil((atom - extent) * grid.gpts - 0.5).astype(int)
        upper = np.floor((atom + extent) * grid.gpts - 0.5).astype(int)
        shape = upper - lower + 1
        candidates = int(np.prod(shape))
        for start in range(0, candidates, chunk_size):
            stop = min(start + chunk_size, candidates)
            indices = (
                np.column_stack(np.unravel_index(np.arange(start, stop), shape)) + lower
            )
            delta = ((indices + 0.5) / grid.gpts) @ cell - cartesian_atom
            distance2 = np.sum(delta**2, axis=1)
            active = distance2 < cutoff**2
            indices, distance2 = indices[active], distance2[active]
            nonzero = distance2 > 0
            values = np.full(len(distance2), np.inf)
            with np.errstate(over="ignore", invalid="ignore"):
                ratio6 = (pair.sigma**2 / distance2[nonzero]) ** 3
                values[nonzero] = 4 * pair.epsilon * ratio6 * (ratio6 - 1)
            wrapped = np.ravel_multi_index(tuple((indices % grid.gpts).T), grid.gpts)
            # add.at also accumulates distinct periodic images of the same atom
            # when a cutoff sphere wraps onto the same voxel more than once.
            np.add.at(flat, wrapped, values)
    return grid


def insertion_probabilities(energies, temperature, *, uniform_fraction=0.05):
    """Return normalized voxel probabilities for energy-biased insertion.

    Shift energies by the minimum before exponentiating. Mix in a uniform
    component over finite-energy voxels to preserve proposal support despite
    numerical underflow. +inf voxels are excluded; NaN and -inf are rejected.
    To sample continuous positions, sample a voxel then uniformly within it;
    its proposal density is its probability divided by the voxel volume.
    Use that density in the Monte Carlo acceptance ratio.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive in Kelvin")
    if not np.isfinite(uniform_fraction) or not 0 <= uniform_fraction <= 1:
        raise ValueError("uniform_fraction must lie between 0 and 1")
    energy = np.asarray(energies, dtype=float)
    if np.any(np.isnan(energy)) or np.any(np.isneginf(energy)):
        raise ValueError("energies must contain only finite values or +inf")
    finite = np.isfinite(energy)
    if not np.any(finite):
        raise ValueError("at least one finite-energy voxel is required")
    weights = np.zeros_like(energy)
    with np.errstate(over="ignore", under="ignore"):
        weights[finite] = np.exp(
            -(energy[finite] - energy[finite].min()) / (BOLTZMANN_EV_K * temperature)
        )
    weights /= weights.sum()
    return (1 - uniform_fraction) * weights + uniform_fraction * finite / finite.sum()
