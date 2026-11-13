Single-Atom Insertion Bias
==========================

Version 1.1 adds a NumPy reference implementation of a single-site
Lennard-Jones insertion field. It can help propose low-energy positions for
energy-biased Monte Carlo in a fixed host. It needs no additional dependencies
for grid construction or sampling.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from atomvoxelizer import lj_insertion_grid, insertion_probabilities

   energy = lj_insertion_grid(
       np.eye(3) * 20, [[10.1, 10.2, 10.3]], ["Si"], probe="Ar",
       resolution=0.5, cutoff=8.0,
   )
   probability = insertion_probabilities(energy.grid, temperature=298.15)
   rng = np.random.default_rng(42)
   flat_index = rng.choice(probability.size, p=probability.ravel())
   index = np.array(np.unravel_index(flat_index, energy.grid.shape))
   position = ((index + rng.random(3)) / energy.gpts) @ energy.cell
   voxel_volume = abs(np.linalg.det(energy.cell)) / probability.size
   proposal_density = probability.ravel()[flat_index] / voxel_volume

The example ``examples/insertion/uff_argon.py`` runs offline using an eight-site
carbon cage. It writes an NPZ archive containing ``energy_ev``, ``probability``,
``cell``, ``gpts``, ``temperature_k``, ``probe``, ``cutoff_angstrom``,
``uniform_fraction``, ``samples_angstrom``, and
``proposal_density_per_angstrom3``. For example:

.. code-block:: bash

   python examples/insertion/uff_argon.py --resolution 0.5 --samples 1000 \
       --output argon_insertion.npz --plot argon_insertion.png

.. image:: _static/argon_insertion.png
   :alt: Argon insertion energy and voxel proposal probability in a carbon cage
   :width: 95%

To use an existing fully periodic orthogonal structure, install
``AtomVoxelizer[examples]`` and pass its path as the positional argument:

.. code-block:: bash

   python examples/insertion/uff_argon.py framework.cif --probe Ar --cutoff 8

Parameters and Conventions
--------------------------

The bundled subset contains H, He, Li, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S,
Cl, Ar, K, Ca, Ti, Fe, Co, Ni, Cu, Zn, Br, Kr, Zr, Ag, I, and Xe. Pass element
symbols, rather than UFF coordination labels. Only the van der Waals parameters
are used; no bonding, charges, electrostatics, or automatic atom typing is
included. Unknown symbols raise an error.

Values follow Table I of `Rappe et al. (1992)
<https://doi.org/10.1021/ja00051a040>`_ and were checked against the
`RDKit UFF parameter table
<https://github.com/rdkit/rdkit/blob/master/Code/ForceField/UFF/Params.cpp>`_.
UFF tabulates the minimum separation x and a well depth in kcal/mol. The API
returns sigma = x / 2**(1/6) in Angstrom and epsilon in eV for
``4 epsilon [(sigma/r)^12 - (sigma/r)^6]``. The default geometric means of both
sigma and epsilon match UFF. ``mixing="lorentz-berthelot"`` selects an arithmetic
sigma mean and a geometric epsilon mean instead.

A custom mapping replaces the bundled lookup:

.. code-block:: python

   from atomvoxelizer import LJParameters

   parameters = {"host": LJParameters(3.0, 0.01),
                 "guest": LJParameters(3.4, 0.012)}
   energy = lj_insertion_grid(
       np.eye(3) * 20, [[10, 10, 10]], ["host"], probe="guest",
       gpts=(40, 40, 40), cutoff=8, parameters=parameters,
   )

Geometry and Sampling Limits
----------------------------

This reference implementation supports three mutually orthogonal cell vectors
(including rotated cells) and full periodicity. It rejects skew cells. It
computes the actual atom-to-voxel-center distance without snapping atoms to the
grid. The potential is truncated and unshifted at the cutoff; there is no tail
correction. The cutoff defaults to half the shortest cell vector and may not
exceed it. This avoids double counting periodic images. All calculations use
float64. Work arrays are chunked, but runtime still scales with atoms times
voxels; this is a reference workflow for modest systems.

``insertion_probabilities`` shifts energies by their finite minimum before
exponentiation, normalizes them, and mixes in a 5% uniform component over finite
voxels by default. This preserves proposal support when Boltzmann weights
underflow. Exact overlaps have infinite energy and their entire voxel is
excluded from this proposal; treat those voxels as hard exclusions or combine
with a separate full-cell uniform proposal if the target distribution includes
them. Setting ``uniform_fraction=0`` gives pure normalized Boltzmann weights.

Uniform jitter within the chosen voxel produces a piecewise constant continuous
proposal with density q(r) = p(voxel) / voxel_volume. The voxel-center energy is
an approximation for proposing positions. Evaluate the actual simulation energy
at the sampled position and include the forward and reverse proposal densities
in the Metropolis-Hastings or grand-canonical acceptance ratio. These samples
alone are not an equilibrium trajectory or a complete GCMC algorithm. Rebuild
the field when the host changes; guest-guest interactions require the simulation
energy model. UFF is a generic bias model here, rather than a validated
adsorption potential for a particular host/guest pair.
