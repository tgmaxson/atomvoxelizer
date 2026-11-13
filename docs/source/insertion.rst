Single-Atom Insertion Bias
==========================

This example uses an LJ insertion grid to sample favorable positions in a
fixed host. In the carbon-cage example below, a 0.25 Angstrom grid reduced the
variance of a Widom insertion estimate by **14x at 298 K and 22x at 77 K**
compared with uniform sampling.

The bias is useful when many insertions share the same host: the grid is built
once and reused. Finer grids gave larger gains in this test; a 1 Angstrom grid
provided little benefit. These are savings in energy evaluations, before grid
setup and sampling costs. The test uses UFF for both the bias and the target
energy; gains with another potential or in interacting-guest GCMC remain
unmeasured. See :doc:`argon_mc` for an empty-box Ar MC example that checks
incremental grid updates against full regeneration.

Carbon-Cage Example
-------------------

Run from the repository root:

.. code-block:: bash

   python examples/insertion/uff_argon.py --resolution 0.5 --samples 1000 \
       --output argon_insertion.npz --plot argon_insertion.png

The script writes insertion energies, voxel probabilities, sampled positions,
and their proposal densities to an NPZ file. It runs offline. To use a periodic
structure file, install ``AtomVoxelizer[examples]`` and pass its path:

.. code-block:: bash

   python examples/insertion/uff_argon.py framework.cif --probe Ar --cutoff 8

.. image:: _static/argon_insertion.png
   :alt: Argon insertion energies and proposal probabilities in a carbon cage
   :width: 95%

.. list-table:: Example and benchmark settings
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Value
   * - Host
     - Eight C atoms at all combinations of x, y, z = 4 or 12 Angstrom
   * - Cell
     - 16 x 16 x 16 Angstrom, fully periodic
   * - Probe
     - Single Ar atom
   * - Potential
     - UFF LJ; geometric mixing of sigma and epsilon
   * - Cutoff
     - 8 Angstrom; unshifted, no tail correction
   * - Example defaults
     - 0.5 Angstrom grid; 298.15 K; 1000 proposals; seed 42
   * - Benchmark grids
     - 1, 0.5, 0.25 Angstrom
   * - Benchmark temperatures
     - 298.15 and 77 K
   * - Sampling budgets
     - 256, 1024, 4096 energy evaluations per repeat; 64 repeats
   * - Benchmark proposal
     - 95% grid Boltzmann bias + 5% uniform over the entire cell
   * - Reference integration
     - Midpoint quadrature at 0.25 and 0.125 Angstrom
   * - Benchmark seed
     - 20261113

Sampling Results
----------------

The benchmark estimates the single-particle insertion factor

.. math::

   Q = \frac{1}{V} \int_V e^{-E(\mathbf{r})/(k_B T)}\,d\mathbf{r}.

It samples continuous positions within voxels, evaluates their actual Ar-C
energies, and corrects each contribution by its proposal density. Both methods
therefore estimate the same Q; the voxel energy only determines where to sample.

.. list-table:: Variance reduction at 4096 energy evaluations per repeat
   :header-rows: 1
   :widths: 40 30 30

   * - Proposal
     - 298.15 K
     - 77 K
   * - Uniform
     - 1.00x
     - 1.00x
   * - Grid, 1 Angstrom
     - 1.40x
     - 1.72x
   * - Grid, 0.5 Angstrom
     - 8.55x
     - 5.26x
   * - Grid, 0.25 Angstrom
     - 14.35x
     - 21.93x

Variance reduction is the ratio of uniform to biased variance across repeats.
A 14x reduction implies roughly 14 times fewer independent energy evaluations
for the same uncertainty. The ratios are estimates from 64 repeats.

.. image:: _static/insertion_sampling.png
   :alt: Insertion-factor error versus energy evaluations for uniform and grid-biased sampling
   :width: 100%

At 77 K, uniform sampling gave Q = 1.86840 +/- 0.00273 and the 0.25 Angstrom
bias gave 1.86560 +/- 0.00058, against a quadrature reference of 1.86567.
Uncertainties are standard errors over 64 repeats. Reference refinement changed
Q by 0.000018 at 298 K and 0.000184 at 77 K, below the Monte Carlo uncertainties.

The finest bias took 0.32 seconds to build in this run. Reuse can amortize this
cost; rebuilding after every host change may erase the benefit. These variance
reductions do not establish wall-clock speedups or GCMC acceptance rates.

Reproduce the comparison with:

.. code-block:: bash

   python benchmarks/benchmark_insertion_sampling.py \
       --temperatures 298.15 77 --resolutions 1 0.5 0.25 \
       --sample-counts 256 1024 4096 --repeats 64 --seed 20261113 \
       --reference-resolutions 0.25 0.125 \
       --output insertion_sampling.json --plot insertion_sampling.png

The :download:`raw results <_static/insertion_sampling.json>` include estimates,
errors, timings, and reference-refinement values.

Building a Radial Mask
----------------------

This constructs the same cage and draws one continuous insertion proposal:

.. code-block:: python

   from itertools import product
   import numpy as np
   from atomvoxelizer import VoxelGrid, LJParameters, lj_pair_energy
   from atomvoxelizer import uff_parameters, insertion_probabilities

   cell = np.eye(3) * 16
   positions = np.array(list(product([4., 12.], repeat=3)))
   host, probe = uff_parameters("C"), uff_parameters("Ar")
   pair = LJParameters(np.sqrt(host.sigma * probe.sigma),
                       np.sqrt(host.epsilon * probe.epsilon))
   mask = lambda r: lj_pair_energy(r, pair, cutoff=8.0)
   energy = VoxelGrid(cell, resolution=0.5, dtype=np.float64)
   energy.add_spheres(positions, np.full(len(positions), 8.0), mask=mask)
   probability = (
       0.95 * insertion_probabilities(energy.grid, 298.15, uniform_fraction=0)
       + 0.05 / energy.grid.size
   )
   rng = np.random.default_rng(42)
   selected = rng.choice(probability.size, p=probability.ravel())
   index = np.array(np.unravel_index(selected, energy.grid.shape))
   position = ((index + rng.random(3)) / energy.gpts) @ cell
   voxel_volume = abs(np.linalg.det(cell)) / probability.size
   q = probability[tuple(index)] / voxel_volume

For an insertion-factor estimate, evaluate the simulation energy at ``position``
and use ``exp(-E / (k_B T)) / (V * q)``. For Monte Carlo moves, include forward
and reverse proposal densities in the acceptance ratio. A biased proposal alone
does not sample the equilibrium distribution.

The explicit uniform term above gives every voxel nonzero probability.
``insertion_probabilities`` defaults to a 5% uniform component over finite-energy
voxels only; exact-overlap voxels remain excluded. The benchmark uses full-cell
support because positions away from an infinite-energy voxel center can still
have finite energy.

LJ Parameters
-------------

.. list-table:: UFF values used in the example
   :header-rows: 1

   * - Element
     - Sigma (Angstrom)
     - Epsilon (eV)
   * - C
     - 3.430851
     - 0.00455323
   * - Ar
     - 3.445996
     - 0.00802236

Parameters follow Table I of `Rappe et al. (1992)
<https://doi.org/10.1021/ja00051a040>`_, checked against the
`RDKit UFF table
<https://github.com/rdkit/rdkit/blob/master/Code/ForceField/UFF/Params.cpp>`_.
The API converts the tabulated minimum separation to sigma = x / 2**(1/6)
and well depth from kcal/mol to eV. Mix sigma geometrically for UFF or
arithmetically for Lorentz-Berthelot; both use a geometric epsilon mean.

Pass element symbols from ``UFF_ELEMENTS``; unknown symbols raise an error.
Only the LJ term is included. For custom parameters, replace ``pair`` in the
mask with an ``LJParameters(sigma, epsilon)`` instance. The grid only receives a
radial callable; Gaussian, inverse-distance, or other profiles use the same
operation. A negative ``value`` subtracts a mask at a given position.

Callable masks currently use the NumPy backend. The built-in ``constant`` and
``distance`` masks retain their existing cached-stencil behavior. Callable
masks evaluate actual subvoxel distances in bounded chunks, and
``iter_sphere_samples`` exposes the same indices and distances when needed.

Periodic Cells
--------------

Orthogonal and triclinic cells are supported with full periodicity. Each atom
contributes an LJ sphere evaluated at actual voxel-center distances. All images
inside the cutoff are summed, including multiple images of an atom when the
cutoff spans the cell.

The default cutoff is half the shortest perpendicular cell height. Larger
explicit cutoffs are allowed. Sphere bounds use reciprocal cell vectors so
skew cells retain all contributing voxels. Calculations use float64 with
chunked work arrays; larger cutoffs and stronger skew increase the cost.
