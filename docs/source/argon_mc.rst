Iterative Ar Monte Carlo
========================

Pairwise fields can be added and removed without rebuilding the entire grid.
In this example, **2627 accepted updates over 3000 MC attempts agreed with full
regeneration to within 1.5e-14 eV**. The incremental grid was never reset.
Cutoff choice had a much larger effect: the low-energy RMS difference between
8 and 10 Angstrom cutoffs was 0.50 meV for the final snapshot.

Run from the repository root:

.. code-block:: bash

   python examples/insertion/argon_gcmc.py --steps 3000 --audit-every 25 \
       --output argon_gcmc.npz --report argon_gcmc.json --plot argon_gcmc.png

.. list-table:: Run settings
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value
   * - Initial state
     - Empty 20 x 20 x 20 Angstrom periodic box
   * - Moves
     - Equally likely insertion or deletion; uniform choice of atom for deletion
   * - Temperature / activity
     - 150 K / 0.001 Angstrom^-3
   * - Pair model
     - UFF Ar; energy-shifted LJ; 10 eV cap per pair; no tail correction
   * - MC cutoff / grid spacing
     - 8 Angstrom / 0.5 Angstrom
   * - Insertion proposals
     - 95% grid Boltzmann bias + 5% uniform
   * - Audits
     - Every 25 attempts, including rejected moves; no grid resets
   * - Cutoff comparison
     - 4, 6, 8, 10 Angstrom on the same final configuration
   * - Seed
     - 42

Finite Core and Cutoff
----------------------

The example uses

.. math::

   u(r) = \begin{cases}
       \min(10\,\mathrm{eV},\ u_{LJ}(r)-u_{LJ}(r_c)), & r < r_c,\\
       0, & r \geq r_c.
   \end{cases}

At r = 0 the pair energy is 10 eV. The cap is applied **before summing pairs**;
a voxel's total energy can exceed 10 eV. This keeps subtraction finite without
rescaling the attractive energies. The energy approaches zero continuously at
the cutoff and is exactly zero at and beyond it. The force is not shifted.
This is an energy-shifted LJ potential, as in the
`LAMMPS shift convention <https://docs.lammps.org/pair_modify.html>`_.

.. code-block:: python

   import numpy as np
   from atomvoxelizer import VoxelGrid, lj_insertion_grid, lj_pair_energy, uff_parameters

   cell = np.eye(3) * 20
   grid = VoxelGrid(cell, resolution=0.5, dtype=np.float64)
   position = [10.1, 10.2, 10.3]
   pair = uff_parameters("Ar")
   assert lj_pair_energy(0.0, pair, 8.0, shift=True, energy_cap=10.0) == 10.0
   assert lj_pair_energy(8.0, pair, 8.0, shift=True, energy_cap=10.0) == 0.0

   # Field contributed by one accepted Ar atom:
   contribution = lj_insertion_grid(
       cell, [position], ["Ar"], gpts=grid.gpts,
       cutoff=8.0, shift=True, energy_cap=10.0,
   )
   grid.grid += contribution.grid  # Subtract this field on deletion.

``shift`` and ``energy_cap`` are optional. Their defaults retain the earlier
unshifted, singular LJ model. The example uses the capped, shifted model for
both its grid and its actual MC energies. Calculations use float64.

Update and Audit
----------------

An accepted insertion adds the new atom's field; an accepted deletion subtracts
it. Rejected moves leave the grid unchanged. Every 25 attempts, the script
regenerates a separate field from all current positions and compares it with
the accumulated field. The comparison never replaces the accumulated grid.

.. image:: _static/argon_gcmc.png
   :alt: Ar population, incremental grid drift, shifted LJ curves, and cutoff sensitivity
   :width: 100%

.. list-table:: Largest discrepancies across all audits
   :header-rows: 1

   * - Metric
     - Result
   * - Maximum absolute grid error
     - 1.42e-14 eV
   * - Whole-grid RMS error
     - 2.53e-15 eV
   * - RMS error where rebuilt E < 0.1 eV
     - 2.52e-15 eV
   * - Sum of absolute differences in voxel probabilities
     - 1.43e-13

The errors accumulate at floating-point precision, far below the meV scale of
the attractive field. For this run, incremental addition/subtraction preserves
the full-regeneration result, including weak-energy regions.

Cutoff Sensitivity
------------------

The following comparison freezes the final nine-atom configuration. Differences
include both the omitted tail and the cutoff-dependent energy shift.

.. list-table:: Grid difference from the 10 Angstrom cutoff
   :header-rows: 1

   * - Cutoff (Angstrom)
     - Whole-grid RMS (meV)
     - RMS where reference E < 0.1 eV (meV)
   * - 4
     - 6.84
     - 6.11
   * - 6
     - 2.01
     - 1.93
   * - 8
     - 0.51
     - 0.50
   * - 10
     - 0
     - 0

A short cutoff changes the attractive field appreciably; this error dominates
update drift. The 10 Angstrom field is a comparison reference, not a converged
infinite-cutoff result. Each pair curve is exactly zero at its own cutoff.

MC Acceptance
-------------

The simulation starts empty and exchanges Ar atoms with a reservoir. Activity
is z = exp(beta*mu)/Lambda^3 in Angstrom^-3. For N atoms, an insertion at r with
proposal density q_X(r) is accepted with probability

.. math::

   A_{ins} = \min\left(1,\frac{z\,e^{-\beta\Delta U}}{(N+1)q_X(r)}\right).

Deletion uses the reciprocal ratio for the reverse insertion, evaluated in the
state **without the deleted atom**. The insertion bias therefore changes as
atoms enter or leave, while detailed balance uses the appropriate state in each
direction. Empty-box deletion attempts are rejected; move selection remains
50/50. Pair energies are evaluated at actual positions, without grid
interpolation. The example restricts the cutoff to at most half the cubic box
length so each pair has one contributing image.

The NPZ file stores final positions and both grids. The
:download:`full report <_static/argon_gcmc.json>` records every audit, accepted
move counts, and cutoff differences. This run validates incremental grid
updates; it does not establish equilibration or a sampling-efficiency gain.
