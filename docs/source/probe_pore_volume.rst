Probe Pore Volume
=================

AtomVoxelizer can estimate probe-center accessible pore volume from an existing
``VoxelGrid`` plus arrays of atomic positions and radii. The structure object is
not passed into the analysis API; callers provide the numerical data directly.

The method inflates each atom by the probe radius and marks voxels where the
probe center can fit:

.. code-block:: text

   exclusion_radius_i = atom_radius_i + probe_radius

The resulting volume is the volume available to the center of the probe. This is
not automatically the same as all pore-volume definitions reported by other
codes. For example, PoreBlazer reports a probe-occupiable ``V_PO`` from a
Connolly/free-volume sampling workflow, while its nitrogen-network lattice
points are the closer direct comparison to AtomVoxelizer's probe-center mask.

Minimal AtomVoxelizer Example
-----------------------------

This example uses a pre-existing grid, positions, and radii arrays:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import VoxelGrid, VoxelGridAnalysis

   # User-supplied numerical data.
   cell = np.array([[12.632, 0.0, 0.0], [0.0, 12.632, 0.0], [0.0, 0.0, 26.186]])
   positions = np.loadtxt("positions.txt")
   radii = np.loadtxt("atom_radii.txt")

   grid = VoxelGrid(cell, gpts=(50, 50, 103), dtype=np.float32)
   analysis = VoxelGridAnalysis(grid)

   result = analysis.analyze_probe_accessibility(
       positions=positions,
       radii=radii,
       probe_radius=1.657,
       surface_method="voxel-faces",
   )

   print(result.accessible_volume)
   print(result.accessible_surface_area)
   print(result.accessible_voxel_count)

``result.accessible_mask`` is a boolean array. ``True`` means the probe center
can occupy that voxel without overlapping any atom. Set ``write_grid=True`` to
store that binary mask back into the grid.

BEA Comparison To PoreBlazer
----------------------------

The comparison below used the BEA CIF in ``examples/zeolite/BEA.cif``. To match
PoreBlazer's default nitrogen probe setup, AtomVoxelizer used:

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - Atom radii
     - PoreBlazer ``UFF.atoms`` sigma / 2
   * - Probe radius
     - ``1.657 A`` = PoreBlazer nitrogen sigma ``3.314 A`` / 2
   * - Grid shape
     - ``50 x 50 x 103``
   * - Surface method
     - ``voxel-faces``

PoreBlazer's corrected lattice for this input is ``50 x 50 x 103`` with an
actual cubelet size of about ``0.25264 A``. Matching this grid shape avoids a
small discretization difference from AtomVoxelizer's default
``ceil(length / resolution)`` grid construction.

PoreBlazer Input
----------------

The PoreBlazer ``input.dat`` used for BEA was:

.. code-block:: text

   BEA.xyz
   12.6320000000 12.6320000000 26.1860000000
   90.0000000000 90.0000000000 90.0000000000

The matching ``defaults.dat`` was:

.. code-block:: text

   UFF.atoms
   2.58, 10.22, 298, 12.8
   3.314
   500
   0.25
   20.0, 0.25
   21908391
   1

The final line enables ``nitrogen_network.xyz`` output. That file is useful
because its point count is a direct nitrogen probe-center lattice comparison.

Matched BEA Results
-------------------

The direct probe-center lattice volumes agreed closely:

.. list-table::
   :header-rows: 1

   * - Quantity
     - AtomVoxelizer
     - PoreBlazer
     - Difference
   * - Probe radius
     - ``1.657 A``
     - ``1.657 A``
     - ``0.000%``
   * - Nitrogen probe-center lattice volume
     - ``499.967712 A^3``
     - ``502.028531 A^3``
     - ``-0.410%``
   * - Nitrogen probe-center lattice fraction
     - ``0.119654``
     - ``0.120148``
     - ``-0.410%``

PoreBlazer also reported network-accessible ``V_PO = 1782.835 A^3`` for this
run. That should not be compared directly to AtomVoxelizer's current
probe-center mask volume because PoreBlazer ``V_PO`` is a Connolly/free-volume
sampling result, not simply the number of nitrogen-accessible lattice points.

The surface areas also differ by definition. AtomVoxelizer's value above used a
voxel-face boundary estimate, while PoreBlazer reports a Monte Carlo accessible
surface area from sampled points on inflated atom surfaces. Agreement of
surface area should be evaluated only after implementing the same sampled
surface-area definition.

Timing
------

Timing was measured on an AMD EPYC 7551P 32-core CPU from the same Python
environment, using the end-to-end runner scripts for the matched BEA setup.
Each tool was run three times.

.. list-table::
   :header-rows: 1

   * - Tool
     - Best time [s]
     - Mean time [s]
     - Notes
   * - AtomVoxelizer
     - ``1.603``
     - ``1.916``
     - Includes ASE CIF loading, UFF radius parsing, probe mask construction, and voxel-face area.
   * - PoreBlazer
     - ``8.415``
     - ``8.444``
     - Includes PoreBlazer volume, surface, PSD, and nitrogen-network output.

The timing is not a pure kernel benchmark because the two workflows do different
amounts of work. It is still useful as an end-to-end comparison for this BEA
probe-volume setup.
