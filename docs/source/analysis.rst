Analysis
========

``VoxelGridAnalysis`` turns voxel grids into geometric quantities such as
connected-region volumes, surface areas, meshes, and probe-center accessible
pore volumes. Install the analysis dependencies when these features are needed:

.. code-block:: bash

   pip install "AtomVoxelizer[analysis]"

Connected Volumes
-----------------

The analysis class can select voxel regions by value, label connected
components, and convert voxel counts to physical volumes using the determinant
of the periodic cell:

.. code-block:: python

   from atomvoxelizer import VoxelGridAnalysis

   analysis = VoxelGridAnalysis(grid)
   regions = analysis.analyze_regions(threshold=0.5)

   for region in regions:
       print(region.voxel_count, region.volume, region.surface_area)

``volume`` is reported in cubic Angstrom when the input cell is in Angstrom.
``surface_area`` is estimated by applying marching cubes to the selected mask
and transforming mesh vertices into real-space coordinates.

Periodic boundary conditions are applied by default. Connected components are
merged across opposite cell faces, and periodic surface areas are measured from
a tiled mask while counting only the central periodic image.

For large convergence scans, ``surface_area_voxel_faces`` provides a faster
periodic estimate by counting exposed voxel faces directly. This avoids the
large 3x3x3 tiled array used by periodic marching cubes, at the cost of a
grid-aligned rather than smoothed surface.

Scalar Distance Surfaces
------------------------

Distance masks can build scalar fields where each voxel stores the distance to
the nearest atom within a cutoff. ``mesh_at_value`` then traces a marching-cubes
surface at a fixed distance:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import VoxelGrid, VoxelGridAnalysis

   grid = VoxelGrid(atoms.cell.array, resolution=0.35)
   grid.grid.fill(np.inf)
   grid.min_spheres(atoms.get_positions(), cutoff_radii, mask="distance")

   analysis = VoxelGridAnalysis(grid)
   vertices, faces = analysis.mesh_at_value(level=2.0, periodic=True)
   area = analysis.mesh_surface_area(vertices, faces)

For periodic meshes, triangles that cross the boundary are clipped to the
primary cell. Vertices are returned in real-space coordinates, and faces are
integer indices into that vertex array.

Zeolite Pore Volume And Surface Area
------------------------------------

For zeolites, a common workflow is:

1. Build an occupied framework mask from atomic cores.
2. Analyze the inverse mask as the pore space.
3. Sum connected-region volumes to estimate geometric pore volume.
4. Sum surface areas to estimate geometric internal surface area.

Example:

.. code-block:: python

   from atomvoxelizer import VoxelGridAnalysis

   analysis = VoxelGridAnalysis(voxel_grid)
   pore_regions = analysis.analyze_regions(max_value=0.0)
   pore_volume_a3 = sum(region.volume for region in pore_regions)
   pore_area_a2 = sum(region.surface_area for region in pore_regions)

   mass_amu = sum(atoms.get_masses())
   pore_volume_cm3_g = analysis.volume_angstrom3_to_cm3_per_g(pore_volume_a3, mass_amu)
   internal_area_m2_g = analysis.area_angstrom2_to_m2_per_g(pore_area_a2, mass_amu)

This is a geometric voxel estimate. It is not a probe-accessible BET surface
area and is not corrected for a finite adsorbate or solvent probe. The zeolite
examples include convergence plots and a fast ``voxel-faces`` surface estimator
for fine resolution scans.

Probe-Center Accessibility
--------------------------

Probe analysis asks where the center of a spherical probe can fit. The caller
supplies atomic positions, one radius per atom, and a probe radius. Atom
exclusion radii are inflated by the probe radius:

.. code-block:: text

   exclusion_radius_i = atom_radius_i + probe_radius

Minimal example:

.. code-block:: python

   from atomvoxelizer import VoxelGrid, VoxelGridAnalysis

   grid = VoxelGrid(cell, resolution=0.25)
   analysis = VoxelGridAnalysis(grid)

   result = analysis.analyze_probe_accessibility(
       positions=positions,
       radii=covalent_radii,
       probe_radius=1.86,
       surface_method="voxel-faces",
   )

   print(result.accessible_volume)
   print(result.accessible_surface_area)

``result.accessible_mask`` is a boolean mask where ``True`` means a probe
center can occupy that voxel without overlapping any atom. The default analysis
does not modify the input grid. Set ``write_grid=True`` to store the binary
accessible mask back into the grid as 1 for accessible voxels and 0 for
excluded voxels.

This is a probe-center calculation. The reported volume is the volume available
to the center of the probe. The reported surface area is the boundary of that
accessible-center region, not a BET measurement and not yet a rolling-probe
contact-area correction.

Use ``probe_accessible_surface_area`` when you want a sampled accessible surface
area based on inflated atom surfaces:

.. code-block:: python

   surface_area = analysis.probe_accessible_surface_area(
       positions=positions,
       radii=radii,
       probe_radius=1.657,
       samples_per_atom=1000,
       surface_radius_scale=1.122,
   )

BEA Probe Comparison To PoreBlazer
----------------------------------

The BEA comparison used ``examples/zeolite/BEA.cif`` and matched PoreBlazer's
default nitrogen probe setup. AtomVoxelizer was supplied only numerical arrays:
cell, atomic positions, atomic radii, and probe radius.

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
     - deterministic sampling, ``1000`` points/atom, ``surface_radius_scale=1.122``

The matching PoreBlazer ``input.dat`` was:

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

The direct probe-center lattice volume and accessible surface area agreed
closely:

.. list-table::
   :header-rows: 1

   * - Quantity
     - AtomVoxelizer
     - PoreBlazer
     - Difference
   * - Nitrogen probe-center lattice volume
     - ``499.967712 A^3``
     - ``502.028531 A^3``
     - ``-0.410%``
   * - Nitrogen probe-center lattice fraction
     - ``0.119654``
     - ``0.120148``
     - ``-0.410%``
   * - Accessible surface area
     - ``499.912904 A^2``
     - ``488.980000 A^2``
     - ``2.236%``

Small differences are expected because the two tools discretize and sample the
structure independently. The close agreement indicates that the two
probe-accessible volume and surface-area workflows are consistent for this BEA
setup.

Timing was measured on an AMD EPYC 7551P 32-core CPU from the same Python
environment, using the end-to-end runner scripts for the matched setup. Each
tool was run three times:

.. list-table::
   :header-rows: 1

   * - Tool
     - Best time [s]
     - Mean time [s]
   * - AtomVoxelizer
     - ``3.537``
     - ``3.576``
   * - PoreBlazer
     - ``8.415``
     - ``8.444``

This timing is not a pure kernel benchmark because the workflows do different
amounts of work. It is still a useful end-to-end comparison for the matched BEA
probe-volume setup. AtomVoxelizer does not currently compute pore size
distributions; PSD support is planned for future work.
