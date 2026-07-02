Analysis
========

``VoxelGridAnalysis`` provides post-processing helpers for voxel grids. The
analysis dependency is optional because marching cubes and connected-component
labeling use scikit-image:

.. code-block:: bash

   pip install ".[analysis]"

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
``surface_area`` is estimated by applying marching cubes to the selected voxel
mask and transforming mesh vertices into real-space coordinates. By default,
connected-component labeling and surface-area estimation apply periodic boundary
conditions. Periodic connected components are merged across opposite cell faces;
periodic surface areas are measured from a tiled mask and counted only for the
central periodic image.

For large convergence scans, ``surface_area_voxel_faces`` provides a faster
periodic estimate by counting exposed voxel faces directly. This avoids the
large 3x3x3 tiled array used by periodic marching cubes, at the cost of a
grid-aligned rather than smoothed surface.

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

See :doc:`examples` for zeolite scripts and convergence plots. The current
zeolite values are geometric voxel estimates rather than probe-accessible BET or
adsorbate-specific pore-volume estimates.

Probe-Center Accessibility
--------------------------

Probe analysis asks where the center of a spherical probe can fit. The caller
supplies atomic positions, one radius per atom, and a probe radius. Atom
exclusion radii are inflated by the probe radius:

.. code-block:: python

   import numpy as np

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

See :doc:`probe_pore_volume` for a BEA comparison against PoreBlazer and a
minimal probe-volume workflow.

Scalar Distance Surfaces
------------------------

Distance masks can be used to build scalar fields where each voxel stores the
distance to the nearest atom within a chosen cutoff. ``mesh_at_value`` then
traces a marching-cubes surface at a fixed distance:

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

See :doc:`examples` for finite Wulff and periodic Pt(211) distance-surface
examples.
