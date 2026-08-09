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

Probe Comparison To PoreBlazer
------------------------------

The comparison suite uses five periodic frameworks: BEA as the reference
zeolite, MWW as a more confined zeolite, FAU as a more open zeolite, UiO-66 as
a confined-cage MOF, and MOF-5 as a very open-cage MOF. The BEA and MWW
structures are included in the repository examples. FAU is the idealized
pure-silica framework structure from the
`IZA-SC database <https://www.iza-structure.org/IZA-SC/framework_3d.php?ID=93>`_,
and the UiO-66 and MOF-5 structures are from COD entries
`4132633 <https://www.crystallography.net/4132633.html>`_ and
`1516305 <https://www.crystallography.net/1516305.html>`_, respectively.

Each run uses the same PoreBlazer-style nitrogen settings. AtomVoxelizer was
supplied only numerical arrays: cell, atomic positions, atomic radii, and probe
radius.

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - Atom radii
     - PoreBlazer ``UFF.atoms`` sigma / 2
   * - Probe radius
     - ``1.657 A`` = PoreBlazer nitrogen sigma ``3.314 A`` / 2
   * - Grid shape
     - Per-framework grid matched to the PoreBlazer lattice at a nominal
       ``0.25 A`` cubelet size
   * - PoreBlazer lattice sampling
     - ``500`` samples/atom
   * - Surface method
     - deterministic sampling, ``1000`` points/atom, ``surface_radius_scale=1.122``

The direct probe-center volume comparison is shown below. The PoreBlazer
column is the volume obtained by counting points in its nitrogen-accessible
lattice (``nitrogen_network.xyz``), which is the closest direct equivalent to
AtomVoxelizer's accessible grid volume. Fractions are normalized by the unit
cell volume.

.. list-table::
   :header-rows: 1
   :widths: 14 20 17 17 10 11 11

   * - Framework
     - Role
     - AtomVoxelizer volume (A^3)
     - PoreBlazer N2 lattice volume (A^3)
     - Difference
     - AtomVoxelizer fraction
     - PoreBlazer fraction
   * - BEA
     - Reference zeolite
     - ``499.967712``
     - ``502.028531``
     - ``-0.410%``
     - ``0.119654``
     - ``0.120148``
   * - MWW
     - More confined zeolite
     - ``428.552877``
     - ``414.293536``
     - ``3.442%``
     - ``0.094839``
     - ``0.091683``
   * - FAU
     - More open zeolite
     - ``2919.608324``
     - ``2886.139921``
     - ``1.160%``
     - ``0.202346``
     - ``0.200027``
   * - UiO-66
     - Confined-cage MOF
     - ``852.058013``
     - ``926.447100``
     - ``-8.030%``
     - ``0.095892``
     - ``0.104264``
   * - MOF-5
     - Very open-cage MOF
     - ``7314.701198``
     - ``7180.556990``
     - ``1.868%``
     - ``0.426009``
     - ``0.418196``

The corresponding accessible-surface comparison is kept in a separate table
because the values use different numerical estimators: AtomVoxelizer samples
inflated atom surfaces deterministically, while PoreBlazer reports its
network-accessible ``S_AC`` estimate.

.. list-table::
   :header-rows: 1
   :widths: 18 23 22 15

   * - Framework
     - AtomVoxelizer sampled area (A^2)
     - PoreBlazer network S_AC (A^2)
     - Difference
   * - BEA
     - ``499.912904``
     - ``488.980000``
     - ``2.236%``
   * - MWW
     - ``389.703755``
     - ``380.020000``
     - ``2.548%``
   * - FAU
     - ``1791.682398``
     - ``1706.190000``
     - ``5.011%``
   * - UiO-66
     - ``767.382180``
     - ``764.160000``
     - ``0.422%``
   * - MOF-5
     - ``3509.281932``
     - ``3528.300000``
     - ``-0.539%``

Across the five structures, the direct lattice-volume differences range from
``-8.030%`` to ``3.442%`` and the surface-area differences range from
``-0.539%`` to ``5.011%``. Small differences are expected because the two tools
discretize and sample the structure independently. PoreBlazer's ``V_PO`` is a
different free-volume estimator and is therefore not placed in the direct
volume table. AtomVoxelizer does not currently compute pore size distributions;
PSD support is planned for future work.
