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

Zeolite Pore Volume And Surface Area
------------------------------------

For zeolites, a common workflow is:

1. Build an occupied framework mask from atomic cores.
2. Analyze the inverse mask as the pore space.
3. Sum connected-region volumes to estimate accessible pore volume.
4. Sum marching-cubes surface areas to estimate internal surface area.

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

The same workflow is provided as ``examples/zeolite_analysis.py``:

.. code-block:: bash

   python examples/zeolite_analysis.py BEA --resolution 0.25

The example can also run a resolution-convergence study and save a plot:

.. code-block:: bash

   python examples/zeolite_analysis.py BEA --convergence 1.0 0.75 0.5 0.35 --plot bea_convergence.png

Experimental Comparison
-----------------------

Experimental BET surface area and pore volume are usually reported as
``m^2/g`` and ``cm^3/g``. Direct comparison to a geometric voxel model requires
normalizing by the mass represented by the simulated unit cell or supercell and
matching the experimental assumptions: framework composition, extra-framework
cations, adsorbate probe size, activation state, defects, and whether the
reported pore volume is micropore, mesopore, or total pore volume.

AtomVoxelizer provides unit-conversion helpers once you know the mass represented
by the simulated structure:

.. code-block:: python

   mass_amu = sum(atoms.get_masses())
   pore_volume_cm3_g = analysis.volume_angstrom3_to_cm3_per_g(pore_volume_a3, mass_amu)
   area_m2_g = analysis.area_angstrom2_to_m2_per_g(pore_area_a2, mass_amu)

The comparison table should be filled with values from the specific material
and synthesis route being modeled:

.. list-table::
   :header-rows: 1

   * - Framework
     - Experimental BET surface area
     - Experimental pore volume
     - Notes
   * - BEA
     - source-specific
     - source-specific
     - Zeolite beta values depend strongly on Si/Al ratio and activation.
   * - MWW/MCM-22
     - source-specific
     - source-specific
     - MCM-22 reports often distinguish micropore and external surface area.

Use the voxel result as a geometric internal-surface estimate, not as a direct
replacement for adsorbate-specific BET analysis.
