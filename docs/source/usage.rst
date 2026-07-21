Usage
=====

See :doc:`install` for PyPI installation, source installation from the GitLab
repository, and optional dependency details.

Create a voxel grid from a periodic cell:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import VoxelGrid

   cell = np.eye(3) * 10.0
   grid = VoxelGrid(cell=cell, resolution=0.25)
   grid.add_sphere(center=np.array([5.0, 5.0, 5.0]), radius=1.0, value=1.0)

Grid Dtype
----------

``VoxelGrid`` uses ``numpy.float32`` values by default. Pass ``dtype`` when a
different grid storage type is useful:

.. code-block:: python

   occupancy = VoxelGrid(cell=cell, resolution=0.25, dtype=np.int16)
   distance = VoxelGrid(cell=cell, resolution=0.25, dtype=np.float64)
   amplitudes = VoxelGrid(cell=cell, resolution=0.25, dtype=np.complex64)

Integer dtypes are useful for count-like masks such as coordination-shell
overlap fields. Floating dtypes are better for distance fields and analysis
workflows. Complex dtypes support arithmetic operations such as ``set_sphere``,
``add_sphere``, ``mul_sphere``, and ``div_sphere``. Ordered operations are not
defined for complex values, so ``min_sphere``, ``clamp_grid``,
``sample_voxels_in_range``, and threshold plotting raise ``TypeError`` for
complex grids.

Sphere Masks
------------

Sphere operations accept ``mask="constant"`` and ``mask="distance"``.
The constant mask writes the supplied value or factor across every voxel in the
sphere. The distance mask writes the real-space distance from the sphere center
at each voxel, in Angstrom when the cell is in Angstrom.

Use ``min_spheres`` with the distance mask to compute the distance to the
nearest atom within a cutoff:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import VoxelGrid, VoxelGridAnalysis

   grid = VoxelGrid(cell=atoms.cell.array, resolution=0.35)
   grid.grid.fill(np.inf)
   grid.min_spheres(atoms.get_positions(), cutoff_radii, mask="distance")

   analysis = VoxelGridAnalysis(grid)
   vertices, faces = analysis.mesh_at_value(2.0, periodic=True)

Periodic scalar meshes are clipped at the primary cell boundary. This avoids
wrapping a boundary-crossing triangle across the cell.

Exporting Grid Data
-------------------

Use ``save_npz`` to preserve a voxel grid as a compact NumPy archive containing
the grid values, cell, grid dimensions, and dtype. The matching ``from_npz``
constructor restores a ``VoxelGrid`` for later analysis:

.. code-block:: python

   grid.save_npz("distance_mask.npz")
   restored = VoxelGrid.from_npz("distance_mask.npz")

For visualization or downstream sampling, ``to_point_cloud`` converts selected
voxels to real-space voxel-center coordinates plus their stored values:

.. code-block:: python

   centers, values = grid.to_point_cloud(min_value=2.5, max_value=3.5)

``voxel_centers`` returns the real-space center of every voxel, or only the
centers for a supplied ``(N, 3)`` integer index array. These export helpers are
NumPy-only and work with scalar ``VoxelGrid`` instances from the default and
Numba backends.

Coordination-Surface Masks
--------------------------

One useful pattern is to add overlapping shells around atoms, then carve the
atomic cores back out. For example, a shell radius of
``1.4 * covalent_radius`` and a core radius of ``1.1 * covalent_radius`` gives
a coordination-number-like surface field. The voxel value is the number of
shells covering that point, so values near 3 mark positions coordinated by
roughly three nearby atoms.

.. code-block:: python

   import numpy as np
   from ase.data import covalent_radii

   from atomvoxelizer import VoxelGrid

   grid = VoxelGrid(cell=atoms.cell.array, resolution=0.25)
   centers = atoms.get_positions()
   radii = np.array([covalent_radii[atom.number] for atom in atoms], dtype=float)

   grid.add_spheres(centers, 1.4 * radii, value=1.0)
   grid.set_spheres(centers, 1.1 * radii, value=0.0)

   samples = list(
       grid.sample_voxels_in_range(
           min_val=2.5,
           max_val=3.5,
           min_dist=2.0,
           seed=123,
       )
   )

The resulting grid is not a solvent- or adsorbate-accessible probe surface. It
is a geometric shell-overlap field for sampling surface-like positions near
atoms. Sampling from ``2.5`` to ``3.5`` selects the coordination-3 surface while
avoiding exact integer boundary issues.

Examples
--------

See :doc:`examples` for complete zeolite, nanoparticle, and periodic surface
workflows.

Run tests and benchmarks with:

.. code-block:: bash

   pytest
   python benchmarks/benchmark_backends.py --workloads zeolite nanoparticle surface \
       --plot mask_generation_scaling.png
   python benchmarks/benchmark_dtypes.py --backend numpy
   python benchmarks/benchmark_structures.py
