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

Sphere Masks
------------

Sphere operations accept ``mask="constant"`` and ``mask="distance"``.
The constant mask writes the supplied value or factor across every voxel in the
sphere. The distance mask writes the real-space distance from the sphere center
at each voxel, in Angstrom when the cell is in Angstrom.

Use ``min_spheres`` with the distance mask to compute a nearest-atom distance
field:

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

The zeolite example in ``examples/zeolite/zeolite_voxel.py`` demonstrates
reading CIF files with ASE, voxelizing covalent-radius shells and cores, and
plotting slices and supercell scaling.

The Wulff example in ``examples/wulff/distance_surface.py`` demonstrates a
nearest-atom distance field around a nanoparticle and writes a marching-cubes
surface mesh:

.. code-block:: bash

   pip install -e ".[examples,analysis]"
   python examples/wulff/distance_surface.py --symbol Pt --size 147 --distance 2.0 --output pt_surface.npz
   python examples/wulff/distance_surface.py --symbol Pt --size 147 --distance 2.0 --show

The periodic surface example in ``examples/surfaces/pt211_distance_surface.py``
uses the same distance-field workflow for a Pt(211) slab with periodic boundary
conditions:

.. code-block:: bash

   python examples/surfaces/pt211_distance_surface.py --distance 1.8 --show

Run tests and benchmarks with:

.. code-block:: bash

   pytest
   python benchmarks/benchmark_backends.py --backends numpy numba taichi cupy
   python benchmarks/benchmark_structures.py
