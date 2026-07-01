Field Voxel Grids
=================

``FieldVoxelGrid`` is an experimental NumPy-only prototype for storing more
than one scalar value at each voxel. It keeps the same periodic indexing and
sphere-stencil logic as ``VoxelGrid``, but the array shape is
``(*gpts, *value_shape)``.

This branch does not implement CuPy or Taichi field-grid backends. The
experimental API is intended for review before deciding whether it should be
merged into the main scalar backend design.

Value Shapes
------------

Use ``value_shape`` to choose what each voxel stores:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import FieldVoxelGrid, VectorVoxelGrid

   scalar = FieldVoxelGrid(cell, resolution=0.25, value_shape=())
   length_one = FieldVoxelGrid(cell, resolution=0.25, value_shape=(1,))
   vector = FieldVoxelGrid(cell, resolution=0.25, value_shape=(3,))
   matrix = FieldVoxelGrid(cell, resolution=0.25, value_shape=(3, 3))

   # Convenience alias for value_shape=(3,)
   vector = VectorVoxelGrid(cell, resolution=0.25)

``value_shape=()`` stores true scalar values. ``value_shape=(1,)`` stores
scalars as length-one vectors, which can be useful when a workflow wants all
fields to have a trailing value dimension. Matrix-valued fields such as
``value_shape=(3, 3)`` can be used for tensor-like quantities.

Constant Masks
--------------

The ``constant`` mask accepts a value matching the field shape and writes,
adds, multiplies, or divides that value over the selected sphere:

.. code-block:: python

   matrix_grid = FieldVoxelGrid(cell, resolution=0.25, value_shape=(3, 3))
   matrix_grid.add_sphere(center, radius=1.2, value=np.eye(3))

   scalar_grid = FieldVoxelGrid(cell, resolution=0.25, value_shape=())
   scalar_grid.add_sphere(center, radius=1.2, value=1.0)

Normal Masks
------------

For ``value_shape=(3,)``, ``mask="normal"`` writes the unit vector pointing
away from the atom center at each selected voxel. The center voxel has a zero
vector because the direction is undefined there.

Summing these normal masks over atoms gives a natural local direction field.
The field can then be normalized while leaving zero vectors unchanged:

.. code-block:: python

   grid = VectorVoxelGrid(cell, resolution=0.25)
   grid.add_spheres(atom_positions, radii, mask="normal")
   grid.normalize_vectors()

Plotting
--------

Vector fields can be inspected with Matplotlib quiver plots:

.. code-block:: python

   grid.plot_quiver_slice(axis="z", index=grid.gpts[2] // 2, stride=2, normalize=True)
   grid.plot_quiver_3D(stride=3, min_norm=0.1, normalize=True, length=0.5)

The plotting helpers are intended for exploration. For tests and custom plots,
use ``quiver_slice_data`` and ``quiver_3d_data`` to retrieve the sampled
positions and vector components without creating a Matplotlib figure.
