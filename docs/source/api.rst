API Reference
=============

.. autoclass:: atomvoxelizer.VoxelGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VoxelGridNumPy
   :members:
   :undoc-members:
   :show-inheritance:

Experimental Field Grids
------------------------

These classes are experimental. The default implementation uses NumPy, and the
Numba field-grid backend is available when Numba is installed. CuPy and Taichi
field-grid backends are not implemented. See :doc:`concepts` for the field-grid
model.

.. autoclass:: atomvoxelizer.FieldVoxelGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VectorVoxelGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.FieldVoxelGridNumba
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VectorVoxelGridNumba
   :members:
   :undoc-members:
   :show-inheritance:

Optional Backends
-----------------

These classes require their optional backend packages to be installed in the
documentation/build environment.

.. autoclass:: atomvoxelizer.VoxelGridNumba
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VoxelGridCuPy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VoxelGridTaichi
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.VoxelGridTaichiGPU
   :members:
   :undoc-members:
   :show-inheritance:

Analysis
--------

.. autoclass:: atomvoxelizer.VoxelGridAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: atomvoxelizer.ProbeAccessibleResult
   :members:
   :undoc-members:
   :show-inheritance:
