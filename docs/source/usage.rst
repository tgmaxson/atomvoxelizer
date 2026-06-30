Usage
=====

Install the package in editable mode while developing:

.. code-block:: bash

   pip install -e ".[dev,examples]"

Install optional acceleration backends directly if you need them:

.. code-block:: bash

   pip install numba
   pip install taichi
   # Choose the CuPy package matching your CUDA runtime, for example:
   pip install cupy-cuda12x

``VoxelGrid`` is always the NumPy backend. Optional acceleration backends are
explicit: ``VoxelGridNumba``, ``VoxelGridTaichi``, and ``VoxelGridCuPy``.

Create a voxel grid from a periodic cell:

.. code-block:: python

   import numpy as np

   from atomvoxelizer import VoxelGrid

   cell = np.eye(3) * 10.0
   grid = VoxelGrid(cell=cell, resolution=0.25)
   grid.add_sphere(center=np.array([5.0, 5.0, 5.0]), radius=1.0, value=1.0)

The zeolite example in ``examples/zeolite_voxel.py`` demonstrates reading CIF
files with ASE, voxelizing covalent-radius shells and cores, and plotting slices
and supercell scaling.

Run tests and benchmarks with:

.. code-block:: bash

   pytest
   python benchmarks/benchmark_backends.py --backends numpy numba taichi cupy
   python benchmarks/benchmark_structures.py
