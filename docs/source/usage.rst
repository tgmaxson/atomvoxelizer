Usage
=====

Install the package in editable mode while developing:

.. code-block:: bash

   pip install -e ".[examples,docs]"

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

