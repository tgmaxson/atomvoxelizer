Installation
============

Install From PyPI
-----------------

For normal use, install the latest released package from PyPI:

.. code-block:: bash

   pip install AtomVoxelizer

Optional feature groups can be installed with extras:

.. code-block:: bash

   pip install "AtomVoxelizer[analysis]"
   pip install "AtomVoxelizer[examples]"
   pip install "AtomVoxelizer[docs]"

The base package includes the NumPy ``VoxelGrid`` backend. Acceleration
backends are optional packages installed directly when needed:

.. code-block:: bash

   pip install numba
   pip install taichi
   # Choose the CuPy package matching your CUDA runtime, for example:
   pip install cupy-cuda12x

Install From Source
-------------------

Use the GitLab repository for development, local documentation builds, or
unreleased changes:

.. code-block:: bash

   git clone https://gitlab.com/tgmaxson/atomvoxelizer.git
   cd atomvoxelizer
   pip install -e ".[dev,examples]"

The canonical source repository is:

   https://gitlab.com/tgmaxson/atomvoxelizer

Optional Feature Map
--------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Install
     - Used by
   * - Analysis
     - ``pip install "AtomVoxelizer[analysis]"`` or ``pip install scikit-image``
     - connected components, marching cubes, scalar mesh extraction
   * - Examples
     - ``pip install "AtomVoxelizer[examples]"``
     - ASE structure loading, CIF examples, Wulff and surface examples
   * - Quickstart tutorial
     - ``pip install ase wulffpack``; install ORB separately for ``--score orb-v3``
     - Wulff nanoparticle construction, EMT scoring, and optional ORB-V3 scoring
   * - Numba backend
     - ``pip install numba``
     - ``VoxelGridNumba``, ``FieldVoxelGridNumba``, ``VectorVoxelGridNumba``
   * - CuPy backend
     - ``pip install cupy-cuda12x`` or the CuPy package matching your CUDA runtime
     - ``VoxelGridCuPy``
   * - Taichi backend
     - ``pip install taichi``
     - ``VoxelGridTaichi`` and ``VoxelGridTaichiGPU``
   * - Documentation
     - ``pip install "AtomVoxelizer[docs]"``
     - local Sphinx documentation builds
   * - Publishing
     - ``pip install "AtomVoxelizer[publish]"``
     - building and checking PyPI artifacts

See :doc:`contributing` for development commands, testing, documentation
builds, and release publishing.
