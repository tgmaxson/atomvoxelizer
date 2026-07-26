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
   ATOMVOXELIZER_BUILD_CPP=1 pip install "AtomVoxelizer[cpp]"

The base package includes the NumPy ``VoxelGrid`` backend. Acceleration
backends that depend on external packages are installed directly when needed:

.. code-block:: bash

   pip install numba
   pip install taichi
   # Choose the CuPy package matching your CUDA runtime, for example:
   pip install cupy-cuda12x

``VoxelGridCpp`` is a compiled pybind11 backend and is optional. Release builds
are pure Python by default. To compile the extension from source, set
``ATOMVOXELIZER_BUILD_CPP=1`` and request the ``cpp`` extra. If the extension
is not available for a platform or cannot be compiled, the pure-Python package
remains usable and importing ``VoxelGridCpp`` raises an explanatory
``ImportError``.

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

GitHub Mirror
-------------

AtomVoxelizer is also mirrored to GitHub:

   https://github.com/tgmaxson/atomvoxelizer

The GitHub repository is maintained for public archiving, publishing, and
discoverability. Active development, merge requests, and release preparation
are handled in GitLab. GitHub issues will be monitored, but contributors should
prefer the GitLab repository for development work.

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
     - ``pip install ase wulffpack``; install ORB separately for ORB-V3 MCMD
     - Wulff nanoparticle construction, voxel adsorption sites, and CO MCMD
   * - Numba backend
     - ``pip install numba``
     - ``VoxelGridNumba``, ``FieldVoxelGridNumba``, ``VectorVoxelGridNumba``
   * - C++ backend
     - ``ATOMVOXELIZER_BUILD_CPP=1 pip install "AtomVoxelizer[cpp]"``
     - ``VoxelGridCpp``
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
