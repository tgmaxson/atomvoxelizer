Installation
============

Install From PyPI
-----------------

For normal use, install the released package from PyPI:

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

Use the GitLab repository for development, local documentation builds, or when
you need changes that have not been released to PyPI yet:

.. code-block:: bash

   git clone https://gitlab.com/tgmaxson/atomvoxelizer.git
   cd atomvoxelizer
   pip install -e ".[dev,examples]"

The canonical source repository is:

   https://gitlab.com/tgmaxson/atomvoxelizer

Common Development Commands
---------------------------

Run the tests:

.. code-block:: bash

   pytest

Build the documentation:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Build and check release artifacts:

.. code-block:: bash

   python -m build
   twine check dist/*

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
   * - Numba backend
     - ``pip install numba``
     - ``VoxelGridNumba``
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
