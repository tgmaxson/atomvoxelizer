Publishing
==========

AtomVoxelizer uses ``pyproject.toml`` and setuptools, so source distributions
and wheels can be built with the standard Python build frontend.

Install publishing tools:

.. code-block:: bash

   pip install -e ".[publish]"

Before publishing, run the CPU test suite and build the docs:

.. code-block:: bash

   pytest
   python -m sphinx -b html docs/source docs/build/html

Build and inspect package artifacts:

.. code-block:: bash

   python -m build
   twine check dist/*

Upload to TestPyPI first:

.. code-block:: bash

   twine upload --repository testpypi dist/*

Install from TestPyPI in a clean environment and run a small import check:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AtomVoxelizer
   python -c "from atomvoxelizer import VoxelGrid; print(VoxelGrid.backend_name)"

When the TestPyPI package looks correct, upload to PyPI:

.. code-block:: bash

   twine upload dist/*

Optional backends are published as extras:

.. code-block:: bash

   pip install "AtomVoxelizer[numba]"
   pip install "AtomVoxelizer[taichi]"
   pip install "AtomVoxelizer[cupy]"

