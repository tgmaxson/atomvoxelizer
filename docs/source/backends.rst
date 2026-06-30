Backends
========

AtomVoxelizer separates the public voxel-grid API from optional acceleration
backends. All backends expose the same core methods: ``add_sphere``,
``set_sphere``, ``add_spheres``, ``set_spheres``, ``mul_sphere``,
``div_sphere``, ``clamp_grid``, and ``to_numpy``.

NumPy
-----

``VoxelGrid`` is the default and always uses NumPy. It has no optional runtime
dependencies and is the reference implementation used by the test suite.

``VoxelGridNumPy`` is kept as an alias for ``VoxelGrid`` for callers that want
to be explicit in benchmarks or tests.

Numba
-----

``VoxelGridNumba`` inherits from ``VoxelGrid`` and overrides the hot mutating
sphere operations with Numba-compiled kernels. It is available when the
``numba`` extra is installed:

.. code-block:: bash

   pip install ".[numba]"

CuPy
----

``VoxelGridCuPy`` stores the voxel grid as a CuPy array and overrides the sphere
and clamp operations with CuPy array operations. It inherits from the Numba
backend when Numba is importable, otherwise from the NumPy backend. Use
``to_numpy`` to copy results back to host memory.

The CuPy backend is optional:

.. code-block:: bash

   pip install ".[cupy]"

GPU functionality is not exercised by the CPU-only test suite.

Taichi
------

``VoxelGridTaichi`` inherits from ``VoxelGrid`` and overrides mutating sphere
operations with Taichi CPU kernels. It is available when the ``taichi`` extra is
installed:

.. code-block:: bash

   pip install ".[taichi]"

The current Taichi backend initializes Taichi with ``arch=ti.cpu`` so it can be
tested consistently on machines without GPU access.

The current Taichi CPU backend is not expected to beat NumPy on every benchmark.
It launches Taichi kernels for many small sphere operations and pays Taichi JIT
and kernel-dispatch overhead. Numba batches equal-radius atoms into compiled
loops, while the NumPy backend uses efficient indexed array updates. Taichi is
included as an experimental backend and is a better target for future batched
or GPU-oriented kernels than for the current small CPU workloads.

Consistency
-----------

The tests compare NumPy, Numba, and Taichi CPU results on the same deterministic
workload. CuPy tests are included but skipped when CuPy is not installed.

Benchmarking
------------

Run a single workload benchmark with an aligned table:

.. code-block:: bash

   python benchmarks/benchmark_backends.py --workload zeolite --backends numpy numba taichi

Run zeolite supercell scaling and save a plot:

.. code-block:: bash

   python benchmarks/benchmark_backends.py --zeolite-scaling --framework BEA --resolution 0.5 --plot zeolite_scaling.png
