Algorithm And Acceleration
==========================

AtomVoxelizer represents an atomistic structure as a scalar field on a periodic
3D grid. The grid is defined by a simulation cell and either a target real-space
resolution or explicit grid-point counts. Atomic operations then paint local
spherical regions into that grid.

Grid Construction
-----------------

``VoxelGrid`` stores the cell matrix, its inverse, the grid shape, and a
``float32`` array of voxel values. Positions are converted to fractional
coordinates with the inverse cell, wrapped into the primary periodic image, and
then mapped to integer voxel indices.

Sphere operations are implemented from cached integer offsets:

1. For a given radius, grid shape, and cell, AtomVoxelizer enumerates all integer
   voxel offsets that lie inside the sphere.
2. The offset list is cached, so repeated atoms with the same radius reuse the
   same stencil.
3. Each atom center is converted to a grid index.
4. Cached offsets are added to that center and wrapped modulo the grid shape,
   which applies periodic boundary conditions.
5. The selected voxels are modified with the requested operation.

The supported mutating operations are explicit functions rather than a generic
callback: ``set_sphere``, ``add_sphere``, ``mul_sphere``, ``div_sphere``, and
``min_sphere``. The batch versions apply the same operations to many centers and
radii.

Mask Types
----------

Sphere operations support two mask types:

``constant``
   Every voxel in the sphere receives the supplied value or factor. This is the
   default and is useful for occupancy masks, coordination shells, and blocking
   atomic cores.

``distance``
   Each voxel receives its real-space distance from the sphere center. The
   distances are cached alongside the sphere offsets. Using ``min_spheres`` with
   ``mask="distance"`` gives a nearest-atom distance field within the chosen
   cutoff radius.

Periodic Analysis
-----------------

``VoxelGridAnalysis`` has two related analysis paths:

* Binary-region analysis labels connected voxel volumes and estimates surface
  area from marching cubes.
* Scalar-field analysis traces a mesh at a specified value with
  ``mesh_at_value``. For a nearest-atom distance field, this traces a surface at
  a fixed distance from the atoms.

For periodic scalar meshes, AtomVoxelizer tiles the grid, runs marching cubes on
the tiled field, keeps triangles associated with the central periodic image, and
clips boundary-crossing triangles to the primary cell. Clipping is important:
wrapping vertices back into the primary cell would create long triangles across
the cell boundary.

Acceleration Strategy
---------------------

The NumPy implementation is the reference backend and is always available. It
uses cached stencils and NumPy indexed updates. It is simple and often fast
enough for small structures.

The Numba backend inherits from ``VoxelGrid`` and replaces the hot mutating
sphere operations with compiled loops. Its batch operations group atoms with the
same radius, reuse cached stencils, and update a flattened grid array. This
removes most Python overhead from repeated sphere painting and is currently the
fastest CPU backend for the benchmark workloads below.

The CuPy backend stores the grid on a CUDA device and uses CuPy indexed updates.
It is useful when the GPU workload is large enough to amortize data movement and
kernel-launch overhead. Small atom-by-atom updates can be slower than CPU
backends because each operation does relatively little work per launch.

The Taichi backend provides CPU and GPU variants using Taichi kernels. The
current implementation is experimental. It pays noticeable JIT and kernel
dispatch overhead for many small sphere operations, so the CPU Taichi backend is
not expected to outperform NumPy or Numba in the current benchmark shape.

Optional Packages
-----------------

The optional package requirements are:

.. list-table::
   :header-rows: 1

   * - Feature
     - Package
   * - Numba backend
     - ``numba``
   * - CuPy backend
     - a CUDA-specific CuPy package, for example ``cupy-cuda12x``
   * - Taichi backend
     - ``taichi``
   * - Analysis and scalar mesh extraction
     - ``scikit-image``
   * - Structure examples and benchmark structure builders
     - ``ase``
   * - Documentation builds
     - ``sphinx``

Benchmark Environment
---------------------

The example benchmark data below was generated on this development machine:

* CPU: AMD EPYC 7551P 32-Core Processor, 32 physical cores, single socket.
* Python: 3.12.12.
* Packages: NumPy 2.4.1, Numba 0.64.0, Taichi 1.7.4, scikit-image 0.26.0,
  ASE 3.27.0.
* GPU backends were not included in these measurements.
* Each timing reports the best of three runs after a small warmup.

BEA Zeolite Scaling
-------------------

Command:

.. code-block:: bash

   python benchmarks/benchmark_backends.py --zeolite-scaling --framework BEA \
       --resolution 0.5 --repeats 3 --backends numpy numba taichi \
       --plot docs/source/_static/zeolite_scaling.png

.. list-table::
   :header-rows: 1

   * - Scale
     - Atoms
     - Grid
     - NumPy best [s]
     - Numba best [s]
     - Taichi CPU best [s]
   * - 1x1x1
     - 192
     - 26x26x53
     - 0.0133
     - 0.0004
     - 0.3116
   * - 1x1x2
     - 384
     - 26x26x105
     - 0.0262
     - 0.0005
     - 0.3989
   * - 1x2x2
     - 768
     - 26x51x105
     - 0.0497
     - 0.0008
     - 0.5129
   * - 2x2x2
     - 1536
     - 51x51x105
     - 0.1032
     - 0.0013
     - 0.8473

.. image:: _static/zeolite_scaling.png
   :alt: BEA zeolite backend scaling benchmark
   :width: 90%

Wulff Construction Benchmark
----------------------------

Command:

.. code-block:: bash

   python benchmarks/benchmark_backends.py --workload wulff --wulff-size 1000 \
       --resolution 0.5 --repeats 3 --backends numpy numba taichi

The generated Wulff construction contained 1103 atoms and used a 75x75x75 grid.

.. list-table::
   :header-rows: 1

   * - Backend
     - Best [s]
     - Mean [s]
     - Max absolute difference vs NumPy
   * - NumPy
     - 0.0973
     - 0.0978
     - 0.000
   * - Numba
     - 0.0027
     - 0.0032
     - 0.000
   * - Taichi CPU
     - 0.7198
     - 0.7383
     - 0.000

Example Outputs
---------------

The zeolite analysis example can be used to inspect resolution convergence:

.. code-block:: bash

   python examples/zeolite/zeolite_analysis.py BEA --convergence 1.0 0.75 0.5 \
       --plot docs/source/_static/zeolite_convergence.png

.. image:: _static/zeolite_convergence.png
   :alt: BEA zeolite pore-volume and surface-area convergence
   :width: 90%

The Wulff distance-surface example exports a mesh and can save or show a 3D
Matplotlib preview:

.. code-block:: bash

   python examples/wulff/distance_surface.py --size 147 --resolution 0.8 \
       --cutoff 3.5 --distance 1.8 --plot docs/source/_static/wulff_distance_surface.png

.. image:: _static/wulff_distance_surface.png
   :alt: Wulff nearest-atom distance isosurface
   :width: 80%
