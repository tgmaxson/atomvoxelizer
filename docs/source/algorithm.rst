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
* Fast grid-face area analysis counts selected/unselected voxel-face boundaries
  directly with ``surface_area_voxel_faces``. This is much faster for very fine
  periodic convergence scans, but it is a voxel-face estimate rather than a
  smoothed triangular surface.

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
removes most Python overhead from repeated sphere painting and is generally the
fastest CPU backend for the current voxelization workloads.

The CuPy backend stores the grid on a CUDA device and uses CuPy indexed updates.
It is useful when the GPU workload is large enough to amortize data movement and
kernel-launch overhead. Small atom-by-atom updates can be slower than CPU
backends because each operation does relatively little work per launch.

The Taichi backend provides CPU and GPU variants using Taichi kernels. The
current implementation is experimental. It pays noticeable JIT and kernel
dispatch overhead for many small sphere operations, so the CPU Taichi backend is
not expected to outperform NumPy or Numba for the current small sphere-update
workloads.

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

See :doc:`examples` for runnable examples, plots, and benchmark commands.
