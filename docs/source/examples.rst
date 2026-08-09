Examples
========

The example scripts live in the repository under ``examples/``. Install the
example and analysis dependencies before running them:

.. code-block:: bash

   pip install "AtomVoxelizer[examples,analysis]"

For development from source:

.. code-block:: bash

   git clone https://gitlab.com/tgmaxson/atomvoxelizer.git
   cd atomvoxelizer
   pip install -e ".[dev,examples]"

Zeolite Voxelization
--------------------

``examples/zeolite/zeolite_voxel.py`` reads a framework CIF with ASE,
voxelizes covalent-radius shells and cores, plots slices, and benchmarks
supercell scaling:

.. code-block:: bash

   python examples/zeolite/zeolite_voxel.py BEA

For a single documentation-style slice through the framework grid:

.. code-block:: bash

   python examples/zeolite/zeolite_slice_visual.py --framework BEA \
       --resolution 0.25 --output docs/source/_static/zeolite_voxel_slice.png

.. image:: _static/zeolite_voxel_slice.png
   :alt: BEA zeolite voxel-grid slice showing atomic cores, void space, and coordination shell voxels
   :width: 85%

Zeolite Geometric Pore Analysis
-------------------------------

``examples/zeolite/zeolite_analysis.py`` estimates geometric pore volume and
geometric internal surface area from the inverse of a framework-core mask:

.. code-block:: bash

   python examples/zeolite/zeolite_analysis.py BEA --resolution 0.25

The result is a geometric voxel estimate. It is not a probe-accessible BET
surface area and is not corrected for a finite adsorbate or solvent probe. Probe
methods may be added in the future in the spirit of established porosity tools
such as `Zeo++ <https://www.zeoplusplus.org/>`_ and
`PoreBlazer <https://github.com/SarkisovGroup/PoreBlazer>`_.

The convergence command below samples resolutions from 1.00 to 0.05 Angstrom in
0.05 Angstrom increments. The example uses the fast ``voxel-faces`` surface
estimator by default; pass ``--surface-method marching-cubes`` for a smoother
triangulated estimate on smaller grids.

.. code-block:: bash

   python examples/zeolite/zeolite_analysis.py BEA --convergence \
       1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 \
       0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 \
       --plot bea_convergence.png

.. image:: _static/zeolite_convergence.png
   :alt: BEA geometric pore-volume and surface-area convergence
   :width: 90%

Probe Pore Volume
-----------------

``VoxelGridAnalysis.analyze_probe_accessibility`` estimates probe-center
accessible volume from a user-supplied grid, atomic positions, radii, and probe
radius. ``probe_accessible_surface_area`` estimates sampled accessible surface
area from inflated atom surfaces.

See :doc:`analysis` for the method and a five-framework comparison against
PoreBlazer, including the matched AtomVoxelizer setup, agreement for
probe-accessible volume and surface area, and per-framework timing results.

Finite Wulff Distance Surface
-----------------------------

``examples/wulff/distance_surface.py`` builds a Wulff nanoparticle, computes a
nearest-atom distance field, and exports a marching-cubes mesh at a requested
distance:

.. code-block:: bash

   python examples/wulff/distance_surface.py --symbol Pt --size 147 \
       --distance 2.0 --output pt_surface.npz
   python examples/wulff/distance_surface.py --symbol Pt --size 147 \
       --distance 2.0 --show

.. image:: _static/wulff_distance_surface.png
   :alt: Wulff nearest-atom distance isosurface
   :width: 80%

Periodic Pt(211) Distance Surface
---------------------------------

``examples/surfaces/pt211_distance_surface.py`` applies the same distance-field
workflow to a periodic stepped Pt(211) slab:

.. code-block:: bash

   python examples/surfaces/pt211_distance_surface.py --distance 1.8 --show

.. image:: _static/pt211_distance_surface.png
   :alt: Periodic Pt(211) nearest-atom distance isosurface
   :width: 85%

Voxel-Guided CO MCMD
--------------------

``examples/mc/orb_v3_co_mcmd.py`` builds a small cube-like WulffPack
nanoparticle, constructs a coordination-surface voxel mask, samples adsorption
sites, and runs CO adsorption/desorption MCMD. The default calculator is the
conservative ORB-V3 infinite-neighbor model on CPU. ASE EMT is available with
``--calculator emt`` for quick control-flow checks. Coverage is counted as
``N_CO / N_surface_atoms`` and adsorption is capped by ``--max-coverage`` so the
number of sampled voxel sites does not define the maximum CO loading. This is a
workflow demonstration; the ORB-V3 tutorial settings are not presented as a
validated CO/Pt adsorption model.

.. code-block:: bash

   python examples/mc/orb_v3_co_mcmd.py --natoms 55 --steps 100 \
       --calculator orb-v3 --device cpu --orb-model-size inf \
       --temperature 500 --target-coverage 0.5 --md-steps 50

By default the script writes ``examples/mc/orb_v3_co_mcmd.traj`` for viewing
the MCMD path with ASE.

The step-by-step explanation is in :doc:`quickstart`.

Benchmarks
----------

Run the mask-generation benchmark with:

.. code-block:: bash

   python benchmarks/benchmark_backends.py --workloads zeolite nanoparticle surface \
       --numba-threads 24 --plot mask_generation_scaling.png

The benchmark scales each workload from small systems to roughly 3000 atoms and
compares a simple direct atom-grid distance scan with ``VoxelGrid`` NumPy,
``VoxelGridNumba``, opt-in ``VoxelGridNumba`` 27-color addition, and
``VoxelGridCpp``.

The CPU cell-coloring thread sweep checks whether the planned 27-color Numba
path benefits from CPU parallelism:

.. code-block:: bash

   python benchmarks/benchmark_numba_cell_coloring.py --thread-scaling \
       --thread-counts 1 2 4 8 16 24 28 \
       --resolution 0.45 --repeats 3 \
       --csv benchmarks/numba_cell_coloring_threads.csv \
       --plot docs/source/_static/numba_cell_coloring_threads.png

The documentation thread-scaling figure was generated on an AMD EPYC 7551P
32-core CPU. It shows that the prepared colored update can benefit from
parallelism for a larger Pt nanoparticle, but the plan-building overhead and
shared-node thread-count sensitivity mean this remains an explicit opt-in path.

The experimental CuPy cell-coloring benchmark compares the CPU Numba backend,
the existing CuPy ``add.at`` path, the 27-color CuPy kernel, and simple
high-level Torch prototypes for a same-radius Pt nanoparticle workload:

.. code-block:: bash

   python benchmarks/benchmark_cupy_cell_coloring.py --sizes 309 923 2869 \
       --resolution 0.6 --repeats 3 \
       --csv benchmarks/cupy_cell_coloring_rtx4090.csv \
       --plot docs/source/_static/cupy_cell_coloring_scaling.png

The documentation figure was generated on a front-partition NVIDIA GeForce RTX
4090. The CPU Numba series was measured in the same Slurm job for a direct
comparison against the GPU implementations.

Run the dtype benchmark to compare grid storage types:

.. code-block:: bash

   python benchmarks/benchmark_dtypes.py --backend numpy
   python benchmarks/benchmark_dtypes.py --backend numba

.. image:: _static/mask_generation_scaling.png
   :alt: AtomVoxelizer mask-generation benchmark
   :width: 90%

.. image:: _static/numba_cell_coloring_threads.png
   :alt: Numba 27-color cell-coloring thread scaling benchmark
   :width: 85%

.. image:: _static/cupy_cell_coloring_scaling.png
   :alt: Experimental CuPy cell-coloring benchmark compared with CPU Numba
   :width: 85%
