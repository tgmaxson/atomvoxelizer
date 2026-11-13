# AtomVoxelizer

AtomVoxelizer builds periodic atom-centered voxel grids for atomistic structures.
The core `VoxelGrid` class stores a 3D NumPy grid over a periodic cell and provides
helpers for adding, setting, scaling, sampling, and plotting spherical regions.

## Installation

Install the latest released package from PyPI:

```bash
pip install AtomVoxelizer
```

Install from the GitLab repository for development or unreleased changes:

```bash
git clone https://gitlab.com/tgmaxson/atomvoxelizer.git
cd atomvoxelizer
pip install -e ".[dev,examples]"
```

Install optional acceleration backends directly if you need them:

```bash
pip install numba
pip install taichi
# Choose the CuPy package matching your CUDA runtime, for example:
pip install cupy-cuda12x
pip install ".[analysis]"
ATOMVOXELIZER_BUILD_CPP=1 pip install ".[cpp]"
```

`VoxelGrid` is always the NumPy backend. Optional acceleration backends are
explicit: `VoxelGridNumba`, `VoxelGridCpp`, `VoxelGridTaichi`, and
`VoxelGridCuPy`. The C++ backend is optional and only available when the
pybind11 extension was built. Use the `cpp` extra together with
`ATOMVOXELIZER_BUILD_CPP=1` to request the build dependency and extension.
`VoxelGridCuPy` also includes an experimental `add_spheres_cell_colored` method
that uses a 27-color spatial-cell schedule to avoid GPU write races without
atomics. Mixed radii are handled by sizing the coloring cells from the largest
radius in the batch and grouping atoms by radius for the actual stencil update.
`VoxelGridNumba` provides the same opt-in `add_spheres_cell_colored` and
prepared-plan interface for CPU parallel experiments. The normal `add_spheres`
path remains the default because the colored path has planning overhead and is
not always faster.

`VoxelGridAnalysis` uses scikit-image for connected-volume and marching-cubes
analysis when the `analysis` extra is installed. The examples extra installs
ASE for CIF loading and Wulff construction examples.

AtomVoxelizer also includes experimental field grids for scalar, vector, and
matrix-valued data at each voxel. Use `FieldVoxelGrid` / `VectorVoxelGrid` for
the NumPy implementation, or `FieldVoxelGridNumba` / `VectorVoxelGridNumba`
when Numba is installed.

## Basic Usage

```python
import numpy as np

from atomvoxelizer import VoxelGrid

cell = np.eye(3) * 10.0
grid = VoxelGrid(cell=cell, resolution=0.25)

grid.add_sphere(center=np.array([5.0, 5.0, 5.0]), radius=1.0, value=1.0)
grid.set_sphere(center=np.array([2.0, 2.0, 2.0]), radius=0.5, value=-1.0)
grid.clamp_grid(min_val=-1.0, max_val=1.0)
```

Use `dtype=` to choose the grid storage dtype when needed. The default is
`np.float32`; integer dtypes are useful for count-like masks, and complex dtypes
support arithmetic sphere operations but not ordered operations such as
`min_sphere`, `clamp_grid`, or value-range sampling.

Sphere operations accept two masks. `mask="constant"` writes the supplied value
or factor across the sphere. `mask="distance"` writes the real-space distance
from the sphere center at each voxel. Combining a distance mask with
`min_spheres` gives a nearest-atom distance field:

```python
from atomvoxelizer import VoxelGridAnalysis

grid.grid.fill(np.inf)
grid.min_spheres(atom_positions, cutoff_radii, mask="distance")

analysis = VoxelGridAnalysis(grid)
vertices, faces = analysis.mesh_at_value(2.0, periodic=True)
surface_area = analysis.mesh_surface_area(vertices, faces)
```

Periodic scalar meshes are clipped at the primary cell boundary so triangles
that cross a periodic boundary are cut at the cell edge.

Voxel grids can be saved as compact NumPy archives and restored later:

```python
grid.save_npz("distance_mask.npz")
restored = VoxelGrid.from_npz("distance_mask.npz")
```

For visualization or downstream sampling, selected voxels can be exported as a
real-space point cloud:

```python
centers, values = grid.to_point_cloud(min_value=2.5, max_value=3.5)
```

`VoxelGridNumba` and `VoxelGridCpp` also support ordered operation batching for
compiled mask recipes:

```python
grid.apply_sphere_operations([
    {"operation": "add", "centers": atom_positions, "radii": 1.4 * radii, "value": 1.0},
    {"operation": "set", "centers": atom_positions, "radii": 1.1 * radii, "value": 0.0},
])
```

## Zeolite Example

The zeolite example and CIF files live in `examples/zeolite/`.

```bash
pip install -e ".[examples]"
python examples/zeolite/zeolite_voxel.py BEA
```

The script reads a framework CIF, builds voxel grids at several resolutions, plots
middle XZ slices, benchmarks supercell scaling, and opens a 3D scatter plot.

The analysis example estimates geometric pore volume and geometric internal
surface area:

```bash
pip install -e ".[examples,analysis]"
python examples/zeolite/zeolite_analysis.py BEA --resolution 0.25
python examples/zeolite/zeolite_analysis.py BEA --convergence 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 --plot bea_convergence.png
```

The analysis example reports geometric voxel estimates, not probe-accessible BET
surface areas. It uses a fast voxel-face surface-area estimate by default. Use
`--surface-method marching-cubes` for a smoother marching-cubes estimate on
smaller grids.

## Wulff Distance-Surface Example

The Wulff example builds a nanoparticle, voxelizes the nearest-atom distance
field, and exports a marching-cubes mesh at a requested distance:

```bash
pip install -e ".[examples,analysis]"
python examples/wulff/distance_surface.py --symbol Pt --size 147 --distance 2.0 --output pt_surface.npz
python examples/wulff/distance_surface.py --symbol Pt --size 147 --distance 2.0 --plot pt_surface.png
python examples/wulff/distance_surface.py --symbol Pt --size 147 --distance 2.0 --show
```

## Periodic Surface Example

The Pt(211) example traces a periodic nearest-atom distance surface for a
stepped slab:

```bash
pip install -e ".[examples,analysis]"
python examples/surfaces/pt211_distance_surface.py --distance 1.8 --show
```

## Tests and Benchmarks

Run the correctness tests with:

```bash
pytest
```

Run the mask-generation benchmark with:

```bash
python benchmarks/benchmark_backends.py --workloads zeolite nanoparticle surface --resolution 0.5 --repeats 3 --numba-threads 24 --plot mask_generation_scaling.png
python benchmarks/benchmark_dtypes.py --backend numpy
python benchmarks/benchmark_numba_memory_order.py
python benchmarks/benchmark_numba_cell_coloring.py --thread-scaling --thread-counts 1 2 4 8 16 24 28
python benchmarks/benchmark_cupy_cell_coloring.py --sizes 923 2869 --resolution 0.6 --repeats 3
```

This benchmark scales zeolite, nanoparticle, and surface systems from small
models to roughly 6000 atoms and compares a direct atom-grid distance scan with
`VoxelGrid` NumPy, `VoxelGridNumba`, the opt-in Numba 27-color path, and
`VoxelGridCpp` when the compiled extension is available. Add `--cupy` on a
CUDA-equipped machine to include the experimental CuPy GPU backend;
its coloring plans are prepared before timing, which includes synchronization
and transfer of the completed mask to host memory.
When `--csv` is supplied, the file is initialized with a header and rewritten
after each completed workload so a running Slurm job can be monitored.
The CSV retains all measured backend series, and the rendered scaling plot
shows all six backends for comparison.

Run the built-in structure benchmark helper with:

```bash
python benchmarks/benchmark_structures.py
```

The main benchmark compares a simple direct atom-grid distance scan with the
CPU backends above. CuPy and Taichi backends remain available experimentally,
with CuPy included explicitly using `--cupy`.

## Documentation

The hosted documentation is available at:

https://atomvoxelizer.readthedocs.io/en/latest/index.html

Documentation is built with Sphinx for Read the Docs.

Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

Read the Docs can use `.readthedocs.yaml` directly.

## Publishing

For the versioning, validation, upload, and tag sequence, see the
[release checklist](docs/source/contributing.rst#release-checklist). The
artifact commands are:

```bash
pip install -e ".[publish]"
python -m build --outdir dist-X.Y.Z
twine check dist-X.Y.Z/*
```

Upload to TestPyPI first, then PyPI:

```bash
twine upload --repository testpypi dist-X.Y.Z/*
twine upload dist-X.Y.Z/*
```

## Repository mirror

The primary development repository is hosted on GitLab. The GitHub
repository is maintained as a synchronized public mirror for archival,
citation, publishing, and release purposes:

https://github.com/tgmaxson/atomvoxelizer

Please submit merge requests through the GitLab repository. GitHub issues
will also be monitored, but active development happens on GitLab.

## Single-atom insertion energies (1.3.0)

Build UFF Lennard-Jones energy grids for single-atom probes and generate
normalized energy-biased insertion proposals:

```python
import numpy as np
from atomvoxelizer import lj_insertion_grid, insertion_probabilities

energy = lj_insertion_grid(
    np.eye(3) * 20, [[10.1, 10.2, 10.3]], ["Si"],
    probe="Ar", resolution=0.5, cutoff=8.0,
)
probability = insertion_probabilities(energy.grid, temperature=298.15)
```

Run the offline carbon-cage example, including an energy/probability plot:

```bash
python examples/insertion/uff_argon.py --plot argon_insertion.png
```

The output contains energies, sampled positions, and proposal densities for
Monte Carlo acceptance corrections. Orthogonal and triclinic periodic cells
are supported. In the carbon-cage benchmark, a 0.25 Angstrom bias reduced
insertion-estimator variance by 14x at 298 K and 22x at 77 K. Reusing the grid
can save energy evaluations; setup costs and gains with other potentials need
separate assessment. See the [insertion guide](https://atomvoxelizer.readthedocs.io/en/latest/insertion.html)
for parameters, measured results, and sampling corrections.

An [iterative Ar MC example](https://atomvoxelizer.readthedocs.io/en/latest/argon_mc.html)
starts from an empty box, updates pair fields after accepted moves, and plots
drift against full regeneration. It uses a 10 eV pair cap and an energy-shifted
cutoff, with a separate comparison of 4–10 Angstrom cutoffs.
