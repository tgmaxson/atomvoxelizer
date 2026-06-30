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
```

`VoxelGrid` is always the NumPy backend. Optional acceleration backends are
explicit: `VoxelGridNumba`, `VoxelGridTaichi`, and `VoxelGridCuPy`.
`VoxelGridAnalysis` uses scikit-image for connected-volume and marching-cubes
analysis when the `analysis` extra is installed. The examples extra installs
ASE for CIF loading and Wulff construction examples.

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

Run the backend benchmark with:

```bash
python benchmarks/benchmark_backends.py --backends numpy numba taichi cupy
python benchmarks/benchmark_backends.py --zeolite-scaling --framework BEA --resolution 0.5 --plot zeolite_scaling.png
python benchmarks/benchmark_backends.py --workload zeolite --backends taichi-gpu
```

Run the built-in structure benchmarks for a zeolite and a roughly 1000 atom Wulff
construction with:

```bash
python benchmarks/benchmark_structures.py
```

Backends whose optional dependencies are not installed are reported as missing.

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

Build and check PyPI artifacts with:

```bash
pip install -e ".[publish]"
python -m build
twine check dist/*
```

Upload to TestPyPI first, then PyPI:

```bash
twine upload --repository testpypi dist/*
twine upload dist/*
```
