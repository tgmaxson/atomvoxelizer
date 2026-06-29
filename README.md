# AtomVoxelizer

AtomVoxelizer builds periodic atom-centered voxel grids for atomistic structures.
The core `VoxelGrid` class stores a 3D NumPy grid over a periodic cell and provides
helpers for adding, setting, scaling, sampling, and plotting spherical regions.

## Installation

Install from this repository:

```bash
pip install .
```

Install optional acceleration backends with extras:

```bash
pip install ".[numba]"
pip install ".[taichi]"
pip install ".[cupy]"
pip install ".[analysis]"
```

`VoxelGrid` is always the NumPy backend. Optional acceleration backends are
explicit: `VoxelGridNumba`, `VoxelGridTaichi`, and `VoxelGridCuPy`.
`VoxelGridAnalysis` provides connected-volume and marching-cubes surface-area
analysis when the `analysis` extra is installed.

For development, examples, tests, and documentation:

```bash
pip install -e ".[dev,examples]"
```

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

## Zeolite Example

The zeolite example and CIF files live in `examples/`.

```bash
pip install -e ".[examples]"
python examples/zeolite_voxel.py BEA
```

The script reads a framework CIF, builds voxel grids at several resolutions, plots
middle XZ slices, benchmarks supercell scaling, and opens a 3D scatter plot.

The analysis example estimates pore volume and internal surface area:

```bash
pip install -e ".[examples,analysis]"
python examples/zeolite_analysis.py BEA --resolution 0.25
python examples/zeolite_analysis.py BEA --convergence 1.0 0.75 0.5 --plot bea_convergence.png
```

## Tests and Benchmarks

Run the correctness tests with:

```bash
pytest
```

Run the backend benchmark with:

```bash
python benchmarks/benchmark_backends.py --backends numpy numba taichi cupy
```

Run the built-in structure benchmarks for a zeolite and a roughly 1000 atom Wulff
construction with:

```bash
python benchmarks/benchmark_structures.py
```

Backends whose optional dependencies are not installed are reported as missing.

## Documentation

Documentation is scaffolded with Sphinx for Read the Docs.

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
