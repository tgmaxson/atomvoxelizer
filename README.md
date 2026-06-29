# AtomVoxelizer

AtomVoxelizer builds periodic atom-centered voxel grids for atomistic structures.
The core `VoxelGrid` class stores a 3D NumPy grid over a periodic cell and provides
helpers for adding, setting, scaling, sampling, and plotting spherical regions.

## Installation

Install from this repository:

```bash
pip install .
```

For development, examples, and documentation:

```bash
pip install -e ".[examples,docs]"
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

## Documentation

Documentation is scaffolded with Sphinx for Read the Docs.

Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

Read the Docs can use `.readthedocs.yaml` directly.

