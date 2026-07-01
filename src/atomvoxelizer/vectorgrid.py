from __future__ import annotations

from functools import lru_cache

import numpy as np

from .voxelgrid import VoxelGrid, _cached_sphere_offsets


@lru_cache(maxsize=200)
def _cached_sphere_offsets_and_vectors(radius, gpts, cell):
    gpts_arr = np.array(gpts, dtype=np.int32)
    cell_arr = np.array(cell, dtype=np.float64)
    offsets = _cached_sphere_offsets(radius, gpts, cell)
    vectors = np.zeros((offsets.shape[0], 3), dtype=np.float32)

    for index, offset in enumerate(offsets):
        disp_frac = offset.astype(np.float64) / gpts_arr
        disp = disp_frac @ cell_arr
        norm = np.linalg.norm(disp)
        if norm > 0.0:
            vectors[index] = disp / norm

    return offsets, vectors


class FieldVoxelGrid(VoxelGrid):
    """Periodic voxel grid where each voxel stores an arbitrary-shaped value.

    This NumPy-only prototype mirrors the geometric indexing behavior of
    :class:`atomvoxelizer.VoxelGrid`, but stores values with shape
    ``(*gpts, *value_shape)``. Examples:

    * ``value_shape=()`` stores scalar values.
    * ``value_shape=(1,)`` stores scalar values as length-1 vectors.
    * ``value_shape=(3,)`` stores real-space vectors and enables the
      ``mask="normal"`` sphere mask.
    * ``value_shape=(3, 3)`` stores a matrix at each voxel.
    """

    backend_name = "numpy-field"

    def __init__(self, cell, resolution=None, gpts=None, dtype=np.float32, value_shape=(3,), components=None):
        dtype = np.dtype(dtype)
        if dtype.kind not in "f":
            raise TypeError("FieldVoxelGrid dtype must be a floating NumPy dtype")
        if components is not None:
            value_shape = (int(components),)
        self.value_shape = self._normalize_value_shape(value_shape)
        if any(dim < 1 for dim in self.value_shape):
            raise ValueError("value_shape dimensions must be at least 1")
        self.components = self.value_shape[0] if len(self.value_shape) == 1 else None
        super().__init__(cell=cell, resolution=resolution, gpts=gpts, dtype=dtype)
        self.grid = np.zeros((*tuple(self.gpts), *self.value_shape), dtype=self.dtype)

    @staticmethod
    def _normalize_value_shape(value_shape):
        if value_shape is None:
            return ()
        if isinstance(value_shape, int):
            return (int(value_shape),)
        return tuple(int(dim) for dim in value_shape)

    def _offset_indices(self, center_idx, offsets):
        indices = (offsets + center_idx) % self.gpts
        return tuple(indices[:, axis] for axis in range(3))

    @staticmethod
    def _validate_field_mask(mask):
        if mask not in {"constant", "normal"}:
            raise ValueError("mask must be 'constant' or 'normal'")

    def _default_constant_value(self):
        if self.value_shape == ():
            return self.dtype.type(1)
        return np.ones(self.value_shape, dtype=self.dtype)

    def _validate_value(self, value):
        values = np.asarray(value, dtype=self.dtype)
        if self.value_shape == ():
            if values.shape != ():
                raise ValueError("value must be scalar for value_shape=()")
            return values
        if self.value_shape == (1,) and values.shape == ():
            return values.reshape(1)
        if values.shape != self.value_shape:
            raise ValueError(f"value must have shape {self.value_shape}")
        return values

    def _sphere_indices_and_values(self, center, radius, value, mask):
        self._validate_field_mask(mask)
        center_idx = self._center_index(center)
        if mask == "constant":
            offsets = self._sphere_offsets(radius)
            values = self._validate_value(value)
        else:
            if self.value_shape != (3,):
                raise ValueError('normal mask requires value_shape=(3,)')
            offsets, normal_vectors = _cached_sphere_offsets_and_vectors(
                float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))
            )
            values = normal_vectors.astype(self.dtype, copy=False) * self.dtype.type(value)
        return self._offset_indices(center_idx, offsets), values

    def set_sphere(self, center, radius, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        self.grid[indices] = values

    def add_sphere(self, center, radius, value=None, mask="constant"):
        if value is None:
            value = self._default_constant_value() if mask == "constant" else 1.0
        indices, values = self._sphere_indices_and_values(center, radius, value, mask)
        np.add.at(self.grid, indices, values)

    def mul_sphere(self, center, radius, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        indices, values = self._sphere_indices_and_values(center, radius, factor, mask)
        self.grid[indices] *= values

    def div_sphere(self, center, radius, factor=None, mask="constant"):
        if factor is None:
            factor = self._default_constant_value() if mask == "constant" else 1.0
        indices, values = self._sphere_indices_and_values(center, radius, factor, mask)
        self.grid[indices] /= values

    def set_spheres(self, centers, radii, value=None, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.set_sphere(center, radius, value=value, mask=mask)

    def add_spheres(self, centers, radii, value=None, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.add_sphere(center, radius, value=value, mask=mask)

    def mul_spheres(self, centers, radii, factor=None, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.mul_sphere(center, radius, factor=factor, mask=mask)

    def div_spheres(self, centers, radii, factor=None, mask="constant"):
        centers, radii = self._validate_spheres(centers, radii)
        for center, radius in zip(centers, radii):
            self.div_sphere(center, radius, factor=factor, mask=mask)

    def min_sphere(self, center, radius, value=1, mask="constant"):
        raise NotImplementedError("min_sphere is not implemented for FieldVoxelGrid")

    def min_spheres(self, centers, radii, value=1, mask="constant"):
        raise NotImplementedError("min_spheres is not implemented for FieldVoxelGrid")

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        raise NotImplementedError("clamp_grid is not implemented for FieldVoxelGrid")

    def sample_voxels_in_range(self, min_val=0.0, max_val=1.0, min_dist=0.0, return_indices=False, seed=None):
        raise NotImplementedError("sample_voxels_in_range is not implemented for FieldVoxelGrid")

    def value_norms(self):
        """Return the Euclidean/Frobenius norm of each voxel value."""
        if self.value_shape == ():
            return np.abs(self.grid)
        axes = tuple(range(3, self.grid.ndim))
        return np.linalg.norm(self.grid, axis=axes)

    def normalize_values(self, inplace=True):
        """Normalize nonzero voxel values and leave zero values unchanged."""
        target = self.grid if inplace else self.grid.copy()
        norms = self._norms_for(target, self.value_shape)
        nonzero = norms > 0.0
        if self.value_shape == ():
            target[nonzero] /= norms[nonzero]
        else:
            target[nonzero] /= norms[nonzero][(...,) + (None,) * len(self.value_shape)]
        if inplace:
            return self
        return target

    @staticmethod
    def _norms_for(values, value_shape):
        if value_shape == ():
            return np.abs(values)
        axes = tuple(range(3, values.ndim))
        return np.linalg.norm(values, axis=axes)

    def vector_norms(self):
        """Return per-voxel value norms.

        Kept as a convenience alias for the first vector-field prototype.
        """
        return self.value_norms()

    def normalize_vectors(self, inplace=True):
        """Normalize nonzero voxel values.

        Kept as a convenience alias for the first vector-field prototype.
        """
        return self.normalize_values(inplace=inplace)

    def scalar_values(self):
        """Return scalar values for ``value_shape=()`` or ``value_shape=(1,)`` fields."""
        if self.value_shape == ():
            return self.grid
        if self.value_shape == (1,):
            return self.grid[..., 0]
        raise ValueError("scalar_values requires value_shape=() or value_shape=(1,)")

    def _check_vector_field(self, operation):
        if self.value_shape != (3,):
            raise ValueError(f"{operation} requires value_shape=(3,)")

    @staticmethod
    def _normalize_selected(vectors):
        if vectors.size == 0:
            return vectors
        normalized = vectors.copy()
        norms = np.linalg.norm(normalized, axis=-1)
        nonzero = norms > 0.0
        normalized[nonzero] /= norms[nonzero, None]
        return normalized

    def _voxel_center_positions(self):
        nx, ny, nz = self.gpts
        ix, iy, iz = np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            np.arange(nz),
            indexing="ij",
        )
        frac = (np.stack([ix, iy, iz], axis=-1) + 0.5) / self.gpts
        return frac @ self.cell

    def _sampled_vector_mask(self, norms, stride, min_norm):
        if stride < 1:
            raise ValueError("stride must be at least 1")
        selected = norms > min_norm
        sampled = np.zeros_like(selected, dtype=bool)
        sampled[(slice(None, None, stride),) * selected.ndim] = True
        return selected & sampled

    def _slice_index(self, axis, index=None, position=None):
        ax_map = {"x": 0, "y": 1, "z": 2}
        if axis not in ax_map:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        ax_idx = ax_map[axis]
        if index is not None and position is not None:
            raise ValueError("Specify either index or position, not both")
        if position is not None:
            index = self.position_to_index(np.eye(3)[ax_idx] * position)[ax_idx]
        if index is None:
            index = self.gpts[ax_idx] // 2
        if not (0 <= index < self.gpts[ax_idx]):
            raise IndexError(f"{axis}-index {index} out of bounds (0 to {self.gpts[ax_idx] - 1})")
        return ax_idx, int(index)

    def quiver_slice_data(self, axis="z", index=None, position=None, stride=1, min_norm=0.0, normalize=False):
        """Return 2D quiver data for a slice through a three-component vector field."""
        self._check_vector_field("quiver_slice_data")
        ax_idx, index = self._slice_index(axis, index=index, position=position)

        axes = [0, 1, 2]
        axes.remove(ax_idx)
        ax1, ax2 = axes
        slicers = [slice(None)] * 3
        slicers[ax_idx] = index
        vectors = self.grid[tuple(slicers)]
        positions = self._voxel_center_positions()[tuple(slicers)]

        norms = np.linalg.norm(vectors, axis=-1)
        selected = self._sampled_vector_mask(norms, stride, min_norm)
        selected_vectors = vectors[selected]
        if normalize:
            selected_vectors = self._normalize_selected(selected_vectors)

        return {
            "x": positions[..., ax1][selected],
            "y": positions[..., ax2][selected],
            "u": selected_vectors[:, ax1] if selected_vectors.size else np.array([], dtype=self.dtype),
            "v": selected_vectors[:, ax2] if selected_vectors.size else np.array([], dtype=self.dtype),
            "norm": norms[selected],
            "axis": axis,
            "index": index,
            "axes": (ax1, ax2),
        }

    def plot_quiver_slice(
        self,
        axis="z",
        index=None,
        position=None,
        stride=1,
        min_norm=0.0,
        normalize=False,
        ax=None,
        scale=None,
    ):
        """Plot a 2D quiver slice through a three-component vector field."""
        import matplotlib.pyplot as plt

        data = self.quiver_slice_data(
            axis=axis,
            index=index,
            position=position,
            stride=stride,
            min_norm=min_norm,
            normalize=normalize,
        )
        if ax is None:
            _fig, ax = plt.subplots()
        ax.quiver(data["x"], data["y"], data["u"], data["v"], data["norm"], angles="xy", scale_units="xy", scale=scale)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("xyz"[data["axes"][0]])
        ax.set_ylabel("xyz"[data["axes"][1]])
        ax.set_title(f"{axis} slice {data['index']}")
        return ax

    def quiver_3d_data(self, stride=1, min_norm=0.0, normalize=False):
        """Return 3D quiver data for a three-component vector field."""
        self._check_vector_field("quiver_3d_data")
        positions = self._voxel_center_positions()
        vectors = self.grid
        norms = np.linalg.norm(vectors, axis=-1)

        selected = self._sampled_vector_mask(norms, stride, min_norm)
        selected_vectors = vectors[selected]
        if normalize:
            selected_vectors = self._normalize_selected(selected_vectors)

        selected_positions = positions[selected]
        return {
            "x": selected_positions[:, 0] if selected_positions.size else np.array([], dtype=self.dtype),
            "y": selected_positions[:, 1] if selected_positions.size else np.array([], dtype=self.dtype),
            "z": selected_positions[:, 2] if selected_positions.size else np.array([], dtype=self.dtype),
            "u": selected_vectors[:, 0] if selected_vectors.size else np.array([], dtype=self.dtype),
            "v": selected_vectors[:, 1] if selected_vectors.size else np.array([], dtype=self.dtype),
            "w": selected_vectors[:, 2] if selected_vectors.size else np.array([], dtype=self.dtype),
            "norm": norms[selected],
        }

    def plot_quiver_3D(self, stride=1, min_norm=0.0, normalize=False, ax=None, length=1.0):
        """Plot a sampled 3D quiver view of a three-component vector field."""
        import matplotlib.pyplot as plt

        data = self.quiver_3d_data(stride=stride, min_norm=min_norm, normalize=normalize)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        ax.quiver(data["x"], data["y"], data["z"], data["u"], data["v"], data["w"], length=length, normalize=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def plot_2D(self, *args, **kwargs):
        raise NotImplementedError("plot_2D is not implemented for FieldVoxelGrid; use plot_quiver_slice")

    def plot_3D(self, *args, **kwargs):
        raise NotImplementedError("plot_3D is not implemented for FieldVoxelGrid; use plot_quiver_3D")


class VectorVoxelGrid(FieldVoxelGrid):
    """Convenience field grid with a three-component vector value by default."""

    def __init__(self, cell, resolution=None, gpts=None, dtype=np.float32, components=3, value_shape=None):
        if value_shape is None:
            value_shape = (int(components),)
        super().__init__(cell=cell, resolution=resolution, gpts=gpts, dtype=dtype, value_shape=value_shape)


FieldVoxelGridNumPy = FieldVoxelGrid
VectorVoxelGridNumPy = VectorVoxelGrid


__all__ = [
    "FieldVoxelGrid",
    "FieldVoxelGridNumPy",
    "VectorVoxelGrid",
    "VectorVoxelGridNumPy",
    "_cached_sphere_offsets_and_vectors",
]
