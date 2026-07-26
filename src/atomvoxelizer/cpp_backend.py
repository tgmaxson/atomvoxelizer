from __future__ import annotations

import numpy as np

try:
    from . import _cpp_backend
except ImportError as exc:  # pragma: no cover - depends on compiled extension
    raise ImportError(
        "VoxelGridCpp requires the AtomVoxelizer C++ extension to be built. "
        "Install from a wheel with C++ support or build from source with pybind11."
    ) from exc

from .voxelgrid import VoxelGrid, _cached_sphere_offsets, _cached_sphere_offsets_and_distances


_OPERATION_CODES = {
    "set": 0,
    "add": 1,
    "mul": 2,
    "div": 3,
    "min": 4,
}


class VoxelGridCpp(VoxelGrid):
    """Voxel grid with pybind11 C++ mutation kernels."""

    backend_name = "cpp"

    def __init__(self, cell, resolution=None, gpts=None, dtype=np.float32):
        super().__init__(cell=cell, resolution=resolution, gpts=gpts, dtype=dtype)
        if np.issubdtype(self.dtype, np.complexfloating):
            raise TypeError("VoxelGridCpp does not support complex grid dtypes")

    def _flat_grid_and_shape(self):
        nx, ny, nz = self.grid.shape
        return np.ascontiguousarray(self.grid).ravel(), int(nx), int(ny), int(nz)

    def _grouped_offsets(self, radius, mask):
        self._validate_mask(mask)
        if mask == "constant":
            return _cached_sphere_offsets(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell))), None
        return _cached_sphere_offsets_and_distances(float(radius), tuple(self.gpts), tuple(map(tuple, self.cell)))

    def _apply_sphere(self, center, radius, scale, mask, operation):
        center_idx = self._center_index(center)
        offsets, distances = self._grouped_offsets(radius, mask)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        _cpp_backend.apply_offsets(grid, center_idx, offsets, distances, float(scale), nx, ny, nz, operation)

    def _apply_spheres(self, centers, radii, scale, mask, operation):
        centers, radii = self._validate_spheres(centers, radii)
        self._validate_mask(mask)
        if centers.shape[0] == 0:
            return
        center_indices = self.positions_to_indices(centers)
        grid, nx, ny, nz = self._flat_grid_and_shape()
        unique_radii, group_ids = np.unique(radii, return_inverse=True)
        offsets_parts = []
        distances_parts = []
        starts = []
        counts = []

        offset_start = 0
        for radius in unique_radii:
            offsets, distances = self._grouped_offsets(radius, mask)
            offsets_parts.append(np.ascontiguousarray(offsets, dtype=np.int32))
            if distances is not None:
                distances_parts.append(np.ascontiguousarray(distances, dtype=np.float32))
            starts.append(offset_start)
            counts.append(offsets.shape[0])
            offset_start += offsets.shape[0]

        all_offsets = np.ascontiguousarray(np.vstack(offsets_parts), dtype=np.int32)
        all_distances = None
        if mask == "distance":
            all_distances = np.ascontiguousarray(np.concatenate(distances_parts), dtype=np.float32)

        _cpp_backend.apply_grouped_offsets(
            grid,
            np.ascontiguousarray(center_indices, dtype=np.int32),
            np.ascontiguousarray(group_ids, dtype=np.int32),
            all_offsets,
            all_distances,
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64),
            float(scale),
            nx,
            ny,
            nz,
            operation,
        )

    def apply_sphere_operations(self, operations):
        """Apply an ordered sequence of batched sphere operations in one C++ call.

        Each operation is a mapping with these keys:

        ``operation``
            One of ``"set"``, ``"add"``, ``"mul"``, ``"div"``, or ``"min"``.
        ``centers`` and ``radii``
            Arrays accepted by the corresponding ``*_spheres`` method.
        ``mask``
            ``"constant"`` or ``"distance"``. Defaults to ``"constant"``
            except for ``"min"``, where it defaults to ``"distance"``.
        ``value`` or ``factor``
            Scalar applied by the operation. ``factor`` is accepted for
            ``"mul"`` and ``"div"`` for consistency with the public API.

        Operations are applied in the order supplied. This is useful for
        recipes such as adding outer coordination shells and then setting inner
        atomic cores to zero without crossing the Python/C++ boundary for each
        stage.
        """
        packed = self._pack_sphere_operations(operations)
        if packed is None:
            return

        grid, nx, ny, nz = self._flat_grid_and_shape()
        _cpp_backend.apply_operation_sequence(grid, *packed, nx, ny, nz)

    def _pack_sphere_operations(self, operations):
        if operations is None:
            raise ValueError("operations must be an iterable of operation mappings")

        center_parts = []
        center_operation_parts = []
        center_group_parts = []
        operation_codes = []
        scales = []
        offset_parts = []
        distance_parts = []
        uses_distances = []
        starts = []
        counts = []
        group_lookup = {}
        offset_start = 0

        for spec in operations:
            if not isinstance(spec, dict):
                raise TypeError("each operation must be a mapping")
            operation = spec.get("operation")
            if operation not in _OPERATION_CODES:
                raise ValueError("operation must be one of 'set', 'add', 'mul', 'div', or 'min'")
            mask = spec.get("mask", "distance" if operation == "min" else "constant")
            self._validate_mask(mask)

            centers, radii = self._validate_spheres(spec.get("centers"), spec.get("radii"))
            if centers.shape[0] == 0:
                continue

            operation_index = len(operation_codes)
            scale = self._operation_scale(spec, operation)
            center_indices = self.positions_to_indices(centers)
            group_ids = np.empty(centers.shape[0], dtype=np.int32)

            for local_group_id, radius in enumerate(np.unique(radii)):
                key = (mask, float(radius))
                group_id = group_lookup.get(key)
                if group_id is None:
                    offsets, distances = self._grouped_offsets(radius, mask)
                    group_id = len(starts)
                    group_lookup[key] = group_id
                    offset_parts.append(np.ascontiguousarray(offsets, dtype=np.int32))
                    if distances is None:
                        distance_parts.append(np.zeros(offsets.shape[0], dtype=np.float32))
                        uses_distances.append(0)
                    else:
                        distance_parts.append(np.ascontiguousarray(distances, dtype=np.float32))
                        uses_distances.append(1)
                    starts.append(offset_start)
                    counts.append(offsets.shape[0])
                    offset_start += offsets.shape[0]

                group_ids[radii == radius] = group_id

            center_parts.append(np.ascontiguousarray(center_indices, dtype=np.int32))
            center_operation_parts.append(np.full(centers.shape[0], operation_index, dtype=np.int32))
            center_group_parts.append(group_ids)
            operation_codes.append(_OPERATION_CODES[operation])
            scales.append(scale)

        if not center_parts:
            return None

        return (
            np.ascontiguousarray(np.vstack(center_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(center_operation_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(center_group_parts), dtype=np.int32),
            np.ascontiguousarray(np.vstack(offset_parts), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(distance_parts), dtype=np.float32),
            np.ascontiguousarray(uses_distances, dtype=np.uint8),
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(operation_codes, dtype=np.int32),
            np.ascontiguousarray(scales, dtype=np.float64),
        )

    @staticmethod
    def _operation_scale(spec, operation):
        if operation in {"mul", "div"}:
            return float(spec.get("factor", spec.get("value", 2.0)))
        return float(spec.get("value", 1.0))

    def set_sphere(self, center, radius, value=1, mask="constant"):
        self._apply_sphere(center, radius, value, mask, "set")

    def add_sphere(self, center, radius, value=1, mask="constant"):
        self._apply_sphere(center, radius, value, mask, "add")

    def mul_sphere(self, center, radius, factor=2, mask="constant"):
        self._apply_sphere(center, radius, factor, mask, "mul")

    def div_sphere(self, center, radius, factor=2, mask="constant"):
        self._apply_sphere(center, radius, factor, mask, "div")

    def min_sphere(self, center, radius, value=1, mask="distance"):
        self._check_ordered_grid("min_sphere")
        self._apply_sphere(center, radius, value, mask, "min")

    def add_spheres(self, centers, radii, value=1, mask="constant"):
        self._apply_spheres(centers, radii, value, mask, "add")

    def set_spheres(self, centers, radii, value=1, mask="constant"):
        self._apply_spheres(centers, radii, value, mask, "set")

    def mul_spheres(self, centers, radii, factor=2, mask="constant"):
        self._apply_spheres(centers, radii, factor, mask, "mul")

    def div_spheres(self, centers, radii, factor=2, mask="constant"):
        self._apply_spheres(centers, radii, factor, mask, "div")

    def min_spheres(self, centers, radii, value=1, mask="distance"):
        self._check_ordered_grid("min_spheres")
        self._apply_spheres(centers, radii, value, mask, "min")

    def clamp_grid(self, min_val=0.0, max_val=1.0):
        self._check_ordered_grid("clamp_grid")
        np.clip(self.grid, min_val, max_val, out=self.grid)


__all__ = ["VoxelGridCpp"]
