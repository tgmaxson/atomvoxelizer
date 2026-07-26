#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

enum class Operation {
    Set,
    Add,
    Mul,
    Div,
    Min,
};

Operation parse_operation(const std::string &operation) {
    if (operation == "set") {
        return Operation::Set;
    }
    if (operation == "add") {
        return Operation::Add;
    }
    if (operation == "mul") {
        return Operation::Mul;
    }
    if (operation == "div") {
        return Operation::Div;
    }
    if (operation == "min") {
        return Operation::Min;
    }
    throw std::invalid_argument("Unknown C++ voxel operation: " + operation);
}

Operation parse_operation_code(std::int32_t operation) {
    switch (operation) {
    case 0:
        return Operation::Set;
    case 1:
        return Operation::Add;
    case 2:
        return Operation::Mul;
    case 3:
        return Operation::Div;
    case 4:
        return Operation::Min;
    default:
        throw std::invalid_argument("Unknown C++ voxel operation code");
    }
}

int wrap_index(int index, int size) {
    int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

template <typename T>
void apply_value(T &target, double value, Operation operation) {
    switch (operation) {
    case Operation::Set:
        target = static_cast<T>(value);
        break;
    case Operation::Add:
        target = static_cast<T>(target + value);
        break;
    case Operation::Mul:
        target = static_cast<T>(target * value);
        break;
    case Operation::Div:
        target = static_cast<T>(target / value);
        break;
    case Operation::Min:
        if (value < static_cast<double>(target)) {
            target = static_cast<T>(value);
        }
        break;
    }
}

template <typename T>
void apply_offsets_typed(
    py::array_t<T, py::array::c_style> grid,
    py::array_t<std::int32_t, py::array::c_style> center_idx,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances_obj,
    double scale,
    int nx,
    int ny,
    int nz,
    Operation operation
) {
    auto grid_view = grid.template mutable_unchecked<1>();
    auto center_view = center_idx.template unchecked<1>();
    auto offsets_view = offsets.template unchecked<2>();
    const bool use_distances = !distances_obj.is_none();

    py::array_t<float, py::array::c_style> distances_array;
    const float *distances = nullptr;
    if (use_distances) {
        distances_array = py::cast<py::array_t<float, py::array::c_style>>(distances_obj);
        distances = distances_array.data();
    }

    for (py::ssize_t n = 0; n < offsets_view.shape(0); ++n) {
        const int x = wrap_index(static_cast<int>(center_view(0) + offsets_view(n, 0)), nx);
        const int y = wrap_index(static_cast<int>(center_view(1) + offsets_view(n, 1)), ny);
        const int z = wrap_index(static_cast<int>(center_view(2) + offsets_view(n, 2)), nz);
        const py::ssize_t flat = (static_cast<py::ssize_t>(x) * ny + y) * nz + z;
        const double value = use_distances ? static_cast<double>(distances[n]) * scale : scale;
        apply_value(grid_view(flat), value, operation);
    }
}

template <typename T>
void apply_many_offsets_typed(
    py::array_t<T, py::array::c_style> grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances_obj,
    double scale,
    int nx,
    int ny,
    int nz,
    Operation operation
) {
    auto grid_view = grid.template mutable_unchecked<1>();
    auto centers_view = center_indices.template unchecked<2>();
    auto offsets_view = offsets.template unchecked<2>();
    const bool use_distances = !distances_obj.is_none();

    py::array_t<float, py::array::c_style> distances_array;
    const float *distances = nullptr;
    if (use_distances) {
        distances_array = py::cast<py::array_t<float, py::array::c_style>>(distances_obj);
        distances = distances_array.data();
    }

    for (py::ssize_t c = 0; c < centers_view.shape(0); ++c) {
        for (py::ssize_t n = 0; n < offsets_view.shape(0); ++n) {
            const int x = wrap_index(static_cast<int>(centers_view(c, 0) + offsets_view(n, 0)), nx);
            const int y = wrap_index(static_cast<int>(centers_view(c, 1) + offsets_view(n, 1)), ny);
            const int z = wrap_index(static_cast<int>(centers_view(c, 2) + offsets_view(n, 2)), nz);
            const py::ssize_t flat = (static_cast<py::ssize_t>(x) * ny + y) * nz + z;
            const double value = use_distances ? static_cast<double>(distances[n]) * scale : scale;
            apply_value(grid_view(flat), value, operation);
        }
    }
}

template <typename T>
void apply_grouped_offsets_typed(
    py::array_t<T, py::array::c_style> grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> group_ids,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances_obj,
    py::array_t<std::int64_t, py::array::c_style> starts,
    py::array_t<std::int64_t, py::array::c_style> counts,
    double scale,
    int nx,
    int ny,
    int nz,
    Operation operation
) {
    auto grid_view = grid.template mutable_unchecked<1>();
    auto centers_view = center_indices.template unchecked<2>();
    auto group_view = group_ids.template unchecked<1>();
    auto offsets_view = offsets.template unchecked<2>();
    auto starts_view = starts.template unchecked<1>();
    auto counts_view = counts.template unchecked<1>();
    const bool use_distances = !distances_obj.is_none();

    py::array_t<float, py::array::c_style> distances_array;
    const float *distances = nullptr;
    if (use_distances) {
        distances_array = py::cast<py::array_t<float, py::array::c_style>>(distances_obj);
        distances = distances_array.data();
    }

    for (py::ssize_t c = 0; c < centers_view.shape(0); ++c) {
        const std::int32_t group = group_view(c);
        const py::ssize_t start = starts_view(group);
        const py::ssize_t stop = start + counts_view(group);
        for (py::ssize_t n = start; n < stop; ++n) {
            const int x = wrap_index(static_cast<int>(centers_view(c, 0) + offsets_view(n, 0)), nx);
            const int y = wrap_index(static_cast<int>(centers_view(c, 1) + offsets_view(n, 1)), ny);
            const int z = wrap_index(static_cast<int>(centers_view(c, 2) + offsets_view(n, 2)), nz);
            const py::ssize_t flat = (static_cast<py::ssize_t>(x) * ny + y) * nz + z;
            const double value = use_distances ? static_cast<double>(distances[n]) * scale : scale;
            apply_value(grid_view(flat), value, operation);
        }
    }
}

template <typename T>
void apply_operation_sequence_typed(
    py::array_t<T, py::array::c_style> grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> center_operation_ids,
    py::array_t<std::int32_t, py::array::c_style> group_ids,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::array_t<float, py::array::c_style> distances,
    py::array_t<std::uint8_t, py::array::c_style> uses_distances,
    py::array_t<std::int64_t, py::array::c_style> starts,
    py::array_t<std::int64_t, py::array::c_style> counts,
    py::array_t<std::int32_t, py::array::c_style> operation_codes,
    py::array_t<double, py::array::c_style> scales,
    int nx,
    int ny,
    int nz
) {
    auto grid_view = grid.template mutable_unchecked<1>();
    auto centers_view = center_indices.template unchecked<2>();
    auto center_ops_view = center_operation_ids.template unchecked<1>();
    auto group_view = group_ids.template unchecked<1>();
    auto offsets_view = offsets.template unchecked<2>();
    auto distances_view = distances.template unchecked<1>();
    auto uses_distances_view = uses_distances.template unchecked<1>();
    auto starts_view = starts.template unchecked<1>();
    auto counts_view = counts.template unchecked<1>();
    auto operation_codes_view = operation_codes.template unchecked<1>();
    auto scales_view = scales.template unchecked<1>();

    for (py::ssize_t c = 0; c < centers_view.shape(0); ++c) {
        const std::int32_t operation_id = center_ops_view(c);
        const Operation operation = parse_operation_code(operation_codes_view(operation_id));
        const double scale = scales_view(operation_id);
        const std::int32_t group = group_view(c);
        const bool use_distances = uses_distances_view(group) != 0;
        const py::ssize_t start = starts_view(group);
        const py::ssize_t stop = start + counts_view(group);

        for (py::ssize_t n = start; n < stop; ++n) {
            const int x = wrap_index(static_cast<int>(centers_view(c, 0) + offsets_view(n, 0)), nx);
            const int y = wrap_index(static_cast<int>(centers_view(c, 1) + offsets_view(n, 1)), ny);
            const int z = wrap_index(static_cast<int>(centers_view(c, 2) + offsets_view(n, 2)), nz);
            const py::ssize_t flat = (static_cast<py::ssize_t>(x) * ny + y) * nz + z;
            const double value = use_distances ? static_cast<double>(distances_view(n)) * scale : scale;
            apply_value(grid_view(flat), value, operation);
        }
    }
}

template <typename Func>
void dispatch_grid_dtype(py::array grid, Func &&func) {
    const char kind = grid.dtype().kind();
    const py::ssize_t itemsize = grid.itemsize();

    if (kind == 'f' && itemsize == 4) {
        func(py::array_t<float, py::array::c_style>(grid));
    } else if (kind == 'f' && itemsize == 8) {
        func(py::array_t<double, py::array::c_style>(grid));
    } else if (kind == 'i' && itemsize == 1) {
        func(py::array_t<std::int8_t, py::array::c_style>(grid));
    } else if (kind == 'i' && itemsize == 2) {
        func(py::array_t<std::int16_t, py::array::c_style>(grid));
    } else if (kind == 'i' && itemsize == 4) {
        func(py::array_t<std::int32_t, py::array::c_style>(grid));
    } else if (kind == 'i' && itemsize == 8) {
        func(py::array_t<std::int64_t, py::array::c_style>(grid));
    } else if (kind == 'u' && itemsize == 1) {
        func(py::array_t<std::uint8_t, py::array::c_style>(grid));
    } else if (kind == 'u' && itemsize == 2) {
        func(py::array_t<std::uint16_t, py::array::c_style>(grid));
    } else if (kind == 'u' && itemsize == 4) {
        func(py::array_t<std::uint32_t, py::array::c_style>(grid));
    } else if (kind == 'u' && itemsize == 8) {
        func(py::array_t<std::uint64_t, py::array::c_style>(grid));
    } else {
        throw std::invalid_argument("VoxelGridCpp supports integer and real floating grid dtypes");
    }
}

void validate_grid(py::array grid) {
    if (grid.ndim() != 1) {
        throw std::invalid_argument("grid must be a flat one-dimensional array");
    }
}

}  // namespace

void apply_offsets(
    py::array grid,
    py::array_t<std::int32_t, py::array::c_style> center_idx,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances,
    double scale,
    int nx,
    int ny,
    int nz,
    const std::string &operation
) {
    validate_grid(grid);
    const Operation op = parse_operation(operation);
    dispatch_grid_dtype(grid, [&](auto typed_grid) {
        apply_offsets_typed(typed_grid, center_idx, offsets, distances, scale, nx, ny, nz, op);
    });
}

void apply_many_offsets(
    py::array grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances,
    double scale,
    int nx,
    int ny,
    int nz,
    const std::string &operation
) {
    validate_grid(grid);
    const Operation op = parse_operation(operation);
    dispatch_grid_dtype(grid, [&](auto typed_grid) {
        apply_many_offsets_typed(typed_grid, center_indices, offsets, distances, scale, nx, ny, nz, op);
    });
}

void apply_grouped_offsets(
    py::array grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> group_ids,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::object distances,
    py::array_t<std::int64_t, py::array::c_style> starts,
    py::array_t<std::int64_t, py::array::c_style> counts,
    double scale,
    int nx,
    int ny,
    int nz,
    const std::string &operation
) {
    validate_grid(grid);
    const Operation op = parse_operation(operation);
    dispatch_grid_dtype(grid, [&](auto typed_grid) {
        apply_grouped_offsets_typed(
            typed_grid, center_indices, group_ids, offsets, distances, starts, counts, scale, nx, ny, nz, op
        );
    });
}

void apply_operation_sequence(
    py::array grid,
    py::array_t<std::int32_t, py::array::c_style> center_indices,
    py::array_t<std::int32_t, py::array::c_style> center_operation_ids,
    py::array_t<std::int32_t, py::array::c_style> group_ids,
    py::array_t<std::int32_t, py::array::c_style> offsets,
    py::array_t<float, py::array::c_style> distances,
    py::array_t<std::uint8_t, py::array::c_style> uses_distances,
    py::array_t<std::int64_t, py::array::c_style> starts,
    py::array_t<std::int64_t, py::array::c_style> counts,
    py::array_t<std::int32_t, py::array::c_style> operation_codes,
    py::array_t<double, py::array::c_style> scales,
    int nx,
    int ny,
    int nz
) {
    validate_grid(grid);
    dispatch_grid_dtype(grid, [&](auto typed_grid) {
        apply_operation_sequence_typed(
            typed_grid, center_indices, center_operation_ids, group_ids, offsets, distances, uses_distances, starts,
            counts, operation_codes, scales, nx, ny, nz
        );
    });
}

PYBIND11_MODULE(_cpp_backend, m) {
    m.doc() = "C++ mutation kernels for AtomVoxelizer";
    m.def("apply_offsets", &apply_offsets, py::arg("grid"), py::arg("center_idx"), py::arg("offsets"),
          py::arg("distances"), py::arg("scale"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
          py::arg("operation"));
    m.def("apply_many_offsets", &apply_many_offsets, py::arg("grid"), py::arg("center_indices"),
          py::arg("offsets"), py::arg("distances"), py::arg("scale"), py::arg("nx"), py::arg("ny"),
          py::arg("nz"), py::arg("operation"));
    m.def("apply_grouped_offsets", &apply_grouped_offsets, py::arg("grid"), py::arg("center_indices"),
          py::arg("group_ids"), py::arg("offsets"), py::arg("distances"), py::arg("starts"), py::arg("counts"),
          py::arg("scale"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("operation"));
    m.def("apply_operation_sequence", &apply_operation_sequence, py::arg("grid"), py::arg("center_indices"),
          py::arg("center_operation_ids"), py::arg("group_ids"), py::arg("offsets"), py::arg("distances"),
          py::arg("uses_distances"), py::arg("starts"), py::arg("counts"), py::arg("operation_codes"),
          py::arg("scales"), py::arg("nx"), py::arg("ny"), py::arg("nz"));
}
