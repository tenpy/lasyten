#pragma once

/// Small helpers to hand owning NumPy ``nb::ndarray`` values out of bindings (leftover #10).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace lasyten::ndarray_utils {

namespace nb = nanobind;

/// Move a host ``std::vector<T>`` into an owning 1-D NumPy array.
template<typename T>
[[nodiscard]] nb::ndarray<nb::numpy, T>
vector_to_numpy_1d(std::vector<T> v)
{
    auto* heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    std::size_t shape[1] = { heap->size() };
    return nb::ndarray<nb::numpy, T>(heap->data(), /*ndim=*/1, shape, owner);
}

/// Owning 2-D NumPy array from row-major ``rows * cols`` data in ``v``.
template<typename T>
[[nodiscard]] nb::ndarray<nb::numpy, T>
vector_to_numpy_2d(std::vector<T> v, std::size_t rows, std::size_t cols)
{
    if (v.size() != rows * cols) {
        throw std::invalid_argument("vector_to_numpy_2d: size does not match rows*cols");
    }
    auto* heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    std::size_t shape[2] = { rows, cols };
    return nb::ndarray<nb::numpy, T>(heap->data(), /*ndim=*/2, shape, owner);
}

} // namespace lasyten::ndarray_utils
