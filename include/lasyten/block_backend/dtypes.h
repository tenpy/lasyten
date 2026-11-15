#pragma once

#include <cstdint>
#include <lasyten/lasyten.h>
#include <string>
#include <vector>

namespace lasyten {

/// The dtype of (entries in) a tensor.
///
/// Encoding: ``value ≈ num_bytes * 2 + int(not is_real)`` (except for `Bool` and `Int64`).
enum class Dtype : std::uint8_t
{
    // value ~= num_bytes * 2 + int(not is_real)
    // (except for Bool and Int64)
    Bool = 2,
    Int64 = 4, // used for index arrays, e.g. argsort
    Float32 = 8,
    Complex64 = 9,
    Float64 = 16,
    Complex128 = 17,
};

namespace dtype {

constexpr bool
is_real(Dtype dtype)
{
    return static_cast<std::uint8_t>(dtype) % 2 == 0;
}
constexpr bool
is_complex(Dtype dtype)
{
    return static_cast<std::uint8_t>(dtype) % 2 == 1;
}

Dtype to_complex(Dtype dtype);
Dtype to_real(Dtype dtype);

std::string repr(Dtype dtype);

/// Parse ``repr`` output (``bool``, ``float64``, …).
Dtype from_repr(std::string const& name);

/// Epsilon: difference between 1.0 and next representable. Bool raises.
float64 eps(Dtype dtype);

#if LASYTEN_HAS_PYTHON
/// Numpy dtype object for this Dtype.
py::object to_numpy_dtype(Dtype dtype);

/// Build Dtype from numpy dtype object. Returns none for None input (Python only).
Dtype from_numpy_dtype(py::object numpy_dtype);

#endif

/// Common dtype that can represent all given dtypes.
Dtype common(const std::vector<Dtype>& dtypes);

#if LASYTEN_HAS_PYTHON
/// Convert a Python scalar to this dtype's scalar type (returns py::object).
py::object convert_python_scalar(Dtype dtype, py::object value);

/// Python type (bool, float, complex) as a type object.
py::object python_type(Dtype dtype);

/// Zero scalar in this dtype (e.g. 0, 0.0, 0j).
py::object zero_scalar(Dtype dtype);

/// One scalar in this dtype (e.g. 1, 1.0, 1+0j).
py::object one_scalar(Dtype dtype);
#endif // LASYTEN_HAS_PYTHON

} // namespace dtype

} // namespace lasyten
