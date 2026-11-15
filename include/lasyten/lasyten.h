#pragma once
#define _USE_MATH_DEFINES

// global definitions and includes for lasyten

#include <cassert>
#include <cstdint>
#include <utility>

#include <lasyten/check.h>

#include <complex>
#include <limits>

// The one place that pulls in pybind11 / nanobind (or, with LASYTEN_WITH_PYTHON=OFF, opaque
// stand-ins for the ``py::`` types): see lasyten/detail/python_api.h.
#include <lasyten/detail/python_api.h>

static_assert(std::numeric_limits<double>::is_iec559); // double is indeed 64 bit

namespace lasyten {

using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;
using float32 = float;
using float64 = double;
using complex64 = std::complex<float32>;
using complex128 = std::complex<float64>;

} // namespace lasyten
