#pragma once

/// nanobind helpers: accept NumPy arrays as BlockInds arguments.
///
/// ``nb::class_<BlockInds>`` owns the Python type. ``nb::implicitly_convertible<py::array,
/// BlockInds>()`` (registered in ``py_block_inds.cpp``) extends *load* so call sites can still
/// pass ndarrays; *cast* always produces bound BlockInds.

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/block_inds_numpy.h>

#include <nanobind/nanobind.h>
