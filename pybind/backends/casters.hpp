#pragma once

/// nanobind helpers: accept NumPy arrays as BlockInds arguments.
///
/// ``nb::class_<BlockInds>`` owns the Python type. ``nb::implicitly_convertible<py::array,
/// BlockInds>()`` (registered in ``py_block_inds.cpp``) extends *load* so call sites can still
/// pass ndarrays; *cast* always produces bound BlockInds. The pybind11-side bridge used by
/// liblasyten is ``include/lasyten/py_nanobind_bridge.h``.

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/block_inds_numpy.h>

#include <nanobind/nanobind.h>

#include "../py_pybind11_bridge.h"
