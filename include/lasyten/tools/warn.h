#pragma once

#include <lasyten/detail/python_api.h>

#include <iostream>
#include <string>
#include <string_view>

namespace lasyten {

/// Emit a UserWarning via Python's warning machinery when the interpreter is active.
/// Falls back to stderr when ``!Py_IsInitialized()`` (pure-C++ / pre-interpreter use) and always
/// when built with ``LASYTEN_WITH_PYTHON=OFF``.
inline void
warn(std::string_view message, int stack_level = 2)
{
#if LASYTEN_HAS_PYTHON
    if (!Py_IsInitialized()) {
        // Fallback for pure-C++ / pre-interpreter use; refine later if needed.
        std::cerr << "UserWarning: " << message << '\n';
        return;
    }
    // Stable-ABI safe; with ``-W error`` (pytest) the warning becomes an exception.
    if (PyErr_WarnEx(PyExc_UserWarning, std::string(message).c_str(), stack_level) < 0) {
        throw py::error_already_set();
    }
#else
    (void)stack_level;
    std::cerr << "UserWarning: " << message << '\n';
#endif
}

} // namespace lasyten
