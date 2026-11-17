#pragma once

/// Helpers for C++ programs that *embed* a Python interpreter and use liblasyten (the ctests, the
/// user-guide example). nanobind has no ``scoped_interpreter`` (it targets extension modules), so
/// this provides the two pieces such a program needs. Requires ``LASYTEN_WITH_PYTHON=ON``.
///
/// Usage::
///
///     lasyten::ScopedInterpreter guard;      // Py_Initialize / Py_FinalizeEx
///     nb::module_::import_("lasyten");        // loads lasyten._core (registers bound classes)
///     lasyten::init_embedded_nanobind();      // attach this program's nanobind to the
///     extension's

#include <lasyten/lasyten.h>

#if !LASYTEN_HAS_PYTHON
#error "<lasyten/embed.h> needs LASYTEN_WITH_PYTHON=ON"
#endif

namespace lasyten {

/// RAII Python interpreter (``Py_Initialize`` on construction, ``Py_FinalizeEx`` on destruction).
class ScopedInterpreter
{
  public:
    ScopedInterpreter() { Py_Initialize(); }
    ScopedInterpreter(ScopedInterpreter const&) = delete;
    ScopedInterpreter& operator=(ScopedInterpreter const&) = delete;
    ~ScopedInterpreter() { Py_FinalizeEx(); }
};

/// The program carries its own copy of nanobind (static library), which must be attached to the
/// nanobind internals shared with the ``lasyten._core`` extension. Call once, after the
/// interpreter is up.
inline void
init_embedded_nanobind()
{
    auto core = nb::module_::import_("lasyten._core");
    if (!nb::register_module(core)) {
        throw nb::python_error();
    }
}

} // namespace lasyten
