#pragma once

/// Pickle support for nanobind-bound lasyten types.
///
/// nanobind types do not pickle automatically. The pickled *logical state* mirrors what
/// ``save_hdf5`` / ``from_hdf5`` persist (see ``hdf5_bind.h``), but is kept as plain Python
/// objects (``numpy`` arrays, ``dict`` keyed by the HDF5 sub-group names, ...). No HDF5 / h5py
/// round trip is involved.
///
/// Three flavours are provided:
///
///  * ``def_value_pickle``    -- value types (``Sector``, ``SectorArray``, ...): ``__getstate__``
///    plus a placement-new ``__setstate__`` (the pattern from the nanobind docs). Unpickling goes
///    through ``cls.__new__(cls)`` followed by ``__setstate__(state)``.
///  * ``def_reduce_ctor``     -- types with a Python-visible constructor: ``__reduce__`` returns
///    ``(type(self), ctor_args)``, i.e. unpickling calls the constructor.
///  * ``def_state_pickle``    -- shared_ptr-held / factory types whose state is a ``dict``:
///    ``__getstate__`` returns the dict, ``_from_pickle_state(dict)`` is a static factory and
///    ``__reduce__`` returns ``(methodcaller("_from_pickle_state", state), (type(self),))``.
///  * ``reduce_via_method``   -- building block for custom ``__reduce__``: unpickling evaluates
///    ``target.method(*args)`` (e.g. ``backend.block_from_numpy(array)``).
///
/// Why ``operator.methodcaller``: nanobind functions (``nb_func``) cannot be pickled by
/// reference, so a plain ``(cls._from_pickle_state, (state,))`` reduce value is not picklable. A
/// *type* can be pickled by reference though, and ``methodcaller`` is a regular picklable object,
/// so ``methodcaller(name, *args)(cls)`` == ``cls.name(*args)`` gives a reconstructor that goes
/// through the static factory. (``__setstate__`` is not used for shared_ptr-held types: their
/// ``__new__`` allocates an uninitialized instance without an owning ``shared_ptr``.)

#include "py_lasyten_nanobind.h"

#include <new>
#include <utility>

namespace lasyten::pickle {

/// The Python type of ``self`` (most derived bound class, so that subclasses round-trip).
inline nb::handle
type_of(nb::handle self)
{
    return nb::handle(reinterpret_cast<PyObject*>(Py_TYPE(self.ptr())));
}

/// ``(operator.methodcaller(name, *args), (target,))``: unpickling evaluates
/// ``target.name(*args)``; ``target`` is typically a class (static factory) or a picklable
/// instance such as a backend (``backend.block_from_numpy(array)``).
template<typename... Args>
nb::tuple
reduce_via_method(nb::handle target, char const* name, Args&&... args)
{
    nb::object caller =
      nb::module_::import_("operator").attr("methodcaller")(name, std::forward<Args>(args)...);
    return nb::make_tuple(caller, nb::make_tuple(target));
}

/// ``(type(self), ctor_args)``: unpickling evaluates ``type(self)(*ctor_args)``.
inline nb::tuple
reduce_via_ctor(nb::handle self, nb::tuple ctor_args)
{
    return nb::make_tuple(type_of(self), std::move(ctor_args));
}

/// Value types: ``__getstate__(self) -> state`` and a placement-new ``__setstate__(self, state)``.
///
/// \param get   ``(T const&) -> nb::object``
/// \param make  ``(nb::object state) -> T`` (T must be move-constructible)
template<typename Cls, typename Get, typename Make>
void
def_value_pickle(Cls& cls, Get get, Make make)
{
    using T = typename Cls::Type;
    cls.def(
      "__getstate__", [get](T const& self) -> nb::object { return get(self); }, "Pickle state.");
    cls.def(
      "__setstate__",
      [make](T& self, nb::object state) { new (&self) T(make(state)); },
      nb::arg("state"),
      "Restore from the pickle state (placement new).");
}

/// Types reconstructed through a Python-visible constructor: ``__reduce__`` returns
/// ``(type(self), args(self))``.
///
/// \param args  ``(T const&) -> nb::tuple`` of constructor arguments
template<typename Cls, typename Args>
void
def_reduce_ctor(Cls& cls, Args args)
{
    using T = typename Cls::Type;
    cls.def(
      "__reduce__",
      [args](nb::object self) -> nb::tuple {
          return reduce_via_ctor(self, args(nb::cast<T const&>(self)));
      },
      "Pickle support: ``(type(self), ctor_args)``.");
}

/// ``dict``-state types with a static factory ``_from_pickle_state(dict)``.
///
/// The dict keys mirror the HDF5 sub-group names used by ``save_hdf5``.
///
/// \param get   ``(T const&) -> nb::dict``
/// \param from  ``(nb::dict) -> std::shared_ptr<T>`` (static factory)
template<typename Cls, typename Get, typename From>
void
def_state_pickle(Cls& cls, Get get, From from)
{
    using T = typename Cls::Type;
    cls.def(
      "__getstate__",
      [get](T const& self) -> nb::dict { return get(self); },
      "Pickle state: dict with the same logical fields as ``save_hdf5``.");
    cls.def_static(
      "_from_pickle_state",
      [from](nb::dict state) { return from(state); },
      nb::arg("state"),
      "Rebuild from the dict returned by ``__getstate__`` (used by ``__reduce__``).");
    cls.def(
      "__reduce__",
      [](nb::object self) -> nb::tuple {
          nb::object state = self.attr("__getstate__")();
          return reduce_via_method(type_of(self), "_from_pickle_state", state);
      },
      "Pickle support: ``(methodcaller('_from_pickle_state', state), (type(self),))``.");
}

/// ``state[key]`` cast to ``V``; raises ``KeyError`` for a missing key.
template<typename V>
V
get_item(nb::dict const& state, char const* key)
{
    nb::str k(key);
    if (!state.contains(k)) {
        throw nb::key_error(key);
    }
    return nb::cast<V>(state[k]);
}

} // namespace lasyten::pickle
