#pragma once

#include <lasyten/lasyten.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

// nanobind casters for the pybind11 object types that still appear in the public C++ API
// (py::object, py::array, py::dict, ...). Phase 4 removes them from include/lasyten/.
#include "py_pybind11_bridge.h"
// nanobind casters for Leg / FusionSymbol: must be visible in *every* binding TU (otherwise
// nanobind silently falls back to the generic "unbound type" caster and fails at runtime).
#include "symmetries/casters.hpp"

namespace lasyten {

// ``lasyten::py`` is pybind11 (declared in lasyten.h; needed by the C++ API) and ``lasyten::nb``
// is nanobind. Binding code registers classes / functions through ``nb::`` and keeps using
// ``py::object`` etc. only where the C++ API demands it; both interoperate on ``PyObject*``.

// the _core.cpp file implements the main nanobind python module lasyten._core with all python
// bindings.

/// Variadic arguments arrive as ``nb::args`` / ``nb::kwargs`` (nanobind's special parameter
/// types). The remaining binding code works with the pybind11 object API, so reinterpret them.
inline py::args
pyargs(nb::args const& a)
{
    return py::reinterpret_borrow<py::args>(py::handle(a.ptr()));
}

inline py::kwargs
pykwargs(nb::kwargs const& k)
{
    return py::reinterpret_borrow<py::kwargs>(py::handle(k.ptr()));
}

/// Constructor binding for a bound class ``T``.
///
/// Classes deriving from ``std::enable_shared_from_this`` must be created through a
/// ``std::shared_ptr`` (``nb::new_``): an in-place ``nb::init`` instance has no owning
/// ``shared_ptr``, so ``shared_from_this()`` in C++ methods would throw ``bad_weak_ptr``.
/// All other types use plain ``nb::init`` (value-like types such as ``Sector``).
template<typename T, typename... Args>
auto
ctor_for()
{
    if constexpr (std::is_abstract_v<T>) {
        // No trampoline (policy: only LinearOperator / PlanarLinearOperator have one), so an
        // abstract C++ class cannot be instantiated -- not even through a Python subclass.
        return nb::new_([](Args...) -> std::shared_ptr<T> {
            throw nb::type_error(
              "abstract class: cannot be instantiated from Python (no trampoline available)");
        });
    } else if constexpr (nb::detail::has_shared_from_this_v<T>) {
        return nb::new_(
          [](Args... args) { return std::make_shared<T>(std::forward<Args>(args)...); });
    } else {
        return nb::init<Args...>();
    }
}

/// Like ``ctor_for``, but takes the type of the ``nb::class_<...>`` object being defined (so the
/// binding code can write ``lasyten::ctor<decltype(cls), Args...>()``). Trampoline classes keep
/// ``nb::init`` so that Python subclasses instantiate the trampoline (alias) type; those are
/// never ``enable_shared_from_this``.
template<typename Cls, typename... Args>
auto
ctor()
{
    using C = std::remove_cvref_t<Cls>;
    using T = typename C::Type;
    if constexpr (!std::is_same_v<typename C::Alias, T>) {
        return nb::init<Args...>();
    } else {
        return ctor_for<T, Args...>();
    }
}

// here, we have declarations of binding functions defined in the corresponding *.cpp files.

void bind_config(nb::module_& m);
void bind_tools(nb::module_& m);
void bind_cost_polynomials(nb::module_& m);
void bind_mappings(nb::module_& m);
void bind_block_backend(nb::module_& m);
void bind_symmetries(nb::module_& m);
void declare_block_inds(nb::module_& m);
void bind_block_inds(nb::module_& m);
void declare_tensor_backend(nb::module_& m);
void bind_tensor_backend(nb::module_& m);
void bind_no_symmetry_backend(nb::module_& m);
void bind_abelian_backend_data(nb::module_& m);
void bind_abelian_backend(nb::module_& m);
void bind_fusion_tree_data(nb::module_& m);
void bind_fusion_tree_backend(nb::module_& m);
void bind_fusion_tree_mapping(nb::module_& m);
void bind_backend_factory(nb::module_& m);
void declare_tensors_labels(nb::module_& m);
void bind_tensors_labels(nb::module_& m);
void declare_tensors_vector_like(nb::module_& m);
void bind_tensors_vector_like(nb::module_& m);
void declare_tensors_tensor(nb::module_& m);
void bind_tensors_tensor(nb::module_& m);
void bind_tensors_direct_sum(nb::module_& m);
void declare_tensors_symmetric_tensor(nb::module_& m);
void bind_tensors_symmetric_tensor(nb::module_& m);
void declare_tensors_diagonal_tensor(nb::module_& m);
void bind_tensors_diagonal_tensor(nb::module_& m);
void declare_tensors_mask(nb::module_& m);
void bind_tensors_mask(nb::module_& m);
void bind_tensors_charged_tensor(nb::module_& m);
void bind_tensors_hidden_leg_tensor(nb::module_& m);
void bind_tensors_helpers(nb::module_& m);
void bind_tensors_constructors(nb::module_& m);
void bind_tensors_ops_elementwise(nb::module_& m);
void bind_tensors_ops_algebra(nb::module_& m);
void bind_tensors_ops_legs(nb::module_& m);
void bind_tensors_decompositions(nb::module_& m);
void bind_tensors_sparse(nb::module_& m);
void bind_tensors_krylov_based(nb::module_& m);
void bind_tensors_planar(nb::module_& m);
void bind_models_degrees_of_freedom(nb::module_& m);
void bind_models_sites(nb::module_& m);
void bind_models_couplings(nb::module_& m);

} // namespace lasyten
