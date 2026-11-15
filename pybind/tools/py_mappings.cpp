#include <lasyten/tools/mappings.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "docstrings/tools/mappings.h"

#include <complex>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

template<typename Mapping>
void
bind_sparse_mapping(nb::module_& m, char const* name, char const* doc)
{
    using KT = typename Mapping::Key;
    using Inner = typename Mapping::Inner;

    nb::class_<Mapping> cls(m, name);
    cls.doc() = doc;

    cls.def(lasyten::ctor<decltype(cls)>())
      .def(lasyten::ctor<decltype(cls), typename Mapping::Outer>(), nb::arg("data"))
      .def_rw("data", &Mapping::data)
      .def_static("from_identity",
                  &Mapping::from_identity,
                  nb::arg("keys"),
                  doc_cpp_ref(R"pydoc(from_identity)pydoc", "lasyten::Mapping::from_identity()"))
      .def("pre_compose",
           &Mapping::pre_compose,
           nb::arg("other"),
           doc_cpp_ref(R"pydoc(pre_compose)pydoc", "lasyten::Mapping::pre_compose()"))
      .def("nonzero_rows",
           &Mapping::nonzero_rows,
           doc_cpp_ref(R"pydoc(nonzero_rows)pydoc", "lasyten::Mapping::nonzero_rows()"))
      .def("nonzero_cols",
           &Mapping::nonzero_cols,
           doc_cpp_ref(R"pydoc(nonzero_cols)pydoc", "lasyten::Mapping::nonzero_cols()"))
      .def("prune",
           &Mapping::prune,
           nb::arg("tol"),
           doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::Mapping::prune()"))
      .def("__len__", &Mapping::size)
      .def(
        "__contains__",
        [](Mapping const& self, KT const& j) { return self.contains(j); },
        nb::arg("j"))
      .def(
        "__getitem__",
        [](Mapping const& self, KT const& j) {
            auto it = self.data.find(j);
            if (it == self.data.end()) {
                throw py::key_error("SparseMapping key not found");
            }
            return it->second; // copy to Python dict
        },
        nb::arg("j"))
      .def(
        "__setitem__",
        [](Mapping& self, KT const& j, Inner const& inner) { self.data[j] = inner; },
        nb::arg("j"),
        nb::arg("value"))
      .def("items",
           [](Mapping const& self) {
               py::list out;
               for (auto const& [j, inner] : self.data) {
                   out.append(py::make_tuple(j, inner));
               }
               return out;
           })
      .def("keys",
           [](Mapping const& self) {
               py::list out;
               for (auto const& [j, inner] : self.data) {
                   (void)inner;
                   out.append(j);
               }
               return out;
           })
      .def("values", [](Mapping const& self) {
          py::list out;
          for (auto const& [j, inner] : self.data) {
              (void)j;
              out.append(inner);
          }
          return out;
      });
}

template<typename IdMapping>
void
bind_identity_mapping(nb::module_& m, char const* name)
{
    using KT = typename IdMapping::Key;

    nb::class_<IdMapping> cls(m, name);
    cls.doc() = doc_cpp_ref(R"pydoc(IdMapping)pydoc", "lasyten::IdMapping");

    cls.def(lasyten::ctor<decltype(cls)>())
      .def(lasyten::ctor<decltype(cls), std::vector<KT> const&>(), nb::arg("keys"))
      .def_rw("keys", &IdMapping::keys)
      .def("pre_compose",
           &IdMapping::pre_compose,
           nb::arg("other"),
           doc_cpp_ref(R"pydoc(pre_compose)pydoc", "lasyten::IdMapping::pre_compose()"))
      .def("nonzero_rows", &IdMapping::nonzero_rows)
      .def("nonzero_cols", &IdMapping::nonzero_cols)
      .def("prune",
           &IdMapping::prune,
           nb::arg("tol"),
           doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::IdMapping::prune()"));
}

} // namespace

void
bind_mappings(nb::module_& m)
{
    bind_sparse_mapping<SparseMappingFusionTree>(
      m,
      "SparseMappingFusionTree",
      doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::IdMapping::prune()"));

    bind_sparse_mapping<SparseMappingFusionTreePair>(
      m,
      "SparseMappingFusionTreePair",
      doc_cpp_ref(R"pydoc(prune)pydoc", "lasyten::IdMapping::prune()"));

    bind_identity_mapping<IdentityMappingFusionTree>(m, "IdentityMappingFusionTree");
    bind_identity_mapping<IdentityMappingFusionTreePair>(m, "IdentityMappingFusionTreePair");
}

} // namespace lasyten
