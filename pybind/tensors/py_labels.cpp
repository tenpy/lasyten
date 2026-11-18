#include <lasyten/tensors/labels.h>

#include "../doc_plus.h"
#include "../py_lasyten_nanobind.h"
#include "../tools/hdf5_bind.h"
#include "../tools/pickle_bind.h"

#include "docstrings/tensors/labels.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

std::string
as_leg_label(py::handle obj)
{
    if (obj.is_none()) {
        return "";
    }
    return py::cast<std::string>(obj);
}

std::vector<int64>
py_indices(LegLabels const& self, py::object idcs)
{
    if (py::isinstance<py::str>(idcs)) {
        return { self.index(py::cast<std::string>(idcs)) };
    }
    if (py::isinstance<py::int_>(idcs)) {
        return { self.index(py::cast<int64>(idcs)) };
    }
    std::vector<std::variant<int64, std::string>> parsed;
    for (auto item : py::reinterpret_borrow<py::iterable>(to_iterable(idcs))) {
        if (py::isinstance<py::str>(item)) {
            parsed.emplace_back(py::cast<std::string>(item));
        } else {
            parsed.emplace_back(py::cast<int64>(item));
        }
    }
    return self.indices(parsed);
}

int64
py_index(LegLabels const& self, py::object idx)
{
    if (py::isinstance<py::str>(idx)) {
        return self.index(py::cast<std::string>(idx));
    }
    return self.index(py::cast<int64>(idx));
}

std::vector<std::string>
args_as_strings(py::args const& args)
{
    std::vector<std::string> out;
    out.reserve(args.size());
    for (auto a : args) {
        out.push_back(py::cast<std::string>(a));
    }
    return out;
}

std::optional<nb::class_<LegLabels>> g_labelled_legs;

} // namespace

void
declare_tensors_labels(nb::module_& m)
{
    g_labelled_legs = nb::class_<LegLabels>(m, "LegLabels");
    (*g_labelled_legs).doc() = DOC(lasyten, LegLabels);
}

void
bind_tensors_labels(nb::module_& m)
{
    m.attr("CONTRACT_SYMBOL") = CONTRACT_SYMBOL;
    m.attr("LEG_SELECT_SYMBOL") = LEG_SELECT_SYMBOL;
    m.attr("OPEN_LEG_SYMBOL") = OPEN_LEG_SYMBOL;
    {
        py::list forbidden;
        for (char const* c : FORBIDDEN_LEG_LABEL_CHARS) {
            forbidden.append(c);
        }
        m.attr("FORBIDDEN_LEG_LABEL_CHARS") = forbidden;
    }

    m.def(
      "is_valid_leg_label",
      [](py::object label) {
          if (label.is_none()) {
              return true;
          }
          if (!py::isinstance<py::str>(label)) {
              return false;
          }
          return is_valid_leg_label(std::string{ py::cast<std::string>(label) });
      },
      nb::arg("label"),
      DOC(lasyten, is_valid_leg_label));

    m.def("_combine_leg_labels",
          &_combine_leg_labels,
          nb::arg("labels"),
          nb::arg("offset") = 0,
          DOC(lasyten, _combine_leg_labels));

    m.def(
      "_split_leg_label",
      [](py::object label, py::object num) {
          std::optional<int64> n;
          if (!num.is_none()) {
              n = py::cast<int64>(num);
          }
          return _split_leg_label(as_leg_label(label), n);
      },
      nb::arg("label"),
      nb::arg("num").none() = nb::none(),
      DOC(lasyten, _split_leg_label));

    m.def(
      "_dual_leg_label",
      [](py::object label) { return _dual_leg_label(as_leg_label(label)); },
      nb::arg("label"),
      DOC(lasyten, _dual_leg_label));

    m.def(
      "_dual_label_list", &_dual_label_list, nb::arg("labels"), DOC(lasyten, _dual_label_list));

    m.def(
      "_get_matching_labels",
      [](LegLabelList const& labels1, LegLabelList const& labels2, int64 /*stacklevel*/) {
          return _get_matching_labels(labels1, labels2);
      },
      nb::arg("labels1"),
      nb::arg("labels2"),
      nb::arg("stacklevel") = 1,
      doc_plus(DOC(lasyten, _get_matching_labels),
               R"pydoc(
The ``stacklevel`` argument is Python-only (accepted for API compatibility; unused).
In Python, unset labels are empty strings (``""``).
)pydoc"));

    auto& labelled_legs = *g_labelled_legs;

    labelled_legs.def(lasyten::ctor<decltype(labelled_legs), LegLabelList>(), nb::arg("labels"))
      .def_prop_ro("num_legs", &LegLabels::num_legs)
      .def_prop_ro("is_fully_labelled", &LegLabels::is_fully_labelled)
      .def("as_list", &LegLabels::as_list, DOC(lasyten, LegLabels, as_list))
      .def("test_sanity", &LegLabels::test_sanity)
      .def(
        "set",
        [](LegLabels& self, LegLabelList labels) -> LegLabels& {
            return self.set(std::move(labels));
        },
        nb::arg("labels"),
        nb::rv_policy::reference,
        DOC(lasyten, LegLabels, set))
      .def(
        "update",
        [](LegLabels& self, std::map<std::string, std::string> const& mapping) -> LegLabels& {
            return self.update(mapping);
        },
        nb::arg("mapping"),
        nb::rv_policy::reference,
        DOC(lasyten, LegLabels, update))
      .def("index", &py_index, nb::arg("idx"), DOC(lasyten, LegLabels, index))
      .def("indices", &py_indices, nb::arg("idcs"), DOC(lasyten, LegLabels, indices))
      .def("has",
           [](LegLabels const& self, nb::args args_nb) {
               auto args = pyargs(args_nb);
               if (args.size() == 0) {
                   throw py::value_error("has() requires at least one label");
               }
               return self.has(args_as_strings(args));
           })
      .def(
        "are",
        [](LegLabels const& self,
           nb::args args_nb,
           bool planar,
           bool check_hidden,
           bool check_charged) {
            auto args = pyargs(args_nb);
            return self.are(args_as_strings(args), planar, check_hidden, check_charged);
        },
        nb::arg("labels"),
        nb::kw_only(),
        nb::arg("planar") = false,
        nb::arg("check_hidden") = false,
        nb::arg("check_charged") = false,
        DOC(lasyten, LegLabels, are))
      .def("__len__", &LegLabels::size)
      .def("__getitem__",
           [](LegLabels const& self, py::object key) -> py::object {
               if (py::isinstance<py::str>(key)) {
                   return py::cast(static_cast<std::string const&>(
                     self[self.index(py::cast<std::string>(key))]));
               }
               if (py::isinstance<py::int_>(key)) {
                   return py::cast(static_cast<std::string const&>(self[py::cast<int64>(key)]));
               }
               throw py::type_error("LegLabels indices must be int or str");
           })
      .def("__setitem__",
           [](LegLabels& self, py::object key, py::object value) {
               std::string lab = as_leg_label(value);
               if (py::isinstance<py::str>(key)) {
                   self[py::cast<std::string>(key)] = std::move(lab);
               } else if (py::isinstance<py::int_>(key)) {
                   self[py::cast<int64>(key)] = std::move(lab);
               } else {
                   throw py::type_error("LegLabels indices must be int or str");
               }
           })
      .def("__contains__",
           [](LegLabels const& self, py::object key) {
               if (!py::isinstance<py::str>(key)) {
                   return false;
               }
               return self.has(py::cast<std::string>(key));
           })
      .def(
        "__eq__",
        [](LegLabels const& self, py::object other) {
            if (py::isinstance<LegLabels>(other)) {
                return self == py::cast<LegLabels const&>(other);
            }
            try {
                return self == py::cast<LegLabelList>(other);
            } catch (py::cast_error const&) {
                return false;
            }
        },
        nb::is_operator())
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<LegLabels>(),
           nb::arg("hdf5_saver"),
           nb::arg("h5gr"),
           nb::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<LegLabels>(),
                  nb::arg("hdf5_loader"),
                  nb::arg("h5gr"),
                  nb::arg("subpath"));

    // Pickle: ``LegLabels(labels)`` -- the ``labels`` list HDF5 persists.
    lasyten::pickle::def_reduce_ctor(
      labelled_legs, [](LegLabels const& self) { return nb::make_tuple(self.as_list()); });
}

} // namespace lasyten
