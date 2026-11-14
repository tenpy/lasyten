#include <lasyten/tensors/labels.h>

#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "../tools/hdf5_bind.h"

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
    return obj.cast<std::string>();
}

std::vector<int64>
py_indices(LegLabels const& self, py::object idcs)
{
    if (py::isinstance<py::str>(idcs)) {
        return { self.index(idcs.cast<std::string>()) };
    }
    if (py::isinstance<py::int_>(idcs)) {
        return { self.index(idcs.cast<int64>()) };
    }
    std::vector<std::variant<int64, std::string>> parsed;
    for (auto item : py::reinterpret_borrow<py::iterable>(to_iterable(idcs))) {
        if (py::isinstance<py::str>(item)) {
            parsed.emplace_back(item.cast<std::string>());
        } else {
            parsed.emplace_back(item.cast<int64>());
        }
    }
    return self.indices(parsed);
}

int64
py_index(LegLabels const& self, py::object idx)
{
    if (py::isinstance<py::str>(idx)) {
        return self.index(idx.cast<std::string>());
    }
    return self.index(idx.cast<int64>());
}

std::vector<std::string>
args_as_strings(py::args const& args)
{
    std::vector<std::string> out;
    out.reserve(args.size());
    for (auto a : args) {
        out.push_back(a.cast<std::string>());
    }
    return out;
}

std::optional<py::class_<LegLabels, py::smart_holder>> g_labelled_legs;

} // namespace

void
declare_tensors_labels(py::module_& m)
{
    g_labelled_legs = py::class_<LegLabels, py::smart_holder>(m, "LegLabels");
    (*g_labelled_legs).doc() = DOC(lasyten, LegLabels);
}

void
bind_tensors_labels(py::module_& m)
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
          return is_valid_leg_label(std::string{ label.cast<std::string>() });
      },
      py::arg("label"),
      DOC(lasyten, is_valid_leg_label));

    m.def("_combine_leg_labels",
          &_combine_leg_labels,
          py::arg("labels"),
          py::arg("offset") = 0,
          DOC(lasyten, _combine_leg_labels));

    m.def(
      "_split_leg_label",
      [](py::object label, py::object num) {
          std::optional<int64> n;
          if (!num.is_none()) {
              n = num.cast<int64>();
          }
          return _split_leg_label(as_leg_label(label), n);
      },
      py::arg("label"),
      py::arg("num") = py::none(),
      DOC(lasyten, _split_leg_label));

    m.def(
      "_dual_leg_label",
      [](py::object label) { return _dual_leg_label(as_leg_label(label)); },
      py::arg("label"),
      DOC(lasyten, _dual_leg_label));

    m.def(
      "_dual_label_list", &_dual_label_list, py::arg("labels"), DOC(lasyten, _dual_label_list));

    m.def(
      "_get_matching_labels",
      [](LegLabelList const& labels1, LegLabelList const& labels2, int64 /*stacklevel*/) {
          return _get_matching_labels(labels1, labels2);
      },
      py::arg("labels1"),
      py::arg("labels2"),
      py::arg("stacklevel") = 1,
      doc_plus(DOC(lasyten, _get_matching_labels),
               R"pydoc(
The ``stacklevel`` argument is Python-only (accepted for API compatibility; unused).
In Python, unset labels are empty strings (``""``).
)pydoc"));

    auto& labelled_legs = *g_labelled_legs;

    labelled_legs.def(py::init<LegLabelList>(), py::arg("labels"))
      .def_property_readonly("num_legs", &LegLabels::num_legs)
      .def_property_readonly("is_fully_labelled", &LegLabels::is_fully_labelled)
      .def("as_list", &LegLabels::as_list, DOC(lasyten, LegLabels, as_list))
      .def("test_sanity", &LegLabels::test_sanity)
      .def(
        "set",
        [](LegLabels& self, LegLabelList labels) -> LegLabels& {
            return self.set(std::move(labels));
        },
        py::arg("labels"),
        py::return_value_policy::reference,
        DOC(lasyten, LegLabels, set))
      .def(
        "update",
        [](LegLabels& self, std::map<std::string, std::string> const& mapping) -> LegLabels& {
            return self.update(mapping);
        },
        py::arg("mapping"),
        py::return_value_policy::reference,
        DOC(lasyten, LegLabels, update))
      .def("index", &py_index, py::arg("idx"), DOC(lasyten, LegLabels, index))
      .def("indices", &py_indices, py::arg("idcs"), DOC(lasyten, LegLabels, indices))
      .def("has",
           [](LegLabels const& self, py::args args) {
               if (args.size() == 0) {
                   throw py::value_error("has() requires at least one label");
               }
               return self.has(args_as_strings(args));
           })
      .def(
        "are",
        [](LegLabels const& self, py::args args, bool planar) {
            return self.are(args_as_strings(args), planar);
        },
        py::kw_only(),
        py::arg("planar") = false,
        DOC(lasyten, LegLabels, are))
      .def("__len__", &LegLabels::size)
      .def("__getitem__",
           [](LegLabels const& self, py::object key) -> py::object {
               if (py::isinstance<py::str>(key)) {
                   return py::cast(
                     static_cast<std::string const&>(self[self.index(key.cast<std::string>())]));
               }
               if (py::isinstance<py::int_>(key)) {
                   return py::cast(static_cast<std::string const&>(self[key.cast<int64>()]));
               }
               throw py::type_error("LegLabels indices must be int or str");
           })
      .def("__setitem__",
           [](LegLabels& self, py::object key, py::object value) {
               std::string lab = as_leg_label(value);
               if (py::isinstance<py::str>(key)) {
                   self[key.cast<std::string>()] = std::move(lab);
               } else if (py::isinstance<py::int_>(key)) {
                   self[key.cast<int64>()] = std::move(lab);
               } else {
                   throw py::type_error("LegLabels indices must be int or str");
               }
           })
      .def("__contains__",
           [](LegLabels const& self, py::object key) {
               if (!py::isinstance<py::str>(key)) {
                   return false;
               }
               return self.has(key.cast<std::string>());
           })
      .def(
        "__eq__",
        [](LegLabels const& self, py::object other) {
            if (py::isinstance<LegLabels>(other)) {
                return self == other.cast<LegLabels const&>();
            }
            try {
                return self == other.cast<LegLabelList>();
            } catch (py::cast_error const&) {
                return false;
            }
        },
        py::is_operator())
      .def("save_hdf5",
           lasyten::hdf5::wrap_save_hdf5_const<LegLabels>(),
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<LegLabels>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
