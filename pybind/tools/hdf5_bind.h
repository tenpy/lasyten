#pragma once

#include <lasyten/lasyten.h>

#include "py_hdf5_bridge.h"

#include "hdf5_py_bridge.h"

#include <string>

namespace lasyten::hdf5 {

namespace py = lasyten::py;

/// ``save_hdf5(self, py_saver, py_h5gr, subpath)`` → C++ Saver/Group + TLS PyBridge.
template<typename C>
auto
wrap_save_hdf5()
{
    return [](C& self, py::object py_saver, py::object py_h5gr, std::string subpath) {
        auto saver = saver_from_python(py_saver);
        auto group = wrap_group(py_h5gr);
        PyBridge bridge(py_saver, py::none(), py_h5gr);
        self.save_hdf5(saver, group, subpath);
    };
}

template<typename C>
auto
wrap_save_hdf5_const()
{
    return [](C const& self, py::object py_saver, py::object py_h5gr, std::string subpath) {
        auto saver = saver_from_python(py_saver);
        auto group = wrap_group(py_h5gr);
        PyBridge bridge(py_saver, py::none(), py_h5gr);
        self.save_hdf5(saver, group, subpath);
    };
}

template<typename C>
auto
wrap_from_hdf5()
{
    return [](py::object py_loader, py::object py_h5gr, std::string subpath) {
        auto loader = loader_from_python(py_loader);
        auto group = wrap_group(py_h5gr);
        PyBridge bridge(py::none(), py_loader, py_h5gr);
        auto obj = C::from_hdf5(loader, group, subpath);
        // C++ from_hdf5 may already have memorized; Python memo still needs the bound object.
        py_loader.attr("memorize_load")(py_h5gr, py::cast(obj));
        return obj;
    };
}

/// Classmethod ``from_hdf5(cls, loader, h5gr, subpath)`` (Site, TensorBackend).
template<typename C>
auto
wrap_from_hdf5_classmethod()
{
    return [](py::object cls, py::object py_loader, py::object py_h5gr, std::string subpath) {
        auto loader = loader_from_python(py_loader);
        auto group = wrap_group(py_h5gr);
        PyBridge bridge(py::none(), py_loader, py_h5gr);
        return C::from_hdf5(cls, loader, group, subpath);
    };
}

} // namespace lasyten::hdf5
