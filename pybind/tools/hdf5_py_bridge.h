#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/tools/hdf5.h>

#include <highfive/highfive.hpp>
#include <string>
#include <type_traits>

namespace lasyten::hdf5 {

namespace py = lasyten::py;

/// RAII: bind Python Hdf5Saver/Loader for nested duck-typed save/load during migration.
/// Nestable: restores the previous TLS binding on destruction.
class PyBridge
{
  public:
    PyBridge(py::object saver, py::object loader, py::object h5gr);
    ~PyBridge();
    PyBridge(PyBridge const&) = delete;
    PyBridge& operator=(PyBridge const&) = delete;

  private:
    py::object prev_saver_obj_;
    py::object prev_loader_obj_;
    py::object prev_h5gr_obj_;
    bool had_saver_ = false;
    bool had_loader_ = false;
    bool had_h5gr_ = false;
};

void py_save_handle(std::string const& path, py::handle obj);

inline void
py_save(std::string const& path, py::handle obj)
{
    py_save_handle(path, obj);
}

template<typename T>
    requires(!std::is_base_of_v<py::handle, std::remove_cvref_t<T>>)
void
py_save(std::string const& path, T&& obj)
{
    py_save_handle(path, py::cast(std::forward<T>(obj)));
}

py::object py_load(std::string const& path);
void py_memorize_load(HighFive::Group& h5gr, py::handle obj);
py::object py_get_attr(HighFive::Group& h5gr, std::string const& name);
void py_set_group_attr(std::string const& name, py::handle value);
py::object py_get_group_attr(std::string const& name);

/// Access TLS Python objects set by ``PyBridge`` (nullptr if unbound).
py::object* tls_py_saver();
py::object* tls_py_loader();
py::object* tls_py_h5gr();

} // namespace lasyten::hdf5
