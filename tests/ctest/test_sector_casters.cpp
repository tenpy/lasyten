#include <cassert>
#include <iostream>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/sector_numpy.h>

namespace py = pybind11;
using namespace lasyten;

int
test_sector_casters(int /*argc*/, char** /*args*/)
{
    py::scoped_interpreter guard{};
    py::module_::import("numpy");
    // Import the nanobind-bound Sector / SectorArray classes and attach this program's copy of
    // nanobind to the extension's internals.
    py::module_::import("lasyten._core");
    lasyten::init_embedded_nanobind();

    Sector s{ 1, -2, 3 };
    py::object obj = py::cast(s);
    assert(lasyten::py_isinstance<Sector>(obj));
    auto arr = sector_to_numpy(s);
    assert(arr.ndim() == 1);
    assert(arr.shape(0) == 3);

    Sector s2 = obj.cast<Sector>();
    assert(s2 == s);

    // ndarray / sequence → Sector via sector_from_numpy (same path as load caster)
    Sector from_list = sector_from_numpy(py::list(py::make_tuple(7, 8)));
    assert((from_list == Sector{ 7, 8 }));

    SectorArray sa(2, 2);
    sa[0] = Sector{ 1, 2 };
    sa[1] = Sector{ 3, 4 };
    py::object sa_obj = py::cast(sa);
    assert(lasyten::py_isinstance<SectorArray>(sa_obj));
    auto sa_arr = sector_array_to_numpy(sa);
    assert(sa_arr.ndim() == 2);
    assert(sa_arr.shape(0) == 2);
    assert(sa_arr.shape(1) == 2);

    SectorArray sa2 = sa_obj.cast<SectorArray>();
    assert(sa2 == sa);

    SectorArray from_np = sector_array_from_numpy(sa_arr);
    assert(from_np == sa);

    std::cout << "test_sector_casters passed." << std::endl;
    return 0;
}
