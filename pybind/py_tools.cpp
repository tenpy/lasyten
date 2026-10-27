#include <lasyten/tools.h>
#include <lasyten/tools/lru_cache.h>

#include "doc_plus.h"
#include "docstrings/tools.h"
#include "py_lasyten_pybind11.h"

namespace py = pybind11;
namespace lasyten {

void
bind_tools(py::module_& m)
{
    // Translate C++ NotImplemented into Python's builtin NotImplementedError.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (NotImplemented const& e) {
            PyErr_SetString(PyExc_NotImplementedError, e.what());
        }
    });

    m.def("format_like_list",
          &lasyten::format_like_list,
          doc_cpp_ref(R"pydoc(format_like_list)pydoc", "lasyten::format_like_list"),
          py::arg("it"));

    m.def("is_iterable", &lasyten::is_iterable, py::arg("a"), "If the given object is iterable.");

    m.def("to_iterable",
          &lasyten::to_iterable,
          py::arg("a"),
          "If `a` is a not iterable or a string, return [a], else return a.");

    m.def("to_valid_idx",
          &lasyten::to_valid_idx,
          py::arg("idx"),
          py::arg("length"),
          "Convert to a valid non-negative index into the given length.");

    auto stats_dict = [](CacheStats const& s) {
        py::dict d;
        d["hits"] = s.hits;
        d["misses"] = s.misses;
        d["evictions"] = s.evictions;
        d["rejected"] = s.rejected;
        d["entries"] = s.entries;
        d["bytes"] = s.bytes;
        d["capacity_bytes"] = s.capacity_bytes;
        return d;
    };
    m.def(
      "symbol_cache_info",
      [stats_dict] { return stats_dict(symbol_cache_stats()); },
      "Statistics (hits, misses, evictions, rejected, entries, bytes, capacity_bytes) of the "
      "RAM cache for SU(N) F symbols, R symbols and fusion outcomes.");
    m.def(
      "clear_symbol_cache", &clear_symbol_cache, "Remove all entries of the SU(N) symbol cache.");
    m.def(
      "tree_plan_cache_info",
      [stats_dict] { return stats_dict(tree_plan_cache_stats()); },
      "Statistics of the RAM cache for fusion-tree mappings ('tree plans').");
    m.def("clear_tree_plan_cache", &clear_tree_plan_cache, "Remove all tree-plan cache entries.");
    m.def(
      "tree_size_cache_info",
      [stats_dict] { return stats_dict(tree_size_cache_stats()); },
      "Statistics of the memo of the number of fusion trees (``fusion_trees.size``).");
    m.def("clear_tree_size_cache", &clear_tree_size_cache, "Remove all entries of the size memo.");

    m.def(
      "packed_symbol_info",
      [] {
          auto const s = packed_file_stats();
          py::dict d;
          d["files"] = s.files;
          d["index_bytes"] = s.index_bytes;
          d["data_bytes_in_ram"] = s.data_bytes_in_ram;
          return d;
      },
      "Memory held by open SU(N) symbol files in the packed layout (files, index_bytes, "
      "data_bytes_in_ram); not part of the symbol cache budget.");
    m.def("_set_packed_data_in_ram_max_bytes",
          &set_packed_data_in_ram_max_bytes,
          py::arg("max_bytes"),
          "Test hook: block data of packed files up to this size is held in RAM; returns the old "
          "value.");

    m.def("permutation_as_swaps",
          &lasyten::permutation_as_swaps,
          py::arg("permutation"),
          DOC(lasyten, permutation_as_swaps));
}

} // namespace lasyten
