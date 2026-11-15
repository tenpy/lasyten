#pragma once

#include <lasyten/lasyten.h>
#include <string>
#include <vector>

namespace lasyten {

class NotImplemented : public std::logic_error
{
  public:
    NotImplemented(std::string name);
};

#if LASYTEN_HAS_PYTHON
/// Format elements of an iterable as if it were a plain list.
///
/// This means surrounding them with brackets and separating them by `', '`.
std::string format_like_list(py::iterable it);

/// If the given object is iterable.
bool is_iterable(py::object a);

/// If `a` is not iterable or a string, return [a], else return a.
py::object to_iterable(py::object a);
#endif // LASYTEN_HAS_PYTHON

/// Convert to a valid index into the given length, with python convention of negative indices from
/// back
int64 to_valid_idx(int64 idx, int64 length);

/// Decompose a permutation into a sequence of adjacent swaps.
///
/// Realizes `permutation` as swaps of neighboring positions: each returned index `j` means
/// ``swap(j, j + 1)``, i.e. the permutation
/// ``[*range(j), j + 1, j, *range(j + 2, n)]``.
/// Applying those swaps in order to ``range(n)`` yields `permutation`.
/// This is the shared helper behind `Coupling::permute` and `PermuteLegsInstructionEngine`.
///
/// @param permutation A permutation of ``range(n)``. ``permutation[k]`` is the original index
/// that ends up at position `k`.
/// @returns Swap positions `j` (each representing ``j <-> j + 1``).
/// Throws ``std::invalid_argument`` if `permutation` is not a permutation of ``range(n)``.
[[nodiscard]] std::vector<int64> permutation_as_swaps(std::vector<int64> const& permutation);

} // namespace lasyten

#include <lasyten/tools/cost_polynomials.h>
#include <lasyten/tools/mappings.h>
#include <lasyten/tools/misc.h>
#include <lasyten/tools/version.h>
