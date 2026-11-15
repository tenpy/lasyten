#include <lasyten/tools.h>

#include <algorithm>
#include <format>
#include <lasyten/detail/python_api.h>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lasyten {

NotImplemented::NotImplemented(std::string name)
  : std::logic_error(std::format("Not implemented: {}", name)) {};

#if LASYTEN_HAS_PYTHON
/// Format elements of an iterable as if it were a plain list.
std::string
format_like_list(py::iterable it)
{
    std::string result = "[";
    bool first = true;
    for (auto&& item : it) {
        if (!first)
            result += ", ";
        result += py::str(item).cast<std::string>();
        first = false;
    }
    result += "]";
    return result;
}

bool
is_iterable(py::object a)
{
    try {
        py::iter(a);
        return true;
    } catch (py::error_already_set& m) {
        // expected error: TypeError if not iterable
        if (!m.matches(PyExc_TypeError))
            throw;
    }
    return false;
}

py::object
to_iterable(py::object a)
{
    if (!py::isinstance<py::str>(a) && is_iterable(a))
        return a;
    py::list result(1);
    result[0] = a;
    return result;
}
#endif // LASYTEN_HAS_PYTHON

int64
to_valid_idx(int64 idx, int64 length)
{
    if (idx < -length || idx >= length)
        throw std::out_of_range("Index " + std::to_string(idx) + " out of bounds for length " +
                                std::to_string(length));
    if (idx < 0)
        idx += length;
    return idx;
}

std::vector<int64>
permutation_as_swaps(std::vector<int64> const& permutation)
{
    int64 const n = static_cast<int64>(permutation.size());
    std::vector<int64> sorted = permutation;
    std::sort(sorted.begin(), sorted.end());
    for (int64 i = 0; i < n; ++i) {
        if (sorted[static_cast<std::size_t>(i)] != i) {
            throw std::invalid_argument("permutation_as_swaps: not a permutation");
        }
    }

    std::vector<int64> working(static_cast<std::size_t>(n));
    std::iota(working.begin(), working.end(), int64(0));
    std::vector<int64> swap_positions;

    for (int64 target_pos = 0; target_pos < n; ++target_pos) {
        int64 const value = permutation[static_cast<std::size_t>(target_pos)];
        int64 cur = target_pos;
        for (int64 i = target_pos; i < n; ++i) {
            if (working[static_cast<std::size_t>(i)] == value) {
                cur = i;
                break;
            }
        }
        while (cur > target_pos) {
            swap_positions.push_back(cur - 1);
            std::swap(working[static_cast<std::size_t>(cur - 1)],
                      working[static_cast<std::size_t>(cur)]);
            --cur;
        }
    }
    return swap_positions;
}

} // namespace lasyten
