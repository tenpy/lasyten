#include <lasyten/symmetries/spaces.h>

#include <lasyten/config.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/tools.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <format>
#include <functional>
#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace lasyten {

namespace {

[[nodiscard]] std::size_t
dim_as_size(float64 dim)
{
    if (!(dim >= 0.) || std::floor(dim) != dim) {
        throw std::invalid_argument(
          std::format("dimension must be a non-negative integer, got {}", dim));
    }
    return static_cast<std::size_t>(dim);
}

[[nodiscard]] std::vector<int64>
arange(std::size_t n)
{
    std::vector<int64> out(n);
    std::iota(out.begin(), out.end(), int64{ 0 });
    return out;
}

/// ``repr`` of a bool, using the Python spelling.
[[nodiscard]] char const*
bool_repr(bool value)
{
    return value ? "True" : "False";
}

[[nodiscard]] py::array_t<int64>
vector_to_array(std::vector<int64> const& v)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = v[i];
    }
    return arr;
}

} // namespace

Leg::Leg(Symmetry::Ptr symmetry_,
         float64 dim_,
         bool is_dual_,
         std::optional<std::vector<int64>> basis_perm)
{
    init_leg(std::move(symmetry_), dim_, is_dual_, std::move(basis_perm));
}

void
Leg::init_leg(Symmetry::Ptr symmetry_,
              float64 dim_,
              bool is_dual_,
              std::optional<std::vector<int64>> basis_perm)
{
    symmetry = std::move(symmetry_);
    dim = dim_;
    is_dual = is_dual_;
    if (!basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
    } else {
        if (!symmetry->can_be_dropped()) {
            throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
        }
        if (basis_perm->size() != dim_as_size(dim)) {
            throw std::invalid_argument(
              std::format("basis_perm length {} does not match dim {}", basis_perm->size(), dim));
        }
        _basis_perm = std::move(basis_perm);
        _inverse_basis_perm = inverse_permutation(*_basis_perm);
    }
}

void
Leg::test_sanity() const
{
    // --- hints from Python Leg.test_sanity ---
    // is a permutation
    // ---
    if (!symmetry->can_be_dropped()) {
        assert(!_basis_perm);
    }
    if (!_basis_perm) {
        assert(!_inverse_basis_perm);
    } else {
        assert(_inverse_basis_perm);
        auto const n = dim_as_size(dim);
        assert(_basis_perm->size() == n);
        assert(_inverse_basis_perm->size() == n);
        // is a permutation
        assert(std::unordered_set<int64>(_basis_perm->begin(), _basis_perm->end()).size() == n);
        assert(std::unordered_set<int64>(_inverse_basis_perm->begin(), _inverse_basis_perm->end())
                 .size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            assert((*_basis_perm)[static_cast<std::size_t>((*_inverse_basis_perm)[i])] ==
                   static_cast<int64>(i));
        }
    }
}

ElementarySpace::Ptr
Leg::as_ElementarySpace(bool is_dual_)
{
    // --- hints from Python Leg.as_ElementarySpace ---
    // can be overridden for performance
    // ---
    // can be overridden for performance
    return as_space_obj()->as_ElementarySpace(is_dual_);
}

std::vector<int64>
Leg::basis_perm() const
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
    }
    if (!_basis_perm) {
        return arange(dim_as_size(dim));
    }
    return *_basis_perm;
}

void
Leg::set_basis_perm(std::optional<std::vector<int64>> basis_perm)
{
    if (!basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
        return;
    }
    if (basis_perm->size() != dim_as_size(dim)) {
        throw std::invalid_argument(
          std::format("basis_perm length {} does not match dim {}", basis_perm->size(), dim));
    }
    _basis_perm = std::move(basis_perm);
    _inverse_basis_perm = inverse_permutation(*_basis_perm);
}

std::vector<int64>
Leg::inverse_basis_perm() const
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
    }
    if (!_inverse_basis_perm) {
        return arange(dim_as_size(dim));
    }
    return *_inverse_basis_perm;
}

void
Leg::set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm)
{
    if (!inverse_basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
        return;
    }
    if (inverse_basis_perm->size() != dim_as_size(dim)) {
        throw std::invalid_argument(std::format(
          "inverse_basis_perm length {} does not match dim {}", inverse_basis_perm->size(), dim));
    }
    _inverse_basis_perm = std::move(inverse_basis_perm);
    _basis_perm = inverse_permutation(*_inverse_basis_perm);
}

Leg::Ptr
Leg::shared_leg()
{
    return std::dynamic_pointer_cast<Leg>(shared_from_this());
}

std::vector<Leg::Ptr>
Leg::flat_legs()
{
    return { shared_leg() };
}

std::vector<Leg::Ptr>
Leg::flat_spaces()
{
    return { shared_leg() };
}

int64
Leg::num_flat_legs() const
{
    return 1;
}

std::vector<int64>
Leg::_flat_leg_permutation(int64 offset) const
{
    return { offset };
}

std::string
Leg::ascii_arrow() const
{
    // --- hints from Python Leg.ascii_arrow ---
    // should have already covered all cases
    // ---
    // Subclasses (ElementarySpace / LegPipe) override. Pure Leg should not appear in diagrams.
    throw std::runtime_error("ascii_arrow not implemented for this Leg subclass");
}

py::object
Leg::apply_basis_perm(py::object arr, int64 axis, bool inverse, bool pre_compose) const
{
    // --- hints from Python Leg.apply_basis_perm ---
    // this implementation assumes _basis_perm. AbelianLegPipe overrides this method.
    // perm is identity permutation
    // ---
    auto const& perm = inverse ? _inverse_basis_perm : _basis_perm;
    if (!perm) {
        return arr;
    }
    auto perm_arr = vector_to_array(*perm);
    if (pre_compose) {
        if (axis != 0) {
            throw std::invalid_argument("pre_compose currently requires axis == 0");
        }
        return perm_arr[py::make_tuple(arr)];
    }
    // Native take along `axis` (replaces numpy.take).
    py::array data = py::array::ensure(arr);
    if (!data) {
        throw std::invalid_argument("apply_basis_perm: expected array-like input");
    }
    auto info = data.request();
    if (axis < 0) {
        axis += static_cast<int64>(info.ndim);
    }
    if (axis < 0 || axis >= info.ndim) {
        throw std::invalid_argument("apply_basis_perm: axis out of range");
    }
    // Use advanced indexing: arrange slices with perm on the chosen axis.
    py::list idx;
    for (py::ssize_t i = 0; i < info.ndim; ++i) {
        if (i == axis) {
            idx.append(perm_arr);
        } else {
            idx.append(py::slice(py::none(), py::none(), py::none()));
        }
    }
    return data[py::tuple(idx)];
}

namespace {

[[nodiscard]] std::vector<int64>
py_array_to_i64(py::array arr)
{
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

[[nodiscard]] std::vector<float64>
py_array_to_f64(py::array arr)
{
    auto casted = py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<float64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

[[nodiscard]] SectorArray
take_or_all(SectorArray const& arr, std::optional<std::vector<std::size_t>> const& perm)
{
    if (!perm) {
        return arr;
    }
    return arr.take(*perm);
}

[[nodiscard]] std::vector<int64>
gather_or_all(std::vector<int64> const& vals, std::optional<std::vector<std::size_t>> const& perm)
{
    if (!perm) {
        return vals;
    }
    std::vector<int64> out(perm->size());
    for (std::size_t i = 0; i < perm->size(); ++i) {
        out[i] = vals[(*perm)[i]];
    }
    return out;
}

[[nodiscard]] bool
is_identity_lexsort(std::vector<std::size_t> const& indices)
{
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            return false;
        }
    }
    return true;
}

} // namespace

Space::Space(Symmetry::Ptr symmetry_,
             SectorArray sector_decomposition_,
             std::optional<std::vector<int64>> multiplicities_,
             std::optional<std::string> sector_order_)
  : symmetry(std::move(symmetry_))
  , sector_decomposition(std::move(sector_decomposition_))
  , sector_order(std::move(sector_order_))
{
    // --- hints from Python Space.__init__ ---
    // slices[0, 0] remains 0, which is correct
    // ---
    if (sector_decomposition.sector_ind_len() != symmetry->sector_ind_len) {
        throw std::invalid_argument(
          std::format("Wrong sectors.shape: Expected (*, {}), got ({}, {}).",
                      symmetry->sector_ind_len,
                      sector_decomposition.size(),
                      sector_decomposition.sector_ind_len()));
    }
    num_sectors = static_cast<int64>(sector_decomposition.size());
    auto const n = static_cast<std::size_t>(num_sectors);
    if (!multiplicities_) {
        multiplicities.assign(n, 1);
    } else {
        multiplicities = std::move(*multiplicities_);
        if (multiplicities.size() != n) {
            throw std::invalid_argument(
              std::format("multiplicities length {} does not match number of sectors {}",
                          multiplicities.size(),
                          n));
        }
    }
    if (symmetry->can_be_dropped()) {
        sector_dims = symmetry->batch_sector_dim(sector_decomposition);
        sector_qdims.assign(sector_dims->begin(), sector_dims->end());
        std::vector<std::array<int64, 2>> sl(n);
        int64 running = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sl[i][0] = running;
            running += multiplicities[i] * (*sector_dims)[i];
            sl[i][1] = running;
        }
        slices = std::move(sl);
        dim = static_cast<float64>(running);
    } else {
        sector_dims = std::nullopt;
        sector_qdims = symmetry->batch_qdim(sector_decomposition);
        slices = std::nullopt;
        float64 total = 0.;
        for (std::size_t i = 0; i < n; ++i) {
            total += sector_qdims[i] * static_cast<float64>(multiplicities[i]);
        }
        dim = total;
    }
}

void
Space::test_sanity() const
{
    // --- hints from Python Space.test_sanity ---
    // sectors
    // nothing to check
    // multiplicities
    // slices
    // slices should be consecutive
    // ---
    assert(dim >= 0.);
    // sectors
    if (static_cast<int64>(sector_decomposition.size()) != num_sectors ||
        sector_decomposition.sector_ind_len() != symmetry->sector_ind_len) {
        throw std::runtime_error("wrong sectors.shape");
    }
    assert(symmetry->are_valid_sectors(sector_decomposition));
    {
        std::vector<std::int64_t> ones(static_cast<std::size_t>(num_sectors), 1);
        auto const [unique, um, perm] = sector_decomposition.unique_sorted(ones);
        assert(static_cast<int64>(unique.size()) == num_sectors);
        (void)um;
        (void)perm;
    }
    if (sector_order == "sorted") {
        assert(is_identity_lexsort(sector_decomposition.lexsort_indices()));
    } else if (sector_order == "dual_sorted") {
        auto expect_sorted = symmetry->dual_sectors(sector_decomposition);
        assert(is_identity_lexsort(expect_sorted.lexsort_indices()));
    } else if (!sector_order) {
        // nothing to check
    } else {
        throw std::runtime_error(std::format("Invalid sector_order: {}", *sector_order));
    }
    // multiplicities
    assert(multiplicities.size() == static_cast<std::size_t>(num_sectors));
    assert(std::ranges::all_of(multiplicities, [](int64 m) { return m > 0; }));
    if (symmetry->can_be_dropped()) {
        assert(slices);
        assert(sector_dims);
        assert(slices->size() == static_cast<std::size_t>(num_sectors));
        auto expect_dims = symmetry->batch_sector_dim(sector_decomposition);
        assert(*sector_dims == expect_dims);
        for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
            assert((*slices)[i][1] - (*slices)[i][0] == (*sector_dims)[i] * multiplicities[i]);
        }
        // slices should be consecutive
        if (num_sectors > 0) {
            assert((*slices)[0][0] == 0);
            for (std::size_t i = 1; i < static_cast<std::size_t>(num_sectors); ++i) {
                assert((*slices)[i][0] == (*slices)[i - 1][1]);
            }
            assert((*slices)[static_cast<std::size_t>(num_sectors) - 1][1] ==
                   static_cast<int64>(dim));
        }
    }
}

bool
Space::is_trivial() const
{
    if (num_sectors > 1) {
        return false;
    }
    if (multiplicities[0] > 1) {
        return false;
    }
    return sector_decomposition[0] == symmetry->trivial_sector;
}

bool
Space::operator==(Space const& /*other*/) const
{
    throw py::type_error(
      "Space does not support \"==\" comparison. Use `is_isomorphic_to` instead.");
}

bool
Space::is_isomorphic_to(Space const& other) const
{
    // --- hints from Python Space.is_isomorphic_to ---
    // have the same sorting convention and can be directly compared
    // case should have been covered above
    // all cases should have been covered.
    // ---
    if (!symmetry->equals(*other.symmetry)) {
        throw SymmetryError("Incompatible symmetries");
    }
    if (num_sectors != other.num_sectors) {
        return false;
    }

    // find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and
    // ``other.sector_decomposition[perm2]`` have the same sorting convention
    std::optional<std::vector<std::size_t>> perm1;
    std::optional<std::vector<std::size_t>> perm2;
    if (!sector_order) {
        if (other.sector_order == "sorted") {
            perm1 = sector_decomposition.lexsort_indices();
            perm2 = std::nullopt;
        } else if (other.sector_order == "dual_sorted") {
            perm1 = symmetry->dual_sectors(sector_decomposition).lexsort_indices();
            perm2 = std::nullopt;
        } else {
            perm1 = sector_decomposition.lexsort_indices();
            perm2 = other.sector_decomposition.lexsort_indices();
        }
    } else if (!other.sector_order) {
        if (sector_order == "sorted") {
            perm1 = std::nullopt;
            perm2 = other.sector_decomposition.lexsort_indices();
        } else if (sector_order == "dual_sorted") {
            perm1 = std::nullopt;
            perm2 = symmetry->dual_sectors(other.sector_decomposition).lexsort_indices();
        } else {
            throw std::runtime_error("unreachable sector_order case");
        }
    } else if (sector_order == other.sector_order) {
        perm1 = std::nullopt;
        perm2 = std::nullopt;
    } else if (sector_order == "sorted") {
        perm1 = std::nullopt;
        perm2 = other.sector_decomposition.lexsort_indices();
    } else if (other.sector_order == "sorted") {
        perm1 = sector_decomposition.lexsort_indices();
        perm2 = std::nullopt;
    } else {
        throw std::runtime_error("unreachable sector_order case");
    }

    if (gather_or_all(multiplicities, perm1) != gather_or_all(other.multiplicities, perm2)) {
        return false;
    }
    return take_or_all(sector_decomposition, perm1) ==
           take_or_all(other.sector_decomposition, perm2);
}

bool
Space::is_subspace_of(Space const& other) const
{
    // --- hints from Python Space.is_subspace_of ---
    // sectors are sorted, so we can just iterate over both of them
    // have checked all sectors of self
    // reaching this line means self has sectors which other does not have
    // OPTIMIZE sort once instead of looking up each time
    // this means self has some sectors that other doesn't have
    // ---
    if (!symmetry->is_equivalent_to(*other.symmetry)) {
        return false;
    }
    if (num_sectors == 0) {
        return true;
    }
    if (sector_order == "sorted" && other.sector_order == "sorted") {
        // sectors are sorted, so we can just iterate over both of them
        std::size_t n_self = 0;
        for (std::size_t i = 0; i < other.sector_decomposition.size(); ++i) {
            if (sector_decomposition[n_self] == other.sector_decomposition[i]) {
                if (multiplicities[n_self] > other.multiplicities[i]) {
                    return false;
                }
                ++n_self;
            }
            if (static_cast<int64>(n_self) == num_sectors) {
                // have checked all sectors of self
                return true;
            }
        }
        // reaching this line means self has sectors which other does not have
        return false;
    }

    // OPTIMIZE sort once instead of looking up each time
    int64 num_sectors_checked = 0;
    for (std::size_t i = 0; i < other.sector_decomposition.size(); ++i) {
        auto const m = sector_multiplicity(other.sector_decomposition[i]);
        if (m == 0) {
            continue;
        }
        if (m > other.multiplicities[i]) {
            return false;
        }
        ++num_sectors_checked;
    }
    if (num_sectors_checked < num_sectors) {
        // this means self has some sectors that other doesn't have
        return false;
    }
    return true;
}

ElementarySpace::Ptr
Space::as_ElementarySpace(bool is_dual_)
{
    SectorArray defining_sectors;
    bool is_sorted = false;
    if (is_dual_) {
        defining_sectors = symmetry->dual_sectors(sector_decomposition);
        is_sorted = sector_order == "dual_sorted";
    } else {
        defining_sectors = sector_decomposition;
        is_sorted = sector_order == "sorted";
    }

    ElementarySpace::Ptr es;
    if (is_sorted) {
        es = std::make_shared<ElementarySpace>(
          symmetry, defining_sectors, multiplicities, is_dual_, std::nullopt);
    } else {
        es = ElementarySpace::from_defining_sectors(
          symmetry, defining_sectors, multiplicities, is_dual_, std::nullopt, true);
    }
    return es;
}

Space::Ptr
Space::shared_space()
{
    // --- hints from Python ElementarySpace.from_defining_sectors ---
    // sort sectors
    // combine duplicate sectors (does not affect basis_perm)
    // the convention is that for sectors with dim > 1, all copies of the first
    // state appear, then all copies of the second state, etc. At this point,
    // this order is not yet fully respected
    // updated basis_slices after sorting defining_sectors
    // take the basis_perm associated with the first states and make them contiguous,
    // then go to the second state, etc.
    // [:-1] to exclude len
    // ---
    return std::dynamic_pointer_cast<Space>(shared_from_this());
}

Space::Ptr
Space::as_Space()
{
    return shared_space();
}

std::optional<int64>
Space::sector_decomposition_where(Sector sector) const
{
    // --- hints from Python Space.sector_decomposition_where ---
    // sector_decomposition should be unique, so one of the above if statements should trigger.
    // If we get here, something is wrong / inconsistent.
    // this should raise an informative error
    // ---
    // OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
    auto idx = sector_decomposition.row_where(sector);
    if (!idx) {
        return std::nullopt;
    }
    return static_cast<int64>(*idx);
}

int64
Space::sector_multiplicity(Sector sector) const
{
    auto idx = sector_decomposition_where(sector);
    if (!idx) {
        return 0;
    }
    return multiplicities[static_cast<std::size_t>(*idx)];
}

int64
Space::total_multiplicity() const
{
    return std::accumulate(multiplicities.begin(), multiplicities.end(), int64{ 0 });
}

namespace {

[[nodiscard]] std::optional<std::vector<int64>>
combined_basis_perm(std::vector<Leg::Ptr> const& legs, bool combine_cstyle)
{
    bool any_custom = false;
    for (auto const& leg : legs) {
        if (leg->has_custom_basis_perm()) {
            any_custom = true;
            break;
        }
    }
    if (!any_custom) {
        return std::nullopt;
    }
    std::vector<std::vector<int64>> perms;
    perms.reserve(legs.size());
    for (auto const& leg : legs) {
        perms.push_back(leg->basis_perm());
    }
    return combine_permutations(perms, combine_cstyle);
}

} // namespace

// note: Leg is a virtual base and can therefore not be initialized here, see Leg::init_leg.
LegPipe::LegPipe(std::vector<Leg::Ptr> legs_, bool is_dual_, bool combine_cstyle_)
  : legs(std::move(legs_))
  , num_legs(static_cast<int64>(legs.size()))
  , combine_cstyle(combine_cstyle_)
{
    if (num_legs <= 0) {
        throw std::invalid_argument("LegPipe requires at least one leg");
    }
    auto const& symmetry0 = legs.at(0)->symmetry;
    float64 dim_prod = 1.;
    for (auto const& leg : legs) {
        if (!leg->symmetry->equals(*symmetry0)) {
            throw std::invalid_argument("all legs of a LegPipe must have the same symmetry");
        }
        dim_prod *= leg->dim;
    }
    init_leg(symmetry0, dim_prod, is_dual_, combined_basis_perm(legs, combine_cstyle));
}

void
LegPipe::test_sanity() const
{
    for (auto const& leg : legs) {
        if (!leg->symmetry->equals(*symmetry)) {
            throw std::invalid_argument("all legs of a LegPipe must have the same symmetry");
        }
        leg->test_sanity();
    }
    Leg::test_sanity();
}

Space::Ptr
LegPipe::as_space_obj()
{
    // Factors stay as Legs (ElementarySpace / nested LegPipe). Nested pipes must not be
    // converted to TensorProduct here — TensorProduct factors are Leg::Ptr only.
    return std::make_shared<TensorProduct>(legs, symmetry);
}

Leg::Ptr
LegPipe::dual_leg() const
{
    std::vector<Leg::Ptr> dual_legs;
    dual_legs.reserve(legs.size());
    for (auto it = legs.rbegin(); it != legs.rend(); ++it) {
        dual_legs.push_back((*it)->dual_leg());
    }
    return std::make_shared<LegPipe>(std::move(dual_legs), !is_dual, !combine_cstyle);
}

bool
LegPipe::is_trivial() const
{
    return std::ranges::all_of(legs, [](Leg::Ptr const& leg) { return leg->is_trivial(); });
}

std::vector<Leg::Ptr>
LegPipe::flat_legs()
{
    std::vector<Leg::Ptr> out;
    for (auto const& leg : legs) {
        auto part = leg->flat_legs();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<Leg::Ptr>
LegPipe::flat_spaces()
{
    std::vector<Leg::Ptr> out;
    for (auto const& leg : legs) {
        auto part = leg->flat_spaces();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

int64
LegPipe::num_flat_legs() const
{
    int64 n = 0;
    for (auto const& leg : legs) {
        n += leg->num_flat_legs();
    }
    return n;
}

std::vector<int64>
LegPipe::_flat_leg_permutation(int64 offset) const
{
    if (num_legs == num_flat_legs()) {
        std::vector<int64> perm(static_cast<std::size_t>(num_legs));
        std::iota(perm.begin(), perm.end(), offset);
        if (!combine_cstyle) {
            std::reverse(perm.begin(), perm.end());
        }
        return perm;
    }
    std::vector<Leg::Ptr> ordered = legs;
    if (!combine_cstyle) {
        std::reverse(ordered.begin(), ordered.end());
    }
    std::vector<int64> offsets;
    offsets.reserve(ordered.size());
    int64 running = offset;
    for (auto const& leg : ordered) {
        offsets.push_back(running);
        running += leg->num_flat_legs();
    }
    if (!combine_cstyle) {
        std::reverse(offsets.begin(), offsets.end());
    }
    std::vector<int64> perm;
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto part = legs[i]->_flat_leg_permutation(offsets[i]);
        perm.insert(perm.end(), part.begin(), part.end());
    }
    return perm;
}

void
LegPipe::set_basis_perm(std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error(std::format("Can not set basis_perm for {}.", "LegPipe"));
}

void
LegPipe::set_inverse_basis_perm(std::optional<std::vector<int64>> /*inverse_basis_perm*/)
{
    throw py::type_error(std::format("Can not set basis_perm for {}.", "LegPipe"));
}

std::string
LegPipe::ascii_arrow() const
{
    return "║";
}

bool
LegPipe::operator==(Leg const& other) const
{
    auto const* o = dynamic_cast<LegPipe const*>(&other);
    if (o == nullptr) {
        return false;
    }
    if (is_abelian_leg_pipe() != o->is_abelian_leg_pipe()) {
        return false;
    }
    if (is_dual != o->is_dual) {
        return false;
    }
    if (combine_cstyle != o->combine_cstyle) {
        return false;
    }
    if (num_legs != o->num_legs) {
        return false;
    }
    for (std::size_t i = 0; i < legs.size(); ++i) {
        if (!(*legs[i] == *o->legs[i])) {
            return false;
        }
    }
    return true;
}

Leg::Ptr
LegPipe::operator[](int64 idx) const
{
    auto const n = static_cast<int64>(legs.size());
    auto const i = to_valid_idx(idx, n);
    return legs[static_cast<std::size_t>(i)];
}

std::string
LegPipe::repr(bool show_symmetry, bool one_line) const
{
    // --- hints from Python LegPipe.__repr__ ---
    // the above should always fit in linewidth ...
    // ---
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "LegPipe";

    if (one_line) {
        if (show_symmetry) {
            auto res = std::format("{}(num_legs={}, is_dual={}, symmetry={}, combine_cstyle={})",
                                   ClsName,
                                   num_legs,
                                   bool_repr(is_dual),
                                   symmetry->repr(),
                                   bool_repr(combine_cstyle));
            if (static_cast<int64>(res.size()) <= linewidth) {
                return res;
            }
            return repr(false, true);
        }
        auto res = std::format("{}(num_legs={}, is_dual={}, combine_cstyle={})",
                               ClsName,
                               num_legs,
                               bool_repr(is_dual),
                               bool_repr(combine_cstyle));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }
        throw std::runtime_error("LegPipe one-line repr exceeds linewidth");
    }

    for (bool force_children_one_line : { false, true }) {
        std::vector<std::string> lines;
        lines.push_back(std::format("{}([", ClsName));
        for (auto const& leg : legs) {
            py::object leg_obj = py::cast(leg);
            std::string rep;
            try {
                rep =
                  py::str(leg_obj.attr("__repr__")(py::arg("show_symmetry") = false,
                                                   py::arg("one_line") = force_children_one_line));
            } catch (py::error_already_set&) {
                rep = py::str(leg_obj.attr("__repr__")());
            }
            std::istringstream iss(rep);
            std::string line;
            while (std::getline(iss, line)) {
                lines.push_back(indent + line);
            }
        }
        if (show_symmetry) {
            lines.push_back(
              std::format("], is_dual={}, symmetry={})", bool_repr(is_dual), symmetry->repr()));
        } else {
            lines.push_back(std::format("], is_dual={})", bool_repr(is_dual)));
        }
        bool maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
        bool linewidth_ok = std::ranges::all_of(
          lines, [&](std::string const& l) { return static_cast<int64>(l.size()) < linewidth; });
        if (maxlines_ok && linewidth_ok) {
            std::string out = lines[0];
            for (std::size_t i = 1; i < lines.size(); ++i) {
                out += '\n';
                out += lines[i];
            }
            return out;
        }
    }
    return repr(show_symmetry, true);
}

namespace {

/// The Python ``no_symmetry``, i.e. the product symmetry with a single ``NoSymmetry`` factor.
[[nodiscard]] Symmetry::Ptr
no_symmetry_product()
{
    return std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() });
}

/// ``_sort_sectors``: lexsort the `sectors`, applying the same permutation to `multiplicities`.
[[nodiscard]] std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
sort_sectors(SectorArray const& sectors, std::vector<int64> const& multiplicities)
{
    auto [sorted, perm] = sectors.sorted();
    std::vector<int64> mults(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        mults[i] = multiplicities[perm[i]];
    }
    return { std::move(sorted), std::move(mults), std::move(perm) };
}

/// ``np.concatenate([[0], np.cumsum(values)])``, i.e. the ``values.size() + 1`` slice boundaries.
[[nodiscard]] std::vector<int64>
slice_boundaries(std::vector<int64> const& values)
{
    std::vector<int64> out(values.size() + 1, 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i + 1] = out[i] + values[i];
    }
    return out;
}

/// ``symmetry.batch_sector_dim(sectors) * multiplicities``, the number of states per sector.
[[nodiscard]] std::vector<int64>
num_states_per_sector(Symmetry const& symmetry,
                      SectorArray const& sectors,
                      std::vector<int64> const& multiplicities)
{
    auto num_states = symmetry.batch_sector_dim(sectors);
    for (std::size_t i = 0; i < num_states.size(); ++i) {
        num_states[i] *= multiplicities[i];
    }
    return num_states;
}

/// ``_parse_inputs_drop_symmetry``. ``nullopt`` means ``'all'``, both on input and output.
[[nodiscard]] std::pair<std::optional<std::vector<int64>>, Symmetry::Ptr>
parse_inputs_drop_symmetry(std::optional<std::vector<int64>> const& which,
                           Symmetry const& symmetry)
{
    if (!which) {
        return { std::nullopt, no_symmetry_product() };
    }
    auto const num_factors = static_cast<int64>(symmetry.num_factors());
    std::vector<int64> valid;
    valid.reserve(which->size());
    for (auto i : *which) {
        valid.push_back(to_valid_idx(i, num_factors));
    }
    if (static_cast<int64>(valid.size()) == num_factors) {
        return { std::nullopt, no_symmetry_product() };
    }
    std::vector<SymmetryFactor::Ptr> remaining;
    for (int64 i = 0; i < num_factors; ++i) {
        if (std::ranges::find(valid, i) == valid.end()) {
            remaining.push_back(symmetry.factors[static_cast<std::size_t>(i)]);
        }
    }
    return { std::move(valid), std::make_shared<Symmetry>(std::move(remaining)) };
}

[[nodiscard]] py::object
slices_to_py(std::optional<std::vector<std::array<int64, 2>>> const& slices)
{
    if (!slices) {
        return py::none();
    }
    py::array_t<int64> arr({ static_cast<py::ssize_t>(slices->size()), py::ssize_t{ 2 } });
    auto buf = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < slices->size(); ++i) {
        buf(static_cast<py::ssize_t>(i), 0) = (*slices)[i][0];
        buf(static_cast<py::ssize_t>(i), 1) = (*slices)[i][1];
    }
    return arr;
}

[[nodiscard]] py::object
optional_perm_to_py(std::optional<std::vector<int64>> const& perm)
{
    if (!perm) {
        return py::none();
    }
    return vector_to_array(*perm);
}

[[nodiscard]] std::optional<std::vector<int64>>
optional_perm_from_py(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_array_to_i64(py::array::ensure(obj));
}

} // namespace

// note: Leg is a virtual base and can therefore not be initialized here, see Leg::init_leg.
// This is also convenient, since the dim is only computed by the Space constructor.
ElementarySpace::ElementarySpace(Symmetry::Ptr symmetry_,
                                 SectorArray defining_sectors_,
                                 std::optional<std::vector<int64>> multiplicities_,
                                 bool is_dual_,
                                 std::optional<std::vector<int64>> basis_perm_)
  : Space(symmetry_,
          is_dual_ ? symmetry_->dual_sectors(defining_sectors_) : defining_sectors_,
          std::move(multiplicities_),
          is_dual_ ? std::optional<std::string>{ "dual_sorted" }
                   : std::optional<std::string>{ "sorted" })
  , defining_sectors(std::move(defining_sectors_))
{
    if (!symmetry_->are_valid_sectors(defining_sectors)) {
        throw std::invalid_argument("defining_sectors contains invalid sectors for this symmetry");
    }
    init_leg(Space::symmetry, Space::dim, is_dual_, std::move(basis_perm_));
}

void
ElementarySpace::test_sanity() const
{
    assert(static_cast<int64>(defining_sectors.size()) == num_sectors);
    assert(defining_sectors.sector_ind_len() == Space::symmetry->sector_ind_len);
    if (is_dual) {
        assert(sector_order == "dual_sorted");
    } else {
        assert(sector_order == "sorted");
    }
    Space::test_sanity();
    Leg::test_sanity();
}

ElementarySpace::Ptr
ElementarySpace::from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis)
{
    // --- hints from Python ElementarySpace.from_basis ---
    // note: numpy.lexsort is stable, i.e. it preserves the order of equal keys.
    // how often each appears in the input sectors_of_basis
    // ---
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("from_basis is meaningless for {}.", symmetry->str()));
    }
    // note: the lexsort is stable, i.e. it preserves the order of equal keys.
    auto const basis_perm = sectors_of_basis.lexsort_indices();
    auto const sorted = sectors_of_basis.take(basis_perm);
    auto const diffs = sorted.find_row_differences(/*include_len=*/true);
    // [:-1] to exclude len
    auto sectors = sorted.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));
    auto const dims = symmetry->batch_sector_dim(sectors);
    std::vector<int64> multiplicities(sectors.size());
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        // how often the sector appears in the input sectors_of_basis
        auto const num_occurrences = static_cast<int64>(diffs[i + 1] - diffs[i]);
        if (num_occurrences % dims[i] != 0) {
            throw std::invalid_argument(
              "Sectors must appear in whole multiplets, i.e. a number of times that is an "
              "integer multiple of their dimension.");
        }
        multiplicities[i] = num_occurrences / dims[i];
    }
    return std::make_shared<ElementarySpace>(
      std::move(symmetry),
      std::move(sectors),
      std::move(multiplicities),
      false,
      std::vector<int64>(basis_perm.begin(), basis_perm.end()));
}

ElementarySpace::Ptr
ElementarySpace::from_independent_symmetries(std::vector<Ptr> const& independent_descriptions)
{
    // --- hints from Python ElementarySpace.from_independent_symmetries ---
    // OPTIMIZE this can be implemented better. if many consecutive basis elements have the same
    // resulting sector, we can skip over all of them.
    // ignore those with no_symmetry
    // all descriptions had no_symmetry
    // TODO is there a way to define this? the straight-forward picture works only if we have
    // a vector space and can identify states.
    // note: this interface is more general than it needs to be. The use case in
    // GroupedSite would allow us to specialize, if that is easier. A given state
    // is in the trivial sector for all but one of the independent_descriptions.
    // ---
    // OPTIMIZE this can be implemented better. if many consecutive basis elements have the same
    //          resulting sector, we can skip over all of them.
    if (independent_descriptions.empty()) {
        throw std::invalid_argument(
          "from_independent_symmetries requires at least one description");
    }
    auto const dim = independent_descriptions[0]->Space::dim;
    if (!std::ranges::all_of(independent_descriptions,
                             [dim](Ptr const& s) { return s->Space::dim == dim; })) {
        throw std::invalid_argument("all independent descriptions must have the same dimension");
    }
    // ignore those with no_symmetry
    auto const no_sym = no_symmetry_product();
    std::vector<Ptr> descriptions;
    for (auto const& s : independent_descriptions) {
        if (!s->Space::symmetry->equals(*no_sym)) {
            descriptions.push_back(s);
        }
    }
    if (descriptions.empty()) {
        // all descriptions had no_symmetry
        return from_trivial_sector(static_cast<int64>(dim));
    }
    std::vector<SymmetryFactor::Ptr> factors;
    for (auto const& s : descriptions) {
        auto const& own = s->Space::symmetry->factors;
        factors.insert(factors.end(), own.begin(), own.end());
    }
    auto symmetry = std::make_shared<Symmetry>(std::move(factors));
    if (!symmetry->can_be_dropped()) {
        // TODO is there a way to define this? the straight-forward picture works only if we have
        //      a vector space and can identify states.
        //      note: this interface is more general than it needs to be. The use case in
        //            GroupedSite would allow us to specialize, if that is easier. A given state
        //            is in the trivial sector for all but one of the independent_descriptions.
        throw SymmetryError(
          std::format("from_independent_symmetries is not supported for {}.", symmetry->str()));
    }
    // concatenate the sectors_of_basis of all descriptions along the sector axis
    std::vector<SectorArray> parts;
    parts.reserve(descriptions.size());
    for (auto const& s : descriptions) {
        parts.push_back(s->sectors_of_basis());
    }
    auto const num_basis_states = dim_as_size(dim);
    SectorArray sectors_of_basis(num_basis_states, symmetry->sector_ind_len);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        auto sector = Sector::zeros(symmetry->sector_ind_len);
        std::size_t offset = 0;
        for (auto const& part : parts) {
            auto const& row = part[i];
            for (std::uint8_t k = 0; k < row.len(); ++k) {
                sector[offset++] = row[k];
            }
        }
        sectors_of_basis[i] = sector;
    }
    return from_basis(std::move(symmetry), std::move(sectors_of_basis));
}

ElementarySpace::Ptr
ElementarySpace::from_largest_common_subspace(std::vector<Space::Ptr> const& spaces, bool is_dual)
{
    // --- hints from Python ElementarySpace.from_largest_common_subspace ---
    // OPTIMIZE implementation for mixed orders? or just override this in ElementarySpace?
    // ---
    if (spaces.empty()) {
        throw std::invalid_argument("Need at least one space");
    }
    if (spaces.size() == 1) {
        return spaces[0]->as_ElementarySpace(is_dual);
    }
    if (spaces.size() > 2) {
        // OPTIMIZE directly implement for many
        auto pair = from_largest_common_subspace({ spaces[0], spaces[1] });
        std::vector<Space::Ptr> remaining{ std::static_pointer_cast<Space>(pair) };
        remaining.insert(remaining.end(), spaces.begin() + 2, spaces.end());
        return from_largest_common_subspace(remaining, is_dual);
    }
    auto const& sp1 = *spaces[0];
    auto const& sp2 = *spaces[1];
    SectorArray sectors = SectorArray::empty(sp1.symmetry->sector_ind_len);
    std::vector<int64> mults;
    if (sp1.sector_order == "sorted" && sp2.sector_order == "sorted") {
        SectorArray::iter_common_sorted(
          sp1.sector_decomposition,
          sp2.sector_decomposition,
          /*a_strict=*/true,
          /*b_strict=*/true,
          [&](std::ptrdiff_t i, std::ptrdiff_t j) {
              sectors.push_back(sp1.sector_decomposition[static_cast<std::size_t>(i)]);
              mults.push_back(std::min(sp1.multiplicities[static_cast<std::size_t>(i)],
                                       sp2.multiplicities[static_cast<std::size_t>(j)]));
          });
    } else {
        // OPTIMIZE implementation for mixed orders?
        for (std::size_t i = 0; i < sp1.sector_decomposition.size(); ++i) {
            auto const& sector = sp1.sector_decomposition[i];
            auto const j = sp2.sector_decomposition_where(sector);
            if (!j) {
                continue;
            }
            sectors.push_back(sector);
            mults.push_back(
              std::min(sp1.multiplicities[i], sp2.multiplicities[static_cast<std::size_t>(*j)]));
        }
    }
    auto res = from_sector_decomposition(
      sp1.symmetry, std::move(sectors), std::move(mults), is_dual, std::nullopt, true);
    // from_sector_decomposition potentially introduces a meaningless basis_perm,
    // which we want to ignore here.
    // OPTIMIZE (JU) then dont compute it in the first place?
    res->Leg::set_basis_perm(std::nullopt);
    return res;
}

ElementarySpace::Ptr
ElementarySpace::from_null_space(Symmetry::Ptr symmetry, bool is_dual)
{
    auto sectors = symmetry->empty_sector_array;
    return std::make_shared<ElementarySpace>(
      std::move(symmetry), std::move(sectors), std::vector<int64>{}, is_dual, std::nullopt);
}

ElementarySpace::Ptr
ElementarySpace::from_defining_sectors(Symmetry::Ptr symmetry,
                                       SectorArray defining_sectors,
                                       std::optional<std::vector<int64>> multiplicities_,
                                       bool is_dual,
                                       std::optional<std::vector<int64>> basis_perm,
                                       bool unique_sectors,
                                       std::vector<std::size_t>* return_sorting_perm)
{
    std::vector<int64> multiplicities =
      multiplicities_.value_or(std::vector<int64>(defining_sectors.size(), 1));
    if (multiplicities.size() != defining_sectors.size()) {
        throw std::invalid_argument(
          std::format("multiplicities length {} does not match number of defining sectors {}",
                      multiplicities.size(),
                      defining_sectors.size()));
    }

    // sort sectors
    std::vector<std::size_t> sort;
    if (symmetry->can_be_dropped()) {
        auto const num_states = num_states_per_sector(*symmetry, defining_sectors, multiplicities);
        auto const basis_slices = slice_boundaries(num_states);
        std::tie(defining_sectors, multiplicities, sort) =
          sort_sectors(defining_sectors, multiplicities);
        if (defining_sectors.size() == 0) {
            basis_perm = std::vector<int64>{};
        } else {
            if (!basis_perm) {
                basis_perm = arange(static_cast<std::size_t>(basis_slices.back()));
            }
            std::vector<int64> sorted_perm;
            sorted_perm.reserve(basis_perm->size());
            for (auto const i : sort) {
                for (auto k = basis_slices[i]; k < basis_slices[i + 1]; ++k) {
                    sorted_perm.push_back((*basis_perm)[static_cast<std::size_t>(k)]);
                }
            }
            basis_perm = std::move(sorted_perm);
        }
    } else {
        std::tie(defining_sectors, multiplicities, sort) =
          sort_sectors(defining_sectors, multiplicities);
        if (basis_perm) {
            throw std::invalid_argument("basis_perm is meaningless for this symmetry");
        }
    }
    // combine duplicate sectors (does not affect basis_perm)
    if (!unique_sectors) {
        auto const mult_slices = slice_boundaries(multiplicities);
        auto const diffs = defining_sectors.find_row_differences(/*include_len=*/true);
        // the convention is that for sectors with dim > 1, all copies of the first
        // state appear, then all copies of the second state, etc. At this point,
        // this order is not yet fully respected
        if (basis_perm && !symmetry->is_abelian()) {
            // updated basis_slices after sorting defining_sectors
            auto const num_states =
              num_states_per_sector(*symmetry, defining_sectors, multiplicities);
            auto const basis_slices = slice_boundaries(num_states);
            for (std::size_t i = 0; i + 1 < diffs.size(); ++i) {
                auto const sector_dim = symmetry->sector_dim(defining_sectors[diffs[i]]);
                if (sector_dim == 1) {
                    continue;
                }
                std::vector<int64> const mults(
                  multiplicities.begin() + static_cast<std::ptrdiff_t>(diffs[i]),
                  multiplicities.begin() + static_cast<std::ptrdiff_t>(diffs[i + 1]));
                std::vector<int64> offsets(mults.size() + 1, 0);
                for (std::size_t j = 0; j < mults.size(); ++j) {
                    offsets[j + 1] = offsets[j] + mults[j] * sector_dim;
                }
                auto const start = static_cast<std::size_t>(basis_slices[diffs[i]]);
                auto const stop = static_cast<std::size_t>(basis_slices[diffs[i + 1]]);
                // take the basis_perm associated with the first states and make them contiguous,
                // then go to the second state, etc.
                std::vector<int64> new_perm;
                new_perm.reserve(stop - start);
                for (int64 k = 0; k < sector_dim; ++k) {
                    for (std::size_t j = 0; j < mults.size(); ++j) {
                        auto const mult = mults[j];
                        for (int64 t = 0; t < mult; ++t) {
                            new_perm.push_back(
                              (*basis_perm)[start +
                                            static_cast<std::size_t>(offsets[j] + k * mult + t)]);
                        }
                    }
                }
                assert(new_perm.size() == stop - start);
                std::ranges::copy(new_perm,
                                  basis_perm->begin() + static_cast<std::ptrdiff_t>(start));
            }
        }
        std::vector<int64> unique_mults(diffs.size() - 1);
        for (std::size_t i = 0; i + 1 < diffs.size(); ++i) {
            unique_mults[i] = mult_slices[diffs[i + 1]] - mult_slices[diffs[i]];
        }
        // [:-1] to exclude len
        defining_sectors =
          defining_sectors.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));
        multiplicities = std::move(unique_mults);
    }
    auto res = std::make_shared<ElementarySpace>(std::move(symmetry),
                                                 std::move(defining_sectors),
                                                 std::move(multiplicities),
                                                 is_dual,
                                                 std::move(basis_perm));
    if (return_sorting_perm != nullptr) {
        *return_sorting_perm = std::move(sort);
    }
    return res;
}

ElementarySpace::Ptr
ElementarySpace::from_sector_decomposition(Symmetry::Ptr symmetry,
                                           SectorArray sector_decomposition,
                                           std::optional<std::vector<int64>> multiplicities,
                                           bool is_dual,
                                           std::optional<std::vector<int64>> basis_perm,
                                           bool unique_sectors)
{
    auto defining_sectors =
      is_dual ? symmetry->dual_sectors(sector_decomposition) : std::move(sector_decomposition);
    return from_defining_sectors(std::move(symmetry),
                                 std::move(defining_sectors),
                                 std::move(multiplicities),
                                 is_dual,
                                 std::move(basis_perm),
                                 unique_sectors);
}

ElementarySpace::Ptr
ElementarySpace::from_trivial_sector(int64 dim,
                                     Symmetry::Ptr symmetry,
                                     bool is_dual,
                                     std::optional<std::vector<int64>> basis_perm)
{
    if (!symmetry) {
        symmetry = no_symmetry_product();
    }
    if (dim == 0) {
        return from_null_space(std::move(symmetry), is_dual);
    }
    auto sectors = SectorArray::from_sector(symmetry->trivial_sector);
    return std::make_shared<ElementarySpace>(std::move(symmetry),
                                             std::move(sectors),
                                             std::vector<int64>{ dim },
                                             is_dual,
                                             std::move(basis_perm));
}

ElementarySpace::Ptr
ElementarySpace::shared_es() const
{
    return std::const_pointer_cast<ElementarySpace>(
      std::dynamic_pointer_cast<const ElementarySpace>(shared_from_this()));
}

SectorArray
ElementarySpace::sectors_of_basis() const
{
    // --- hints from Python ElementarySpace.sectors_of_basis ---
    // build in internal basis, then permute
    // ---
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("sectors_of_basis is meaningless for {}.", Space::symmetry->str()));
    }
    // build in internal basis, then permute
    SectorArray res(dim_as_size(Space::dim), Space::symmetry->sector_ind_len);
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
        auto const& sector = sector_decomposition[i];
        for (auto k = (*slices)[i][0]; k < (*slices)[i][1]; ++k) {
            res[static_cast<std::size_t>(k)] = sector;
        }
    }
    if (!_inverse_basis_perm) {
        return res;
    }
    std::vector<std::size_t> perm(_inverse_basis_perm->begin(), _inverse_basis_perm->end());
    return res.take(perm);
}

std::string
ElementarySpace::repr(bool show_symmetry, bool one_line) const
{
    // --- hints from Python ElementarySpace.__repr__ ---
    // try to show everything, then less and less
    // there is no chance to print all sectors in one line
    // try one line
    // try multi line
    // one of the above returns should have triggered
    // ---
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "ElementarySpace";

    struct Options
    {
        bool full_sectors;
        bool summarized_sectors;
        bool symmetry;
    };
    // try to show everything, then less and less
    std::array<Options, 4> const options{ { { true, false, show_symmetry },
                                            { false, true, show_symmetry },
                                            { false, false, show_symmetry },
                                            { false, false, false } } };
    for (auto const& opt : options) {
        if (opt.full_sectors && 3 * static_cast<int64>(defining_sectors.size()) *
                                    static_cast<int64>(defining_sectors.sector_ind_len()) >
                                  linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        std::vector<std::string> items;
        if (opt.symmetry) {
            items.push_back(std::format("symmetry={}", Space::symmetry->repr()));
        }
        if (opt.full_sectors) {
            py::list def_sector_strs;
            for (auto const& a : defining_sectors) {
                def_sector_strs.append(Space::symmetry->sector_str(a));
            }
            py::list sector_dec_strs;
            for (auto const& a : sector_decomposition) {
                sector_dec_strs.append(Space::symmetry->sector_str(a));
            }
            items.push_back(std::format("defining_sectors={}", format_like_list(def_sector_strs)));
            items.push_back(
              std::format("sector_decomposition={}", format_like_list(sector_dec_strs)));
            items.push_back(
              std::format("multiplicities={}", format_like_list(py::cast(multiplicities))));
            if (_basis_perm) {
                items.push_back(
                  std::format("basis_perm={}", format_like_list(py::cast(*_basis_perm))));
            }
        }
        if (opt.summarized_sectors) {
            items.push_back(std::format("num_sectors={}", num_sectors));
            if (_basis_perm) {
                items.emplace_back("basis_perm=[...]");
            }
        }
        items.push_back(std::format("is_dual={}", bool_repr(is_dual)));

        // try one line
        std::string res = ClsName + "(";
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) {
                res += ", ";
            }
            res += items[i];
        }
        res += ")";
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(items.size()) + 2 <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(items, [&](std::string const& item) {
                return static_cast<int64>(indent.size() + item.size() + 1) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                std::string out = ClsName + "(\n";
                for (auto const& item : items) {
                    out += indent + indent + item + ",\n";
                }
                out += ")";
                return out;
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("ElementarySpace repr: no suitable format found");
}

bool
ElementarySpace::operator==(Leg const& other) const
{
    // --- hints from Python ElementarySpace.__eq__ ---
    // check this first to safely compare later
    // both permutations are trivial, thus equal
    // ---
    auto const* o = dynamic_cast<ElementarySpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_es(*o);
}

bool
ElementarySpace::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<ElementarySpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_es(*o);
}

bool
ElementarySpace::equals_es(ElementarySpace const& other) const
{
    // DirectSumSpace equality is structural (summands), not fused sector content.
    if (is_direct_sum_space() || other.is_direct_sum_space()) {
        if (!(is_direct_sum_space() && other.is_direct_sum_space())) {
            return false;
        }
        return dynamic_cast<DirectSumSpace const&>(*this).equals_dss(
          dynamic_cast<DirectSumSpace const&>(other));
    }
    if (is_dual != other.is_dual) {
        return false;
    }
    if (!Space::symmetry->equals(*other.Space::symmetry)) {
        return false;
    }
    // check this first to safely compare later
    if (num_sectors != other.num_sectors) {
        return false;
    }
    if (multiplicities != other.multiplicities) {
        return false;
    }
    if (!(defining_sectors == other.defining_sectors)) {
        return false;
    }
    if (_basis_perm || other._basis_perm) {
        if (basis_perm() != other.basis_perm()) {
            return false;
        }
    }
    // else: both permutations are trivial, thus equal
    return true;
}

ElementarySpace::Ptr
ElementarySpace::as_ElementarySpace(bool is_dual_)
{
    if (is_dual_ == is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

ElementarySpace::Ptr
ElementarySpace::as_ket_space()
{
    if (!is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

ElementarySpace::Ptr
ElementarySpace::as_bra_space()
{
    if (is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

Space::Ptr
ElementarySpace::change_symmetry(Symmetry::Ptr symmetry, SectorMapFn sector_map, bool injective)
{
    return from_defining_sectors(std::move(symmetry),
                                 sector_map(defining_sectors),
                                 multiplicities,
                                 is_dual,
                                 _basis_perm,
                                 injective);
}

ElementarySpace::Ptr
ElementarySpace::direct_sum(std::vector<Ptr> const& others,
                            std::optional<OptionalLabels> summand_labels) const
{
    std::vector<Ptr> all;
    all.reserve(1 + others.size());
    all.push_back(shared_es());
    all.insert(all.end(), others.begin(), others.end());
    return DirectSumSpace::from_spaces(std::move(all), is_dual, std::move(summand_labels));
}

Space::Ptr
ElementarySpace::drop_symmetry(std::optional<std::vector<int64>> which)
{
    auto const [which_factors, remaining_symmetry] =
      parse_inputs_drop_symmetry(which, *Space::symmetry);
    if (!which_factors) {
        return from_trivial_sector(
          static_cast<int64>(Space::dim), remaining_symmetry, is_dual, _basis_perm);
    }
    // the sector components that are kept
    std::vector<bool> mask(Space::symmetry->sector_ind_len, true);
    for (auto const i : *which_factors) {
        auto const idx = static_cast<std::size_t>(i);
        for (auto k = Space::symmetry->sector_slices[idx];
             k < Space::symmetry->sector_slices[idx + 1];
             ++k) {
            mask[k] = false;
        }
    }
    std::vector<std::size_t> keep;
    for (std::size_t k = 0; k < mask.size(); ++k) {
        if (mask[k]) {
            keep.push_back(k);
        }
    }
    SectorMapFn sector_map = [keep](SectorArray const& sectors) {
        SectorArray res(sectors.size(), static_cast<std::uint8_t>(keep.size()));
        for (std::size_t i = 0; i < sectors.size(); ++i) {
            auto sector = Sector::zeros(static_cast<std::uint8_t>(keep.size()));
            for (std::size_t k = 0; k < keep.size(); ++k) {
                sector[k] = sectors[i][keep[k]];
            }
            res[i] = sector;
        }
        return res;
    };
    return change_symmetry(remaining_symmetry, std::move(sector_map));
}

Space::Ptr
ElementarySpace::dual_space() const
{
    return dual_es();
}

Leg::Ptr
ElementarySpace::dual_leg() const
{
    return dual_es();
}

ElementarySpace::Ptr
ElementarySpace::dual_es() const
{
    return std::make_shared<ElementarySpace>(
      Space::symmetry, defining_sectors, multiplicities, !is_dual, _basis_perm);
}

std::pair<int64, int64>
ElementarySpace::parse_index(int64 idx) const
{
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("parse_index is meaningless for {}.", Space::symmetry->str()));
    }
    idx = to_valid_idx(idx, static_cast<int64>(Space::dim));
    if (_inverse_basis_perm) {
        idx = (*_inverse_basis_perm)[static_cast<std::size_t>(idx)];
    }
    // bisect the (increasing) starts of the slices
    auto const& sl = *slices;
    std::size_t lo = 0;
    std::size_t hi = sl.size();
    while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if (sl[mid][0] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    auto const sector_idx = static_cast<int64>(lo) - 1;
    assert(sector_idx >= 0);
    auto const multiplicity_idx = idx - sl[static_cast<std::size_t>(sector_idx)][0];
    return { sector_idx, multiplicity_idx };
}

Sector
ElementarySpace::idx_to_sector(int64 idx) const
{
    auto const [sector_idx, _] = parse_index(idx);
    return sector_decomposition[static_cast<std::size_t>(sector_idx)];
}

ElementarySpace::Ptr
ElementarySpace::take_slice(py::array blockmask) const
{
    // --- hints from Python ElementarySpace.take_slice ---
    // should be guaranteed by check above already, but to be sure...
    // note blockmask is in the private basis order.
    // ---
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("take_slice is meaningless for {}.", Space::symmetry->str()));
    }
    auto casted = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(blockmask);
    if (!casted || casted.ndim() != 1) {
        throw py::type_error("blockmask must be a 1D array of bool");
    }
    auto const public_mask = casted.unchecked<1>();
    auto const num_basis_states = dim_as_size(Space::dim);
    if (static_cast<std::size_t>(public_mask.shape(0)) != num_basis_states) {
        throw std::invalid_argument("blockmask has wrong length");
    }
    // note: mask is in the internal basis order from here on, i.e. we applied the basis_perm.
    std::vector<bool> mask(num_basis_states);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        auto const public_idx = _basis_perm ? (*_basis_perm)[i] : static_cast<int64>(i);
        mask[i] = public_mask(static_cast<py::ssize_t>(public_idx));
    }
    SectorArray sectors = SectorArray::empty(Space::symmetry->sector_ind_len);
    std::vector<int64> mults;
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
        auto const d_a = (*sector_dims)[i];
        auto const [start, stop] = (*slices)[i];
        int64 num_kept = 0;
        for (auto k = start; k < stop; k += d_a) {
            // multiplets need to be kept or discarded as a whole
            bool const keep = mask[static_cast<std::size_t>(k)];
            for (int64 t = 1; t < d_a; ++t) {
                if (mask[static_cast<std::size_t>(k + t)] != keep) {
                    throw std::invalid_argument(
                      "Multiplets need to be kept or discarded as a whole.");
                }
            }
            if (keep) {
                num_kept += d_a;
            }
        }
        auto const mult = num_kept / d_a;
        if (mult > 0) {
            sectors.push_back(defining_sectors[i]);
            mults.push_back(mult);
        }
    }
    // build basis_perm for small leg.
    // it is determined by demanding
    //    a) that the following diagram commutes
    //
    //        (self, public) ---- self.basis_perm ---->  (self, internal)
    //         |                                           |
    //         v public_blockmask                          v projection_internal
    //         |                                           |
    //        (res, public) ----- small_leg_perm ----->  (res, internal)
    //
    //    b) that projection_internal is also just a mask (i.e it preserves ordering)
    //       which is given by public_blockmask[self.basis_perm]
    //
    // this allows us to internally (e.g. in the abelian backend) store only 1D boolean masks
    // as blocks.
    //
    // note mask is in the private basis order.
    auto const perm = basis_perm();
    std::vector<int64> kept_perm;
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        if (mask[i]) {
            kept_perm.push_back(perm[i]);
        }
    }
    return std::make_shared<ElementarySpace>(
      Space::symmetry, std::move(sectors), std::move(mults), is_dual, rank_data(kept_perm));
}

ElementarySpace::Ptr
ElementarySpace::with_opposite_duality() const
{
    // --- hints from Python ElementarySpace.with_opposite_duality ---
    // already have the self.symmetry.dual_sectors(self.defining_sectors)
    // ---
    SectorArray dual_defining_sectors;
    if (is_dual) {
        // already have the symmetry->dual_sectors(defining_sectors)
        dual_defining_sectors = sector_decomposition;
    } else {
        dual_defining_sectors = Space::symmetry->dual_sectors(defining_sectors);
    }
    // note: dual_defining_sectors are not sorted, but they are unique.
    return from_defining_sectors(Space::symmetry,
                                 std::move(dual_defining_sectors),
                                 multiplicities,
                                 !is_dual,
                                 _basis_perm,
                                 /*unique_sectors=*/true);
}

ElementarySpace::Ptr
ElementarySpace::with_is_dual(bool is_dual_) const
{
    if (is_dual_ == is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

Space::Ptr
ElementarySpace::as_space_obj()
{
    return shared_es();
}

bool
ElementarySpace::is_trivial() const
{
    return Space::is_trivial();
}

std::string
ElementarySpace::ascii_arrow() const
{
    return is_dual ? "^" : "v";
}

void
ElementarySpace::save_hdf5(lasyten::hdf5::Saver& saver,
                           HighFive::Group& h5gr,
                           std::string const& subpath) const
{
    hdf5_export::save_sector_array(saver, subpath + "defining_sectors", defining_sectors);
    hdf5_export::save_sector_array(saver, subpath + "sector_decomposition", sector_decomposition);
    if (sector_order) {
        saver.save_string(subpath + "sector_order", *sector_order);
    } else {
        saver.save_none(subpath + "sector_order");
    }
    hdf5_export::save_optional_i64_vector(saver, subpath + "_basis_perm", _basis_perm);
    hdf5_export::save_optional_i64_vector(
      saver, subpath + "_inverse_basis_perm", _inverse_basis_perm);
    hdf5_export::save_i64_vector(saver, subpath + "multiplicities", multiplicities);
    hdf5_export::save_symmetry(saver, subpath + "symmetry", Space::symmetry);
    saver.save_int64(subpath + "dim", static_cast<std::int64_t>(Space::dim));
    saver.save_int64(subpath + "num_sectors", static_cast<std::int64_t>(num_sectors));
    if (slices) {
        std::vector<std::int64_t> flat;
        flat.reserve(slices->size() * 2);
        for (auto const& sl : *slices) {
            flat.push_back(sl[0]);
            flat.push_back(sl[1]);
        }
        saver.save_array(subpath + "slices",
                         hdf5_export::i64_matrix_to_buffer(flat, slices->size(), 2));
    } else {
        saver.save_none(subpath + "slices");
    }
    if (sector_dims) {
        hdf5_export::save_i64_vector(saver, subpath + "sector_dims", *sector_dims);
    } else {
        saver.save_none(subpath + "sector_dims");
    }
    hdf5_io::h5_set_attr(h5gr.getId(), "is_dual", is_dual);
}

ElementarySpace::Ptr
ElementarySpace::from_hdf5(lasyten::hdf5::Loader& loader,
                           HighFive::Group& h5gr,
                           std::string const& subpath)
{
    auto symmetry = hdf5_export::load_symmetry(loader, subpath + "symmetry");
    auto defining_sectors = hdf5_export::load_sector_array(loader, subpath + "defining_sectors");
    auto multiplicities = hdf5_export::load_i64_vector(loader, subpath + "multiplicities");
    auto basis_perm = hdf5_export::load_optional_i64_vector(loader, subpath + "_basis_perm");
    auto const is_dual_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "is_dual");
    bool const is_dual = is_dual_attr.value_or(0) != 0;
    auto obj = std::make_shared<ElementarySpace>(std::move(symmetry),
                                                 std::move(defining_sectors),
                                                 std::move(multiplicities),
                                                 is_dual,
                                                 std::move(basis_perm));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

[[nodiscard]] std::vector<int64>
dss_cumsum_with_leading_zero(std::vector<int64> const& mults)
{
    std::vector<int64> out;
    out.reserve(mults.size() + 1);
    out.push_back(0);
    int64 running = 0;
    for (auto m : mults) {
        running += m;
        out.push_back(running);
    }
    return out;
}

[[nodiscard]] std::pair<std::vector<ElementarySpace::Ptr>, OptionalLabels>
flatten_direct_sum_spaces(std::vector<ElementarySpace::Ptr> const& spaces,
                          OptionalLabels const& labels)
{
    std::vector<ElementarySpace::Ptr> flat;
    OptionalLabels flat_labels;
    flat.reserve(spaces.size());
    flat_labels.reserve(spaces.size());
    for (std::size_t i = 0; i < spaces.size(); ++i) {
        auto const& s = spaces[i];
        auto const& label = labels[i];
        if (!s) {
            throw std::invalid_argument("DirectSumSpace summands must be non-null");
        }
        if (std::dynamic_pointer_cast<LegPipe>(s)) {
            throw std::invalid_argument(
              "DirectSumSpace summands must be plain ElementarySpaces (not pipes)");
        }
        if (auto dss = std::dynamic_pointer_cast<DirectSumSpace>(s)) {
            if (label) {
                throw std::invalid_argument(
                  "Can not attach an explicit summand label to a nested DirectSumSpace summand; "
                  "its own summand_labels are propagated to the flattened slots instead.");
            }
            auto [nested_spaces, nested_labels] =
              flatten_direct_sum_spaces(dss->spaces, dss->summand_labels);
            flat.insert(flat.end(), nested_spaces.begin(), nested_spaces.end());
            flat_labels.insert(flat_labels.end(), nested_labels.begin(), nested_labels.end());
        } else {
            flat.push_back(s);
            flat_labels.push_back(label);
        }
    }
    return { std::move(flat), std::move(flat_labels) };
}

/// Duplicate entries (ignoring unset labels) and invalid-syntax labels raise.
void
validate_summand_labels(OptionalLabels const& labels)
{
    std::unordered_set<std::string> seen;
    for (auto const& l : labels) {
        if (!l) {
            continue;
        }
        if (!is_valid_leg_label(l)) {
            throw std::invalid_argument(std::format("Invalid summand label: {}", *l));
        }
        if (!seen.insert(*l).second) {
            throw std::invalid_argument(std::format("Duplicate summand label: {}", *l));
        }
    }
}

} // namespace

DirectSumSpace::Prepared
DirectSumSpace::prepare(std::vector<ElementarySpace::Ptr> spaces_in,
                        bool is_dual_,
                        std::optional<OptionalLabels> summand_labels_in)
{
    OptionalLabels labels_in =
      summand_labels_in.value_or(OptionalLabels(spaces_in.size(), std::nullopt));
    if (labels_in.size() != spaces_in.size()) {
        throw std::invalid_argument("summand_labels must have the same length as spaces");
    }
    auto [flat, flat_labels] = flatten_direct_sum_spaces(spaces_in, labels_in);
    if (flat.empty()) {
        throw std::invalid_argument("DirectSumSpace requires at least one summand");
    }
    validate_summand_labels(flat_labels);
    auto const& sym = flat[0]->Space::symmetry;
    if (!std::ranges::all_of(
          flat, [&](ElementarySpace::Ptr const& o) { return o->Space::symmetry->equals(*sym); })) {
        throw std::invalid_argument("DirectSumSpace requires matching symmetries");
    }
    if (!std::ranges::all_of(
          flat, [is_dual_](ElementarySpace::Ptr const& o) { return o->is_dual == is_dual_; })) {
        throw std::invalid_argument("DirectSumSpace requires matching duality");
    }

    std::optional<std::vector<int64>> basis_perm_;
    if (sym->can_be_dropped()) {
        std::vector<int64> perm;
        auto offset = int64{ 0 };
        for (auto const& s : flat) {
            for (auto const idx : s->basis_perm()) {
                perm.push_back(idx + offset);
            }
            offset += static_cast<int64>(s->Space::dim);
        }
        basis_perm_ = std::move(perm);
    }

    auto sectors = flat[0]->defining_sectors;
    auto mults = flat[0]->multiplicities;
    for (std::size_t i = 1; i < flat.size(); ++i) {
        sectors = sectors.concat(flat[i]->defining_sectors);
        mults.insert(mults.end(), flat[i]->multiplicities.begin(), flat[i]->multiplicities.end());
    }

    // Build the fused ElementarySpace view to obtain sorted/merged defining data, then
    // copy those into Prepared. We do not keep the temporary plain ES.
    auto fused = ElementarySpace::from_defining_sectors(
      sym, std::move(sectors), std::move(mults), is_dual_, std::move(basis_perm_));

    Prepared prepared;
    prepared.spaces = std::move(flat);
    prepared.symmetry = fused->Space::symmetry;
    prepared.defining_sectors = fused->defining_sectors;
    prepared.multiplicities = fused->multiplicities;
    if (sym->can_be_dropped()) {
        prepared.basis_perm = fused->basis_perm();
    }
    prepared.summand_labels = std::move(flat_labels);
    return prepared;
}

DirectSumSpace::DirectSumSpace(Prepared prepared, bool is_dual_)
  : ElementarySpace(prepared.symmetry,
                    prepared.defining_sectors,
                    prepared.multiplicities,
                    is_dual_,
                    prepared.basis_perm)
  , spaces(std::move(prepared.spaces))
  , summand_labels(std::move(prepared.summand_labels))
{
    for (std::size_t i = 0; i < summand_labels.size(); ++i) {
        if (summand_labels[i]) {
            _summand_labelmap[*summand_labels[i]] = static_cast<int64>(i);
        }
    }
}

DirectSumSpace::DirectSumSpace(std::vector<ElementarySpace::Ptr> spaces_,
                               bool is_dual_,
                               std::optional<OptionalLabels> summand_labels_)
  : DirectSumSpace(prepare(std::move(spaces_), is_dual_, std::move(summand_labels_)), is_dual_)
{
}

DirectSumSpace::Ptr
DirectSumSpace::from_spaces(std::vector<ElementarySpace::Ptr> spaces_,
                            bool is_dual_,
                            std::optional<OptionalLabels> summand_labels_)
{
    return std::make_shared<DirectSumSpace>(
      std::move(spaces_), is_dual_, std::move(summand_labels_));
}

int64
DirectSumSpace::get_summand_idx(SummandRef which) const
{
    if (std::holds_alternative<std::string>(which)) {
        auto const& label = std::get<std::string>(which);
        auto it = _summand_labelmap.find(label);
        if (it == _summand_labelmap.end()) {
            std::string available;
            for (std::size_t i = 0; i < summand_labels.size(); ++i) {
                if (i > 0) {
                    available += ", ";
                }
                available += summand_labels[i] ? *summand_labels[i] : std::string("None");
            }
            throw std::invalid_argument(
              std::format("No summand with label {}. Labels are [{}]", label, available));
        }
        return it->second;
    }
    auto i = std::get<int64>(which);
    auto const n = static_cast<int64>(spaces.size());
    if (i < 0) {
        i += n;
    }
    if (i < 0 || i >= n) {
        throw std::invalid_argument(
          std::format("summand index {} out of range for DirectSumSpace with {} summands",
                      std::get<int64>(which),
                      n));
    }
    return i;
}

bool
DirectSumSpace::has_summand_label(std::string const& label) const
{
    return _summand_labelmap.contains(label);
}

DirectSumSpace::Ptr
DirectSumSpace::shared_dss() const
{
    return std::dynamic_pointer_cast<DirectSumSpace>(shared_es());
}

void
DirectSumSpace::test_sanity() const
{
    if (spaces.empty()) {
        throw std::logic_error("DirectSumSpace::test_sanity: empty spaces");
    }
    if (summand_labels.size() != spaces.size()) {
        throw std::logic_error("DirectSumSpace::test_sanity: summand_labels length mismatch");
    }
    validate_summand_labels(summand_labels);
    for (auto const& s : spaces) {
        if (!s) {
            throw std::logic_error("DirectSumSpace::test_sanity: null summand");
        }
        if (s->is_direct_sum_space()) {
            throw std::logic_error("DirectSumSpace::test_sanity: nested DirectSumSpace");
        }
        if (std::dynamic_pointer_cast<LegPipe>(s)) {
            throw std::logic_error("DirectSumSpace::test_sanity: pipe summand");
        }
        if (s->is_dual != is_dual) {
            throw std::logic_error("DirectSumSpace::test_sanity: duality mismatch");
        }
        if (!s->Space::symmetry->equals(*Space::symmetry)) {
            throw std::logic_error("DirectSumSpace::test_sanity: symmetry mismatch");
        }
        s->test_sanity();
    }
    // Fused view must match collapsing direct_sum of the plain summands.
    auto plain = as_plain_ElementarySpace();
    if (plain->num_sectors != num_sectors || plain->multiplicities != multiplicities ||
        !(plain->defining_sectors == defining_sectors)) {
        throw std::logic_error("DirectSumSpace::test_sanity: fused view mismatch");
    }
    if (Space::symmetry->can_be_dropped() && plain->basis_perm() != basis_perm()) {
        throw std::logic_error("DirectSumSpace::test_sanity: basis_perm mismatch");
    }
    ElementarySpace::test_sanity();
}

std::vector<std::vector<int64>>
DirectSumSpace::mult_slices() const
{
    std::vector<std::vector<int64>> out;
    out.reserve(static_cast<std::size_t>(num_sectors));
    for (auto const& sector : sector_decomposition) {
        std::vector<int64> mults;
        mults.reserve(spaces.size());
        for (auto const& space : spaces) {
            auto idx = space->sector_decomposition_where(sector);
            mults.push_back(idx.has_value() ? space->multiplicities[static_cast<std::size_t>(*idx)]
                                            : int64{ 0 });
        }
        out.push_back(dss_cumsum_with_leading_zero(mults));
    }
    return out;
}

ElementarySpace::Ptr
DirectSumSpace::as_plain_ElementarySpace() const
{
    std::optional<std::vector<int64>> perm;
    if (Space::symmetry->can_be_dropped()) {
        perm = basis_perm();
    }
    return ElementarySpace::from_defining_sectors(Space::symmetry,
                                                  defining_sectors,
                                                  multiplicities,
                                                  is_dual,
                                                  std::move(perm),
                                                  /*unique_sectors=*/true);
}

Space::Ptr
DirectSumSpace::as_space_obj()
{
    return shared_dss();
}

ElementarySpace::Ptr
DirectSumSpace::as_ElementarySpace(bool is_dual_)
{
    return with_is_dual(is_dual_);
}

Space::Ptr
DirectSumSpace::dual_space() const
{
    return dual_dss();
}

Leg::Ptr
DirectSumSpace::dual_leg() const
{
    return dual_dss();
}

DirectSumSpace::Ptr
DirectSumSpace::dual_dss() const
{
    // NB: not with_opposite_duality(), which gives an isomorphic space (same sectors) with
    // opposite is_dual, rather than the dual space.
    std::vector<ElementarySpace::Ptr> new_spaces;
    new_spaces.reserve(spaces.size());
    for (auto const& s : spaces) {
        new_spaces.push_back(s->dual_es());
    }
    return from_spaces(std::move(new_spaces), !is_dual, summand_labels);
}

Space::Ptr
DirectSumSpace::change_symmetry(Symmetry::Ptr symmetry_, SectorMapFn sector_map, bool injective)
{
    std::vector<ElementarySpace::Ptr> new_spaces;
    new_spaces.reserve(spaces.size());
    for (auto const& s : spaces) {
        new_spaces.push_back(std::dynamic_pointer_cast<ElementarySpace>(
          s->change_symmetry(symmetry_, sector_map, injective)));
    }
    return from_spaces(std::move(new_spaces), is_dual, summand_labels);
}

Space::Ptr
DirectSumSpace::drop_symmetry(std::optional<std::vector<int64>> which)
{
    std::vector<ElementarySpace::Ptr> new_spaces;
    new_spaces.reserve(spaces.size());
    for (auto const& s : spaces) {
        new_spaces.push_back(std::dynamic_pointer_cast<ElementarySpace>(s->drop_symmetry(which)));
    }
    return from_spaces(std::move(new_spaces), is_dual, summand_labels);
}

ElementarySpace::Ptr
DirectSumSpace::take_slice(py::array blockmask) const
{
    warn("Using `DirectSumSpace.take_slice` loses the direct-sum structure and results in "
         "a plain ElementarySpace. Explicitly convert using `as_plain_ElementarySpace` to "
         "suppress this warning.");
    return as_plain_ElementarySpace()->take_slice(std::move(blockmask));
}

void
DirectSumSpace::set_basis_perm(std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for DirectSumSpace.");
}

void
DirectSumSpace::set_inverse_basis_perm(std::optional<std::vector<int64>> /*inverse_basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for DirectSumSpace.");
}

ElementarySpace::Ptr
DirectSumSpace::with_opposite_duality() const
{
    std::vector<ElementarySpace::Ptr> new_spaces;
    new_spaces.reserve(spaces.size());
    for (auto const& s : spaces) {
        new_spaces.push_back(s->with_opposite_duality());
    }
    return from_spaces(std::move(new_spaces), !is_dual, summand_labels);
}

bool
DirectSumSpace::operator==(Leg const& other) const
{
    auto const* o = dynamic_cast<DirectSumSpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_dss(*o);
}

bool
DirectSumSpace::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<DirectSumSpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_dss(*o);
}

bool
DirectSumSpace::equals_dss(DirectSumSpace const& other) const
{
    if (is_dual != other.is_dual) {
        return false;
    }
    if (spaces.size() != other.spaces.size()) {
        return false;
    }
    for (std::size_t i = 0; i < spaces.size(); ++i) {
        if (!spaces[i]->equals_es(*other.spaces[i])) {
            return false;
        }
    }
    return true;
}

std::string
DirectSumSpace::repr(bool show_symmetry, bool one_line) const
{
    std::ostringstream oss;
    oss << "DirectSumSpace(";
    if (!one_line) {
        oss << '\n';
    }
    for (std::size_t i = 0; i < spaces.size(); ++i) {
        if (!one_line) {
            oss << "  ";
        }
        if (i > 0) {
            oss << (one_line ? ", " : ",\n  ");
        }
        oss << spaces[i]->repr(show_symmetry, /*one_line=*/true);
    }
    if (!one_line) {
        oss << '\n';
    }
    oss << ')';
    return oss.str();
}

DirectSumSpace::Ptr
DirectSumSpace::from_basis(Symmetry::Ptr /*symmetry*/, SectorArray /*sectors_of_basis*/)
{
    throw py::type_error("from_basis is not supported for DirectSumSpace");
}

DirectSumSpace::Ptr
DirectSumSpace::from_null_space(Symmetry::Ptr /*symmetry*/, bool /*is_dual*/)
{
    throw py::type_error("from_null_space is not supported for DirectSumSpace");
}

DirectSumSpace::Ptr
DirectSumSpace::from_defining_sectors(Symmetry::Ptr /*symmetry*/,
                                      SectorArray /*defining_sectors*/,
                                      std::optional<std::vector<int64>> /*multiplicities*/,
                                      bool /*is_dual*/,
                                      std::optional<std::vector<int64>> /*basis_perm*/,
                                      bool /*unique_sectors*/,
                                      std::vector<std::size_t>* /*return_sorting_perm*/)
{
    throw py::type_error("from_defining_sectors is not supported for DirectSumSpace");
}

DirectSumSpace::Ptr
DirectSumSpace::from_trivial_sector(int64 /*dim*/,
                                    Symmetry::Ptr /*symmetry*/,
                                    bool /*is_dual*/,
                                    std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("from_trivial_sector is not supported for DirectSumSpace");
}

void
DirectSumSpace::save_hdf5(lasyten::hdf5::Saver& saver,
                          HighFive::Group& h5gr,
                          std::string const& subpath) const
{
    ElementarySpace::save_hdf5(saver, h5gr, subpath);
    HighFive::Group spaces_g;
    std::string spaces_sub;
    saver.save_sequence_begin(subpath + "spaces",
                              hdf5_io::REPR_LIST,
                              static_cast<std::int64_t>(spaces.size()),
                              spaces_g,
                              spaces_sub);
    lasyten::hdf5::Saver spaces_saver(spaces_g);
    for (std::size_t i = 0; i < spaces.size(); ++i) {
        hdf5_export::save_elementary_space(spaces_saver, std::to_string(i), spaces[i]);
    }
    hdf5_io::h5_set_attr(h5gr.getId(), "is_direct_sum_space", true);
    hdf5_export::save_optional_labels(saver, subpath + "summand_labels", summand_labels);
}

DirectSumSpace::Ptr
DirectSumSpace::from_hdf5(lasyten::hdf5::Loader& loader,
                          HighFive::Group& h5gr,
                          std::string const& subpath)
{
    hid_t spaces_id = loader.open(subpath + "spaces");
    auto const n = hdf5_io::h5_get_attr_int64(spaces_id, hdf5_io::ATTR_LEN).value_or(0);
    HighFive::Group spaces_g = hdf5_io::group_from_hid(spaces_id);
    lasyten::hdf5::Loader spaces_loader(spaces_g);
    std::vector<ElementarySpace::Ptr> spaces_;
    spaces_.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        spaces_.push_back(hdf5_export::load_elementary_space(spaces_loader, std::to_string(i)));
    }
    auto const is_dual_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "is_dual");
    bool const is_dual = is_dual_attr.value_or(0) != 0;
    std::optional<OptionalLabels> summand_labels_;
    if (hdf5_io::h5_contains(loader.root(), subpath + "summand_labels")) {
        auto labs = hdf5_export::load_optional_labels(loader, subpath + "summand_labels");
        summand_labels_ = labs.empty() && !spaces_.empty()
                            ? OptionalLabels(spaces_.size(), std::nullopt)
                            : std::move(labs);
    }
    auto obj = from_spaces(std::move(spaces_), is_dual, std::move(summand_labels_));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

/// The symmetry of a :class:`TensorProduct` factor.
[[nodiscard]] Symmetry::Ptr
factor_symmetry(Leg::Ptr const& factor)
{
    return factor->symmetry;
}

[[nodiscard]] Leg::Ptr
factor_change_symmetry(Leg::Ptr const& factor,
                       Symmetry::Ptr const& symmetry,
                       SectorMapFn const& sector_map,
                       bool injective)
{
    if (auto space = std::dynamic_pointer_cast<Space>(factor)) {
        auto changed = space->change_symmetry(symmetry, sector_map, injective);
        auto leg = std::dynamic_pointer_cast<Leg>(changed);
        if (!leg) {
            throw std::invalid_argument(
              "change_symmetry on a TensorProduct factor must yield a Leg");
        }
        return leg;
    }
    auto pipe = std::dynamic_pointer_cast<LegPipe>(factor);
    if (!pipe) {
        throw std::invalid_argument("TensorProduct factor is not a Leg");
    }
    std::vector<Leg::Ptr> new_legs;
    new_legs.reserve(pipe->legs.size());
    for (auto const& leg : pipe->legs) {
        new_legs.push_back(factor_change_symmetry(leg, symmetry, sector_map, injective));
    }
    return std::make_shared<LegPipe>(std::move(new_legs), pipe->is_dual, pipe->combine_cstyle);
}

[[nodiscard]] Leg::Ptr
factor_drop_symmetry(Leg::Ptr const& factor, std::optional<std::vector<int64>> const& which)
{
    if (auto space = std::dynamic_pointer_cast<Space>(factor)) {
        auto changed = space->drop_symmetry(which);
        auto leg = std::dynamic_pointer_cast<Leg>(changed);
        if (!leg) {
            throw std::invalid_argument(
              "drop_symmetry on a TensorProduct factor must yield a Leg");
        }
        return leg;
    }
    auto pipe = std::dynamic_pointer_cast<LegPipe>(factor);
    if (!pipe) {
        throw std::invalid_argument("TensorProduct factor is not a Leg");
    }
    std::vector<Leg::Ptr> new_legs;
    new_legs.reserve(pipe->legs.size());
    for (auto const& leg : pipe->legs) {
        new_legs.push_back(factor_drop_symmetry(leg, which));
    }
    return std::make_shared<LegPipe>(std::move(new_legs), pipe->is_dual, pipe->combine_cstyle);
}

/// ``factor.__repr__(show_symmetry=..., one_line=...)``, with fallbacks.
[[nodiscard]] std::string
factor_repr(py::handle factor, bool show_symmetry, bool one_line)
{
    // the C++ classes bind the parametrized version as ``repr``, the Python ones as ``__repr__``
    for (char const* name : { "repr", "__repr__" }) {
        if (!py::hasattr(factor, name)) {
            continue;
        }
        try {
            return py::str(factor.attr(name)(py::arg("show_symmetry") = show_symmetry,
                                             py::arg("one_line") = one_line));
        } catch (py::error_already_set&) {
            // the attribute does not accept these arguments; fall through
        }
    }
    return py::str(factor.attr("__repr__")());
}

/// ``np.prod(values)``, i.e. ``1`` for an empty input.
[[nodiscard]] int64
product(std::vector<int64> const& values)
{
    int64 res = 1;
    for (auto const v : values) {
        res *= v;
    }
    return res;
}

/// ``all(a == b for a, b in zip(sectors, other))``, i.e. ignoring surplus entries.
[[nodiscard]] bool
sectors_match(SectorArray const& sectors, SectorArray const& other)
{
    if (sectors.size() != other.size()) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        if (!(sectors[i] == other[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string
join(std::vector<std::string> const& parts, std::string const& sep)
{
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += sep;
        }
        out += parts[i];
    }
    return out;
}

[[nodiscard]] py::object
dim_to_py(float64 dim)
{
    if (std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

/// ``TensorProduct._calc_sectors``, for factors that are already flattened to spaces.
[[nodiscard]] std::pair<SectorArray, std::vector<int64>>
calc_sectors_of_spaces(Symmetry const& symmetry, std::span<const Space::Ptr> spaces)
{
    if (spaces.empty()) {
        return { SectorArray::from_sector(symmetry.trivial_sector), std::vector<int64>{ 1 } };
    }

    if (spaces.size() == 1) {
        auto const& space = *spaces.front();
        if (space.sector_order == "sorted") {
            return { space.sector_decomposition, space.multiplicities };
        }
        auto const perm = space.sector_decomposition.lexsort_indices();
        return { space.sector_decomposition.take(perm),
                 gather_or_all(space.multiplicities, perm) };
    }

    if (symmetry.is_abelian()) {
        // all combinations of one sector per space, ordered like ``make_grid(_, cstyle=False)``
        std::vector<std::size_t> num_sectors(spaces.size());
        std::size_t num_combinations = 1;
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            num_sectors[n] = static_cast<std::size_t>(spaces[n]->num_sectors);
            num_combinations *= num_sectors[n];
        }
        std::vector<SectorArray> uncoupled;
        uncoupled.reserve(spaces.size());
        std::vector<int64> multiplicities(num_combinations, 1);
        std::size_t stride = 1;
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            SectorArray column(num_combinations, symmetry.sector_ind_len);
            for (std::size_t m = 0; m < num_combinations; ++m) {
                auto const i = (m / stride) % num_sectors[n];
                column[m] = spaces[n]->sector_decomposition[i];
                multiplicities[m] *= spaces[n]->multiplicities[i];
            }
            uncoupled.push_back(std::move(column));
            stride *= num_sectors[n];
        }
        auto const sectors = symmetry.multiple_fusion_broadcast(uncoupled);
        auto [unique, mults, perm] = sectors.unique_sorted(multiplicities);
        (void)perm;
        return { std::move(unique), std::move(mults) };
    }

    // define recursively
    auto const [sectors, mults] =
      calc_sectors_of_spaces(symmetry, spaces.first(spaces.size() - 1));
    auto const& last = *spaces.back();
    SectorArray combined = SectorArray::empty(symmetry.sector_ind_len);
    std::vector<int64> combined_mults;
    for (std::size_t j = 0; j < last.sector_decomposition.size(); ++j) {
        auto const s2 = last.sector_decomposition[j];
        auto const m2 = last.multiplicities[j];
        for (std::size_t i = 0; i < sectors.size(); ++i) {
            auto const s1 = sectors[i];
            auto const m12 = mults[i] * m2;
            for (auto const& c : symmetry.fusion_outcomes(s1, s2)) {
                combined.push_back(c);
                // OPTIMIZE support batched N symbol?
                combined_mults.push_back(
                  symmetry.has_unique_fusion() ? m12 : m12 * symmetry._n_symbol(s1, s2, c));
            }
        }
    }
    auto [unique, unique_mults, perm] = combined.unique_sorted(combined_mults);
    (void)perm;
    return { std::move(unique), std::move(unique_mults) };
}

/// ``TensorProduct._calc_sectors``.
[[nodiscard]] std::pair<SectorArray, std::vector<int64>>
calc_sectors_of_factors(Symmetry const& symmetry, std::vector<Leg::Ptr> const& factors)
{
    // LegPipes do not have sectors -> flatten them for the purpose of calculating sectors
    std::vector<Space::Ptr> spaces;
    for (auto const& factor : factors) {
        for (auto const& leg : factor->flat_spaces()) {
            // need the sector decomposition of each factor. easiest way: convert to Space
            // OPTIMIZE is this optimal? should we store the as_Space() for later use?
            spaces.push_back(as_space(leg));
        }
    }
    return calc_sectors_of_spaces(symmetry, spaces);
}

} // namespace

Space::Ptr
as_space(Leg::Ptr const& leg)
{
    if (auto space = std::dynamic_pointer_cast<Space>(leg)) {
        return space;
    }
    return leg->as_space_obj();
}

TensorProduct::Prepared
TensorProduct::prepare(std::vector<Leg::Ptr> const& factors,
                       Symmetry::Ptr symmetry,
                       std::optional<SectorArray> sector_decomposition,
                       std::optional<std::vector<int64>> multiplicities)
{
    if (!symmetry) {
        if (factors.empty()) {
            throw std::invalid_argument("If spaces is empty, the symmetry arg is required.");
        }
        symmetry = factor_symmetry(factors.front());
    }
    for (auto const& factor : factors) {
        if (!factor_symmetry(factor)->equals(*symmetry)) {
            throw SymmetryError("Incompatible symmetries.");
        }
    }
    if (!sector_decomposition || !multiplicities) {
        if (sector_decomposition || multiplicities) {
            warn("Need both _sectors and _multiplicities to skip recomputation. "
                 "Got just one.",
                 /*stack_level=*/1);
        }
        auto [sectors, mults] = calc_sectors_of_factors(*symmetry, factors);
        sector_decomposition = std::move(sectors);
        multiplicities = std::move(mults);
    }
    return { std::move(symmetry), std::move(*sector_decomposition), std::move(*multiplicities) };
}

TensorProduct::TensorProduct(std::vector<Leg::Ptr> factors_, Prepared prepared)
  : Space(std::move(prepared.symmetry),
          std::move(prepared.sector_decomposition),
          std::move(prepared.multiplicities),
          "sorted")
  , factors(std::move(factors_))
  , num_factors(static_cast<int64>(factors.size()))
{
}

TensorProduct::TensorProduct(std::vector<Leg::Ptr> factors_,
                             Symmetry::Ptr symmetry_,
                             std::optional<SectorArray> sector_decomposition_,
                             std::optional<std::vector<int64>> multiplicities_)
  : TensorProduct(factors_,
                  prepare(factors_,
                          std::move(symmetry_),
                          std::move(sector_decomposition_),
                          std::move(multiplicities_)))
{
}

void
TensorProduct::test_sanity() const
{
    assert(static_cast<int64>(factors.size()) == num_factors);
    for (auto const& factor : factors) {
        factor->test_sanity();
    }
    Space::test_sanity();
}

TensorProduct::Ptr
TensorProduct::from_partial_products(std::vector<Ptr> const& products)
{
    // --- hints from Python TensorProduct.from_partial_products ---
    // forming isomorphic performs the fusion more efficiently, since it uses the partially
    // fused [f.sectors for f in factors] instead of the flat [s.factors for f in factors for s in
    // f.factors]
    // ---
    if (products.empty()) {
        throw std::invalid_argument("Need at least one TensorProduct");
    }
    std::vector<Leg::Ptr> legs = products.front()->factors;
    auto symmetry = products.front()->symmetry;
    std::vector<Space::Ptr> partial;
    partial.reserve(products.size());
    partial.push_back(products.front());
    for (std::size_t i = 1; i < products.size(); ++i) {
        legs.insert(legs.end(), products[i]->factors.begin(), products[i]->factors.end());
        if (!products[i]->symmetry->equals(*symmetry)) {
            throw SymmetryError("Mismatched symmetries");
        }
        partial.push_back(products[i]);
    }
    auto [sectors, mults] = calc_sectors_of_spaces(*symmetry, partial);
    return std::make_shared<TensorProduct>(
      std::move(legs), std::move(symmetry), std::move(sectors), std::move(mults));
}

Space::Ptr
TensorProduct::dual_space() const
{
    auto const dual = symmetry->dual_sectors(sector_decomposition);
    auto [sectors, mults, perm] = sort_sectors(dual, multiplicities);
    (void)perm;
    std::vector<Leg::Ptr> dual_factors;
    dual_factors.reserve(factors.size());
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        dual_factors.push_back((*it)->dual());
    }
    return std::make_shared<TensorProduct>(
      std::move(dual_factors), symmetry, std::move(sectors), std::move(mults));
}

int64
TensorProduct::block_size(std::variant<int64, Sector> coupled) const
{
    if (auto const* idx = std::get_if<int64>(&coupled)) {
        return multiplicities[static_cast<std::size_t>(to_valid_idx(*idx, num_sectors))];
    }
    return sector_multiplicity(std::get<Sector>(coupled));
}

Space::Ptr
TensorProduct::change_symmetry(Symmetry::Ptr symmetry_, SectorMapFn sector_map, bool injective)
{
    auto sectors = sector_map(sector_decomposition);
    auto mults = multiplicities;
    std::vector<std::size_t> perm;
    if (!injective) {
        std::tie(sectors, mults, perm) = sectors.unique_sorted(mults);
    } else {
        std::tie(sectors, mults, perm) = sort_sectors(sectors, mults);
    }
    std::vector<Leg::Ptr> new_factors;
    new_factors.reserve(factors.size());
    for (auto const& factor : factors) {
        new_factors.push_back(factor_change_symmetry(factor, symmetry_, sector_map, injective));
    }
    // note: unlike the Python version, which passes ``self.symmetry``, we pass the *new*
    // symmetry here. Otherwise the constructor rejects the new factors.
    return std::make_shared<TensorProduct>(
      std::move(new_factors), std::move(symmetry_), std::move(sectors), std::move(mults));
}

Space::Ptr
TensorProduct::drop_symmetry(std::optional<std::vector<int64>> which)
{
    auto const [which_factors, remaining_symmetry] = parse_inputs_drop_symmetry(which, *symmetry);
    SectorArray sectors;
    std::vector<int64> mults;
    if (!which_factors) {
        // note: unlike the Python version, we use the trivial sector of the *remaining*
        // symmetry, which is the only one with the right sector_ind_len.
        sectors = SectorArray::from_sector(remaining_symmetry->trivial_sector);
        mults = { static_cast<int64>(dim) };
    } else {
        // the sector components that are kept
        std::vector<bool> mask(symmetry->sector_ind_len, true);
        for (auto const i : *which_factors) {
            auto const idx = static_cast<std::size_t>(i);
            for (auto k = symmetry->sector_slices[idx]; k < symmetry->sector_slices[idx + 1];
                 ++k) {
                mask[k] = false;
            }
        }
        std::vector<std::size_t> keep;
        for (std::size_t k = 0; k < mask.size(); ++k) {
            if (mask[k]) {
                keep.push_back(k);
            }
        }
        SectorArray kept(sector_decomposition.size(), static_cast<std::uint8_t>(keep.size()));
        for (std::size_t i = 0; i < sector_decomposition.size(); ++i) {
            auto sector = Sector::zeros(static_cast<std::uint8_t>(keep.size()));
            for (std::size_t k = 0; k < keep.size(); ++k) {
                sector[k] = sector_decomposition[i][keep[k]];
            }
            kept[i] = sector;
        }
        std::vector<std::size_t> perm;
        std::tie(sectors, mults, perm) = kept.unique_sorted(multiplicities);
    }
    std::vector<Leg::Ptr> new_factors;
    new_factors.reserve(factors.size());
    for (auto const& factor : factors) {
        new_factors.push_back(factor_drop_symmetry(factor, which_factors));
    }
    return std::make_shared<TensorProduct>(
      std::move(new_factors), remaining_symmetry, std::move(sectors), std::move(mults));
}

bool
TensorProduct::has_pipes() const
{
    return std::ranges::any_of(
      factors, [](Leg::Ptr const& f) { return std::dynamic_pointer_cast<LegPipe>(f) != nullptr; });
}

std::vector<Leg::Ptr>
TensorProduct::flat_legs() const
{
    std::vector<Leg::Ptr> out;
    for (auto const& factor : factors) {
        auto part = factor->flat_legs();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<Leg::Ptr>
TensorProduct::flat_spaces() const
{
    std::vector<Leg::Ptr> out;
    for (auto const& factor : factors) {
        auto part = factor->flat_spaces();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

int64
TensorProduct::num_flat_legs() const
{
    int64 n = 0;
    for (auto const& factor : factors) {
        n += factor->num_flat_legs();
    }
    return n;
}

std::vector<std::vector<int64>>
TensorProduct::flat_legs_nesting() const
{
    int64 i = 0;
    std::vector<std::vector<int64>> res;
    res.reserve(factors.size());
    for (auto const& factor : factors) {
        auto const num = factor->num_flat_legs();
        std::vector<int64> idcs(static_cast<std::size_t>(num));
        std::iota(idcs.begin(), idcs.end(), i);
        res.push_back(std::move(idcs));
        i += num;
    }
    return res;
}

std::vector<int64>
TensorProduct::flat_leg_idcs(int64 i) const
{
    i = to_valid_idx(i, num_factors);
    int64 start = 0;
    for (int64 k = 0; k < i; ++k) {
        start += factors[static_cast<std::size_t>(k)]->num_flat_legs();
    }
    auto const num = factors[static_cast<std::size_t>(i)]->num_flat_legs();
    std::vector<int64> res(static_cast<std::size_t>(num));
    std::iota(res.begin(), res.end(), start);
    return res;
}

int64
TensorProduct::forest_block_size(SectorArray const& uncoupled, Sector coupled) const
{
    // --- hints from Python TensorProduct.forest_block_size ---
    // OPTIMIZE ?
    // ---
    // OPTIMIZE ?
    auto const num_trees = static_cast<int64>(fusion_trees(symmetry, uncoupled, coupled).size());
    return num_trees * tree_block_size(uncoupled);
}

IndexSlice
TensorProduct::forest_block_slice(SectorArray const& uncoupled, Sector coupled) const
{
    // --- hints from Python TensorProduct.forest_block_slice ---
    // no break occurred
    // ---
    int64 offset = 0;
    bool found = false;
    for (auto const& item : iter_uncoupled()) {
        if (sectors_match(item.uncoupled, uncoupled)) {
            found = true;
            break;
        }
        auto const tree_block = product(item.multiplicities);
        auto const num_trees =
          static_cast<int64>(fusion_trees(symmetry, item.uncoupled, coupled).size());
        offset += num_trees * tree_block;
    }
    if (!found) {
        throw std::invalid_argument("Uncoupled sectors incompatible");
    }
    auto const size = forest_block_size(uncoupled, coupled);
    return { offset, offset + size };
}

TensorProduct::Ptr
TensorProduct::insert_multiply(Leg::Ptr other, int64 pos) const
{
    auto const self_ptr = std::const_pointer_cast<TensorProduct>(
      std::dynamic_pointer_cast<TensorProduct const>(shared_from_this()));
    std::vector<Space::Ptr> partial{ self_ptr, as_space(other) };
    auto [sectors, mults] = calc_sectors_of_spaces(*symmetry, partial);
    // Python uses list slicing, i.e. ``factors[:pos] + [other] + factors[pos:]``.
    // In particular, ``pos == -1`` (as used by right_multiply) inserts before the last factor.
    auto const n = static_cast<int64>(factors.size());
    auto const at =
      static_cast<std::ptrdiff_t>(pos < 0 ? std::max<int64>(0, n + pos) : std::min(pos, n));
    std::vector<Leg::Ptr> new_factors;
    new_factors.reserve(factors.size() + 1);
    new_factors.insert(new_factors.end(), factors.begin(), factors.begin() + at);
    new_factors.push_back(std::move(other));
    new_factors.insert(new_factors.end(), factors.begin() + at, factors.end());
    return std::make_shared<TensorProduct>(
      std::move(new_factors), symmetry, std::move(sectors), std::move(mults));
}

std::vector<TreeBlockItem>
TensorProduct::iter_tree_blocks(SectorArray const& coupled) const
{
    // --- hints from Python TensorProduct.iter_tree_blocks ---
    // OPTIMIZE some users in FTBackend ignore some of the yielded values.
    // is that ok performance wise or should we have special case iterators?
    // start index of the current tree block within the block
    // ---
    // OPTIMIZE some users in FTBackend ignore some of the yielded values.
    //          is that ok performance wise or should we have special case iterators?
    std::vector<std::uint8_t> are_dual;
    for (auto const& leg : flat_legs()) {
        are_dual.push_back(leg->is_dual ? 1 : 0);
    }
    auto const uncoupled_items = iter_uncoupled();
    std::vector<TreeBlockItem> out;
    for (std::size_t i = 0; i < coupled.size(); ++i) {
        int64 start = 0; // start index of the current tree block within the block
        for (auto const& item : uncoupled_items) {
            auto const tree_block = product(item.multiplicities);
            for (auto const& tree :
                 fusion_trees(symmetry, item.uncoupled, coupled[i], are_dual).all_trees()) {
                out.push_back({ tree,
                                { start, start + tree_block },
                                item.multiplicities,
                                static_cast<int64>(i) });
                start += tree_block;
            }
        }
    }
    return out;
}

std::vector<ForestBlockItem>
TensorProduct::iter_forest_blocks(SectorArray const& coupled) const
{
    auto const uncoupled_items = iter_uncoupled();
    std::vector<ForestBlockItem> out;
    for (std::size_t i = 0; i < coupled.size(); ++i) {
        int64 start = 0;
        for (auto const& item : uncoupled_items) {
            auto const tree_block = product(item.multiplicities);
            auto const num_trees =
              static_cast<int64>(fusion_trees(symmetry, item.uncoupled, coupled[i]).size());
            auto const width = num_trees * tree_block;
            if (width == 0) {
                continue;
            }
            out.push_back({ item.uncoupled, { start, start + width }, static_cast<int64>(i) });
            start += width;
        }
    }
    return out;
}

std::vector<UncoupledItem>
TensorProduct::iter_uncoupled(bool yield_slices) const
{
    auto const legs = flat_legs();
    std::vector<UncoupledItem> out;

    if (legs.empty()) {
        // note: for a TensorProduct of zero spaces we *do* yield once, with empty arrays.
        UncoupledItem item{ symmetry->empty_sector_array, {}, std::nullopt };
        if (yield_slices) {
            item.slices = std::vector<IndexSlice>{};
        }
        out.push_back(std::move(item));
        return out;
    }

    std::vector<Space::Ptr> spaces;
    spaces.reserve(legs.size());
    for (auto const& leg : legs) {
        spaces.push_back(as_space(leg));
    }
    // ``it.product``, i.e. the last index varies the fastest
    std::vector<std::size_t> strides(spaces.size());
    std::size_t total = 1;
    for (std::size_t n = spaces.size(); n-- > 0;) {
        strides[n] = total;
        total *= static_cast<std::size_t>(spaces[n]->num_sectors);
    }
    out.reserve(total);
    for (std::size_t m = 0; m < total; ++m) {
        SectorArray uncoupled(spaces.size(), symmetry->sector_ind_len);
        std::vector<int64> mults(spaces.size());
        std::optional<std::vector<IndexSlice>> slices_;
        if (yield_slices) {
            slices_.emplace(spaces.size());
        }
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            auto const i = (m / strides[n]) % static_cast<std::size_t>(spaces[n]->num_sectors);
            uncoupled[n] = spaces[n]->sector_decomposition[i];
            mults[n] = spaces[n]->multiplicities[i];
            if (yield_slices) {
                auto const& slc = (*spaces[n]->slices)[i];
                (*slices_)[n] = IndexSlice{ slc[0], slc[1] };
            }
        }
        out.push_back({ std::move(uncoupled), std::move(mults), std::move(slices_) });
    }
    return out;
}

TensorProduct::Ptr
TensorProduct::left_multiply(Leg::Ptr other) const
{
    return insert_multiply(std::move(other), 0);
}

TensorProduct::Ptr
TensorProduct::permuted(std::vector<int64> const& perm) const
{
    if (static_cast<int64>(perm.size()) != num_factors) {
        throw std::invalid_argument("perm has wrong length");
    }
    std::vector<bool> seen(perm.size(), false);
    std::vector<Leg::Ptr> new_factors;
    new_factors.reserve(perm.size());
    for (auto const i : perm) {
        auto const idx = static_cast<std::size_t>(to_valid_idx(i, num_factors));
        if (seen[idx]) {
            throw std::invalid_argument("perm is not a permutation");
        }
        seen[idx] = true;
        new_factors.push_back(factors[idx]);
    }
    return std::make_shared<TensorProduct>(
      std::move(new_factors), symmetry, sector_decomposition, multiplicities);
}

TensorProduct::Ptr
TensorProduct::right_multiply(Leg::Ptr other) const
{
    return insert_multiply(std::move(other), -1);
}

int64
TensorProduct::tree_block_size(SectorArray const& uncoupled) const
{
    // --- hints from Python TensorProduct.tree_block_size ---
    // OPTIMIZE ?
    // ---
    // OPTIMIZE ?
    auto const legs = flat_legs();
    auto const n = std::min(legs.size(), uncoupled.size());
    int64 res = 1;
    for (std::size_t i = 0; i < n; ++i) {
        res *= as_space(legs[i])->sector_multiplicity(uncoupled[i]);
    }
    return res;
}

IndexSlice
TensorProduct::tree_block_slice(FusionTree const& tree) const
{
    // --- hints from Python TensorProduct.tree_block_slice ---
    // OPTIMIZE ?
    // no break occurred
    // ---
    // OPTIMIZE ?
    int64 start = 0;
    int64 tree_block = 1;
    bool found = false;
    for (auto const& item : iter_uncoupled()) {
        tree_block = product(item.multiplicities);
        if (sectors_match(item.uncoupled, tree.uncoupled)) {
            found = true;
            break;
        }
        auto const num_trees =
          static_cast<int64>(fusion_trees(symmetry, item.uncoupled, tree.coupled).size());
        start += num_trees * tree_block;
    }
    if (!found) {
        throw std::invalid_argument("Uncoupled sectors incompatible");
    }
    auto const tree_idx =
      fusion_trees(symmetry, tree.uncoupled, tree.coupled, tree.are_dual).index(tree);
    start += tree_block * static_cast<int64>(tree_idx);
    return { start, start + tree_block };
}

bool
TensorProduct::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<TensorProduct const*>(&other);
    if (o == nullptr) {
        return false;
    }
    if (num_factors != o->num_factors) {
        return false;
    }
    if (!symmetry->equals(*o->symmetry)) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!(*factors[i] == *o->factors[i])) {
            return false;
        }
    }
    return true;
}

Leg::Ptr
TensorProduct::operator[](int64 idx) const
{
    return factors[static_cast<std::size_t>(to_valid_idx(idx, num_factors))];
}

std::string
TensorProduct::repr(bool show_symmetry, bool one_line) const
{
    // --- hints from Python TensorProduct.__repr__ ---
    // there is no chance to print all sectors in one line
    // populate two lists; one intended for single line, one for multiline
    // try one line
    // try multi line
    // one of the above returns should have triggered
    // ---
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "TensorProduct";

    struct Options
    {
        bool full_sectors;
        bool summarized_sectors;
        bool show_all_factors;
        bool symmetry;
    };
    std::array<Options, 6> const options{ { { true, false, true, show_symmetry },
                                            { false, true, true, show_symmetry },
                                            { true, false, false, show_symmetry },
                                            { false, true, false, show_symmetry },
                                            { false, false, false, show_symmetry },
                                            { false, false, false, false } } };
    for (auto const& opt : options) {
        if (opt.full_sectors && 3 * static_cast<int64>(sector_decomposition.size()) *
                                    static_cast<int64>(sector_decomposition.sector_ind_len()) >
                                  linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        // populate two lists; one intended for single line, one for multiline
        std::vector<std::string> one_line_items;
        std::vector<std::string> lines{ ClsName + "(" };
        if (opt.symmetry) {
            one_line_items.push_back(std::format("symmetry={}", symmetry->repr()));
            lines.push_back(std::format("{}symmetry={},", indent, symmetry->repr()));
        }
        if (opt.show_all_factors) {
            std::vector<std::string> reprs;
            reprs.reserve(factors.size());
            for (auto const& factor : factors) {
                reprs.push_back(
                  factor_repr(py::cast(factor), /*show_symmetry=*/false, /*one_line=*/true));
            }
            one_line_items.push_back(std::format("factors=[{}]", join(reprs, ", ")));
            lines.push_back(std::format("{}factors=[", indent));
            for (auto const& r : reprs) {
                lines.push_back(std::format("{}{}{},", indent, indent, r));
            }
            lines.push_back(std::format("{}],", indent));
        } else {
            one_line_items.push_back(std::format("num_factors={}", num_factors));
            lines.push_back(std::format("{}num_factors={},", indent, num_factors));
        }
        if (opt.full_sectors) {
            py::list sector_strs;
            for (auto const& a : sector_decomposition) {
                sector_strs.append(symmetry->sector_str(a));
            }
            std::vector<std::string> const new_items{
                std::format("sector_decomposition={}", format_like_list(sector_strs)),
                std::format("multiplicities={}", format_like_list(py::cast(multiplicities)))
            };
            one_line_items.insert(one_line_items.end(), new_items.begin(), new_items.end());
            for (auto const& item : new_items) {
                lines.push_back(indent + item + ",");
            }
        }
        if (opt.summarized_sectors) {
            one_line_items.push_back(std::format("num_sectors={}", num_sectors));
            lines.push_back(std::format("{}num_sectors={},", indent, num_sectors));
        }
        lines.emplace_back(")");

        // try one line
        auto const res = std::format("{}({})", ClsName, join(one_line_items, ", "));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(lines, [&](std::string const& l) {
                return static_cast<int64>(l.size()) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                return join(lines, "\n");
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("TensorProduct repr: no suitable format found");
}

std::pair<SectorArray, std::vector<int64>>
TensorProduct::calc_sectors(std::vector<Leg::Ptr> const& factors_) const
{
    // --- hints from Python TensorProduct._calc_sectors ---
    // OPTIMIZE is this optimal? should we store the f.as_Space() for later use?
    // ---
    return calc_sectors_of_factors(*symmetry, factors_);
}

void
TensorProduct::save_hdf5(lasyten::hdf5::Saver& saver,
                         HighFive::Group& /*h5gr*/,
                         std::string const& subpath) const
{
    HighFive::Group factor_g;
    std::string factor_sub;
    saver.save_sequence_begin(subpath + "factors",
                              hdf5_io::REPR_LIST,
                              static_cast<std::int64_t>(factors.size()),
                              factor_g,
                              factor_sub);
    lasyten::hdf5::Saver factor_saver(factor_g);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        hdf5_export::save_leg(factor_saver, std::to_string(i), factors[i]);
    }
    if (slices) {
        std::vector<std::int64_t> flat;
        flat.reserve(slices->size() * 2);
        for (auto const& sl : *slices) {
            flat.push_back(sl[0]);
            flat.push_back(sl[1]);
        }
        saver.save_array(subpath + "slices",
                         hdf5_export::i64_matrix_to_buffer(flat, slices->size(), 2));
    } else {
        saver.save_none(subpath + "slices");
    }
    hdf5_export::save_symmetry(saver, subpath + "symmetry", symmetry);
    saver.save_int64(subpath + "num_sectors", static_cast<std::int64_t>(num_sectors));
    saver.save_int64(subpath + "num_factors", static_cast<std::int64_t>(num_factors));
    hdf5_export::save_sector_array(saver, subpath + "sector_decomposition", sector_decomposition);
    if (sector_order) {
        saver.save_string(subpath + "sector_order", *sector_order);
    } else {
        saver.save_none(subpath + "sector_order");
    }
    if (std::floor(dim) == dim) {
        saver.save_int64(subpath + "dim", static_cast<std::int64_t>(dim));
    } else {
        saver.save_float64(subpath + "dim", dim);
    }
    hdf5_export::save_i64_vector(saver, subpath + "multiplicities", multiplicities);
    if (sector_dims) {
        hdf5_export::save_i64_vector(saver, subpath + "sector_dims", *sector_dims);
    } else {
        saver.save_none(subpath + "sector_dims");
    }
}

TensorProduct::Ptr
TensorProduct::from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath)
{
    auto symmetry = hdf5_export::load_symmetry(loader, subpath + "symmetry");
    hid_t factors_id = loader.open(subpath + "factors");
    auto const n = hdf5_io::h5_get_attr_int64(factors_id, hdf5_io::ATTR_LEN).value_or(0);
    HighFive::Group factors_g = hdf5_io::group_from_hid(factors_id);
    lasyten::hdf5::Loader factors_loader(factors_g);
    std::vector<Leg::Ptr> factors;
    factors.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        factors.push_back(hdf5_export::load_leg(factors_loader, std::to_string(i)));
    }
    auto sector_decomposition =
      hdf5_export::load_sector_array(loader, subpath + "sector_decomposition");
    auto multiplicities = hdf5_export::load_i64_vector(loader, subpath + "multiplicities");
    auto obj = std::make_shared<TensorProduct>(std::move(factors),
                                               std::move(symmetry),
                                               std::move(sector_decomposition),
                                               std::move(multiplicities));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

/// Entry ``n`` of row ``m`` of ``lasyten.tools.misc.make_grid(shape, cstyle)``.
///
/// `strides` must be ``make_stride(shape, cstyle)``. Since the grid enumerates all multi-indices
/// exactly once, the row index is just the flat index for those strides.
[[nodiscard]] int64
grid_entry(int64 m,
           std::vector<int64> const& strides,
           std::vector<int64> const& shape,
           std::size_t n)
{
    return (m / strides[n]) % shape[n];
}

/// The legs of an :class:`AbelianLegPipe` must all be :class:`ElementarySpace`\ s.
[[nodiscard]] ElementarySpace::Ptr
as_es_leg(Leg::Ptr const& leg)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(leg);
    if (!es) {
        throw py::type_error("The legs of an AbelianLegPipe must be ElementarySpaces.");
    }
    return es;
}

} // namespace

AbelianLegPipe::Prepared
AbelianLegPipe::prepare(std::vector<ElementarySpace::Ptr> const& legs,
                        bool is_dual,
                        bool combine_cstyle)
{
    // --- hints from Python AbelianLegPipe._calc_sectors ---
    // number of blocks in pipe = np.product(legs_num_sectors)
    // this is different from num_sectors
    // possible combinations of indices
    // advanced indexing:
    // ``grid.T[li]`` is a 1D array containing the block_indices `b_li` of leg ``li`` for all
    // blocks the above are the future self.sector_decomposition but we want to compute (and in
    // particular sort according to) the defining_sectors start with 0 include len, to index slices
    // now exclude len, to index sectors by diffs
    // not for the first entry => np.cumsum starts with 0
    // calculate the slices within blocks: subtract the start of each block
    // ---
    if (legs.empty()) {
        throw std::invalid_argument("Need at least one leg");
    }
    auto symmetry = legs.front()->Space::symmetry;
    if (!symmetry->is_abelian() || !symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("AbelianLegPipe is not supported for {}.", symmetry->str()));
    }
    auto const num_legs = legs.size();

    std::vector<int64> legs_num_sectors(num_legs);
    float64 dim = 1.;
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_num_sectors[n] = legs[n]->num_sectors;
        dim *= legs[n]->Space::dim;
    }
    auto sector_strides = make_stride(legs_num_sectors, combine_cstyle);

    // number of blocks in the pipe, ``prod(legs_num_sectors)``. Different from num_sectors.
    int64 nblocks = 1;
    for (auto const num : legs_num_sectors) {
        nblocks *= num;
    }
    auto const num_blocks = static_cast<std::size_t>(nblocks);

    // determine block_ind_map -- it's essentially the grid.
    // block_ind_map[:, :2] and [:, -1] are set later.
    BlockInds block_ind_map = BlockInds::zeros(num_blocks, 3 + num_legs);
    // the multiplicity for given (i1, i2, ...) is the product of ``multiplicities[il]``
    std::vector<int64> multiplicities(num_blocks, 1);
    std::vector<SectorArray> uncoupled;
    uncoupled.reserve(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        SectorArray column(num_blocks, symmetry->sector_ind_len);
        for (std::size_t m = 0; m < num_blocks; ++m) {
            auto const i = static_cast<std::size_t>(
              grid_entry(static_cast<int64>(m), sector_strides, legs_num_sectors, n));
            block_ind_map(m, 2 + n) = static_cast<int64>(i);
            multiplicities[m] *= legs[n]->multiplicities[i];
            column[m] = legs[n]->sector_decomposition[i];
        }
        uncoupled.push_back(std::move(column));
    }

    // calculate new defining_sectors. At this point, they have duplicates and are not sorted.
    auto sectors = symmetry->multiple_fusion_broadcast(uncoupled);
    if (is_dual) {
        // the above are the future sector_decomposition, but we want to compute
        // (and in particular sort according to) the defining_sectors
        sectors = symmetry->dual_sectors(sectors);
    }

    // sort sectors
    auto const sort = sectors.lexsort_indices();
    std::vector<int64> fusion_outcomes_sort(sort.begin(), sort.end());
    {
        std::vector<int64> sorted_mults(num_blocks);
        for (std::size_t m = 0; m < num_blocks; ++m) {
            sorted_mults[m] = multiplicities[sort[m]];
        }
        block_ind_map = block_ind_map.take(sort);
        multiplicities = std::move(sorted_mults);
        sectors = sectors.take(sort);
    }

    // compute slices in the whole internal basis (we subtract the start of each block below)
    auto const slices = slice_boundaries(multiplicities);
    for (std::size_t m = 0; m < num_blocks; ++m) {
        block_ind_map(m, 0) = slices[m];
        block_ind_map(m, 1) = slices[m + 1];
    }

    // bunch sectors with equal sectors together
    auto const diffs = sectors.find_row_differences(/*include_len=*/true);
    std::vector<int64> block_ind_map_slices(diffs.begin(), diffs.end());
    auto const num_unique = diffs.size() - 1;
    std::vector<int64> block_starts(diffs.size());
    for (std::size_t k = 0; k < diffs.size(); ++k) {
        block_starts[k] = slices[diffs[k]];
    }
    std::vector<int64> unique_mults(num_unique);
    for (std::size_t k = 0; k < num_unique; ++k) {
        unique_mults[k] = block_starts[k + 1] - block_starts[k];
    }
    // [:-1] to exclude len
    auto unique_sectors =
      sectors.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));

    // the new block index J, plus the slices within blocks (subtract the start of each block)
    for (std::size_t k = 0; k < num_unique; ++k) {
        for (std::size_t m = diffs[k]; m < diffs[k + 1]; ++m) {
            block_ind_map(m, 2 + num_legs) = static_cast<int64>(k);
            block_ind_map(m, 0) -= block_starts[k];
            block_ind_map(m, 1) -= block_starts[k];
        }
    }

    auto basis_perm = calc_basis_perm(legs, combine_cstyle, dim, unique_mults, block_ind_map);

    std::vector<Leg::Ptr> leg_ptrs(legs.begin(), legs.end());
    return { std::move(leg_ptrs),
             std::move(symmetry),
             std::move(unique_sectors),
             std::move(unique_mults),
             std::move(basis_perm),
             std::move(sector_strides),
             std::move(fusion_outcomes_sort),
             std::move(block_ind_map_slices),
             std::move(block_ind_map) };
}

// note: the LegPipe base sets a combined basis_perm, which the ElementarySpace base then
// overwrites with the fusion basis_perm. This matches the order of the Python constructor.
AbelianLegPipe::AbelianLegPipe(Prepared prepared, bool is_dual_, bool combine_cstyle_)
  : LegPipe(prepared.legs, is_dual_, combine_cstyle_)
  , ElementarySpace(prepared.symmetry,
                    prepared.defining_sectors,
                    prepared.multiplicities,
                    is_dual_,
                    prepared.basis_perm)
  , sector_strides(std::move(prepared.sector_strides))
  , fusion_outcomes_sort(std::move(prepared.fusion_outcomes_sort))
  , block_ind_map_slices(std::move(prepared.block_ind_map_slices))
  , block_ind_map(std::move(prepared.block_ind_map))
{
}

AbelianLegPipe::AbelianLegPipe(std::vector<ElementarySpace::Ptr> legs_,
                               bool is_dual_,
                               bool combine_cstyle_)
  : AbelianLegPipe(prepare(legs_, is_dual_, combine_cstyle_), is_dual_, combine_cstyle_)
{
    // --- hints from Python AbelianLegPipe.__init__ ---
    // also sets some attributes
    // ---
}

std::vector<int64>
AbelianLegPipe::fusion_outcomes_perm(std::vector<ElementarySpace::Ptr> const& legs,
                                     bool combine_cstyle,
                                     float64 dim,
                                     std::vector<int64> const& multiplicities,
                                     BlockInds const& block_ind_map)
{
    auto const num_legs = legs.size();
    std::vector<int64> legs_dims(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_dims[n] = static_cast<int64>(legs[n]->Space::dim);
    }
    auto const dim_strides = make_stride(legs_dims, combine_cstyle);
    std::vector<int64> perm(dim_as_size(dim));

    // slices_starts is slices[:, 0], but we need to compute it here, since the
    // ElementarySpace base may not be initialized yet at this point
    auto const slices_starts = slice_boundaries(multiplicities);

    std::vector<int64> mult_shape(num_legs);
    std::vector<int64> sector_starts(num_legs);
    for (std::size_t m = 0; m < block_ind_map.nrows(); ++m) {
        auto const row = block_ind_map.row(m);
        // shift the slice start:stop from within the block back to the whole internal basis
        auto const J = static_cast<std::size_t>(row[2 + num_legs]);
        auto const start = row[0] + slices_starts[J];

        // Now for each basis element in start:stop, we construct where it was before sorting.
        // multiplicity_grid :: each row stands for a combination of uncoupled basis elements;
        //                     they are the indices of that basis element *within* the sector.
        // sector_starts[n] is the index of the first basis vector for legs[n] that is in the
        // current sector, namely legs[n].sector_decomposition[idcs[n]]
        int64 count = 1;
        for (std::size_t n = 0; n < num_legs; ++n) {
            auto const idx = static_cast<std::size_t>(row[2 + n]);
            mult_shape[n] = legs[n]->multiplicities[idx];
            sector_starts[n] = (*legs[n]->slices)[idx][0];
            count *= mult_shape[n];
        }
        assert(count == row[1] - row[0]);
        auto const mult_strides = make_stride(mult_shape, combine_cstyle);
        // basis_grid :: each row stands for a combination of uncoupled basis elements; they are
        //               the indices of that basis element within its legs internal basis.
        // Note that the relevant strides are ``dim_strides``, which come from a *different*
        // shape than the multiplicity_grid.
        for (int64 k = 0; k < count; ++k) {
            int64 flat = 0;
            for (std::size_t n = 0; n < num_legs; ++n) {
                flat +=
                  (grid_entry(k, mult_strides, mult_shape, n) + sector_starts[n]) * dim_strides[n];
            }
            perm[static_cast<std::size_t>(start + k)] = flat;
        }
    }
    return perm;
}

std::vector<int64>
AbelianLegPipe::calc_basis_perm(std::vector<ElementarySpace::Ptr> const& legs,
                                bool combine_cstyle,
                                float64 dim,
                                std::vector<int64> const& multiplicities,
                                BlockInds const& block_ind_map)
{
    // --- hints from Python AbelianLegPipe._calc_basis_perm ---
    // see diagram in docstring, we follow the path parallel to ``pipe.basis_perm``.
    // apply basis perm of each leg
    // apply fusion_outcomes_perm (``sort`` in the diagram)
    // ---
    // see the diagram in the docstring of the Python ``_calc_basis_perm``; we follow the path
    // parallel to ``pipe.basis_perm``: inverse of fusion, basis_perm of each leg, fusion, sort.
    auto const num_legs = legs.size();
    std::vector<int64> legs_dims(num_legs);
    std::vector<std::vector<int64>> perms(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_dims[n] = static_cast<int64>(legs[n]->Space::dim);
        perms[n] = legs[n]->basis_perm();
    }
    auto const dim_strides = make_stride(legs_dims, combine_cstyle);
    auto const num_basis_states = dim_as_size(dim);

    // ``np.reshape(np.arange(dim), dims, order)[np.ix_(*perms)].reshape(dim, order)``
    std::vector<int64> combined(num_basis_states);
    for (std::size_t m = 0; m < num_basis_states; ++m) {
        int64 flat = 0;
        for (std::size_t n = 0; n < num_legs; ++n) {
            auto const i = static_cast<std::size_t>(
              grid_entry(static_cast<int64>(m), dim_strides, legs_dims, n));
            flat += perms[n][i] * dim_strides[n];
        }
        combined[m] = flat;
    }

    auto const fusion_perm =
      fusion_outcomes_perm(legs, combine_cstyle, dim, multiplicities, block_ind_map);
    std::vector<int64> res(num_basis_states);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        res[i] = combined[static_cast<std::size_t>(fusion_perm[i])];
    }
    return res;
}

std::vector<ElementarySpace::Ptr>
AbelianLegPipe::es_legs() const
{
    std::vector<ElementarySpace::Ptr> out;
    out.reserve(legs.size());
    for (auto const& leg : legs) {
        out.push_back(as_es_leg(leg));
    }
    return out;
}

std::vector<int64>
AbelianLegPipe::get_fusion_outcomes_perm(std::vector<int64> const& multiplicities_) const
{
    // --- hints from Python AbelianLegPipe._get_fusion_outcomes_perm ---
    // since ElementarySpace.__init__ was not called yet at this point
    // shift the slice start:stop from within the block back to within the whole internal basis
    // they are the indices of that basis element within its legs internal basis
    // now we need to map the multi-indices (rows of basis_grid) to single indices into
    // the unsorted list of fusion outcomes. Note that the relevant strides are ``dim_strides``,
    // and that these strides come from a *different* shape than the multiplicity_grid.
    // That is, we want to do ``perm[start + n] = np.sum(basis_grid[n] * dim_strides)``.
    // Turns out we can do it batched:
    // ---
    return fusion_outcomes_perm(
      es_legs(), combine_cstyle, Space::dim, multiplicities_, block_ind_map);
}

void
AbelianLegPipe::test_sanity() const
{
    // --- hints from Python AbelianLegPipe.test_sanity ---
    // check self.sector_strides
    // C style grid -> lexsorted after reversing column order (see notes)
    // F style grid -> is lexsorted
    // ---
    auto const es = es_legs();
    for (auto const& leg : es) {
        if (auto const* nested = dynamic_cast<LegPipe const*>(leg.get()); nested != nullptr) {
            assert(nested->is_abelian_leg_pipe());
        }
        leg->test_sanity();
    }
    auto const n = static_cast<std::size_t>(num_legs);
    // check sector_strides
    assert(sector_strides.size() == n);
    std::vector<int64> legs_num_sectors(n);
    int64 nblocks = 1;
    for (std::size_t i = 0; i < n; ++i) {
        legs_num_sectors[i] = es[i]->num_sectors;
        nblocks *= legs_num_sectors[i];
    }
    assert(sector_strides == make_stride(legs_num_sectors, combine_cstyle));
    // check block_ind_map_slices
    // note: we do not check for full correctness, just for consistency as slices
    assert(block_ind_map_slices.size() == static_cast<std::size_t>(num_sectors) + 1);
    assert(block_ind_map_slices.front() == 0);
    assert(block_ind_map_slices.back() == nblocks);
    assert(std::ranges::is_sorted(block_ind_map_slices));
    // check block_ind_map
    assert(block_ind_map.nrows() == static_cast<std::size_t>(nblocks));
    assert(block_ind_map.ncols() == 3 + n);
    // the rows are sorted first by J, then by the i, in C-style order if combine_cstyle
    // (see the class docstring). Equivalently, the keys built below are non-decreasing.
    auto const sort_key = [&](std::size_t m) {
        std::vector<int64> key{ block_ind_map(m, 2 + n) };
        for (std::size_t i = 0; i < n; ++i) {
            key.push_back(combine_cstyle ? block_ind_map(m, 2 + i) : block_ind_map(m, 1 + n - i));
        }
        return key;
    };
    for (std::size_t m = 0; m < block_ind_map.nrows(); ++m) {
        auto const J = static_cast<std::size_t>(block_ind_map(m, 2 + n));
        if (m > 0) {
            auto const prev = sort_key(m - 1);
            auto const cur = sort_key(m);
            assert(std::ranges::lexicographical_compare(prev, cur));
        }
        if (m > 0 && block_ind_map(m, 2 + n) == block_ind_map(m - 1, 2 + n)) {
            assert(block_ind_map(m, 0) == block_ind_map(m - 1, 1));
        } else {
            assert(block_ind_map(m, 0) == 0);
        }
        std::vector<Sector> uncoupled(n);
        for (std::size_t i = 0; i < n; ++i) {
            uncoupled[i] =
              es[i]->sector_decomposition[static_cast<std::size_t>(block_ind_map(m, 2 + i))];
        }
        assert(Space::symmetry->multiple_fusion(uncoupled) == sector_decomposition[J]);
    }
    // call to super class(es)
    LegPipe::test_sanity();
    ElementarySpace::test_sanity();
}

Space::Ptr
AbelianLegPipe::as_space_obj()
{
    return shared_es();
}

ElementarySpace::Ptr
AbelianLegPipe::as_ElementarySpace(bool is_dual_)
{
    return with_is_dual(is_dual_);
}

Space::Ptr
AbelianLegPipe::dual_space() const
{
    return dual_pipe();
}

Leg::Ptr
AbelianLegPipe::dual_leg() const
{
    return dual_pipe();
}

AbelianLegPipe::Ptr
AbelianLegPipe::dual_pipe() const
{
    std::vector<ElementarySpace::Ptr> dual_legs;
    dual_legs.reserve(legs.size());
    for (auto it = legs.rbegin(); it != legs.rend(); ++it) {
        dual_legs.push_back(as_es_leg((*it)->dual_leg()));
    }
    return std::make_shared<AbelianLegPipe>(std::move(dual_legs), !is_dual, !combine_cstyle);
}

bool
AbelianLegPipe::is_trivial() const
{
    return ElementarySpace::is_trivial();
}

std::vector<Leg::Ptr>
AbelianLegPipe::flat_spaces()
{
    // --- hints from Python AbelianLegPipe.flat_spaces ---
    // Unlike the plain LegPipe, we do not need to flatten AbelianLegPipes, if we just
    // want to flatten until we get spaces
    // ---
    // Unlike the plain LegPipe, we do not need to flatten AbelianLegPipes, if we just
    // want to flatten until we get spaces
    return { shared_leg() };
}

std::string
AbelianLegPipe::ascii_arrow() const
{
    // ``Leg.ascii_arrow`` in Python: a filled arrow for a pipe that is also an ElementarySpace
    return is_dual ? "▲" : "▼";
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_independent_symmetries(std::vector<Ptr> const& independent_descriptions)
{
    if (independent_descriptions.empty()) {
        throw std::invalid_argument(
          "from_independent_symmetries requires at least one description");
    }
    auto const is_dual = independent_descriptions.front()->is_dual;
    if (!std::ranges::all_of(independent_descriptions,
                             [is_dual](Ptr const& i) { return i->is_dual == is_dual; })) {
        throw std::invalid_argument("independent descriptions must have matching duality");
    }
    auto const num_legs = independent_descriptions.front()->num_legs;
    if (!std::ranges::all_of(independent_descriptions,
                             [num_legs](Ptr const& i) { return i->num_legs == num_legs; })) {
        throw std::invalid_argument("independent descriptions must have the same number of legs");
    }
    std::vector<ElementarySpace::Ptr> legs;
    legs.reserve(static_cast<std::size_t>(num_legs));
    for (std::size_t k = 0; k < static_cast<std::size_t>(num_legs); ++k) {
        std::vector<ElementarySpace::Ptr> group;
        std::vector<Ptr> pipes;
        group.reserve(independent_descriptions.size());
        for (auto const& description : independent_descriptions) {
            auto leg = as_es_leg(description->legs[k]);
            if (auto pipe = std::dynamic_pointer_cast<AbelianLegPipe>(leg)) {
                pipes.push_back(std::move(pipe));
            }
            group.push_back(std::move(leg));
        }
        if (pipes.size() == group.size()) {
            legs.push_back(from_independent_symmetries(pipes));
        } else {
            legs.push_back(ElementarySpace::from_independent_symmetries(group));
        }
    }
    return std::make_shared<AbelianLegPipe>(std::move(legs), is_dual);
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_basis(Symmetry::Ptr /*symmetry*/, SectorArray /*sectors_of_basis*/)
{
    throw py::type_error("from_basis is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_null_space(Symmetry::Ptr /*symmetry*/, bool /*is_dual*/)
{
    throw py::type_error("from_null_space is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_defining_sectors(Symmetry::Ptr /*symmetry*/,
                                      SectorArray /*defining_sectors*/,
                                      std::optional<std::vector<int64>> /*multiplicities*/,
                                      bool /*is_dual*/,
                                      std::optional<std::vector<int64>> /*basis_perm*/,
                                      bool /*unique_sectors*/,
                                      std::vector<std::size_t>* /*return_sorting_perm*/)
{
    throw py::type_error("from_defining_sectors is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_trivial_sector(int64 /*dim*/,
                                    Symmetry::Ptr /*symmetry*/,
                                    bool /*is_dual*/,
                                    std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("from_trivial_sector is not supported for AbelianLegPipe");
}

Space::Ptr
AbelianLegPipe::change_symmetry(Symmetry::Ptr symmetry_, SectorMapFn sector_map, bool injective)
{
    std::vector<ElementarySpace::Ptr> new_legs;
    new_legs.reserve(legs.size());
    for (auto const& leg : es_legs()) {
        new_legs.push_back(std::dynamic_pointer_cast<ElementarySpace>(
          leg->change_symmetry(symmetry_, sector_map, injective)));
    }
    return std::make_shared<AbelianLegPipe>(std::move(new_legs), is_dual, combine_cstyle);
}

Space::Ptr
AbelianLegPipe::drop_symmetry(std::optional<std::vector<int64>> which)
{
    // --- hints from Python AbelianLegPipe.drop_symmetry ---
    // OPTIMIZE can we avoid recomputation of fusion?
    // ---
    // OPTIMIZE can we avoid recomputation of fusion?
    std::vector<ElementarySpace::Ptr> new_legs;
    new_legs.reserve(legs.size());
    for (auto const& leg : es_legs()) {
        new_legs.push_back(std::dynamic_pointer_cast<ElementarySpace>(leg->drop_symmetry(which)));
    }
    return std::make_shared<AbelianLegPipe>(std::move(new_legs), is_dual, combine_cstyle);
}

void
AbelianLegPipe::set_basis_perm(std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for AbelianLegPipe.");
}

void
AbelianLegPipe::set_inverse_basis_perm(std::optional<std::vector<int64>> /*inverse_basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for AbelianLegPipe.");
}

ElementarySpace::Ptr
AbelianLegPipe::take_slice(py::array blockmask) const
{
    warn("Using `AbelianLegPipe.take_slice` loses the product (pipe) structure and results in "
         "a plain ElementarySpace. Explicitly convert using `as_ElementarySpace` to suppress "
         "this warning.");
    // note: unlike the Python version, we call the ElementarySpace implementation directly.
    // Python goes through ``as_ElementarySpace(is_dual=self.is_dual)``, which returns ``self``
    // and therefore recurses infinitely.
    return ElementarySpace::take_slice(std::move(blockmask));
}

ElementarySpace::Ptr
AbelianLegPipe::with_opposite_duality() const
{
    return std::make_shared<AbelianLegPipe>(es_legs(), !is_dual, combine_cstyle);
}

bool
AbelianLegPipe::operator==(Leg const& other) const
{
    // note: LegPipe::operator== already compares combine_cstyle and checks that both sides
    // are (not) AbelianLegPipes.
    return LegPipe::operator==(other);
}

bool
AbelianLegPipe::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<LegPipe const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return LegPipe::operator==(*o);
}

std::string
AbelianLegPipe::repr(bool show_symmetry, bool one_line) const
{
    // --- hints from Python AbelianLegPipe.__repr__ ---
    // sector_mode:  0=show full arrays , 1=show only nums, 2=dont show
    // child_mode: 0=show full , 1=force one-line each, 2=show only num
    // summarize_basis_perm: bool
    // this should not happen
    // dont add anything
    // ---
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "AbelianLegPipe";

    struct Options
    {
        /// 0=show full arrays, 1=show only nums, 2=dont show
        int sector_mode;
        /// 0=show full, 1=force one-line each, 2=show only num
        int child_mode;
        bool summarize_basis_perm;
        bool symmetry;
    };
    std::array<Options, 7> const options{ { { 0, 0, false, show_symmetry },
                                            { 0, 0, true, show_symmetry },
                                            { 0, 1, true, show_symmetry },
                                            { 0, 2, true, show_symmetry },
                                            { 1, 2, true, show_symmetry },
                                            { 2, 2, true, show_symmetry },
                                            { 2, 2, true, false } } };
    for (auto const& opt : options) {
        if (opt.sector_mode == 0 && 3 * static_cast<int64>(sector_decomposition.size()) *
                                        static_cast<int64>(sector_decomposition.sector_ind_len()) >
                                      linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        // populate two lists; one intended for single line, one for multiline.
        // this is because lines behaves differently when dealing with the children / legs
        std::vector<std::string> one_line_items;
        std::vector<std::string> lines{ ClsName + "(" };

        if (opt.symmetry) {
            one_line_items.push_back(std::format("symmetry={}", Space::symmetry->repr()));
            lines.push_back(std::format("{}symmetry={},", indent, Space::symmetry->repr()));
        }

        if (opt.child_mode < 2) {
            std::vector<std::string> reprs;
            reprs.reserve(legs.size());
            for (auto const& leg : legs) {
                reprs.push_back(factor_repr(
                  py::cast(leg), /*show_symmetry=*/false, /*one_line=*/opt.child_mode > 0));
            }
            one_line_items.push_back(std::format("factors=[{}]", join(reprs, ", ")));
            lines.push_back(std::format("{}factors=[", indent));
            for (auto const& r : reprs) {
                lines.push_back(std::format("{}{}{},", indent, indent, r));
            }
            lines.push_back(std::format("{}],", indent));
        } else {
            one_line_items.push_back(std::format("num_legs={}", num_legs));
            lines.push_back(std::format("{}num_legs={},", indent, num_legs));
        }

        if (opt.sector_mode == 0) {
            py::list sector_dec_strs;
            for (auto const& a : sector_decomposition) {
                sector_dec_strs.append(Space::symmetry->sector_str(a));
            }
            py::list def_sector_strs;
            for (auto const& a : defining_sectors) {
                def_sector_strs.append(Space::symmetry->sector_str(a));
            }
            std::vector<std::string> const new_items{
                std::format("sector_decomposition={}", format_like_list(sector_dec_strs)),
                std::format("defining_sectors={}", format_like_list(def_sector_strs)),
                std::format("multiplicities={}", format_like_list(py::cast(multiplicities)))
            };
            one_line_items.insert(one_line_items.end(), new_items.begin(), new_items.end());
            for (auto const& item : new_items) {
                lines.push_back(indent + item + ",");
            }
        } else if (opt.sector_mode == 1) {
            one_line_items.push_back(std::format("num_sectors={}", num_sectors));
            lines.push_back(std::format("{}num_sectors={},", indent, num_sectors));
        }

        if (_basis_perm) {
            if (opt.summarize_basis_perm) {
                one_line_items.emplace_back("basis_perm=[...]");
                lines.push_back(std::format("{}basis_perm=[...],", indent));
            } else {
                auto const perm = format_like_list(py::cast(*_basis_perm));
                one_line_items.push_back(std::format("basis_perm={}", perm));
                lines.push_back(std::format("{}basis_perm={},", indent, perm));
            }
        }

        one_line_items.push_back(std::format("is_dual={}", bool_repr(is_dual)));
        lines.push_back(std::format("{}is_dual={},", indent, bool_repr(is_dual)));
        lines.emplace_back(")");

        // try one line
        auto const res = std::format("{}({})", ClsName, join(one_line_items, ", "));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(lines, [&](std::string const& l) {
                return static_cast<int64>(l.size()) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                return join(lines, "\n");
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("AbelianLegPipe repr: no suitable format found");
}

void
AbelianLegPipe::save_hdf5(lasyten::hdf5::Saver& saver,
                          HighFive::Group& h5gr,
                          std::string const& subpath) const
{
    ElementarySpace::save_hdf5(saver, h5gr, subpath);
    HighFive::Group leg_g;
    std::string leg_sub;
    saver.save_sequence_begin(subpath + "legs",
                              hdf5_io::REPR_LIST,
                              static_cast<std::int64_t>(legs.size()),
                              leg_g,
                              leg_sub);
    lasyten::hdf5::Saver leg_saver(leg_g);
    for (std::size_t i = 0; i < legs.size(); ++i) {
        hdf5_export::save_leg(leg_saver, std::to_string(i), legs[i]);
    }
    hdf5_io::h5_set_attr(h5gr.getId(), "combine_cstyle", combine_cstyle);
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_hdf5(lasyten::hdf5::Loader& loader,
                          HighFive::Group& h5gr,
                          std::string const& subpath)
{
    hid_t legs_id = loader.open(subpath + "legs");
    auto const n = hdf5_io::h5_get_attr_int64(legs_id, hdf5_io::ATTR_LEN).value_or(0);
    HighFive::Group legs_g = hdf5_io::group_from_hid(legs_id);
    lasyten::hdf5::Loader legs_loader(legs_g);
    std::vector<ElementarySpace::Ptr> legs;
    legs.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        legs.push_back(hdf5_export::load_elementary_space(legs_loader, std::to_string(i)));
    }
    auto const is_dual_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "is_dual");
    bool const is_dual = is_dual_attr.value_or(0) != 0;
    auto const combine_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "combine_cstyle");
    bool const combine_cstyle = combine_attr.value_or(0) != 0;
    auto obj = std::make_shared<AbelianLegPipe>(std::move(legs), is_dual, combine_cstyle);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

[[nodiscard]] bool
is_plain_leg_pipe(Leg::Ptr const& leg)
{
    // Python: ``not isinstance(leg, ElementarySpace) and isinstance(leg, LegPipe)``.
    // AbelianLegPipe is both, so it takes the ElementarySpace path.
    return static_cast<bool>(std::dynamic_pointer_cast<LegPipe>(leg)) &&
           !static_cast<bool>(std::dynamic_pointer_cast<ElementarySpace>(leg));
}

[[nodiscard]] std::size_t
leg_dim_as_size(Leg const& leg)
{
    if (!(leg.dim >= 0.) || std::floor(leg.dim) != leg.dim) {
        throw std::invalid_argument(
          std::format("leg dimension must be a non-negative integer, got {}", leg.dim));
    }
    return static_cast<std::size_t>(leg.dim);
}

/// Dense ND complex buffer for LegPipe swap_gate composition (rank may exceed FusionSymbol's 4).
struct NdComplex
{
    std::vector<std::size_t> shape;
    std::vector<complex128> data; // C-order

    [[nodiscard]] std::size_t size() const
    {
        std::size_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        return n;
    }

    [[nodiscard]] std::size_t offset(std::vector<std::size_t> const& idx) const
    {
        std::size_t off = 0;
        for (std::size_t a = 0; a < shape.size(); ++a) {
            off = off * shape[a] + idx[a];
        }
        return off;
    }

    [[nodiscard]] complex128 get(std::vector<std::size_t> const& idx) const
    {
        return data[offset(idx)];
    }

    void set(std::vector<std::size_t> const& idx, complex128 v) { data[offset(idx)] = v; }
};

[[nodiscard]] NdComplex
fusion_symbol_to_nd(FusionSymbol const& src)
{
    NdComplex out;
    out.shape.assign(src.shape().begin(), src.shape().begin() + src.rank());
    out.data.resize(src.size());
    auto c = src.as_complex();
    auto span = c.as_complex128();
    std::copy(span.begin(), span.end(), out.data.begin());
    return out;
}

[[nodiscard]] FusionSymbol
nd_to_fusion_symbol(NdComplex const& src)
{
    if (src.shape.size() < 1 || src.shape.size() > 4) {
        throw std::invalid_argument("nd_to_fusion_symbol: rank must be 1..4");
    }
    FusionSymbol::Shape shape{ { 1, 1, 1, 1 } };
    for (std::size_t i = 0; i < src.shape.size(); ++i) {
        shape[i] = src.shape[i];
    }
    bool any_imag = false;
    for (auto v : src.data) {
        if (v.imag() != 0.) {
            any_imag = true;
            break;
        }
    }
    if (any_imag) {
        return FusionSymbol::from_complex128(
          static_cast<std::uint8_t>(src.shape.size()), shape, src.data);
    }
    std::vector<float64> real(src.data.size());
    for (std::size_t i = 0; i < src.data.size(); ++i) {
        real[i] = src.data[i].real();
    }
    return FusionSymbol::from_float64(
      static_cast<std::uint8_t>(src.shape.size()), shape, std::move(real));
}

[[nodiscard]] std::size_t
norm_axis(int ax, std::size_t rank)
{
    if (ax < 0) {
        ax += static_cast<int>(rank);
    }
    if (ax < 0 || static_cast<std::size_t>(ax) >= rank) {
        throw std::out_of_range("axis out of range");
    }
    return static_cast<std::size_t>(ax);
}

[[nodiscard]] NdComplex
nd_tensordot(NdComplex const& a, NdComplex const& b, std::size_t ax_a, std::size_t ax_b)
{
    if (ax_a >= a.shape.size() || ax_b >= b.shape.size()) {
        throw std::out_of_range("nd_tensordot: axis out of range");
    }
    if (a.shape[ax_a] != b.shape[ax_b]) {
        throw std::invalid_argument("nd_tensordot: contracted dimensions mismatch");
    }
    std::size_t const K = a.shape[ax_a];
    NdComplex out;
    out.shape.reserve(a.shape.size() + b.shape.size() - 2);
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
        if (i != ax_a) {
            out.shape.push_back(a.shape[i]);
        }
    }
    for (std::size_t i = 0; i < b.shape.size(); ++i) {
        if (i != ax_b) {
            out.shape.push_back(b.shape[i]);
        }
    }
    out.data.assign(out.size(), complex128{ 0., 0. });

    std::vector<std::size_t> ia(a.shape.size()), ib(b.shape.size()), io(out.shape.size());
    std::function<void(std::size_t, std::size_t, std::size_t)> rec =
      [&](std::size_t pa, std::size_t pb, std::size_t po) {
          if (pa == a.shape.size() && pb == b.shape.size()) {
              complex128 sum{ 0., 0. };
              for (std::size_t k = 0; k < K; ++k) {
                  ia[ax_a] = k;
                  ib[ax_b] = k;
                  sum += a.get(ia) * b.get(ib);
              }
              out.set(io, sum);
              return;
          }
          if (pa < a.shape.size()) {
              if (pa == ax_a) {
                  rec(pa + 1, pb, po);
                  return;
              }
              for (std::size_t i = 0; i < a.shape[pa]; ++i) {
                  ia[pa] = i;
                  io[po] = i;
                  rec(pa + 1, pb, po + 1);
              }
              return;
          }
          if (pb == ax_b) {
              rec(pa, pb + 1, po);
              return;
          }
          for (std::size_t i = 0; i < b.shape[pb]; ++i) {
              ib[pb] = i;
              io[po] = i;
              rec(pa, pb + 1, po + 1);
          }
      };
    rec(0, 0, 0);
    return out;
}

[[nodiscard]] NdComplex
nd_moveaxis(NdComplex const& src, int src_ax, int dst_ax)
{
    // Match NumPy moveaxis: normalize both axes on the original rank, remove sources
    // from order, then insert each source at the (original) destination index.
    auto const r = src.shape.size();
    auto const s = norm_axis(src_ax, r);
    auto const d = norm_axis(dst_ax, r);
    if (s == d) {
        return src;
    }
    std::vector<std::size_t> perm;
    perm.reserve(r);
    for (std::size_t n = 0; n < r; ++n) {
        if (n != s) {
            perm.push_back(n);
        }
    }
    perm.insert(perm.begin() + static_cast<std::ptrdiff_t>(d), s);

    NdComplex out;
    out.shape.resize(r);
    for (std::size_t i = 0; i < r; ++i) {
        out.shape[i] = src.shape[perm[i]];
    }
    out.data.resize(src.data.size());
    std::vector<std::size_t> idx_out(r), idx_in(r);
    std::function<void(std::size_t)> rec = [&](std::size_t axis) {
        if (axis == r) {
            for (std::size_t i = 0; i < r; ++i) {
                idx_in[perm[i]] = idx_out[i];
            }
            out.set(idx_out, src.get(idx_in));
            return;
        }
        for (std::size_t i = 0; i < out.shape[axis]; ++i) {
            idx_out[axis] = i;
            rec(axis + 1);
        }
    };
    rec(0);
    return out;
}

[[nodiscard]] NdComplex
nd_transpose(NdComplex const& src, std::vector<int> const& axes)
{
    auto const r = src.shape.size();
    if (axes.size() != r) {
        throw std::invalid_argument("nd_transpose: axes length mismatch");
    }
    std::vector<std::size_t> perm(r);
    for (std::size_t i = 0; i < r; ++i) {
        perm[i] = norm_axis(axes[i], r);
    }
    NdComplex out;
    out.shape.resize(r);
    for (std::size_t i = 0; i < r; ++i) {
        out.shape[i] = src.shape[perm[i]];
    }
    out.data.resize(src.data.size());
    std::vector<std::size_t> idx_out(r), idx_in(r);
    std::function<void(std::size_t)> rec = [&](std::size_t axis) {
        if (axis == r) {
            for (std::size_t i = 0; i < r; ++i) {
                idx_in[perm[i]] = idx_out[i];
            }
            out.set(idx_out, src.get(idx_in));
            return;
        }
        for (std::size_t i = 0; i < out.shape[axis]; ++i) {
            idx_out[axis] = i;
            rec(axis + 1);
        }
    };
    rec(0);
    return out;
}

[[nodiscard]] NdComplex
nd_reshape(NdComplex const& src, std::vector<std::size_t> new_shape, bool cstyle)
{
    std::size_t new_n = 1;
    for (auto s : new_shape) {
        new_n *= s;
    }
    if (new_n != src.size()) {
        throw std::invalid_argument("nd_reshape: size mismatch");
    }
    NdComplex out;
    out.shape = std::move(new_shape);
    out.data.resize(new_n);

    auto unravel = [](std::size_t flat, std::vector<std::size_t> const& shape, bool c_order) {
        std::vector<std::size_t> idx(shape.size());
        if (c_order) {
            for (std::size_t a = shape.size(); a-- > 0;) {
                idx[a] = flat % shape[a];
                flat /= shape[a];
            }
        } else {
            for (std::size_t a = 0; a < shape.size(); ++a) {
                idx[a] = flat % shape[a];
                flat /= shape[a];
            }
        }
        return idx;
    };
    auto ravel_c = [](std::vector<std::size_t> const& idx, std::vector<std::size_t> const& shape) {
        std::size_t flat = 0;
        for (std::size_t a = 0; a < shape.size(); ++a) {
            flat = flat * shape[a] + idx[a];
        }
        return flat;
    };

    // NumPy reshape(order=O): same element sequence when both arrays are traversed in order O.
    for (std::size_t seq = 0; seq < new_n; ++seq) {
        auto src_idx = unravel(seq, src.shape, cstyle);
        auto dst_idx = unravel(seq, out.shape, cstyle);
        out.data[ravel_c(dst_idx, out.shape)] = src.data[ravel_c(src_idx, src.shape)];
    }
    return out;
}

} // namespace

FusionSymbol
swap_gate(Leg::Ptr V, Leg::Ptr W)
{
    // --- hints from Python swap_gate ---
    // special case: pipes
    // since we call this function recursively, we do not need to distinguish if W is a pipe at
    // this point [W, Vz, W*, Vz*] [W, Vi, W*, Vi*] [W, Vi, (W*), Vi*] @ [(W), {Vs}, W*, {Vs}*] ->
    // [W, Vi, Vi*, {Vs}, W*, {Vs}*] [W, Vi, (Vi*), {Vs}, W*, {Vs}*] -> [W, Vi, {Vs}, W*, (Vi*),
    // {Vs}*] since we call this function recursively, we do not need to distinguish if V is a pipe
    // at this point [Wa, V, Wa*, V*] [Wi, V, Wi*, V]
    // [{Ws}, (V), {Ws}*, V*] @ [Wi, V, Wi*, (V*)] -> [{Ws}, {Ws*}, V*, Wi, V, Wi*]
    // [{Ws}, {Ws*}, V*, Wi, V, Wi*] -> [{Ws}, Wi, V, {Ws*}, Wi*, V*]
    // build in internal basis order, permute after
    // OPTIMIZE these loops are probably inefficient, and there may be some numpy magic that does
    // it better...
    // ---
    if (!V || !W) {
        throw py::type_error("swap_gate requires two legs");
    }
    if (!V->symmetry->equals(*W->symmetry)) {
        throw SymmetryError("Incompatible symmetries.");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("braid can not be written as array for {}.", V->symmetry->str()));
    }
    auto const dV = leg_dim_as_size(*V);
    auto const dW = leg_dim_as_size(*W);

    if (is_plain_leg_pipe(V)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(V);
        auto const& legs = pipe->legs;
        NdComplex res = fusion_symbol_to_nd(swap_gate(legs.back(), W));
        int n = 0;
        for (auto it = legs.rbegin() + 1; it != legs.rend(); ++it, ++n) {
            NdComplex sw = fusion_symbol_to_nd(swap_gate(*it, W));
            res = nd_tensordot(sw, res, /*ax_a=*/2, /*ax_b=*/0);
            res = nd_moveaxis(res, /*src=*/2, /*dst=*/-2 - n);
        }
        return nd_to_fusion_symbol(
          nd_reshape(res, { dW, dV, dW, dV }, /*cstyle=*/pipe->combine_cstyle));
    }
    if (is_plain_leg_pipe(W)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(W);
        auto const& legs = pipe->legs;
        NdComplex res = fusion_symbol_to_nd(swap_gate(V, legs.front()));
        for (std::size_t n = 1; n < legs.size(); ++n) {
            NdComplex sw = fusion_symbol_to_nd(swap_gate(V, legs[n]));
            res = nd_tensordot(res, sw, /*ax_a=*/n, /*ax_b=*/sw.shape.size() - 1);
            std::vector<int> axes;
            axes.reserve(res.shape.size());
            for (std::size_t i = 0; i < n; ++i) {
                axes.push_back(static_cast<int>(i));
            }
            axes.push_back(-3);
            axes.push_back(-2);
            for (std::size_t i = n; i < 2 * n; ++i) {
                axes.push_back(static_cast<int>(i));
            }
            axes.push_back(-1);
            axes.push_back(-4);
            res = nd_transpose(res, axes);
        }
        return nd_to_fusion_symbol(
          nd_reshape(res, { dW, dV, dW, dV }, /*cstyle=*/pipe->combine_cstyle));
    }

    auto Ves = std::dynamic_pointer_cast<ElementarySpace>(V);
    auto Wes = std::dynamic_pointer_cast<ElementarySpace>(W);
    if (!Ves || !Wes) {
        throw py::type_error("swap_gate expects ElementarySpace or LegPipe legs");
    }

    bool any_complex = false;
    for (std::size_t ia = 0; ia < Ves->defining_sectors.size() && !any_complex; ++ia) {
        for (std::size_t ib = 0; ib < Wes->defining_sectors.size(); ++ib) {
            if (!Ves->Space::symmetry
                   ->swap_gate(Ves->defining_sectors[ia], Wes->defining_sectors[ib])
                   .is_real()) {
                any_complex = true;
                break;
            }
        }
    }
    Dtype const dt = any_complex ? Dtype::Complex128 : Dtype::Float64;
    FusionSymbol res = FusionSymbol::zeros(4,
                                           FusionSymbol::Shape{ { static_cast<std::size_t>(dW),
                                                                  static_cast<std::size_t>(dV),
                                                                  static_cast<std::size_t>(dW),
                                                                  static_cast<std::size_t>(dV) } },
                                           dt);

    auto copy_block = [&](FusionSymbol const& block, int64 j0, int64 i0, int64 db, int64 da) {
        FusionSymbol b = any_complex ? block.as_complex() : block;
        for (int64 j = 0; j < db; ++j) {
            for (int64 i = 0; i < da; ++i) {
                for (int64 j2 = 0; j2 < db; ++j2) {
                    for (int64 i2 = 0; i2 < da; ++i2) {
                        res.set(static_cast<std::size_t>(j0 + j),
                                static_cast<std::size_t>(i0 + i),
                                static_cast<std::size_t>(j0 + j2),
                                static_cast<std::size_t>(i0 + i2),
                                b.get_complex(static_cast<std::size_t>(j),
                                              static_cast<std::size_t>(i),
                                              static_cast<std::size_t>(j2),
                                              static_cast<std::size_t>(i2)));
                    }
                }
            }
        }
    };

    int64 i = 0;
    for (std::size_t ia = 0; ia < Ves->defining_sectors.size(); ++ia) {
        auto const& a = Ves->defining_sectors[ia];
        auto const ma = Ves->multiplicities[ia];
        auto const da = static_cast<int64>(Ves->Space::symmetry->sector_dim(a));
        int64 j = 0;
        for (std::size_t ib = 0; ib < Wes->defining_sectors.size(); ++ib) {
            auto const& b = Wes->defining_sectors[ib];
            auto const mb = Wes->multiplicities[ib];
            auto swap = Ves->Space::symmetry->swap_gate(a, b);
            auto const db = static_cast<int64>(Wes->Space::symmetry->sector_dim(b));
            int64 i2 = i;
            for (int64 na = 0; na < ma; ++na) {
                int64 j2 = j;
                for (int64 nb = 0; nb < mb; ++nb) {
                    copy_block(swap, j2, i2, db, da);
                    j2 += db;
                }
                i2 += da;
            }
            j += db * mb;
        }
        i += da * ma;
    }

    auto const& Winv = Wes->inverse_basis_perm();
    auto const& Vinv = Ves->inverse_basis_perm();
    FusionSymbol out = FusionSymbol::zeros(4,
                                           FusionSymbol::Shape{ { static_cast<std::size_t>(dW),
                                                                  static_cast<std::size_t>(dV),
                                                                  static_cast<std::size_t>(dW),
                                                                  static_cast<std::size_t>(dV) } },
                                           dt);
    for (std::size_t i0 = 0; i0 < static_cast<std::size_t>(dW); ++i0) {
        for (std::size_t i1 = 0; i1 < static_cast<std::size_t>(dV); ++i1) {
            for (std::size_t i2 = 0; i2 < static_cast<std::size_t>(dW); ++i2) {
                for (std::size_t i3 = 0; i3 < static_cast<std::size_t>(dV); ++i3) {
                    out.set(i0,
                            i1,
                            i2,
                            i3,
                            res.get_complex(static_cast<std::size_t>(Winv[i0]),
                                            static_cast<std::size_t>(Vinv[i1]),
                                            static_cast<std::size_t>(Winv[i2]),
                                            static_cast<std::size_t>(Vinv[i3])));
                }
            }
        }
    }
    return out;
}

FusionSymbol
twist_gate_diag(Leg::Ptr V)
{
    if (!V) {
        throw py::type_error("twist_gate_diag requires a leg");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", V->symmetry->str()));
    }
    if (is_plain_leg_pipe(V)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(V);
        FusionSymbol res = twist_gate_diag(pipe->legs.front());
        for (std::size_t n = 1; n < pipe->legs.size(); ++n) {
            FusionSymbol next = twist_gate_diag(pipe->legs[n]);
            auto ra = res.as_complex();
            auto na = next.as_complex();
            auto rs = ra.as_complex128();
            auto ns = na.as_complex128();
            std::size_t const nI = rs.size();
            std::size_t const nJ = ns.size();
            std::vector<complex128> out(nI * nJ);
            if (pipe->combine_cstyle) {
                for (std::size_t ii = 0; ii < nI; ++ii) {
                    for (std::size_t jj = 0; jj < nJ; ++jj) {
                        out[ii * nJ + jj] = rs[ii] * ns[jj];
                    }
                }
            } else {
                for (std::size_t jj = 0; jj < nJ; ++jj) {
                    for (std::size_t ii = 0; ii < nI; ++ii) {
                        out[jj * nI + ii] = rs[ii] * ns[jj];
                    }
                }
            }
            bool any_imag = false;
            for (auto v : out) {
                if (v.imag() != 0.0) {
                    any_imag = true;
                    break;
                }
            }
            if (any_imag) {
                res = FusionSymbol::from_complex128(
                  1, FusionSymbol::Shape{ { out.size(), 1, 1, 1 } }, std::move(out));
            } else {
                std::vector<float64> real(out.size());
                for (std::size_t k = 0; k < out.size(); ++k) {
                    real[k] = out[k].real();
                }
                res = FusionSymbol::from_float64(
                  1, FusionSymbol::Shape{ { real.size(), 1, 1, 1 } }, std::move(real));
            }
        }
        return res;
    }

    // ElementarySpace or AbelianLegPipe
    auto Ves = std::dynamic_pointer_cast<ElementarySpace>(V);
    if (!Ves) {
        throw py::type_error("twist_gate_diag expects ElementarySpace or LegPipe");
    }
    auto const dV = static_cast<std::size_t>(leg_dim_as_size(*Ves));
    if (!Ves->slices) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", Ves->Space::symmetry->str()));
    }

    bool any_complex = false;
    std::vector<complex128> values(dV, complex128{ 0.0, 0.0 });
    for (std::size_t n = 0; n < Ves->sector_decomposition.size(); ++n) {
        auto const& a = Ves->sector_decomposition[n];
        auto const i = static_cast<std::size_t>((*Ves->slices)[n][0]);
        auto const j = static_cast<std::size_t>((*Ves->slices)[n][1]);
        complex128 const twist = Ves->Space::symmetry->topological_twist(a);
        if (twist.imag() != 0.0) {
            any_complex = true;
        }
        for (std::size_t k = i; k < j; ++k) {
            values[k] = twist;
        }
    }
    auto const& perm = Ves->inverse_basis_perm();
    if (any_complex) {
        std::vector<complex128> out(dV);
        for (std::size_t i = 0; i < dV; ++i) {
            out[i] = values[static_cast<std::size_t>(perm[i])];
        }
        return FusionSymbol::from_complex128(
          1, FusionSymbol::Shape{ { dV, 1, 1, 1 } }, std::move(out));
    }
    std::vector<float64> out(dV);
    for (std::size_t i = 0; i < dV; ++i) {
        out[i] = values[static_cast<std::size_t>(perm[i])].real();
    }
    return FusionSymbol::from_float64(1, FusionSymbol::Shape{ { dV, 1, 1, 1 } }, std::move(out));
}

FusionSymbol
twist_gate(Leg::Ptr V)
{
    if (!V) {
        throw py::type_error("twist_gate requires a leg");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", V->symmetry->str()));
    }
    auto diag = twist_gate_diag(std::move(V));
    auto const n = diag.extent(0);
    FusionSymbol out(2, FusionSymbol::Shape{ { n, n, 1, 1 } }, diag.dtype());
    for (std::size_t i = 0; i < n; ++i) {
        out.set(i, i, diag.get_complex(i));
    }
    return out;
}

std::vector<int64>
flat_leg_permutation(std::vector<Leg::Ptr> const& legs)
{
    std::vector<int64> offsets(legs.size(), 0);
    int64 running = 0;
    for (std::size_t i = 0; i < legs.size(); ++i) {
        offsets[i] = running;
        running += legs[i]->num_flat_legs();
    }
    std::vector<int64> perm;
    perm.reserve(static_cast<std::size_t>(running));
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto part = legs[i]->_flat_leg_permutation(offsets[i]);
        perm.insert(perm.end(), part.begin(), part.end());
    }
    return perm;
}

std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
unique_sorted_sectors(SectorArray const& unsorted_sectors,
                      std::vector<int64> const& unsorted_multiplicities)
{
    auto [sectors, mults, perm] = unsorted_sectors.unique_sorted(unsorted_multiplicities);
    return { std::move(sectors), std::vector<int64>(mults.begin(), mults.end()), std::move(perm) };
}

std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
sort_sectors_public(SectorArray const& sectors, std::vector<int64> const& multiplicities)
{
    return sort_sectors(sectors, multiplicities);
}

std::pair<std::optional<std::vector<int64>>, Symmetry::Ptr>
parse_inputs_drop_symmetry_public(std::optional<std::vector<int64>> which, Symmetry::Ptr symmetry)
{
    if (!symmetry) {
        throw py::type_error("parse_inputs_drop_symmetry requires a symmetry");
    }
    return parse_inputs_drop_symmetry(std::move(which), *symmetry);
}

} // namespace lasyten

// =============================================================================
// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)
// =============================================================================
// --- TensorProduct.__init__ ---
// need to set this early, for use in _calc_sectors
// =============================================================================
