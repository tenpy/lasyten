// ---- 00_keep_helpers.cpp ----
#include <lasyten/backends/abelian.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/backends/block_inds_numpy.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

[[nodiscard]] py::object
py_bool_dtype()
{
    return py::reinterpret_borrow<py::object>((PyObject*)&PyBool_Type);
}

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(ElementarySpace::Ptr const& leg, int64 idx)
{
    if (!leg) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return leg->parse_index(idx);
}

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(Space::Ptr const& leg, int64 idx)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(leg);
    if (!es) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return es->parse_index(idx);
}

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(Leg const& leg, int64 idx)
{
    auto es = leg.as_elementary_space();
    if (!es) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return es->parse_index(idx);
}

BlockInds
take_rows(BlockInds const& arr, py::array const& perm)
{
    auto p = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(perm);
    auto buf = p.unchecked<1>();
    std::vector<int64> idx(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        idx[static_cast<std::size_t>(i)] = buf(i);
    return arr.take_i64(idx);
}

py::array_t<int64>
take_rows(py::array_t<int64> const& arr, py::array const& perm)
{
    auto p = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(perm);
    auto pbuf = p.unchecked<1>();
    if (arr.ndim() == 1) {
        auto abuf = arr.unchecked<1>();
        py::array_t<int64> out(pbuf.shape(0));
        auto obuf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < pbuf.shape(0); ++i)
            obuf(i) = abuf(pbuf(i));
        return out;
    }
    if (arr.ndim() != 2) {
        throw std::invalid_argument("take_rows: expected 1D or 2D array");
    }
    auto abuf = arr.unchecked<2>();
    py::array_t<int64> out({ pbuf.shape(0), abuf.shape(1) });
    auto obuf = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < pbuf.shape(0); ++i) {
        auto r = pbuf(i);
        for (py::ssize_t c = 0; c < abuf.shape(1); ++c)
            obuf(i, c) = abuf(r, c);
    }
    return out;
}

[[nodiscard]] py::array_t<int64>
zeros_i64_array(std::vector<py::ssize_t> const& shape)
{
    py::array_t<int64> arr(shape);
    if (arr.size() > 0) {
        std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(), int64{ 0 });
    }
    return arr;
}

[[nodiscard]] py::array_t<int64>
empty_i64_array(std::vector<py::ssize_t> const& shape)
{
    return py::array_t<int64>(shape);
}

} // namespace

AbelianBackendData::AbelianBackendData(Dtype dtype_,
                                       std::string device_,
                                       std::vector<BlockBackend::BlockPtr> blocks_,
                                       BlockInds block_inds_,
                                       bool is_sorted)
  : dtype(dtype_)
  , device(std::move(device_))
  , blocks(std::move(blocks_))
  , block_inds(std::move(block_inds_))
{
    if (block_inds.nrows() != blocks.size()) {
        throw std::invalid_argument(
          "AbelianBackendData: len(blocks) must equal block_inds.nrows()");
    }

    if (!is_sorted) {
        auto perm = block_inds.lexsort_indices();
        block_inds = block_inds.take(perm);
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(blocks.size());
        for (std::size_t i : perm)
            sorted_blocks.push_back(blocks[i]);
        blocks = std::move(sorted_blocks);
    }
}

std::optional<int64>
AbelianBackendData::get_block_num(BlockInds const& query) const
{
    // --- hints from Python AbelianBackendData.get_block_num ---
    // OPTIMIZE use sorted for lookup?
    // ---
    if (query.nrows() != 1) {
        throw std::invalid_argument("AbelianBackendData::get_block_num: expected a single row");
    }
    if (query.ncols() != block_inds.ncols()) {
        throw std::invalid_argument("AbelianBackendData::get_block_num: ncols mismatch");
    }
    auto idx = block_inds.row_where(query.row(0));
    if (!idx)
        return std::nullopt;
    return static_cast<int64>(*idx);
}

BlockBackend::BlockPtr
AbelianBackendData::get_block(BlockInds const& query) const
{
    auto block_num = get_block_num(query);
    if (!block_num.has_value()) {
        return nullptr;
    }
    return blocks[static_cast<std::size_t>(*block_num)];
}

void
AbelianBackendData::save_hdf5(lasyten::hdf5::Saver& saver,
                              HighFive::Group& /*h5gr*/,
                              std::string const& subpath) const
{
    hdf5_export::save_block_inds(saver, subpath + "block_inds", block_inds);
    HighFive::Group blocks_g;
    std::string blocks_sub;
    saver.save_sequence_begin(subpath + "blocks",
                              hdf5_io::REPR_LIST,
                              static_cast<std::int64_t>(blocks.size()),
                              blocks_g,
                              blocks_sub);
    lasyten::hdf5::Saver blocks_saver(blocks_g);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        hdf5_export::save_block(blocks_saver, std::to_string(i), blocks[i]);
    }
    hdf5_export::save_dtype_string(saver, subpath + "dtype", dtype);
    saver.save_string(subpath + "device", device);
}

AbelianBackendData::Ptr
AbelianBackendData::from_hdf5(lasyten::hdf5::Loader& loader,
                              HighFive::Group& h5gr,
                              std::string const& subpath)
{
    auto block_inds = hdf5_export::load_block_inds(loader, subpath + "block_inds");
    hid_t blocks_id = loader.open(subpath + "blocks");
    auto const n = hdf5_io::h5_get_attr_int64(blocks_id, hdf5_io::ATTR_LEN).value_or(0);
    HighFive::Group blocks_g = hdf5_io::group_from_hid(blocks_id);
    lasyten::hdf5::Loader blocks_loader(blocks_g);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        blocks.push_back(hdf5_export::load_block(blocks_loader, std::to_string(i)));
    }
    hid_t device_id = loader.open(subpath + "device");
    auto device = loader.load_string(device_id);
    H5Idec_ref(device_id);
    auto dtype = hdf5_export::load_dtype_string(loader, subpath + "dtype");

    if (!blocks.empty()) {
        device = blocks.front()->device();
    }

    auto obj = std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(blocks), std::move(block_inds), /*is_sorted=*/true);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

namespace {

BlockInds
zeros_i64(std::size_t rows, std::size_t cols)
{
    return BlockInds::zeros(rows, cols);
}

BlockInds
asarray_i64(py::object obj)
{
    return block_inds_from_numpy(obj);
}

py::array_t<int64>
asarray_i64_1d(py::object obj)
{
    auto arr = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw std::invalid_argument("asarray_i64_1d: expected array-like");
    }
    if (arr.ndim() == 0) {
        py::array_t<int64> out(1);
        out.mutable_at(0) = arr.at();
        return out;
    }
    if (arr.ndim() != 1) {
        return arr.reshape({ -1 });
    }
    return arr;
}

py::array_t<int64>
i64_vec_to_numpy(std::vector<int64> const& v)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i)
        buf(static_cast<py::ssize_t>(i)) = v[i];
    return arr;
}

py::array_t<int64>
asarray_i64_np(py::object obj)
{
    auto arr = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw std::invalid_argument("asarray_i64_np: expected array-like");
    }
    return arr;
}

std::vector<int64>
asarray_i64_vec(py::object obj)
{
    auto arr = asarray_i64_1d(obj);
    auto buf = arr.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        out[static_cast<std::size_t>(i)] = buf(i);
    return out;
}

std::vector<int64>
rank_concat(py::object parts)
{
    std::vector<int64> all;
    for (auto item : parts) {
        auto chunk = asarray_i64_vec(py::reinterpret_borrow<py::object>(item));
        all.insert(all.end(), chunk.begin(), chunk.end());
    }
    return rank_data(all);
}

[[nodiscard]] BlockInds
block_inds_from_py_rows(py::object rows)
{
    return block_inds_from_numpy(
      py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(rows));
}

BlockInds
to_block_inds(py::array_t<int64> const& arr)
{
    return block_inds_from_numpy(arr);
}

BlockInds
to_block_inds(BlockInds bi)
{
    return bi;
}

BlockInds
to_block_inds(py::object obj)
{
    return block_inds_from_numpy(obj);
}

py::slice
slice_pair(py::object pair)
{
    return py::slice(pair.attr("__getitem__")(0).cast<py::ssize_t>(),
                     pair.attr("__getitem__")(1).cast<py::ssize_t>(),
                     1);
}

BlockBackend::BlockPtr
b_get(BlockBackend::BlockPtr const& b, py::object key)
{
    return b->get_item(key);
}

void
b_set(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b->set_item(key, py::cast(v));
}

void
b_set_add(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b_set(b, key, (*b_get(b, key)) + (*v));
}

bool
is_zero_scalar(BlockBackend::Scalar const& a)
{
    return a.as_complex128() == 0.;
}

AbelianBackendData::Ptr
make_data(Dtype dtype,
          std::string device,
          std::vector<BlockBackend::BlockPtr> blocks,
          BlockInds block_inds,
          bool is_sorted = false)
{
    return std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(blocks), std::move(block_inds), is_sorted);
}

AbelianBackendData::Ptr
make_data(Dtype dtype,
          std::string device,
          std::vector<BlockBackend::BlockPtr> blocks,
          py::array_t<int64> block_inds,
          bool is_sorted = false)
{
    return make_data(
      dtype, std::move(device), std::move(blocks), to_block_inds(block_inds), is_sorted);
}

std::vector<int64>
mults_of(py::object leg)
{
    return leg.attr("multiplicities").cast<std::vector<int64>>();
}

std::vector<int64>
mults_of(Leg::Ptr const& leg)
{
    return as_space(leg)->multiplicities;
}

int64
nsec(py::object leg)
{
    return leg.attr("num_sectors").cast<int64>();
}

int64
nsec(Leg::Ptr const& leg)
{
    return as_space(leg)->num_sectors;
}

bool
sector_sorted(py::object leg)
{
    py::object so = leg.attr("sector_order");
    return (!so.is_none()) && so.cast<std::string>() == "sorted";
}

py::object
take_rows_obj(py::object arr, py::object perm)
{
    auto a = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto p = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(perm);
    return take_rows(a, p);
}

std::vector<BlockBackend::BlockPtr>
permute_blocks(std::vector<BlockBackend::BlockPtr> const& blocks, py::array const& perm)
{
    auto p = perm.cast<py::array_t<int64>>();
    auto buf = p.unchecked<1>();
    std::vector<BlockBackend::BlockPtr> out;
    out.reserve(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        out.push_back(blocks[static_cast<std::size_t>(buf(i))]);
    return out;
}

} // namespace

// ---- 01_valid_wrap_ctor.cpp ----

BlockInds
valid_block_inds(TensorProduct::Ptr codomain, TensorProduct::Ptr domain)
{
    // --- hints from Python _valid_block_inds ---
    // OPTIMIZE: this is brute-force going through all possible combinations of block indices
    // spaces are sorted, so we can probably reduce that search space quite a bit...
    // ---
    // Prefer calling the Python reference for exact fusion broadcast semantics via numpy zip,
    // but implement locally with make_grid + Symmetry::multiple_fusion_broadcast.
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<int64> nums;
    nums.reserve(legs.size());
    for (auto const& leg : legs)
        nums.push_back(nsec(leg));
    auto grid = make_grid(nums, /*cstyle=*/false);
    auto const n_combos = grid.size();
    auto const nlegs = nums.size();
    auto symmetry = codomain->symmetry;

    auto select_sectors = [&](std::vector<Leg::Ptr> const& factors,
                              std::function<int64(std::size_t, std::size_t)> const& col) {
        std::vector<SectorArray> parts;
        parts.reserve(factors.size());
        for (std::size_t fi = 0; fi < factors.size(); ++fi) {
            auto const& sectors = as_space(factors[fi])->sector_decomposition;
            SectorArray selected = SectorArray::empty(sectors.sector_ind_len());
            selected.reserve(n_combos);
            for (std::size_t r = 0; r < n_combos; ++r)
                selected.push_back(sectors[static_cast<std::size_t>(col(r, fi))]);
            parts.push_back(std::move(selected));
        }
        return symmetry->multiple_fusion_broadcast(parts);
    };

    SectorArray codomain_coupled;
    if (codomain->num_factors > 0) {
        codomain_coupled = select_sectors(
          codomain->factors, [&](std::size_t r, std::size_t fi) { return grid[r][fi]; });
    } else {
        codomain_coupled = SectorArray::repeat(symmetry->trivial_sector, n_combos);
    }

    SectorArray domain_coupled;
    if (domain->num_factors > 0) {
        // domain factors correspond to reversed grid columns
        domain_coupled = select_sectors(
          domain->factors, [&](std::size_t r, std::size_t fi) { return grid[r][nlegs - 1 - fi]; });
    } else {
        domain_coupled = SectorArray::repeat(symmetry->trivial_sector, n_combos);
    }

    std::vector<std::size_t> valid_idx;
    for (std::size_t i = 0; i < n_combos; ++i) {
        if (codomain_coupled[i] == domain_coupled[i])
            valid_idx.push_back(i);
    }
    BlockInds bi(valid_idx.size(), nlegs);
    for (std::size_t r = 0; r < valid_idx.size(); ++r) {
        auto const& row = grid[valid_idx[r]];
        for (std::size_t c = 0; c < nlegs; ++c)
            bi(r, c) = row[c];
    }
    return bi.take(bi.lexsort_indices());
}

TensorBackend::DataPtr
AbelianBackend::wrap(AbelianBackendData::Ptr d)
{
    if (!d)
        throw std::invalid_argument("AbelianBackend::wrap: null");
    return d;
}

AbelianBackendData::Ptr
AbelianBackend::unwrap(DataPtr d)
{
    if (!d)
        throw std::invalid_argument("AbelianBackend::unwrap: null DataPtr");
    auto* p = dynamic_cast<AbelianBackendData*>(d.get());
    if (!p) {
        const auto& obj = *d;
        throw std::invalid_argument(std::format(
          "AbelianBackend::unwrap: expected AbelianBackendData, got {}", typeid(obj).name()));
    }
    return std::static_pointer_cast<AbelianBackendData>(d);
}

AbelianBackendData::Ptr
AbelianBackend::data_from_tensor(TensorCPtr tensor)
{
    if (auto st = std::dynamic_pointer_cast<const SymmetricTensor>(tensor))
        return unwrap(st->data);
    if (auto m = std::dynamic_pointer_cast<const Mask>(tensor))
        return unwrap(m->data);
    throw std::invalid_argument(
      "AbelianBackend::data_from_tensor: expected SymmetricTensor or Mask");
}

namespace {

void
require_matrix_form(SymmetricTensorCPtr a, char const* op)
{
    if (a->num_codomain_legs() != 1 || a->num_domain_legs() != 1) {
        throw std::invalid_argument(
          std::format("{} requires a single-leg codomain and domain (matrix form); got {}x{}",
                      op,
                      a->num_codomain_legs(),
                      a->num_domain_legs()));
    }
}

} // namespace

AbelianBackend::AbelianBackend(std::shared_ptr<BlockBackend> block_backend_)
  : TensorBackend(std::move(block_backend_))
{
}

// ---- 02_early.cpp ----

void
AbelianBackend::test_tensor_sanity(TensorCPtr a, bool is_diagonal)
{
    // --- hints from Python AbelianBackend.test_tensor_sanity ---
    // check device and dtype
    // check leg types
    // recursion into nested pipes is handled via AbelianLegPipe.test_sanity(), which
    // is called via (co)domain.test_sanity() during Tensor.test_sanity()
    // check block_inds
    // check block_inds fulfill charge rule
    // check blocks and charge rule
    // ---
    TensorBackend::test_tensor_sanity(a, is_diagonal);
    // Skip deep checks if the tensor still holds Python-side data.
    py::object raw = py::cast(a).attr("data");
    AbelianBackendData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    check(a->device == data->device, "test_tensor_sanity: device mismatch");
    check(data->device == block_backend->as_device(data->device),
          "test_tensor_sanity: device not normalized by block backend");
    check(a->dtype == data->dtype, "test_tensor_sanity: dtype mismatch");
    int64 num_legs = a->num_legs;
    for (int64 n = 0; n < num_legs; ++n) {
        auto leg = a->get_leg_co_domain(n);
        if (leg && leg.is_pipe() && !leg.is_abelian_leg_pipe())
            throw std::runtime_error("pipes must be AbelianLegPipe");
    }
    auto const& bi = data->block_inds;
    check(bi.nrows() == data->blocks.size(),
          "test_tensor_sanity: block_inds rows != number of blocks");
    check(static_cast<int64>(bi.ncols()) == num_legs,
          "test_tensor_sanity: block_inds cols != num_legs");
    check(bi.all_ge(0), "test_tensor_sanity: negative block index");
    std::vector<int64> maxes;
    for (auto const& leg : conventional_leg_order(a))
        maxes.push_back(nsec(leg));
    check(bi.all_lt_per_column(maxes), "test_tensor_sanity: block index out of range");
    check(bi.is_lexsorted(), "test_tensor_sanity: block_inds not lexsorted");
    if (is_diagonal)
        check(bi.columns_equal(0, 1), "test_tensor_sanity: diagonal block_inds mismatch");
    auto legs = conventional_leg_order(a);
    for (std::size_t i = 0; i < data->blocks.size(); ++i) {
        std::vector<int64> shape;
        if (is_diagonal) {
            auto diag = std::dynamic_pointer_cast<const DiagonalTensor>(a);
            if (!diag) {
                throw std::logic_error(
                  "AbelianBackend::test_tensor_sanity: is_diagonal=true but tensor is not a "
                  "DiagonalTensor");
            }
            auto mults = mults_of(py::cast(diag->leg()));
            shape = { mults[static_cast<std::size_t>(bi(i, 0))] };
        } else {
            for (std::size_t li = 0; li < legs.size(); ++li) {
                auto mults = mults_of(legs[li]);
                shape.push_back(mults[static_cast<std::size_t>(bi(i, li))]);
            }
        }
        block_backend->test_block_sanity(data->blocks[i], shape, a->dtype, a->device);
    }
}

void
AbelianBackend::test_mask_sanity(MaskCPtr a)
{
    // --- hints from Python AbelianBackend.test_mask_sanity ---
    // check charge rule
    // ---
    TensorBackend::test_mask_sanity(a);
    py::object raw = py::cast(a).attr("data");
    AbelianBackendData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    check(a->device == data->device, "test_mask_sanity: device mismatch");
    check(data->dtype == Dtype::Bool, "test_mask_sanity: dtype must be bool");
    auto const& bi = data->block_inds;
    check(bi.nrows() == data->blocks.size(),
          "test_mask_sanity: block_inds rows != number of blocks");
    check(static_cast<int64>(bi.ncols()) == a->num_legs,
          "test_mask_sanity: block_inds cols != num_legs");
    bool is_projection = a->is_projection;
    auto large_leg = py::cast(a->large_leg());
    auto small_leg = py::cast(a->small_leg());
    for (std::size_t i = 0; i < data->blocks.size(); ++i) {
        int64 bi_small = is_projection ? bi(i, 0) : bi(i, 1);
        int64 bi_large = is_projection ? bi(i, 1) : bi(i, 0);
        check(bi_large >= bi_small, "test_mask_sanity: inconsistent sector indices");
        int64 expect_len = mults_of(large_leg)[static_cast<std::size_t>(bi_large)];
        int64 expect_sum = mults_of(small_leg)[static_cast<std::size_t>(bi_small)];
        block_backend->test_block_sanity(
          data->blocks[i], std::vector<int64>{ expect_len }, Dtype::Bool, data->device);
        check(block_backend->sum_all(data->blocks[i]).as_int64() == expect_sum,
              "test_mask_sanity: mask block sum mismatch");
    }
}

Leg::Ptr
AbelianBackend::make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, Leg::Ptr pipe)
{
    // --- hints from Python AbelianBackend.make_pipe ---
    // OPTIMIZE rm check
    // ---
    std::vector<ElementarySpace::Ptr> es;
    es.reserve(legs.size());
    for (auto const& l : legs) {
        auto p = l.as_elementary_space();
        if (!p)
            throw std::invalid_argument("make_pipe: legs must be ElementarySpace");
        es.push_back(std::move(p));
    }
    if (auto ab = std::dynamic_pointer_cast<AbelianLegPipe>(pipe.as_elementary_space())) {
        if (ab->combine_cstyle != !is_dual) {
            throw std::invalid_argument(
              "make_pipe: AbelianLegPipe combine_cstyle does not match is_dual");
        }
        if (ab->is_dual != is_dual) {
            throw std::invalid_argument("make_pipe: AbelianLegPipe is_dual does not match");
        }
        return ab;
    }
    return std::make_shared<AbelianLegPipe>(std::move(es), is_dual, !is_dual);
}

TensorBackend::DataPtr
AbelianBackend::act_block_diagonal_square_matrix(SymmetricTensorCPtr a,
                                                 BlockUnaryFn block_method,
                                                 std::optional<DtypeMapFn> dtype_map)
{
    // --- hints from Python AbelianBackend.act_block_diagonal_square_matrix ---
    // use that all_block_inds is just ascending -> all_block_inds[j, 0] == j
    // ---
    auto a_data = data_from_tensor(a);
    auto leg = py::cast(a->domain).attr("factors").attr("__getitem__")(0);
    BlockInds all_block_inds = BlockInds::arange_diag(static_cast<std::size_t>(nsec(leg)));
    std::vector<BlockBackend::BlockPtr> res_blocks;
    BlockInds::iter_common_noncommon_sorted(
      a_data->block_inds,
      all_block_inds,
      [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
          BlockBackend::BlockPtr block;
          if (!i) {
              int64 m = mults_of(leg)[static_cast<std::size_t>(*j)];
              block = block_backend->zeros({ m, m }, a->dtype);
          } else {
              block = a_data->blocks[static_cast<std::size_t>(*i)];
          }
          res_blocks.push_back(block_method(block));
      });
    Dtype dtype = a->dtype;
    if (dtype_map)
        dtype = (*dtype_map)(dtype);
    for (auto& b : res_blocks)
        b = block_backend->to_dtype(b, dtype);
    return wrap(make_data(dtype, a_data->device, std::move(res_blocks), all_block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::add_trivial_leg(TensorCPtr a,
                                int64 legs_pos,
                                bool add_to_domain,
                                int64 co_domain_pos,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain)
{
    // --- hints from Python AbelianBackend.add_trivial_leg ---
    // since the new column is constant, block_inds are still sorted.
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->add_axis(b, legs_pos));
    BlockInds block_inds = a_data->block_inds.insert_column(static_cast<std::size_t>(legs_pos), 0);
    return wrap(make_data(a_data->dtype, a_data->device, std::move(blocks), block_inds, true));
}

bool
AbelianBackend::almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    bool ok = true;
    BlockInds::iter_common_noncommon_sorted(
      a_data->block_inds,
      b_data->block_inds,
      [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
          if (!ok)
              return;
          if (!j) {
              if (block_backend->max_abs(a_data->blocks[static_cast<std::size_t>(*i)])
                    .as_float64() > atol)
                  ok = false;
          } else if (!i) {
              if (block_backend->max_abs(b_data->blocks[static_cast<std::size_t>(*j)])
                    .as_float64() > atol)
                  ok = false;
          } else if (!block_backend->allclose(a_data->blocks[static_cast<std::size_t>(*i)],
                                              b_data->blocks[static_cast<std::size_t>(*j)],
                                              rtol,
                                              atol)) {
              ok = false;
          }
      });
    return ok;
}

TensorBackend::DataPtr
AbelianBackend::apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask)
{
    // --- hints from Python AbelianBackend.apply_mask_to_DiagonalTensor ---
    // append only for one leg, repeat later
    // ---
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    BlockInds t_contr = t_data->block_inds.take_columns(std::array<std::size_t, 1>{ 0 });
    auto m_col = m_data->block_inds.column(1);
    BlockInds m_contr(std::move(m_col), m_data->block_inds.nrows(), 1);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    BlockInds::iter_common_sorted(t_contr,
                                  m_contr,
                                  /*a_strict=*/true,
                                  /*b_strict=*/true,
                                  [&](std::ptrdiff_t i, std::ptrdiff_t j) {
                                      res_blocks.push_back(block_backend->apply_mask(
                                        t_data->blocks[static_cast<std::size_t>(i)],
                                        m_data->blocks[static_cast<std::size_t>(j)],
                                        0));
                                      int64 v = m_data->block_inds(static_cast<std::size_t>(j), 0);
                                      res_rows.push_back({ v, v });
                                  });
    BlockInds res_block_inds = res_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(res_rows);
    return wrap(
      make_data(tensor->dtype, t_data->device, std::move(res_blocks), res_block_inds, true));
}

// ---- 03_mid.cpp ----

TensorBackend::DataPtr
AbelianBackend::copy_data(TensorCPtr a, std::optional<std::string> device)
{
    // --- hints from Python AbelianBackend.copy_data ---
    // OPTIMIZE do we need to copy the block_inds ??
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->copy_block(b, device));
    std::string dev = device.has_value() ? block_backend->as_device(device) : a_data->device;
    return wrap(
      make_data(a_data->dtype, std::move(dev), std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::dagger(TensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->dagger(b));
    BlockInds block_inds = a_data->block_inds.reverse_columns();
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
AbelianBackend::data_item(DataPtr a)
{
    auto data = unwrap(a);
    if (data->blocks.size() > 1)
        throw std::runtime_error("Inconsistent data.");
    if (data->blocks.empty())
        return block_backend->as_scalar(dtype::zero_scalar(data->dtype), data->dtype);
    return block_backend->item(data->blocks[0]);
}

bool
AbelianBackend::diagonal_all(DiagonalTensorCPtr a)
{
    // --- hints from Python AbelianBackend.diagonal_all ---
    // missing blocks are filled with False
    // now it is enough to check that all existing blocks are all-True
    // ---
    auto data = data_from_tensor(a);
    if (static_cast<int64>(data->block_inds.nrows()) < nsec(py::cast(a->leg())))
        return false;
    for (auto const& b : data->blocks)
        if (!block_backend->all(b))
            return false;
    return true;
}

bool
AbelianBackend::diagonal_any(DiagonalTensorCPtr a)
{
    auto data = data_from_tensor(a);
    for (auto const& b : data->blocks)
        if (block_backend->any(b))
            return true;
    return false;
}

TensorBackend::DataPtr
AbelianBackend::diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                           BlockUnaryFn func,
                                           bool maps_zero_to_zero)
{
    // --- hints from Python AbelianBackend.diagonal_elementwise_unary ---
    // use that block_inds is just arange -> block_inds[i, 0] == i
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    BlockInds block_inds;
    if (maps_zero_to_zero) {
        blocks.reserve(a_data->blocks.size());
        for (auto const& b : a_data->blocks) {
            blocks.push_back(func(b));
        }
        block_inds = a_data->block_inds;
    } else {
        block_inds = BlockInds::arange_diag(static_cast<std::size_t>(nsec(py::cast(a->leg()))));
        BlockInds::iter_common_noncommon_sorted(
          block_inds,
          a_data->block_inds,
          [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
              BlockBackend::BlockPtr block;
              if (!j) {
                  block = block_backend->zeros(
                    { mults_of(py::cast(a->leg()))[static_cast<std::size_t>(*i)] }, a->dtype);
              } else {
                  block = a_data->blocks[static_cast<std::size_t>(*j)];
              }
              blocks.push_back(func(block));
          });
    }
    Dtype dt;
    if (blocks.empty()) {
        dt = block_backend->get_dtype(func(block_backend->zeros({ 1 }, a->dtype)));
    } else {
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds, true));
}

std::string
AbelianBackend::get_device_from_data(DataPtr a)
{
    return block_backend->as_device(unwrap(a)->device);
}

Dtype
AbelianBackend::get_dtype_from_data(DataPtr a)
{
    return unwrap(a)->dtype;
}

bool
AbelianBackend::supports_symmetry(Symmetry::Ptr symmetry)
{
    return symmetry->is_abelian() && symmetry->has_trivial_braid();
}

py::object
AbelianBackend::state_tensor_product(BlockBackend::BlockPtr /*state1*/,
                                     BlockBackend::BlockPtr /*state2*/,
                                     Leg::Ptr /*pipe*/)
{
    // --- hints from Python AbelianBackend.state_tensor_product ---
    // clearly define what this should do in tensors.py first!
    // ---
    throw NotImplemented("state_tensor_product not implemented");
}

BlockBackend::BlockPtr
AbelianBackend::to_dense_block_trivial_sector(TensorCPtr tensor)
{
    if (tensor->num_legs != 1) {
        throw std::invalid_argument("to_dense_block_trivial_sector requires a 1-leg tensor");
    }
    auto data = data_from_tensor(tensor);
    auto space = as_space(tensor->codomain->factors[0]);
    auto bi = space->sector_decomposition_where(space->symmetry->trivial_sector);
    if (!bi.has_value()) {
        throw std::invalid_argument(
          "to_dense_block_trivial_sector: space does not contain the trivial sector");
    }
    int64 const row_v = *bi;
    auto block = data->get_block(BlockInds::from_row(std::span<const int64>(&row_v, 1)));
    if (!block) {
        int64 dim = space->multiplicities[static_cast<std::size_t>(*bi)];
        return block_backend->zeros({ dim }, data->dtype);
    }
    return block;
}

TensorBackend::DataPtr
AbelianBackend::zero_data(TensorProduct::Ptr codomain,
                          TensorProduct::Ptr domain,
                          Dtype dtype,
                          std::string device,
                          bool all_blocks)
{
    if (!all_blocks) {
        return wrap(make_data(dtype,
                              std::move(device),
                              {},
                              zeros_i64(0, codomain->num_factors + domain->num_factors),
                              true));
    }
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<BlockBackend::BlockPtr> zero_blocks;
    for (std::size_t r = 0; r < block_inds.nrows(); ++r) {
        std::vector<int64> shape;
        for (std::size_t c = 0; c < block_inds.ncols(); ++c)
            shape.push_back(mults_of(legs[c])[static_cast<std::size_t>(block_inds(r, c))]);
        zero_blocks.push_back(block_backend->zeros(shape, dtype, device));
    }
    return wrap(make_data(dtype, std::move(device), std::move(zero_blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::zero_diagonal_data(TensorProduct::Ptr /*co_domain*/,
                                   Dtype dtype,
                                   std::string device)
{
    return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
AbelianBackend::zero_mask_data(Space::Ptr /*large_leg*/, std::string device)
{
    return wrap(make_data(Dtype::Bool, std::move(device), {}, zeros_i64(0, 2), true));
}

BlockInds
AbelianBackend::leg_pipe_map_incoming_block_inds(AbelianLegPipe const& pipe,
                                                 BlockInds const& incoming_block_inds) const
{
    // --- hints from Python AbelianBackend.leg_pipe_map_incoming_block_inds ---
    // calculate indices of _block_ind_map by using the appropriate strides
    // now permute them to indices in _block_ind_map
    // ---
    if (static_cast<int64>(incoming_block_inds.ncols()) != pipe.num_legs) {
        throw std::invalid_argument(
          "leg_pipe_map_incoming_block_inds: ncols does not match pipe.num_legs");
    }
    // Pack multi-leg sector indices with pipe sector_strides, then apply inverse fusion sort.
    std::vector<int64> strides = pipe.sector_strides;
    auto packed = incoming_block_inds.pack(strides);
    std::vector<int64> inv(pipe.fusion_outcomes_sort.size());
    for (std::size_t i = 0; i < pipe.fusion_outcomes_sort.size(); ++i)
        inv[static_cast<std::size_t>(pipe.fusion_outcomes_sort[i])] = static_cast<int64>(i);
    BlockInds out(packed.size(), 1);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        auto const idx = static_cast<std::size_t>(packed[i]);
        out(i, 0) = inv[idx];
    }
    return out;
}

TensorBackend::DataPtr
AbelianBackend::to_dtype(TensorCPtr a, Dtype dtype)
{
    // --- hints from Python AbelianBackend.to_dtype ---
    // shallow copy if dtype stays same
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::to_block_backend(DataPtr data,
                                 std::shared_ptr<BlockBackend> bb,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    auto d = unwrap(data);
    Dtype dt = dtype.value_or(d->dtype);
    std::string dev =
      bb->as_device(device.has_value() ? device : std::optional<std::string>(d->device));
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(d->blocks.size());
    for (auto const& b : d->blocks)
        blocks.push_back(bb->as_block(py::cast(b), dt, dev));
    return wrap(make_data(dt, std::move(dev), std::move(blocks), d->block_inds));
}

TensorBackend::DataPtr
AbelianBackend::move_to_device(TensorCPtr a, std::string device)
{
    auto a_data = data_from_tensor(a);
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i)
        a_data->blocks[i] =
          block_backend->as_block(py::cast(a_data->blocks[i]), std::nullopt, device);
    a_data->device = block_backend->as_device(device);
    return wrap(a_data);
}

TensorBackend::DataPtr
AbelianBackend::full_data_from_diagonal_tensor(DiagonalTensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_diagonal(b));
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::full_data_from_mask(MaskCPtr a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_mask(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a, std::optional<float64> tol)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->get_diagonal(b, tol));
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

BlockBackend::Scalar
AbelianBackend::diagonal_tensor_trace_full(DiagonalTensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto total = block_backend->as_scalar(0.0, a->dtype);
    for (auto const& b : a_data->blocks)
        total = total + block_backend->sum_all(b);
    return total;
}

TensorBackend::DataPtr
AbelianBackend::mask_dagger(MaskCPtr mask)
{
    // --- hints from Python AbelianBackend.mask_dagger ---
    // the legs swap between domain and codomain. need to swap the two columns of block_inds.
    // since both columns are unique and ascending, the resulting block_inds are still sorted.
    // ---
    auto data = data_from_tensor(mask);
    BlockInds block_inds = data->block_inds.reverse_columns();
    return wrap(make_data(mask->dtype, mask->device, data->blocks, block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::permute_legs(TensorCPtr a,
                             std::vector<int64> codomain_idcs,
                             std::vector<int64> domain_idcs,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain,
                             bool mixes_codomain_domain,
                             std::vector<std::optional<int64>> levels,
                             std::vector<std::optional<bool>> bend_right)
{
    auto a_data = data_from_tensor(a);
    std::vector<int64> axes_perm = codomain_idcs;
    for (auto it = domain_idcs.rbegin(); it != domain_idcs.rend(); ++it)
        axes_perm.push_back(*it);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->permute_axes(b, axes_perm));
    BlockInds block_inds = a_data->block_inds.take_columns_i64(axes_perm);
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), block_inds, false));
}

std::tuple<Space::Ptr, TensorBackend::DataPtr>
AbelianBackend::diagonal_transpose(DiagonalTensorCPtr tens)
{
    // --- hints from Python AbelianBackend.diagonal_transpose ---
    // OPTIMIZE copy needed?
    // ---
    auto leg = py::cast(tens->leg()).cast<Space::Ptr>();
    return { leg->dual_space(), copy_data(tens) };
}

// ---- Remaining methods ----
// Native ports from lasyten/backends/abelian.py. See
// docs/cpp_conversion/convert_AbelianBackend.md.

TensorBackend::DataPtr
AbelianBackend::combine_legs(TensorCPtr tensor,
                             std::vector<std::vector<int64>> leg_idcs_combine,
                             std::vector<Leg::Ptr> pipes,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain)
{
    // --- hints from Python AbelianBackend.combine_legs ---
    // which combined legs are formed in C and F style
    // build new block_inds, compatible with old_blocks, but contain duplicates and are not sorted
    // res_block_inds[:, :i] is already set
    // old_block_inds[:, :j] are already considered
    // uncombined legs since last group: block_inds are simply unchanged
    // current combined group
    // product space in the domain has opposite order of its spaces compared to the
    // convention in block_inds
    // for each row of block_inds, find the corresponding row of pipe.block_ind_map
    // trailing uncombined legs:
    // sort the new block_inds
    // determine, for each old block, which slices of the new block it should occupy
    // have already set info for new_legs[:i]
    // have already considered old_legs[:j]
    // uncombined legs since last group: slice is all of 0:mult
    // block_slices[:, i, 0] = 0 is already set
    // trailing uncombined legs
    // identify the duplicates in res_block_inds
    // all those old_blocks are embedded into a single new block
    // includes both 0 and len, to have slices later
    // build the new blocks
    // we lexsort( .T)-ed res_block_inds while it still had duplicates, and then indexed by diffs,
    // which is sorted and thus preserves lexsort( .T)-ing of res_block_inds
    // ---
    for (auto const& p : pipes) {
        if (!std::dynamic_pointer_cast<AbelianLegPipe>(p.as_elementary_space()))
            throw std::invalid_argument("abelian backend requires AbelianLegPipe");
    }
    auto t_data = data_from_tensor(tensor);
    int64 num_result_legs = tensor->num_legs;
    for (auto const& group : leg_idcs_combine)
        num_result_legs -= static_cast<int64>(group.size()) - 1;
    auto old_blocks = t_data->blocks;
    std::vector<bool> cstyles;
    BlockInds res_block_inds(t_data->block_inds.nrows(),
                             static_cast<std::size_t>(num_result_legs));
    int64 i = 0, j = 0;
    std::vector<std::vector<int64>> map_inds;
    int64 num_codomain = tensor->num_codomain_legs();
    auto const& t_bi = t_data->block_inds;
    for (std::size_t gi = 0; gi < leg_idcs_combine.size(); ++gi) {
        auto const& group = leg_idcs_combine[gi];
        auto pipe = std::dynamic_pointer_cast<AbelianLegPipe>(pipes[gi].as_elementary_space());
        int64 num_uncombined = group[0] - j;
        if (num_uncombined > 0) {
            std::vector<std::size_t> src_cols(static_cast<std::size_t>(num_uncombined));
            std::iota(src_cols.begin(), src_cols.end(), static_cast<std::size_t>(j));
            res_block_inds.assign_columns(static_cast<std::size_t>(i),
                                          t_bi.take_columns(src_cols));
        }
        i += num_uncombined;
        j += num_uncombined;
        bool in_domain = group[0] >= num_codomain;
        cstyles.push_back(pipe->combine_cstyle != in_domain);
        std::vector<std::size_t> group_cols;
        group_cols.reserve(group.size());
        for (int64 c = group.front(); c <= group.back(); ++c)
            group_cols.push_back(static_cast<std::size_t>(c));
        BlockInds bi_group = t_bi.take_columns(group_cols);
        if (in_domain)
            bi_group = bi_group.reverse_columns();
        auto multi_indices = bi_group.pack(pipe->sector_strides);
        std::vector<int64> inv_fusion(pipe->fusion_outcomes_sort.size());
        for (std::size_t k = 0; k < pipe->fusion_outcomes_sort.size(); ++k) {
            inv_fusion[static_cast<std::size_t>(pipe->fusion_outcomes_sort[k])] =
              static_cast<int64>(k);
        }
        std::vector<int64> block_ind_map_rows(multi_indices.size());
        for (std::size_t k = 0; k < multi_indices.size(); ++k) {
            block_ind_map_rows[k] = inv_fusion[static_cast<std::size_t>(multi_indices[k])];
        }
        map_inds.push_back(block_ind_map_rows);
        res_block_inds.set_column(static_cast<std::size_t>(i),
                                  pipe->block_ind_map.take_i64(block_ind_map_rows)
                                    .column(pipe->block_ind_map.ncols() - 1));
        i += 1;
        j += static_cast<int64>(group.size());
    }
    if (i < num_result_legs) {
        std::vector<std::size_t> src_cols;
        for (int64 c = j; c < static_cast<int64>(t_bi.ncols()); ++c)
            src_cols.push_back(static_cast<std::size_t>(c));
        res_block_inds.assign_columns(static_cast<std::size_t>(i), t_bi.take_columns(src_cols));
    }
    auto sort = res_block_inds.lexsort_indices();
    res_block_inds = res_block_inds.take(sort);
    {
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(old_blocks.size());
        for (auto idx : sort)
            sorted_blocks.push_back(old_blocks[idx]);
        old_blocks = std::move(sorted_blocks);
    }
    {
        std::vector<std::vector<int64>> map_inds_sorted;
        map_inds_sorted.reserve(map_inds.size());
        for (auto const& rows : map_inds) {
            std::vector<int64> sorted_rows(sort.size());
            for (std::size_t k = 0; k < sort.size(); ++k)
                sorted_rows[k] = rows[sort[k]];
            map_inds_sorted.push_back(std::move(sorted_rows));
        }
        map_inds = std::move(map_inds_sorted);
    }

    py::array_t<int64> block_slices =
      zeros_i64_array({ static_cast<py::ssize_t>(old_blocks.size()),
                        static_cast<py::ssize_t>(num_result_legs),
                        2 });
    i = 0;
    j = 0;
    for (std::size_t gi = 0; gi < leg_idcs_combine.size(); ++gi) {
        auto const& group = leg_idcs_combine[gi];
        auto pipe = std::dynamic_pointer_cast<AbelianLegPipe>(pipes[gi].as_elementary_space());
        auto const& block_ind_map_rows = map_inds[gi];
        int64 num_uncombined = group[0] - j;
        for (int64 u = 0; u < num_uncombined; ++u) {
            py::object mults =
              py::cast(tensor).attr("get_leg_co_domain")(j).attr("multiplicities");
            block_slices.attr("__setitem__")(
              py::make_tuple(py::ellipsis(), i, 1),
              mults.attr("__getitem__")(
                i64_vec_to_numpy(res_block_inds.column(static_cast<std::size_t>(i)))));
            ++i;
            ++j;
        }
        BlockInds slice_cols = pipe->block_ind_map.take_i64(block_ind_map_rows)
                                 .take_columns(std::array<std::size_t, 2>{ 0, 1 });
        block_slices.attr("__setitem__")(py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                        i,
                                                        py::slice(std::nullopt, std::nullopt, 1)),
                                         block_inds_to_numpy(slice_cols));
        ++i;
        j += static_cast<int64>(group.size());
    }
    int64 num_legs = tensor->num_legs;
    while (j < num_legs) {
        py::object mults = py::cast(tensor).attr("get_leg_co_domain")(j).attr("multiplicities");
        block_slices.attr("__setitem__")(py::make_tuple(py::ellipsis(), i, 1),
                                         mults.attr("__getitem__")(i64_vec_to_numpy(
                                           res_block_inds.column(static_cast<std::size_t>(i)))));
        ++i;
        ++j;
    }

    auto diffs_vec = res_block_inds.find_row_differences(/*include_len=*/true);
    py::ssize_t res_num_blocks = static_cast<py::ssize_t>(diffs_vec.size()) - 1;
    {
        std::vector<std::size_t> keep(diffs_vec.begin(), diffs_vec.end() - 1);
        res_block_inds = res_block_inds.take(keep);
    }
    py::array_t<int64> res_block_shapes =
      zeros_i64_array({ res_num_blocks, static_cast<py::ssize_t>(num_result_legs) });
    auto legs = conventional_leg_order(new_codomain, new_domain);
    for (std::size_t li = 0; li < legs.size(); ++li) {
        res_block_shapes.attr("__setitem__")(
          py::make_tuple(py::ellipsis(), static_cast<py::ssize_t>(li)),
          py::cast(legs[li])
            .attr("multiplicities")
            .attr("__getitem__")(i64_vec_to_numpy(res_block_inds.column(li))));
    }
    std::vector<BlockBackend::BlockPtr> res_blocks;
    auto shapes = asarray_i64_np(res_block_shapes);
    auto sbuf = shapes.unchecked<2>();
    auto slices_arr = asarray_i64_np(block_slices);
    // block_slices may be 3D - use numpy indexing instead
    Dtype dt = tensor->dtype;
    std::string device = tensor->device;
    for (py::ssize_t n = 0; n < res_num_blocks; ++n) {
        std::vector<int64> shape;
        for (py::ssize_t c = 0; c < sbuf.shape(1); ++c)
            shape.push_back(sbuf(n, c));
        auto new_block = block_backend->zeros(shape, dt, device);
        int64 start = static_cast<int64>(diffs_vec[static_cast<std::size_t>(n)]);
        int64 stop = static_cast<int64>(diffs_vec[static_cast<std::size_t>(n) + 1]);
        for (int64 row = start; row < stop; ++row) {
            py::list slc_list;
            py::object row_slices = block_slices.attr("__getitem__")(row);
            for (py::ssize_t ax = 0; ax < num_result_legs; ++ax) {
                py::object be = row_slices.attr("__getitem__")(ax);
                slc_list.append(py::slice(be.attr("__getitem__")(0).cast<py::ssize_t>(),
                                          be.attr("__getitem__")(1).cast<py::ssize_t>(),
                                          1));
            }
            auto combined = block_backend->combine_legs(
              old_blocks[static_cast<std::size_t>(row)], leg_idcs_combine, cstyles);
            b_set(new_block, py::tuple(slc_list), combined);
        }
        res_blocks.push_back(new_block);
    }
    return wrap(make_data(dt, t_data->device, std::move(res_blocks), res_block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    if (a->num_codomain_legs() == 0 && b->num_domain_legs() == 0) {
        // Python returns a Scalar here; wrap as 0-leg data with one scalar block.
        auto s = inner(a, b, false);
        auto block = block_backend->as_block(s.to_numpy(), a->dtype);
        return wrap(
          make_data(a->dtype, data_from_tensor(a)->device, { block }, zeros_i64(1, 0), true));
    }
    if (a->num_domain_legs() == 0)
        return _compose_no_contraction(a, b);
    return _compose_worker(a, b);
}

namespace {

AbelianBackendData::Ptr
abelian_compose_worker(AbelianBackend& self,
                       AbelianBackendData::Ptr a_data,
                       AbelianBackendData::Ptr b_data,
                       TensorProduct::Ptr new_codomain,
                       std::vector<Leg::Ptr> const& contr_spaces,
                       TensorProduct::Ptr new_domain)
{
    auto& bb = *self.block_backend;
    Dtype a_dtype = a_data->dtype;
    Dtype b_dtype = b_data->dtype;
    Dtype res_dtype = dtype::common({ a_dtype, b_dtype });
    if (a_data->blocks.empty() || b_data->blocks.empty())
        return AbelianBackend::unwrap(
          self.zero_data(new_codomain, new_domain, res_dtype, a_data->device));

    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    if (a_dtype != res_dtype)
        for (auto& B : a_blocks)
            B = bb.to_dtype(B, res_dtype);
    if (b_dtype != res_dtype)
        for (auto& B : b_blocks)
            B = bb.to_dtype(B, res_dtype);

    int64 num_contr = static_cast<int64>(contr_spaces.size());
    auto [a_block_inds_keep, a_contr_bi] =
      a_data->block_inds.hsplit(static_cast<std::size_t>(new_codomain->num_factors));
    auto [b_contr_bi, b_block_inds_keep] =
      b_data->block_inds.hsplit(static_cast<std::size_t>(num_contr));

    std::vector<int64> nsecs;
    for (auto const& l : contr_spaces)
        nsecs.push_back(nsec(l));
    auto strides_vec = make_stride(nsecs, /*cstyle=*/false);
    std::vector<int64> strides_rev = strides_vec;
    std::reverse(strides_rev.begin(), strides_rev.end());
    auto a_contr_keys = a_contr_bi.pack(strides_rev);
    auto b_contr_keys = b_contr_bi.pack(strides_vec);

    // Match np.lexsort(hstack([contr[:,None], keep]).T): last keep col primary, contr least.
    BlockInds a_sort_key =
      BlockInds::hstack({ BlockInds(a_contr_keys, a_contr_keys.size(), 1), a_block_inds_keep });
    auto a_sort = a_sort_key.lexsort_indices();
    a_block_inds_keep = a_block_inds_keep.take(a_sort);
    {
        std::vector<int64> sorted_keys;
        sorted_keys.reserve(a_contr_keys.size());
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(a_blocks.size());
        for (auto idx : a_sort) {
            sorted_keys.push_back(a_contr_keys[idx]);
            sorted_blocks.push_back(a_blocks[idx]);
        }
        a_contr_keys = std::move(sorted_keys);
        a_blocks = std::move(sorted_blocks);
    }

    auto a_slices = a_block_inds_keep.find_row_differences(/*include_len=*/true);
    auto b_slices = b_block_inds_keep.find_row_differences(/*include_len=*/true);

    std::vector<std::vector<BlockBackend::BlockPtr>> a_blocks_g, b_blocks_g;
    std::vector<std::vector<int64>> a_contr_g, b_contr_g;
    for (std::size_t g = 0; g + 1 < a_slices.size(); ++g) {
        auto i0 = a_slices[g], i1 = a_slices[g + 1];
        std::vector<BlockBackend::BlockPtr> grp;
        grp.reserve(i1 - i0);
        for (std::size_t k = i0; k < i1; ++k)
            grp.push_back(a_blocks[k]);
        a_blocks_g.push_back(std::move(grp));
        a_contr_g.emplace_back(a_contr_keys.begin() + static_cast<std::ptrdiff_t>(i0),
                               a_contr_keys.begin() + static_cast<std::ptrdiff_t>(i1));
    }
    for (std::size_t g = 0; g + 1 < b_slices.size(); ++g) {
        auto j0 = b_slices[g], j1 = b_slices[g + 1];
        std::vector<BlockBackend::BlockPtr> grp;
        grp.reserve(j1 - j0);
        for (std::size_t k = j0; k < j1; ++k)
            grp.push_back(b_blocks[k]);
        b_blocks_g.push_back(std::move(grp));
        b_contr_g.emplace_back(b_contr_keys.begin() + static_cast<std::ptrdiff_t>(j0),
                               b_contr_keys.begin() + static_cast<std::ptrdiff_t>(j1));
    }
    {
        std::vector<std::size_t> keep(a_slices.begin(), a_slices.end() - 1);
        a_block_inds_keep = a_block_inds_keep.take(keep);
    }
    {
        std::vector<std::size_t> keep(b_slices.begin(), b_slices.end() - 1);
        b_block_inds_keep = b_block_inds_keep.take(keep);
    }

    std::vector<std::vector<int64>> a_shape_keep, b_shape_keep;
    a_shape_keep.reserve(a_blocks_g.size());
    b_shape_keep.reserve(b_blocks_g.size());
    for (auto const& grp : a_blocks_g) {
        auto sh = bb.get_shape(grp[0]);
        a_shape_keep.emplace_back(sh.begin(), sh.begin() + new_codomain->num_factors);
    }
    for (auto const& grp : b_blocks_g) {
        auto sh = bb.get_shape(grp[0]);
        b_shape_keep.emplace_back(sh.begin() + num_contr, sh.end());
    }

    if (new_codomain->num_factors == 0) {
        for (auto& grp : a_blocks_g)
            for (auto& B : grp)
                B = bb.reshape(B, { -1 });
    } else {
        for (std::size_t g = 0; g < a_blocks_g.size(); ++g) {
            int64 prod = 1;
            for (auto s : a_shape_keep[g])
                prod *= s;
            for (auto& B : a_blocks_g[g])
                B = bb.reshape(B, { prod, -1 });
        }
    }
    if (new_domain->num_factors == 0) {
        std::vector<int64> perm;
        for (int64 p = num_contr - 1; p >= 0; --p)
            perm.push_back(p);
        for (auto& grp : b_blocks_g)
            for (auto& B : grp)
                B = bb.reshape(bb.permute_axes(B, perm), { -1 });
    } else {
        std::vector<int64> perm;
        for (int64 p = num_contr - 1; p >= 0; --p)
            perm.push_back(p);
        for (int64 p = num_contr; p < num_contr + new_domain->num_factors; ++p)
            perm.push_back(p);
        for (std::size_t g = 0; g < b_blocks_g.size(); ++g) {
            int64 prod = 1;
            for (auto s : b_shape_keep[g])
                prod *= s;
            for (auto& B : b_blocks_g[g])
                B = bb.reshape(bb.permute_axes(B, perm), { -1, prod });
        }
    }

    SectorArray a_charges;
    if (new_codomain->num_factors > 0) {
        std::vector<SectorArray> parts;
        for (int64 f = 0; f < new_codomain->num_factors; ++f) {
            auto const& secs =
              as_space(new_codomain->factors[static_cast<std::size_t>(f)])->sector_decomposition;
            SectorArray selected = SectorArray::empty(secs.sector_ind_len());
            for (std::size_t r = 0; r < a_block_inds_keep.nrows(); ++r)
                selected.push_back(secs[static_cast<std::size_t>(
                  a_block_inds_keep(r, static_cast<std::size_t>(f)))]);
            parts.push_back(std::move(selected));
        }
        a_charges = new_codomain->symmetry->multiple_fusion_broadcast(parts);
    } else {
        a_charges =
          SectorArray::repeat(new_codomain->symmetry->trivial_sector, a_block_inds_keep.nrows());
    }
    SectorArray b_charges;
    if (new_domain->num_factors > 0) {
        std::vector<SectorArray> parts;
        for (int64 f = 0; f < new_domain->num_factors; ++f) {
            auto const& secs =
              as_space(new_domain->factors[static_cast<std::size_t>(f)])->sector_decomposition;
            SectorArray selected = SectorArray::empty(secs.sector_ind_len());
            // b_block_inds_keep[:, ::-1] column f corresponds to domain factor f
            for (std::size_t r = 0; r < b_block_inds_keep.nrows(); ++r)
                selected.push_back(secs[static_cast<std::size_t>(b_block_inds_keep(
                  r, b_block_inds_keep.ncols() - 1 - static_cast<std::size_t>(f)))]);
            parts.push_back(std::move(selected));
        }
        b_charges = new_domain->symmetry->multiple_fusion_broadcast(parts);
    } else {
        b_charges =
          SectorArray::repeat(new_domain->symmetry->trivial_sector, b_block_inds_keep.nrows());
    }

    auto a_charge_lookup = list_to_dict_list(a_charges);

    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    for (std::size_t col_b = 0; col_b < b_charges.size(); ++col_b) {
        auto it = a_charge_lookup.find(b_charges[col_b]);
        if (it == a_charge_lookup.end())
            continue;
        for (int64 row_a : it->second) {
            std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> common_pairs;
            iter_common_sorted_1d(
              a_contr_g[static_cast<std::size_t>(row_a)],
              b_contr_g[col_b],
              [&](std::ptrdiff_t k1, std::ptrdiff_t k2) { common_pairs.emplace_back(k1, k2); });
            if (common_pairs.empty())
                continue;
            auto [k1, k2] = common_pairs[0];
            auto block = bb.matrix_dot(
              a_blocks_g[static_cast<std::size_t>(row_a)][static_cast<std::size_t>(k1)],
              b_blocks_g[col_b][static_cast<std::size_t>(k2)]);
            for (std::size_t pi = 1; pi < common_pairs.size(); ++pi) {
                std::tie(k1, k2) = common_pairs[pi];
                auto add = bb.matrix_dot(
                  a_blocks_g[static_cast<std::size_t>(row_a)][static_cast<std::size_t>(k1)],
                  b_blocks_g[col_b][static_cast<std::size_t>(k2)]);
                block = (*block) + (*add);
            }
            std::vector<int64> out_shape = a_shape_keep[static_cast<std::size_t>(row_a)];
            out_shape.insert(
              out_shape.end(), b_shape_keep[col_b].begin(), b_shape_keep[col_b].end());
            block = bb.reshape(block, out_shape);
            res_blocks.push_back(block);
            auto row_a_span = a_block_inds_keep.row(static_cast<std::size_t>(row_a));
            auto row_b_span = b_block_inds_keep.row(col_b);
            std::vector<int64> row;
            row.reserve(row_a_span.size() + row_b_span.size());
            row.insert(row.end(), row_a_span.begin(), row_a_span.end());
            row.insert(row.end(), row_b_span.begin(), row_b_span.end());
            res_rows.push_back(std::move(row));
        }
    }

    BlockInds block_inds;
    if (res_blocks.empty()) {
        block_inds = zeros_i64(0, new_codomain->num_factors + new_domain->num_factors);
    } else {
        block_inds = BlockInds::from_rows(res_rows);
    }
    return make_data(res_dtype, a_data->device, std::move(res_blocks), block_inds, true);
}

} // namespace

TensorBackend::DataPtr
AbelianBackend::_compose_worker(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    // --- hints from Python AbelianBackend._compose_worker ---
    // if there are no actual blocks to contract, we can directly return 0
    // convert blocks to common dtype
    // need to contract the domain legs of a with the codomain legs of b.
    // due to the leg ordering
    // Deal with the columns of the block inds that are kept/contracted separately
    // Merge the block_inds on the contracted legs to a single column, using strides.
    // Note: The order in a.data.block_inds is opposite from the order in b.data.block_inds!
    // I.e. a.data.block_inds[-1-n] and b.data.block_inds[n] describe one leg to contract
    // We choose F-style strides, by appearance in b.data.block_inds.
    // This guarantees that the b.data.block_inds sorting is preserved.
    // We do not care about the sorting of the a.data.block_inds, since we need to re-sort anyway,
    // to group by a_block_inds_keep.
    // 1D array
    // sort the a.data.block_inds *first* by the _keep, *then* by the _contr columns
    // The b_block_inds_* and b_blocks are already sorted like that.
    // now group everything that has matching *_block_inds_keep
    // Reshape blocks to matrices.
    // Reason: We could use block_tdot to do the pairwise block contractions.
    // This would then internally reshape to matrices, to use e.g. GEMM.
    // One of the a_blocks may be contracted with many different b_blocks, and require
    // the same reshape every time. Instead, we do it once at this point.
    // All blocks in a_blocks[n] have the same kept legs -> same kept shape
    // special case: reshape to vector.
    // need to permute the leg order of one group of permuted legs.
    // OPTIMIZE does it matter, which?
    // choose to permute the legs of the b-blocks
    // special case: reshape to vector
    // compute coupled sectors for all rows of the block inds // for all blocks
    // lookup table ``tuple(sector) -> idcs_in_a_charges``
    // rows_a changes faster than cols_b, such that the resulting block_inds are lex-sorted
    // empty list if no match
    // Use first pair of common indices to initialize a block.
    // for further pairs of common indices, add the result onto the existing block
    // finish up:
    // ---
    std::vector<Leg::Ptr> contr_spaces = b->codomain->factors;
    return wrap(abelian_compose_worker(
      *this, data_from_tensor(a), data_from_tensor(b), a->codomain, contr_spaces, b->domain));
}

TensorBackend::DataPtr
AbelianBackend::_compose_no_contraction(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    // --- hints from Python AbelianBackend._compose_no_contraction ---
    // grid is lexsorted, with rows as all combinations of a/b block indices.
    // Since the grid was in F-style, and the a_block_inds, b_block_inds are sorted,
    // the res_block_inds are sorted.
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    Dtype res_dtype = dtype::common({ a_data->dtype, b_data->dtype });
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    if (a_data->dtype != res_dtype)
        for (auto& T : a_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    if (b_data->dtype != res_dtype)
        for (auto& T : b_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    auto const& a_bi = a_data->block_inds;
    auto const& b_bi = b_data->block_inds;
    auto l_a = static_cast<int64>(a_bi.nrows());
    auto num_a = static_cast<py::ssize_t>(a_bi.ncols());
    auto l_b = static_cast<int64>(b_bi.nrows());
    auto num_b = static_cast<py::ssize_t>(b_bi.ncols());
    auto grid = make_grid({ l_a, l_b }, /*cstyle=*/false);
    BlockInds res_bi(grid.size(), static_cast<std::size_t>(num_a + num_b));
    for (std::size_t r = 0; r < grid.size(); ++r) {
        auto ia = static_cast<std::size_t>(grid[r][0]);
        auto ib = static_cast<std::size_t>(grid[r][1]);
        for (py::ssize_t c = 0; c < num_a; ++c)
            res_bi(r, static_cast<std::size_t>(c)) = a_bi(ia, static_cast<std::size_t>(c));
        for (py::ssize_t c = 0; c < num_b; ++c)
            res_bi(r, static_cast<std::size_t>(num_a + c)) = b_bi(ib, static_cast<std::size_t>(c));
    }
    std::vector<BlockBackend::BlockPtr> res_blocks;
    res_blocks.reserve(grid.size());
    for (std::size_t r = 0; r < grid.size(); ++r)
        res_blocks.push_back(block_backend->outer(a_blocks[static_cast<std::size_t>(grid[r][0])],
                                                  b_blocks[static_cast<std::size_t>(grid[r][1])]));
    return wrap(make_data(res_dtype, a_data->device, std::move(res_blocks), res_bi, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                            DiagonalTensorCPtr b,
                                            BlockBinaryFn func,
                                            bool partial_zero_is_zero)
{
    // --- hints from Python AbelianBackend.diagonal_elementwise_binary ---
    // OPTIMIZE should we drop zero blocks after?
    // next block of a to process
    // block_ind of that block => it belongs to leg.sector_decomposition[bi_a]
    // next block of b to process
    // block_ind of that block => it belongs to leg.sector_decomposition[bi_b]
    // a has no further blocks
    // b has no further blocks
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto leg = py::cast(a->leg());
    auto mults = mults_of(leg);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto const& a_bi = a_data->block_inds;
    auto const& b_bi = b_data->block_inds;
    std::size_t ia = 0, ib = 0;
    int64 bi_a = a_bi.nrows() == 0 ? -1 : a_bi(0, 0);
    int64 bi_b = b_bi.nrows() == 0 ? -1 : b_bi(0, 0);
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> block_ind_list;
    Dtype a_dtype = a->dtype;
    for (std::size_t i = 0; i < mults.size(); ++i) {
        BlockBackend::BlockPtr block_a;
        if (static_cast<int64>(i) == bi_a) {
            block_a = a_blocks[ia];
            ++ia;
            bi_a = (ia >= a_bi.nrows()) ? -1 : a_bi(ia, 0);
        } else if (partial_zero_is_zero) {
            continue;
        } else {
            block_a = block_backend->zeros({ mults[i] }, a_dtype);
        }
        BlockBackend::BlockPtr block_b;
        if (static_cast<int64>(i) == bi_b) {
            block_b = b_blocks[ib];
            ++ib;
            bi_b = (ib >= b_bi.nrows()) ? -1 : b_bi(ib, 0);
        } else if (partial_zero_is_zero) {
            continue;
        } else {
            block_b = block_backend->zeros({ mults[i] }, a_dtype);
        }
        blocks.push_back(func(block_a, block_b));
        block_ind_list.push_back(static_cast<int64>(i));
    }
    BlockInds block_inds;
    Dtype dt;
    if (blocks.empty()) {
        block_inds = zeros_i64(0, 2);
        auto sample = func(block_backend->ones_block({ 1 }, a_dtype),
                           block_backend->ones_block({ 1 }, b->dtype));
        dt = block_backend->get_dtype(sample);
    } else {
        block_inds = BlockInds::column_stack(
          std::vector<std::span<const int64>>{ block_ind_list, block_ind_list });
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_from_block(BlockBackend::BlockPtr a,
                                    TensorProduct::Ptr co_domain,
                                    float64 /*tol*/)
{
    py::object leg = py::cast(co_domain->factors[0]);
    Dtype dt = block_backend->get_dtype(a);
    auto block_inds = BlockInds::arange_diag(static_cast<std::size_t>(co_domain->num_sectors));
    std::vector<BlockBackend::BlockPtr> blocks;
    auto const& bi = block_inds;
    for (py::ssize_t r = 0; r < static_cast<py::ssize_t>(bi.nrows()); ++r) {
        auto slc = slice_pair(leg.attr("slices").attr("__getitem__")(bi(r, 0)));
        blocks.push_back(b_get(a, slc));
    }
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                                TensorProduct::Ptr co_domain)
{
    py::object leg = py::cast(co_domain->factors[0]);
    auto block_inds = BlockInds::arange_diag(static_cast<std::size_t>(nsec(leg)));
    auto sectors = leg.attr("sector_decomposition").cast<SectorArray>();
    auto mults = mults_of(leg);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t i = 0; i < mults.size(); ++i) {
        blocks.push_back(func(std::vector<int64>{ mults[i] }, sectors[i]));
    }
    BlockBackend::BlockPtr sample =
      blocks.empty() ? func(std::vector<int64>{ 1 }, co_domain->symmetry->trivial_sector)
                     : blocks[0];
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds,
                          true));
}

BlockBackend::BlockPtr
AbelianBackend::diagonal_tensor_to_block(DiagonalTensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto leg = py::cast(a->leg());
    auto res =
      block_backend->zeros({ static_cast<int64>(leg.attr("dim").cast<float64>()) }, a->dtype);
    auto const& bi = a_data->block_inds;
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        auto slc =
          slice_pair(leg.attr("slices").attr("__getitem__")(bi(static_cast<py::ssize_t>(i), 0)));
        b_set(res, slc, a_data->blocks[i]);
    }
    return res;
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::diagonal_to_mask(DiagonalTensorCPtr tens)
{
    auto tens_data = data_from_tensor(tens);
    py::object large_leg = py::cast(tens->leg());
    py::object basis_perm = large_leg.attr("_basis_perm");
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> large_leg_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto const& bi = tens_data->block_inds;
    for (std::size_t n = 0; n < tens_data->blocks.size(); ++n) {
        auto const& diag_block = tens_data->blocks[n];
        if (!block_backend->any(diag_block))
            continue;
        int64 bii = bi(static_cast<py::ssize_t>(n), 0);
        blocks.push_back(diag_block);
        large_leg_block_inds.push_back(bii);
        sectors_vec.push_back(defining[static_cast<std::size_t>(bii)]);
        multiplicities.push_back(block_backend->sum_all(diag_block).as_int64());
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(diag_block, py_bool_dtype());
            // fallback: to_numpy with bool dtype
            try {
                mask = block_backend->to_numpy(diag_block, dtype::to_numpy_dtype(Dtype::Bool))
                         .cast<py::array>();
            } catch (...) {
                mask = block_backend->to_numpy(diag_block).cast<py::array>();
            }
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bii));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    SectorArray sectors = py::cast(tens->symmetry).attr("empty_sector_array").cast<SectorArray>();
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    BlockInds block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = rank_concat(basis_perm_ranks);
            basis_perm_opt = ranked;
        }
        {
            std::vector<int64> ar(sectors.size());
            std::iota(ar.begin(), ar.end(), int64{ 0 });
            block_inds = BlockInds::column_stack(
              { std::span<const int64>(ar), std::span<const int64>(large_leg_block_inds) });
        }
    }
    auto data = make_data(Dtype::Bool, tens_data->device, std::move(blocks), block_inds, true);
    auto small_leg =
      std::make_shared<ElementarySpace>(py::cast(tens->symmetry).cast<Symmetry::Ptr>(),
                                        std::move(sectors),
                                        multiplicities,
                                        large_leg.attr("is_dual").cast<bool>(),
                                        basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::eigh(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    // --- hints from Python AbelianBackend.eigh ---
    // in tensors.py, we do pre-processing such that the following holds:
    // such that we can use the same block_inds
    // for missing blocks, i.e. a zero block, the eigenvalues are zero, so we can just skip
    // adding that block to the eigenvalues.
    // for the eigenvectors, we choose the computational basis vectors, i.e. the matrix
    // representation within that block is the identity matrix.
    // we initialize all blocks to eye and override those where `a` has blocks.
    // ---
    require_matrix_form(a, "eigh");
    auto a_data = data_from_tensor(a);
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    auto new_leg = domain->as_ElementarySpace(new_leg_dual);
    auto v_wrapped = eye_data(domain, a->dtype, a_data->device);
    auto v_data = unwrap(v_wrapped);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    auto const& bi = a_data->block_inds;
    std::optional<std::string> sort_opt = sort;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        auto [vals, vects] = block_backend->eigh(a_data->blocks[n], sort_opt);
        w_blocks.push_back(vals);
        v_data->blocks[static_cast<std::size_t>(bi(static_cast<py::ssize_t>(n), 0))] = vects;
    }
    auto w_data = make_data(
      dtype::to_real(a->dtype), a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), wrap(v_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::eig(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    require_matrix_form(a, "eig");
    auto a_data = data_from_tensor(a);
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    auto new_leg = domain->as_ElementarySpace(new_leg_dual);
    Dtype cdtype = dtype::to_complex(a->dtype);
    auto v_wrapped = eye_data(domain, cdtype, a_data->device);
    auto v_data = unwrap(v_wrapped);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    auto const& bi = a_data->block_inds;
    std::optional<std::string> sort_opt = sort;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        auto [vals, vects] = block_backend->eig(a_data->blocks[n], sort_opt);
        w_blocks.push_back(vals);
        v_data->blocks[static_cast<std::size_t>(bi(static_cast<py::ssize_t>(n), 0))] = vects;
    }
    auto w_data = make_data(cdtype, a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), wrap(v_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::eigvalsh(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    require_matrix_form(a, "eigvalsh");
    auto a_data = data_from_tensor(a);
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    auto new_leg = domain->as_ElementarySpace(new_leg_dual);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    std::optional<std::string> sort_opt = sort;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        w_blocks.push_back(block_backend->eigvalsh(a_data->blocks[n], sort_opt));
    }
    auto w_data = make_data(
      dtype::to_real(a->dtype), a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::eigvals(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    require_matrix_form(a, "eigvals");
    auto a_data = data_from_tensor(a);
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    auto new_leg = domain->as_ElementarySpace(new_leg_dual);
    Dtype cdtype = dtype::to_complex(a->dtype);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    std::optional<std::string> sort_opt = sort;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        w_blocks.push_back(block_backend->eigvals(a_data->blocks[n], sort_opt));
    }
    auto w_data = make_data(cdtype, a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), new_leg };
}

TensorBackend::DataPtr
AbelianBackend::eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device)
{
    // --- hints from Python AbelianBackend.eye_data ---
    // Note: the identity has the same matrix elements in all ONB, so ne need to consider
    // the basis perms.
    // results[i1,...im,jm,...,j1] = delta_{i1,j1} ... delta{im,jm}
    // need exactly the "diagonal" blocks, where sector of i1 matches the one of j1 etc.
    // to guarantee sorting later, it is easier to generate the block inds of the domain
    // domain_block_inds is by construction np.lexsort( .T)-ed.
    // since the last co_domain.num_spaces columns of block_inds are already unique, the first
    // columns are not relevant to np.lexsort( .T)-ing, thus the block_inds above is sorted.
    // ---
    std::vector<int64> domain_dims;
    domain_dims.reserve(static_cast<std::size_t>(co_domain->num_factors));
    for (auto it = co_domain->factors.rbegin(); it != co_domain->factors.rend(); ++it)
        domain_dims.push_back(nsec(*it));
    // domain_block_inds: all multi-indices, F-style like np.indices(...).T.reshape(-1, n)
    auto grid = make_grid(domain_dims, /*cstyle=*/false);
    BlockInds domain_block_inds = BlockInds::from_rows(grid);
    BlockInds block_inds =
      BlockInds::hstack({ domain_block_inds.reverse_columns(), domain_block_inds });
    std::vector<BlockBackend::BlockPtr> blocks;
    auto const& bi = block_inds;
    blocks.reserve(static_cast<std::size_t>(static_cast<py::ssize_t>(bi.nrows())));
    for (py::ssize_t r = 0; r < static_cast<py::ssize_t>(bi.nrows()); ++r) {
        std::vector<int64> shape;
        shape.reserve(static_cast<std::size_t>(co_domain->num_factors));
        for (int64 f = 0; f < co_domain->num_factors; ++f) {
            auto mults = mults_of(co_domain->factors[static_cast<std::size_t>(f)]);
            shape.push_back(mults[static_cast<std::size_t>(bi(r, f))]);
        }
        blocks.push_back(block_backend->eye_block(shape, dtype, device));
    }
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::from_dense_block(BlockBackend::BlockPtr a,
                                 TensorProduct::Ptr codomain,
                                 TensorProduct::Ptr domain,
                                 float64 tol)
{
    Dtype dt = block_backend->get_dtype(a);
    std::string device = block_backend->get_device(a);
    auto projected = block_backend->zeros(block_backend->get_shape(a), dt);
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<BlockBackend::BlockPtr> blocks;
    auto const& bi = block_inds;
    blocks.reserve(static_cast<std::size_t>(static_cast<py::ssize_t>(bi.nrows())));
    for (py::ssize_t r = 0; r < static_cast<py::ssize_t>(bi.nrows()); ++r) {
        py::tuple slices(static_cast<py::ssize_t>(legs.size()));
        for (py::ssize_t c = 0; c < static_cast<py::ssize_t>(legs.size()); ++c) {
            slices[c] = slice_pair(py::cast(legs[static_cast<std::size_t>(c)])
                                     .attr("slices")
                                     .attr("__getitem__")(bi(r, c)));
        }
        auto block = b_get(a, slices);
        blocks.push_back(block);
        b_set(projected, slices, block);
    }
    auto n_diff = block_backend->norm((*a) - (*projected));
    auto n_a = block_backend->norm(a);
    if (n_diff.as_float64() > tol * n_a.as_float64())
        throw std::invalid_argument("Block is not symmetric up to tolerance.");
    return wrap(make_data(dt, std::move(device), std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::from_dense_block_trivial_sector(BlockBackend::BlockPtr block, Space::Ptr leg)
{
    auto bi = leg->sector_decomposition_where(leg->symmetry->trivial_sector);
    if (!bi.has_value()) {
        throw std::invalid_argument(
          "from_dense_block_trivial_sector: leg does not contain the trivial sector");
    }
    auto block_inds = BlockInds::from_row(std::array<int64, 1>{ *bi });
    return wrap(make_data(block_backend->get_dtype(block),
                          block_backend->get_device(block),
                          { block },
                          block_inds,
                          true));
}

TensorBackend::DataPtr
AbelianBackend::from_grid(std::vector<std::vector<py::object>> grid,
                          TensorProduct::Ptr new_codomain,
                          TensorProduct::Ptr new_domain,
                          std::vector<std::vector<int64>> left_mult_slices,
                          std::vector<std::vector<int64>> right_mult_slices,
                          Dtype dtype,
                          std::string device)
{
    // --- hints from Python AbelianBackend.from_grid ---
    // all block inds apart from the ones for the row and col
    // must be identical to the ones of op
    // find block or create it if it does not exist yet
    // ---
    std::vector<BlockBackend::BlockPtr> blocks;
    auto block_inds = zeros_i64(0, new_codomain->num_factors + new_domain->num_factors);
    int64 n_cod = new_codomain->num_factors;
    int64 n_dom = new_domain->num_factors;
    py::tuple codom_slcs(static_cast<py::ssize_t>(std::max<int64>(0, n_cod - 1)));
    for (py::ssize_t i = 0; i < codom_slcs.size(); ++i)
        codom_slcs[i] = py::slice(std::nullopt, std::nullopt, std::nullopt);
    py::tuple dom_slcs(static_cast<py::ssize_t>(std::max<int64>(0, n_dom - 1)));
    for (py::ssize_t i = 0; i < dom_slcs.size(); ++i)
        dom_slcs[i] = py::slice(std::nullopt, std::nullopt, std::nullopt);
    auto legs = conventional_leg_order(new_codomain, new_domain);
    py::object new_cod0 = py::cast(new_codomain->factors[0]);
    py::object new_dom_last = py::cast(new_domain->factors[static_cast<std::size_t>(n_dom - 1)]);

    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid[i].size(); ++j) {
            py::object op = grid[i][j];
            if (op.is_none())
                continue;
            auto op_t = op.cast<TensorCPtr>();
            auto op_data = data_from_tensor(op_t);
            auto const& op_bi = op_data->block_inds;
            for (std::size_t bi_row = 0; bi_row < op_data->blocks.size(); ++bi_row) {
                auto left_sector =
                  py::cast(op_t->codomain)
                    .attr("__getitem__")(0)
                    .attr("sector_decomposition")
                    .attr("__getitem__")(op_bi(static_cast<py::ssize_t>(bi_row), 0));
                int64 left_ind =
                  new_cod0.attr("sector_decomposition_where")(left_sector).cast<int64>();
                int64 right_bi = op_bi(static_cast<py::ssize_t>(bi_row), n_cod);
                auto right_sector = py::cast(op_t->domain)
                                      .attr("__getitem__")(-1)
                                      .attr("sector_decomposition")
                                      .attr("__getitem__")(right_bi);
                int64 right_ind =
                  new_dom_last.attr("sector_decomposition_where")(right_sector).cast<int64>();

                py::list new_bi_list;
                new_bi_list.append(left_ind);
                for (int64 c = 1; c < n_cod; ++c)
                    new_bi_list.append(op_bi(static_cast<py::ssize_t>(bi_row), c));
                new_bi_list.append(right_ind);
                for (int64 c = n_cod + 1; c < static_cast<py::ssize_t>(op_bi.ncols()); ++c)
                    new_bi_list.append(op_bi(static_cast<py::ssize_t>(bi_row), c));
                std::vector<int64> new_bi_row;
                new_bi_row.reserve(static_cast<std::size_t>(py::len(new_bi_list)));
                for (auto item : new_bi_list)
                    new_bi_row.push_back(py::reinterpret_borrow<py::object>(item).cast<int64>());
                BlockInds new_bi = BlockInds::from_row(new_bi_row);

                std::size_t block_idx;
                auto match = block_inds.row_where(new_bi.row(0));
                if (!match.has_value()) {
                    block_idx = blocks.size();
                    block_inds = block_inds.concat(new_bi);
                    std::vector<int64> shape;
                    for (std::size_t c = 0; c < new_bi.ncols(); ++c)
                        shape.push_back(mults_of(legs[c])[static_cast<std::size_t>(new_bi(0, c))]);
                    blocks.push_back(block_backend->zeros(shape, dtype, device));
                } else {
                    block_idx = *match;
                }

                auto row_slc =
                  py::slice(right_mult_slices[static_cast<std::size_t>(right_ind)][j],
                            right_mult_slices[static_cast<std::size_t>(right_ind)][j + 1],
                            1);
                auto col_slc =
                  py::slice(left_mult_slices[static_cast<std::size_t>(left_ind)][i],
                            left_mult_slices[static_cast<std::size_t>(left_ind)][i + 1],
                            1);
                py::tuple block_slcs(
                  static_cast<py::ssize_t>(2 + codom_slcs.size() + dom_slcs.size()));
                py::ssize_t pos = 0;
                block_slcs[pos++] = col_slc;
                for (py::ssize_t k = 0; k < codom_slcs.size(); ++k)
                    block_slcs[pos++] = codom_slcs[k];
                block_slcs[pos++] = row_slc;
                for (py::ssize_t k = 0; k < dom_slcs.size(); ++k)
                    block_slcs[pos++] = dom_slcs[k];
                b_set_add(blocks[block_idx], block_slcs, op_data->blocks[bi_row]);
            }
        }
    }
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, false));
}

TensorBackend::DataPtr
AbelianBackend::from_random_normal(TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain,
                                   float64 sigma,
                                   Dtype dtype,
                                   std::string device)
{
    auto self = this;
    return from_sector_block_func(
      [self, sigma, dtype, device](std::vector<int64> const& shape, Sector const& /*coupled*/) {
          return self->block_backend->random_normal(shape, dtype, sigma, device);
      },
      codomain,
      domain);
}

TensorBackend::DataPtr
AbelianBackend::from_sector_block_func(SectorBlockFactoryFn func,
                                       TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain)
{
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    int64 M = codomain->num_factors;
    std::vector<BlockBackend::BlockPtr> blocks;
    auto const& bi = block_inds;
    for (py::ssize_t r = 0; r < static_cast<py::ssize_t>(bi.nrows()); ++r) {
        std::vector<int64> shape;
        for (py::ssize_t c = 0; c < static_cast<py::ssize_t>(bi.ncols()); ++c)
            shape.push_back(
              mults_of(legs[static_cast<std::size_t>(c)])[static_cast<std::size_t>(bi(r, c))]);
        std::vector<Sector> secs;
        for (int64 i = 0; i < M; ++i) {
            auto const& sectors =
              as_space(codomain->factors[static_cast<std::size_t>(i)])->sector_decomposition;
            secs.push_back(sectors[static_cast<std::size_t>(bi(r, i))]);
        }
        auto coupled = codomain->symmetry->multiple_fusion(secs);
        blocks.push_back(func(shape, coupled));
    }
    BlockBackend::BlockPtr sample;
    if (blocks.empty()) {
        std::vector<int64> shape(static_cast<std::size_t>(M + domain->num_factors), 1);
        sample = func(shape, codomain->symmetry->trivial_sector);
    } else {
        sample = blocks[0];
    }
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds,
                          true));
}

TensorBackend::DataPtr
AbelianBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype dtype,
  std::string device)
{
    // --- hints from Python AbelianBackend.from_tree_pairs ---
    // check if we covered all keys in the dict
    // SymmetricTensor.from_tree_pairs should have done enough input checks to prevent this
    // OPTIMIZE if the code works, we could remove this check
    // ---
    auto block_inds_all = valid_block_inds(codomain, domain);
    std::vector<BlockBackend::BlockPtr> blocks;
    py::list bi_rows;
    std::set<std::pair<FusionTree, FusionTree>> pairs_done;
    auto const& bi = block_inds_all;
    for (py::ssize_t r = 0; r < static_cast<py::ssize_t>(bi.nrows()); ++r) {
        SectorArray unc_c = SectorArray::empty(codomain->symmetry->sector_ind_len);
        std::vector<std::uint8_t> dual_c;
        for (int64 n = 0; n < codomain->num_factors; ++n) {
            py::object f = py::cast(codomain->factors[static_cast<std::size_t>(n)]);
            auto secs = f.attr("sector_decomposition").cast<SectorArray>();
            unc_c.push_back(secs[static_cast<std::size_t>(bi(r, n))]);
            dual_c.push_back(static_cast<std::uint8_t>(f.attr("is_dual").cast<bool>()));
        }
        SectorArray unc_d = SectorArray::empty(domain->symmetry->sector_ind_len);
        std::vector<std::uint8_t> dual_d;
        for (int64 n = 0; n < domain->num_factors; ++n) {
            py::object f = py::cast(domain->factors[static_cast<std::size_t>(n)]);
            auto secs = f.attr("sector_decomposition").cast<SectorArray>();
            unc_d.push_back(
              secs[static_cast<std::size_t>(bi(r, static_cast<py::ssize_t>(bi.ncols()) - 1 - n))]);
            dual_d.push_back(static_cast<std::uint8_t>(f.attr("is_dual").cast<bool>()));
        }
        FusionTree X = FusionTree::from_abelian_symmetry(codomain->symmetry, unc_c, dual_c);
        FusionTree Y = FusionTree::from_abelian_symmetry(domain->symmetry, unc_d, dual_d);
        auto pair = std::make_pair(X, Y);
        pairs_done.insert(pair);
        auto it = trees.find(pair);
        if (it == trees.end())
            continue;
        bi_rows.append(
          py::cast(std::vector<int64>(block_inds_all.row(static_cast<std::size_t>(r)).begin(),
                                      block_inds_all.row(static_cast<std::size_t>(r)).end())));
        blocks.push_back(it->second);
    }
    for (auto const& kv : trees) {
        if (pairs_done.find(kv.first) == pairs_done.end())
            throw std::runtime_error("from_tree_pairs: unexpected tree pair");
    }
    BlockInds block_inds;
    if (bi_rows.size() == 0)
        block_inds = zeros_i64(0, codomain->num_factors + domain->num_factors);
    else
        block_inds = block_inds_from_py_rows(bi_rows);
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, false));
}

BlockBackend::Scalar
AbelianBackend::get_element(SymmetricTensorCPtr a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    py::list rows;
    std::vector<std::vector<int64>> pairs;
    pairs.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto [b, w] = parse_leg_index(legs[i], idcs[i]);
        pairs.push_back({ b, w });
    }
    BlockInds pos = BlockInds::from_rows(pairs);
    // pos shape (num_legs, 2): [:,0]=block_idx, [:,1]=within
    auto block_idcs = BlockInds::from_row(pos.column(0));
    auto a_data = data_from_tensor(a);
    auto block = a_data->get_block(block_idcs);
    if (!block) {
        Dtype dt = a->dtype;
        return block_backend->as_scalar(dtype::zero_scalar(dt), dt);
    }
    auto within = pos.column(1);
    return block_backend->get_block_element(block, within);
}

BlockBackend::Scalar
AbelianBackend::get_element_diagonal(DiagonalTensorCPtr a, int64 idx)
{
    auto [block_idx, idx_within] = parse_leg_index(a->leg(), idx);
    auto query = BlockInds::from_row(std::array<int64, 2>{ block_idx, block_idx });
    auto block = data_from_tensor(a)->get_block(query);
    if (!block) {
        Dtype dt = a->dtype;
        return block_backend->as_scalar(dtype::zero_scalar(dt), dt);
    }
    return block_backend->get_block_element(block, { idx_within });
}

BlockBackend::Scalar
AbelianBackend::get_element_mask(MaskCPtr a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    std::vector<std::vector<int64>> pairs;
    pairs.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto [b, w] = parse_leg_index(legs[i], idcs[i]);
        pairs.push_back({ b, w });
    }
    BlockInds pos = BlockInds::from_rows(pairs);
    BlockInds block_idcs = BlockInds::from_row(pos.column(0));
    auto block = data_from_tensor(a)->get_block(block_idcs);
    if (!block)
        return block_backend->as_scalar(false);
    auto within = pos.column(1);
    int64 small, large;
    if (a->is_projection) {
        small = within[0];
        large = within[1];
    } else {
        large = within[0];
        small = within[1];
    }
    return block_backend->get_block_mask_element(block, large, small);
}

BlockBackend::Scalar
AbelianBackend::inner(SymmetricTensorCPtr a, SymmetricTensorCPtr b, bool do_dagger)
{
    // --- hints from Python AbelianBackend.inner ---
    // F-style strides for block_inds -> preserve sorting
    // these are not sorted:
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    std::vector<int64> nsecs;
    for (auto const& leg : py::cast(a->legs()))
        nsecs.push_back(leg.attr("num_sectors").cast<int64>());
    auto strides_vec = make_stride(nsecs, /*cstyle=*/false);
    auto a_keys = a_data->block_inds.pack(strides_vec);
    std::vector<int64> b_keys;
    if (do_dagger) {
        b_keys = b_data->block_inds.pack(strides_vec);
    } else {
        std::vector<int64> rev = strides_vec;
        std::reverse(rev.begin(), rev.end());
        b_keys = b_data->block_inds.pack(rev);
        std::vector<std::size_t> order(b_keys.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
            return b_keys[i] < b_keys[j];
        });
        std::vector<int64> sorted_keys;
        sorted_keys.reserve(b_keys.size());
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(b_blocks.size());
        for (auto i : order) {
            sorted_keys.push_back(b_keys[i]);
            sorted_blocks.push_back(b_blocks[i]);
        }
        b_keys = std::move(sorted_keys);
        b_blocks = std::move(sorted_blocks);
    }
    auto res = block_backend->as_scalar(0.0, a->dtype);
    iter_common_sorted_1d(a_keys, b_keys, [&](std::ptrdiff_t i, std::ptrdiff_t j) {
        res = res + block_backend->inner(a_blocks[static_cast<std::size_t>(i)],
                                         b_blocks[static_cast<std::size_t>(j)],
                                         do_dagger);
    });
    return res;
}

TensorBackend::DataPtr
AbelianBackend::inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                        Space::Ptr space,
                                                        ElementarySpace::Ptr charge_leg)
{
    if (charge_leg->num_sectors != 1) {
        throw std::invalid_argument(
          "inv_part_from_dense_block_single_sector: charge_leg must have exactly one sector");
    }
    auto bi = space->sector_decomposition_where(charge_leg->sector_decomposition[0]);
    if (!bi.has_value()) {
        throw std::invalid_argument(
          "inv_part_from_dense_block_single_sector: space missing charge sector");
    }
    auto expect_shape = std::vector<int64>{ space->multiplicities[static_cast<std::size_t>(*bi)] };
    if (block_backend->get_shape(vector) != expect_shape) {
        throw std::invalid_argument("inv_part_from_dense_block_single_sector: vector shape does "
                                    "not match sector multiplicity");
    }
    auto block_inds = BlockInds::from_row(std::array<int64, 2>{ *bi, 0 });
    return wrap(make_data(block_backend->get_dtype(vector),
                          block_backend->get_device(vector),
                          { block_backend->add_axis(vector, 1) },
                          block_inds,
                          true));
}

BlockBackend::BlockPtr
AbelianBackend::inv_part_to_dense_block_single_sector(SymmetricTensorCPtr tensor)
{
    auto data = data_from_tensor(tensor);
    if (data->blocks.size() > 1) {
        throw std::invalid_argument(
          "inv_part_to_dense_block_single_sector: expected at most one block");
    }
    if (data->blocks.size() == 1) {
        return b_get(data->blocks[0],
                     py::make_tuple(py::slice(std::nullopt, std::nullopt, std::nullopt), 0));
    }
    auto sector = py::cast(tensor->domain)
                    .attr("__getitem__")(0)
                    .attr("sector_decomposition")
                    .attr("__getitem__")(0)
                    .cast<Sector>();
    int64 dim = py::cast(tensor->codomain)
                  .attr("__getitem__")(0)
                  .attr("sector_multiplicity")(py::cast(sector))
                  .cast<int64>();
    return block_backend->zeros({ dim }, data->dtype);
}

TensorBackend::DataPtr
AbelianBackend::linear_combination(BlockBackend::Scalar a,
                                   TensorCPtr v,
                                   BlockBackend::Scalar b,
                                   TensorCPtr w)
{
    // --- hints from Python AbelianBackend.linear_combination ---
    // ensure common dtypes
    // ---
    auto v_data = data_from_tensor(v);
    auto w_data = data_from_tensor(w);
    auto v_blocks = v_data->blocks;
    auto w_blocks = w_data->blocks;
    Dtype common_dtype = dtype::common({ v->dtype, w->dtype });
    if (v_data->dtype != common_dtype)
        for (auto& T : v_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    if (w_data->dtype != common_dtype)
        for (auto& T : w_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    BlockInds::iter_common_noncommon_sorted(
      v_data->block_inds,
      w_data->block_inds,
      [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
          if (!j) {
              res_blocks.push_back(block_backend->mul(a, v_blocks[static_cast<std::size_t>(*i)]));
              auto row = v_data->block_inds.row(static_cast<std::size_t>(*i));
              res_rows.emplace_back(row.begin(), row.end());
          } else if (!i) {
              res_blocks.push_back(block_backend->mul(b, w_blocks[static_cast<std::size_t>(*j)]));
              auto row = w_data->block_inds.row(static_cast<std::size_t>(*j));
              res_rows.emplace_back(row.begin(), row.end());
          } else {
              res_blocks.push_back(
                block_backend->linear_combination(a,
                                                  v_blocks[static_cast<std::size_t>(*i)],
                                                  b,
                                                  w_blocks[static_cast<std::size_t>(*j)]));
              auto row = v_data->block_inds.row(static_cast<std::size_t>(*i));
              res_rows.emplace_back(row.begin(), row.end());
          }
      });
    BlockInds res_block_inds =
      res_rows.empty() ? zeros_i64(0, v->num_legs) : BlockInds::from_rows(res_rows);
    return wrap(
      make_data(common_dtype, v_data->device, std::move(res_blocks), res_block_inds, true));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::lq(SymmetricTensorCPtr tensor, TensorProduct::Ptr new_co_domain)
{
    // --- hints from Python AbelianBackend.lq ---
    // since self.can_decompose_tensors is False
    // running index, indicating we have already processed a_blocks[:i]
    // due to the loop setup we have:
    // a.codomain.sector_decomposition[j] == new_leg.sector_decomposition[n]
    // a.domain.sector_decomposition[k] == new_leg.sector_decomposition[n]
    // but we still need the leg indices (which may differ depending on the sector_order)
    // block_inds is lexsorted and in this case duplicate-free
    // -> running index i is correct iff domain is correctly sorted
    // we have a block for that sector -> decompose it
    // we do not have a block for that sector
    // => L_block == 0 and we dont even set it.
    // can choose arbitrary blocks for q, as long as they are isometric
    // ---
    require_matrix_form(tensor, "lq");
    auto a_data = data_from_tensor(tensor);
    py::object new_leg = py::cast(new_co_domain->factors[0]);
    auto cod0 = py::cast(tensor->codomain).attr("__getitem__")(0);
    auto dom0 = py::cast(tensor->domain).attr("__getitem__")(0);
    auto a_blocks = a_data->blocks;
    auto a_block_inds = a_data->block_inds;
    std::vector<BlockBackend::BlockPtr> l_blocks, q_blocks;
    py::list l_block_inds, q_block_inds;
    int64 i = 0;
    auto const& codom_secs = tensor->codomain->sector_decomposition;
    auto const& dom_secs = tensor->domain->sector_decomposition;
    int64 n_enum = 0;
    SectorArray::iter_common_sorted(
      codom_secs, dom_secs, true, true, [&](std::ptrdiff_t ji, std::ptrdiff_t ki) {
          int64 j = static_cast<int64>(ji);
          int64 k = static_cast<int64>(ki);
          int64 n = n_enum++;
          py::object sector = py::cast(codom_secs[static_cast<std::size_t>(j)]);
          if (cod0.attr("sector_order").cast<std::string>() != "sorted")
              j = cod0.attr("sector_decomposition_where")(sector).cast<int64>();
          if (dom0.attr("sector_order").cast<std::string>() != "sorted") {
              k = dom0.attr("sector_decomposition_where")(sector).cast<int64>();
              i = static_cast<int64>(a_block_inds.searchsorted_column(1, k));
          }
          if (new_leg.attr("sector_order").cast<std::string>() != "sorted")
              n = new_leg.attr("sector_decomposition_where")(sector).cast<int64>();

          if (i < static_cast<int64>(a_block_inds.nrows()) &&
              a_block_inds(static_cast<std::size_t>(i), 0) == j) {
              auto [l, q] = block_backend->matrix_lq(a_blocks[static_cast<std::size_t>(i)], false);
              l_blocks.push_back(l);
              q_blocks.push_back(q);
              l_block_inds.append(py::make_tuple(j, n));
              ++i;
          } else {
              int64 new_leg_dim = mults_of(new_leg)[static_cast<std::size_t>(n)];
              auto eye = block_backend->eye_matrix(
                mults_of(dom0)[static_cast<std::size_t>(k)], tensor->dtype, std::nullopt);
              q_blocks.push_back(
                b_get(eye,
                      py::make_tuple(py::slice(0, new_leg_dim, 1),
                                     py::slice(std::nullopt, std::nullopt, std::nullopt))));
          }
          q_block_inds.append(py::make_tuple(n, k));
      });
    BlockInds l_bi = l_blocks.empty() ? zeros_i64(0, 2) : block_inds_from_py_rows(l_block_inds);
    BlockInds q_bi = q_blocks.empty() ? zeros_i64(0, 2) : block_inds_from_py_rows(q_block_inds);
    bool l_sorted = new_leg.attr("sector_order").cast<std::string>() == "sorted";
    bool q_sorted = dom0.attr("sector_order").cast<std::string>() == "sorted";
    return { wrap(make_data(tensor->dtype, a_data->device, std::move(l_blocks), l_bi, l_sorted)),
             wrap(make_data(tensor->dtype, a_data->device, std::move(q_blocks), q_bi, q_sorted)) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_binary_operand(MaskCPtr mask1, MaskCPtr mask2, BlockBinaryFn func)
{
    // --- hints from Python AbelianBackend.mask_binary_operand ---
    // next block of mask1 to process
    // its block_ind for the large leg.
    // mask1 has no further blocks
    // mask2 has no further blocks
    // ---
    py::object large_leg = py::cast(mask1->large_leg());
    py::object basis_perm = large_leg.attr("_basis_perm");
    auto mask1_data = data_from_tensor(mask1);
    auto mask2_data = data_from_tensor(mask2);
    auto const& mask1_bi = mask1_data->block_inds;
    auto const& mask2_bi = mask2_data->block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> large_leg_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    std::size_t i1 = 0, i2 = 0;
    int64 b1_i1 = mask1_bi.nrows() == 0 ? -1 : mask1_bi(0, 1);
    int64 b2_i2 = mask2_bi.nrows() == 0 ? -1 : mask2_bi(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg.attr("slices");
    auto mults = mults_of(large_leg);
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block1, block2;
        if (static_cast<int64>(sector_idx) == b1_i1) {
            block1 = mask1_data->blocks[i1];
            ++i1;
            b1_i1 = (i1 >= mask1_bi.nrows()) ? -1 : mask1_bi(i1, 1);
        } else {
            block1 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        }
        if (static_cast<int64>(sector_idx) == b2_i2) {
            block2 = mask2_data->blocks[i2];
            ++i2;
            // Python bug used mask1_block_inds here; use mask2.
            b2_i2 = (i2 >= mask2_bi.nrows()) ? -1 : mask2_bi(i2, 1);
        } else {
            block2 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        }
        auto new_block = func(block1, block2);
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        large_leg_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(new_block).cast<py::array>();
            auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    SectorArray sectors = py::cast(mask1->symmetry).attr("empty_sector_array").cast<SectorArray>();
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    BlockInds block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = rank_concat(basis_perm_ranks);
            basis_perm_opt = ranked;
        }
        std::vector<int64> arange(sectors.size());
        std::iota(arange.begin(), arange.end(), int64{ 0 });
        block_inds = BlockInds::column_stack(
          std::vector<std::span<const int64>>{ arange, large_leg_block_inds });
    }
    auto data = make_data(Dtype::Bool, mask1->device, std::move(blocks), block_inds, true);
    auto small_leg =
      std::make_shared<ElementarySpace>(py::cast(mask1->symmetry).cast<Symmetry::Ptr>(),
                                        std::move(sectors),
                                        multiplicities,
                                        large_leg.attr("is_dual").cast<bool>(),
                                        basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, true);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, false);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::_mask_contract(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx, bool large_leg)
{
    // --- hints from Python AbelianBackend._mask_contract ---
    // sort by the contracted rows
    // otherwise, if leg_idx == -1, the tensor_block_inds_contr are sorted
    // otherwise it is already sorted
    // need to iterate only over the "common" blocks. If either block is zero, so is the result
    // OPTIMIZE (JU) block_inds might actually be sorted but i am not sure right now
    // ---
    auto parsed = tensor->_parse_leg_idx(leg_idx);
    bool in_domain = std::get<0>(parsed);
    int64 co_domain_idx = std::get<1>(parsed);
    leg_idx = std::get<2>(parsed);
    int64 mask_contr;
    if (in_domain) {
        if (mask->is_projection == large_leg) {
            throw std::invalid_argument(
              "_mask_contract: domain leg requires inclusion mask when large_leg is set, "
              "or projection mask when large_leg is unset");
        }
        mask_contr = 0;
    } else {
        if (mask->is_projection != large_leg) {
            throw std::invalid_argument(
              "_mask_contract: codomain leg requires projection mask when large_leg is set, "
              "or inclusion mask when large_leg is unset");
        }
        mask_contr = 1;
    }
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    auto tensor_blocks = t_data->blocks;
    BlockInds tensor_block_inds = t_data->block_inds;
    BlockInds tensor_block_inds_contr = tensor_block_inds.take_columns(
      std::array<std::size_t, 1>{ static_cast<std::size_t>(leg_idx) });
    auto mask_blocks = m_data->blocks;
    BlockInds mask_block_inds = m_data->block_inds;
    BlockInds mask_block_inds_contr = mask_block_inds.take_columns(
      std::array<std::size_t, 1>{ static_cast<std::size_t>(mask_contr) });
    if (leg_idx != tensor->num_legs - 1) {
        auto sort = tensor_block_inds_contr.lexsort_indices();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(tensor_blocks.size());
        for (auto idx : sort)
            sorted_blocks.push_back(tensor_blocks[idx]);
        tensor_blocks = std::move(sorted_blocks);
        tensor_block_inds = tensor_block_inds.take(sort);
        tensor_block_inds_contr = tensor_block_inds.take_columns(
          std::array<std::size_t, 1>{ static_cast<std::size_t>(leg_idx) });
    }
    if (mask_contr == 0) {
        auto sort = mask_block_inds_contr.lexsort_indices();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(mask_blocks.size());
        for (auto idx : sort)
            sorted_blocks.push_back(mask_blocks[idx]);
        mask_blocks = std::move(sorted_blocks);
        mask_block_inds = mask_block_inds.take(sort);
        mask_block_inds_contr = mask_block_inds.take_columns(
          std::array<std::size_t, 1>{ static_cast<std::size_t>(mask_contr) });
    }
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    BlockInds::iter_common_sorted(
      tensor_block_inds_contr,
      mask_block_inds_contr,
      /*a_strict=*/false,
      /*b_strict=*/true,
      [&](std::ptrdiff_t ii, std::ptrdiff_t jj) {
          BlockBackend::BlockPtr block;
          if (large_leg)
              block = block_backend->apply_mask(tensor_blocks[static_cast<std::size_t>(ii)],
                                                mask_blocks[static_cast<std::size_t>(jj)],
                                                leg_idx);
          else
              block = block_backend->enlarge_leg(tensor_blocks[static_cast<std::size_t>(ii)],
                                                 mask_blocks[static_cast<std::size_t>(jj)],
                                                 leg_idx);
          auto row_span = tensor_block_inds.row(static_cast<std::size_t>(ii));
          std::vector<int64> bi_row(row_span.begin(), row_span.end());
          bi_row[static_cast<std::size_t>(leg_idx)] = mask_block_inds(
            static_cast<std::size_t>(jj), static_cast<std::size_t>(1 - mask_contr));
          res_blocks.push_back(block);
          res_rows.push_back(std::move(bi_row));
      });
    BlockInds res_block_inds =
      res_rows.empty() ? zeros_i64(0, tensor->num_legs) : BlockInds::from_rows(res_rows);
    auto data =
      make_data(tensor->dtype, tensor->device, std::move(res_blocks), res_block_inds, false);
    TensorProduct::Ptr codomain, domain;
    if (in_domain) {
        codomain = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
        auto spaces = tensor->domain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] =
          large_leg ? Leg{ mask->small_leg() } : Leg{ mask->large_leg() };
        domain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    } else {
        domain = tensor->domain;
        auto spaces = tensor->codomain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] =
          large_leg ? Leg{ mask->small_leg() } : Leg{ mask->large_leg() };
        codomain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    }
    return { wrap(data), codomain, domain };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    py::object large_leg_obj = py::cast(large_leg);
    py::object basis_perm = large_leg_obj.attr("_basis_perm");
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> large_leg_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg_obj.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg_obj.attr("slices");
    for (std::size_t bi_large = 0; bi_large < defining.size(); ++bi_large) {
        auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(bi_large)));
        auto block = b_get(a, slc);
        int64 mult = block_backend->sum_all(block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(block);
        large_leg_block_inds.push_back(static_cast<int64>(bi_large));
        sectors_vec.push_back(defining[bi_large]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(block).cast<py::array>();
            basis_perm_ranks.append(
              large_leg_obj.attr("basis_perm").attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    SectorArray sectors =
      large_leg_obj.attr("symmetry").attr("empty_sector_array").cast<SectorArray>();
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    BlockInds block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = rank_concat(basis_perm_ranks);
            basis_perm_opt = ranked;
        }
        {
            std::vector<int64> ar(sectors.size());
            std::iota(ar.begin(), ar.end(), int64{ 0 });
            block_inds = BlockInds::column_stack(
              { std::span<const int64>(ar), std::span<const int64>(large_leg_block_inds) });
        }
    }
    auto data =
      make_data(Dtype::Bool, block_backend->get_device(a), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(large_leg->symmetry,
                                                       std::move(sectors),
                                                       multiplicities,
                                                       large_leg_obj.attr("is_dual").cast<bool>(),
                                                       basis_perm_opt);
    return { wrap(data), small_leg };
}

BlockBackend::BlockPtr
AbelianBackend::mask_to_block(MaskCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto large_leg = py::cast(a->large_leg());
    auto res = block_backend->zeros({ static_cast<int64>(large_leg.attr("dim").cast<float64>()) },
                                    Dtype::Bool);
    bool is_projection = a->is_projection;
    auto const& bi = a_data->block_inds;
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        int64 bi_large =
          is_projection ? bi(static_cast<py::ssize_t>(i), 1) : bi(static_cast<py::ssize_t>(i), 0);
        auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bi_large));
        b_set(res, slc, a_data->blocks[i]);
    }
    return res;
}

TensorBackend::DataPtr
AbelianBackend::mask_to_diagonal(MaskCPtr a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    auto large_leg_bi =
      a->is_projection ? a_data->block_inds.column(1) : a_data->block_inds.column(0);
    BlockInds block_inds =
      BlockInds::column_stack(std::vector<std::span<const int64>>{ large_leg_bi, large_leg_bi });
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds));
}

std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
AbelianBackend::mask_transpose(MaskCPtr tens)
{
    auto data = data_from_tensor(tens);
    auto block_inds = data->block_inds.reverse_columns();
    auto out = make_data(tens->dtype, data->device, data->blocks, block_inds, false);
    auto cod = py::cast(tens->codomain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    auto dom = py::cast(tens->domain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    return { cod, dom, wrap(out) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_unary_operand(MaskCPtr mask, BlockUnaryFn func)
{
    // --- hints from Python AbelianBackend.mask_unary_operand ---
    // mask has no further blocks
    // ---
    py::object large_leg = py::cast(mask->large_leg());
    py::object basis_perm = large_leg.attr("_basis_perm");
    auto mask_data = data_from_tensor(mask);
    auto const& mask_bi = mask_data->block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> large_leg_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    std::size_t i = 0;
    int64 b_i = mask_bi.nrows() == 0 ? -1 : mask_bi(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg.attr("slices");
    auto mults = mults_of(large_leg);
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block;
        if (static_cast<int64>(sector_idx) == b_i) {
            block = mask_data->blocks[i];
            ++i;
            b_i = (i >= mask_bi.nrows()) ? -1 : mask_bi(i, 1);
        } else {
            block = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        }
        auto new_block = func(block);
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        large_leg_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            py::array mask_np = block_backend->to_numpy(new_block).cast<py::array>();
            auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(
              large_leg.attr("basis_perm").attr("__getitem__")(slc).attr("__getitem__")(mask_np));
        }
    }
    SectorArray sectors = py::cast(mask->symmetry).attr("empty_sector_array").cast<SectorArray>();
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    BlockInds block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = rank_concat(basis_perm_ranks);
            basis_perm_opt = ranked;
        }
        std::vector<int64> arange(sectors.size());
        std::iota(arange.begin(), arange.end(), int64{ 0 });
        block_inds = BlockInds::column_stack(
          std::vector<std::span<const int64>>{ arange, large_leg_block_inds });
    }
    auto data = make_data(Dtype::Bool, mask_data->device, std::move(blocks), block_inds, true);
    auto small_leg =
      std::make_shared<ElementarySpace>(py::cast(mask->symmetry).cast<Symmetry::Ptr>(),
                                        std::move(sectors),
                                        multiplicities,
                                        large_leg.attr("is_dual").cast<bool>(),
                                        basis_perm_opt);
    return { wrap(data), small_leg };
}

TensorBackend::DataPtr
AbelianBackend::mul(BlockBackend::Scalar a, TensorCPtr b)
{
    auto b_data = data_from_tensor(b);
    if (is_zero_scalar(a))
        return zero_data(py::cast(b->codomain).cast<TensorProduct::Ptr>(),
                         py::cast(b->domain).cast<TensorProduct::Ptr>(),
                         b->dtype,
                         b_data->device);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& T : b_data->blocks)
        blocks.push_back(block_backend->mul(a, T));
    Dtype dt;
    if (blocks.empty()) {
        dt = dtype::is_real(a.dtype()) ? b_data->dtype : dtype::to_complex(b_data->dtype);
    } else {
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, b_data->device, std::move(blocks), b_data->block_inds, true));
}

BlockBackend::Scalar
AbelianBackend::norm(TensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto block_norms =
      block_backend->zeros({ static_cast<int64>(a_data->blocks.size()) }, a->dtype);
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        auto n = block_backend->norm(a_data->blocks[i], 2., std::nullopt);
        block_norms->set_item(static_cast<int64>(i), n);
    }
    return block_backend->norm(block_norms, 2., std::nullopt);
}

TensorBackend::DataPtr
AbelianBackend::outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    // --- hints from Python AbelianBackend.outer ---
    // convert to common dtype
    // res_block_inds are in general not sorted.
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto const& a_bi = a_data->block_inds;
    auto const& b_bi = b_data->block_inds;
    auto l_a = static_cast<int64>(a_bi.nrows());
    auto N_a = static_cast<py::ssize_t>(a_bi.ncols());
    auto l_b = static_cast<int64>(b_bi.nrows());
    auto N_b = static_cast<py::ssize_t>(b_bi.ncols());
    int64 K_a = a->num_codomain_legs();
    Dtype res_dtype = dtype::common({ a->dtype, b->dtype });
    if (a->dtype != res_dtype)
        for (auto& T : a_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    if (b->dtype != res_dtype)
        for (auto& T : b_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    auto grid = make_grid({ l_a, l_b }, /*cstyle=*/false);
    BlockInds res_bi(grid.size(), static_cast<std::size_t>(N_a + N_b));
    {
        for (std::size_t r = 0; r < grid.size(); ++r) {
            auto ia = static_cast<std::size_t>(grid[r][0]);
            auto ib = static_cast<std::size_t>(grid[r][1]);
            for (py::ssize_t c = 0; c < K_a; ++c)
                res_bi(r, static_cast<std::size_t>(c)) = a_bi(ia, static_cast<std::size_t>(c));
            for (py::ssize_t c = 0; c < N_b; ++c)
                res_bi(r, static_cast<std::size_t>(K_a + c)) =
                  b_bi(ib, static_cast<std::size_t>(c));
            for (py::ssize_t c = K_a; c < N_a; ++c)
                res_bi(r, static_cast<std::size_t>(K_a + N_b + (c - K_a))) =
                  a_bi(ia, static_cast<std::size_t>(c));
        }
    }
    std::vector<BlockBackend::BlockPtr> res_blocks;
    res_blocks.reserve(grid.size());
    for (std::size_t r = 0; r < grid.size(); ++r) {
        res_blocks.push_back(
          block_backend->tensor_outer(a_blocks[static_cast<std::size_t>(grid[r][0])],
                                      b_blocks[static_cast<std::size_t>(grid[r][1])],
                                      K_a));
    }
    return wrap(make_data(res_dtype, a_data->device, std::move(res_blocks), res_bi, false));
}

TensorBackend::DataPtr
AbelianBackend::partial_compose(SymmetricTensorCPtr a,
                                SymmetricTensorCPtr b,
                                int64 a_first_leg,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain)
{
    // --- hints from Python AbelianBackend.partial_compose ---
    // construct new data and spaces with the legs to be contracted at the end of a and the
    // beginning of b the computation of these modified tensorproducts cannot be avoided since they
    // may differ from the ones computed in _tensors.py by bending
    // ---
    auto a_data0 = data_from_tensor(a);
    auto b_data0 = data_from_tensor(b);
    int64 a_n_cod = a->num_codomain_legs();
    int64 a_n_legs = a->num_legs;
    int64 b_n_cod = b->num_codomain_legs();
    int64 b_n_dom = b->num_domain_legs();
    int64 b_n_legs = b->num_legs;

    int64 num_contr_legs;
    int64 num_add_legs;
    std::vector<int64> perm_b;
    AbelianBackendData::Ptr b_data;
    if (a_first_leg < a_n_cod) {
        num_contr_legs = b_n_dom;
        num_add_legs = b_n_cod;
        for (int64 idx = b_n_cod; idx < b_n_legs; ++idx)
            perm_b.push_back(idx);
        for (int64 idx = 0; idx < b_n_cod; ++idx)
            perm_b.push_back(idx);
        std::vector<BlockBackend::BlockPtr> b_blocks;
        for (auto const& blk : b_data0->blocks)
            b_blocks.push_back(block_backend->permute_axes(blk, perm_b));
        auto bi = b_data0->block_inds.take_columns_i64(perm_b);
        b_data = make_data(b_data0->dtype, b_data0->device, std::move(b_blocks), bi, false);
    } else {
        num_contr_legs = b_n_cod;
        num_add_legs = b_n_dom;
        for (int64 idx = 0; idx < b_n_legs; ++idx)
            perm_b.push_back(idx);
        b_data = b_data0;
    }

    std::vector<int64> perm_a;
    for (int64 idx = 0; idx < a_first_leg; ++idx)
        perm_a.push_back(idx);
    for (int64 idx = a_first_leg + num_contr_legs; idx < a_n_legs; ++idx)
        perm_a.push_back(idx);
    for (int64 idx = a_first_leg; idx < a_first_leg + num_contr_legs; ++idx)
        perm_a.push_back(idx);
    std::vector<BlockBackend::BlockPtr> a_blocks;
    for (auto const& blk : a_data0->blocks)
        a_blocks.push_back(block_backend->permute_axes(blk, perm_a));
    auto a_bi = a_data0->block_inds.take_columns_i64(perm_a);
    auto a_data = make_data(a_data0->dtype, a_data0->device, std::move(a_blocks), a_bi, false);

    std::vector<Leg::Ptr> mod_codomain_legs;
    for (std::size_t i = 0; i < perm_a.size(); ++i) {
        if (static_cast<int64>(i) < a_n_legs - num_contr_legs)
            mod_codomain_legs.push_back(
              py::cast(a).attr("_as_codomain_leg")(perm_a[i]).cast<Leg::Ptr>());
    }
    auto mod_codomain = std::make_shared<TensorProduct>(mod_codomain_legs, a->symmetry);

    std::vector<Leg::Ptr> mod_domain_legs;
    for (std::size_t i = 0; i < perm_b.size(); ++i) {
        if (static_cast<int64>(i) >= num_contr_legs)
            mod_domain_legs.push_back(
              py::cast(b).attr("_as_domain_leg")(perm_b[i]).cast<Leg::Ptr>());
    }
    std::reverse(mod_domain_legs.begin(), mod_domain_legs.end());
    auto mod_domain = std::make_shared<TensorProduct>(mod_domain_legs,
                                                      py::cast(a->symmetry).cast<Symmetry::Ptr>());

    std::vector<Leg::Ptr> contr_spaces;
    for (std::size_t i = 0; i < perm_b.size(); ++i) {
        if (static_cast<int64>(i) < num_contr_legs)
            contr_spaces.push_back(
              py::cast(b).attr("get_leg_co_domain")(perm_b[i]).cast<Leg::Ptr>());
    }

    auto res_data =
      abelian_compose_worker(*this, a_data, b_data, mod_codomain, contr_spaces, mod_domain);

    std::vector<int64> perm_res;
    for (int64 idx = 0; idx < a_first_leg; ++idx)
        perm_res.push_back(idx);
    for (int64 idx = a_n_legs - num_contr_legs; idx < a_n_legs - num_contr_legs + num_add_legs;
         ++idx)
        perm_res.push_back(idx);
    for (int64 idx = a_first_leg; idx < a_n_legs - num_contr_legs; ++idx)
        perm_res.push_back(idx);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    for (auto const& blk : res_data->blocks)
        res_blocks.push_back(block_backend->permute_axes(blk, perm_res));
    auto res_bi = res_data->block_inds.take_columns_i64(perm_res);
    return wrap(
      make_data(res_data->dtype, res_data->device, std::move(res_blocks), res_bi, false));
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::partial_trace(SymmetricTensorCPtr tensor,
                              std::vector<std::pair<int64, int64>> pairs,
                              std::vector<std::optional<int64>> levels)
{
    // --- hints from Python AbelianBackend.partial_trace ---
    // if pairs[n] has one leg each in codomain and domain or if they are both on the same side
    // only blocks "on the diagonal" of the trace contribute.
    // figure out which blocks are on the diagonal.
    // we do logical_and, so we start with all true
    // legs are the same -> can compare the block_inds
    // legs have opposite duality. need to compare sectors explicitly
    // OPTIMIZE (JU) spaces could store (or cache!) the sector permutation between
    // itself and its dual, then we could compare on the level of block_inds
    // dictionary res_block_inds_row -> Block
    // by charge rule, should be impossible to get multiple blocks.
    // ---
    int64 N = tensor->num_legs;
    int64 K = tensor->num_codomain_legs();
    std::vector<int64> idcs1, idcs2;
    std::vector<bool> opposite_sides;
    std::set<int64> used;
    for (auto const& pr : pairs) {
        idcs1.push_back(pr.first);
        idcs2.push_back(pr.second);
        opposite_sides.push_back((pr.first < K) != (pr.second < K));
        used.insert(pr.first);
        used.insert(pr.second);
    }
    std::vector<int64> remaining;
    for (int64 n = 0; n < N; ++n)
        if (!used.count(n))
            remaining.push_back(n);

    auto t_data = data_from_tensor(tensor);
    auto blocks = t_data->blocks;
    BlockInds bi1 = t_data->block_inds.take_columns_i64(idcs1);
    BlockInds bi2 = t_data->block_inds.take_columns_i64(idcs2);
    BlockInds bi_rem =
      remaining.empty() ? BlockInds::empty(0) : t_data->block_inds.take_columns_i64(remaining);

    std::vector<bool> on_diagonal(blocks.size(), true);
    for (std::size_t n = 0; n < opposite_sides.size(); ++n) {
        if (opposite_sides[n]) {
            for (std::size_t r = 0; r < bi1.nrows(); ++r)
                on_diagonal[r] = on_diagonal[r] && (bi1(r, n) == bi2(r, n));
        } else {
            auto leg1 = py::cast(tensor).attr("get_leg_co_domain")(idcs1[n]);
            auto leg2 = py::cast(tensor).attr("get_leg_co_domain")(idcs2[n]);
            auto secs1 = leg1.attr("sector_decomposition").cast<SectorArray>();
            auto secs2 = leg2.attr("sector_decomposition").cast<SectorArray>();
            SectorArray s1 = SectorArray::empty(secs1.sector_ind_len());
            SectorArray s2 = SectorArray::empty(secs2.sector_ind_len());
            for (std::size_t r = 0; r < bi1.nrows(); ++r) {
                s1.push_back(secs1[static_cast<std::size_t>(bi1(r, n))]);
                s2.push_back(secs2[static_cast<std::size_t>(bi2(r, n))]);
            }
            auto dual_s2 = py::cast(tensor->symmetry).cast<Symmetry::Ptr>()->dual_sectors(s2);
            for (std::size_t r = 0; r < s1.size(); ++r)
                on_diagonal[r] = on_diagonal[r] && (s1[r] == dual_s2[r]);
        }
    }

    std::map<std::vector<int64>, BlockBackend::BlockPtr> res_map;
    for (std::size_t row = 0; row < blocks.size(); ++row) {
        if (!on_diagonal[row])
            continue;
        std::vector<int64> key;
        if (bi_rem.ncols() > 0) {
            auto br = bi_rem.row(row);
            key.assign(br.begin(), br.end());
        }
        auto block = block_backend->trace_partial(blocks[row], idcs1, idcs2, remaining);
        auto it = res_map.find(key);
        if (it != res_map.end())
            it->second = (*(it->second)) + (*block);
        else
            res_map.emplace(std::move(key), block);
    }

    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_keys;
    for (auto const& kv : res_map) {
        res_blocks.push_back(kv.second);
        res_keys.push_back(kv.first);
    }

    Dtype dt = tensor->dtype;
    if (remaining.empty()) {
        if (res_blocks.empty()) {
            auto s = block_backend->as_scalar(dtype::zero_scalar(dt), dt);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            auto data = make_data(dt, t_data->device, { block }, zeros_i64(1, 0), true);
            return { wrap(data), nullptr, nullptr };
        }
        if (res_blocks.size() == 1) {
            auto s = block_backend->item(res_blocks[0]);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            auto data = make_data(dt, t_data->device, { block }, zeros_i64(1, 0), true);
            return { wrap(data), nullptr, nullptr };
        }
        throw std::runtime_error("partial_trace: multiple blocks for scalar result");
    }

    BlockInds res_block_inds;
    if (res_blocks.empty())
        res_block_inds = zeros_i64(0, static_cast<std::size_t>(remaining.size()));
    else
        res_block_inds = BlockInds::from_rows(res_keys);

    auto data = make_data(dt, t_data->device, std::move(res_blocks), res_block_inds, false);

    std::vector<Leg::Ptr> cod_legs;
    for (int64 n = 0; n < K; ++n)
        if (std::find(remaining.begin(), remaining.end(), n) != remaining.end())
            cod_legs.push_back(tensor->codomain->factors[static_cast<std::size_t>(n)]);
    std::vector<Leg::Ptr> dom_legs;
    int64 n_dom = tensor->domain->num_factors;
    for (int64 n = 0; n < n_dom; ++n) {
        int64 leg_idx = N - 1 - n;
        if (std::find(remaining.begin(), remaining.end(), leg_idx) != remaining.end())
            dom_legs.push_back(tensor->domain->factors[static_cast<std::size_t>(n)]);
    }
    auto sym = tensor->symmetry;
    auto new_codomain = std::make_shared<TensorProduct>(cod_legs, sym);
    auto new_domain = std::make_shared<TensorProduct>(dom_legs, sym);
    return { wrap(data), new_codomain, new_domain };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::qr(SymmetricTensorCPtr a, TensorProduct::Ptr new_co_domain)
{
    // --- hints from Python AbelianBackend.qr ---
    // => R_block == 0 and we dont even set it.
    // ---
    require_matrix_form(a, "qr");
    auto a_data = data_from_tensor(a);
    py::object new_leg = py::cast(new_co_domain->factors[0]);
    auto cod0 = py::cast(a->codomain).attr("__getitem__")(0);
    auto dom0 = py::cast(a->domain).attr("__getitem__")(0);
    auto a_blocks = a_data->blocks;
    auto a_block_inds = a_data->block_inds;
    std::vector<BlockBackend::BlockPtr> q_blocks, r_blocks;
    py::list q_block_inds, r_block_inds;
    int64 i = 0;
    auto const& codom_secs = a->codomain->sector_decomposition;
    auto const& dom_secs = a->domain->sector_decomposition;
    int64 n_enum = 0;
    SectorArray::iter_common_sorted(
      codom_secs, dom_secs, true, true, [&](std::ptrdiff_t ji, std::ptrdiff_t ki) {
          int64 j = static_cast<int64>(ji);
          int64 k = static_cast<int64>(ki);
          int64 n = n_enum++;
          py::object sector = py::cast(codom_secs[static_cast<std::size_t>(j)]);
          if (cod0.attr("sector_order").cast<std::string>() != "sorted")
              j = cod0.attr("sector_decomposition_where")(sector).cast<int64>();
          if (dom0.attr("sector_order").cast<std::string>() != "sorted") {
              k = dom0.attr("sector_decomposition_where")(sector).cast<int64>();
              i = static_cast<int64>(a_block_inds.searchsorted_column(1, k));
          }
          if (new_leg.attr("sector_order").cast<std::string>() != "sorted")
              n = new_leg.attr("sector_decomposition_where")(sector).cast<int64>();

          if (i < static_cast<int64>(a_block_inds.nrows()) &&
              a_block_inds(static_cast<std::size_t>(i), 0) == j) {
              auto [q, r] = block_backend->matrix_qr(a_blocks[static_cast<std::size_t>(i)], false);
              q_blocks.push_back(q);
              r_blocks.push_back(r);
              r_block_inds.append(py::make_tuple(n, k));
              ++i;
          } else {
              int64 new_leg_dim = mults_of(new_leg)[static_cast<std::size_t>(n)];
              auto eye = block_backend->eye_matrix(
                mults_of(cod0)[static_cast<std::size_t>(j)], a->dtype, std::nullopt);
              q_blocks.push_back(
                b_get(eye,
                      py::make_tuple(py::slice(std::nullopt, std::nullopt, std::nullopt),
                                     py::slice(0, new_leg_dim, 1))));
          }
          q_block_inds.append(py::make_tuple(j, n));
      });
    BlockInds q_bi = q_blocks.empty() ? zeros_i64(0, 2) : block_inds_from_py_rows(q_block_inds);
    BlockInds r_bi = r_blocks.empty() ? zeros_i64(0, 2) : block_inds_from_py_rows(r_block_inds);
    bool q_sorted = new_leg.attr("sector_order").cast<std::string>() == "sorted";
    bool r_sorted = dom0.attr("sector_order").cast<std::string>() == "sorted";
    return { wrap(make_data(a->dtype, a_data->device, std::move(q_blocks), q_bi, q_sorted)),
             wrap(make_data(a->dtype, a_data->device, std::move(r_blocks), r_bi, r_sorted)) };
}

BlockBackend::Scalar
AbelianBackend::reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                      BlockToScalarFn block_func,
                                      ScalarReduceFn func)
{
    auto data = data_from_tensor(tensor);
    auto mults = mults_of(py::cast(tensor->leg()));
    std::vector<BlockBackend::Scalar> numbers;
    py::ssize_t i = 0;
    auto const& bi = data->block_inds;
    for (std::size_t j = 0; j < mults.size(); ++j) {
        BlockBackend::BlockPtr block;
        if (i < static_cast<py::ssize_t>(bi.nrows()) &&
            static_cast<std::size_t>(bi(static_cast<std::size_t>(i), 0)) == j) {
            block = data->blocks[static_cast<std::size_t>(i)];
            ++i;
        } else {
            block = block_backend->zeros({ mults[j] }, tensor->dtype);
        }
        numbers.push_back(block_func(block));
    }
    return func(numbers);
}

TensorBackend::DataPtr
AbelianBackend::scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg)
{
    // --- hints from Python AbelianBackend.scale_axis ---
    // due to lexsort(a_block_inds.T), a_block_inds_cont is already sorted in this case
    // only need to iterate over common blocks, the non-common multiply to 0.
    // note: unlike the tdot implementation, we do not combine and reshape here.
    // this is because we know the result will have the same block-structure as `a`, and
    // we only need to scale the blocks on one axis, not perform a general tensordot.
    // but this also means that we may encounter duplicates in a_block_inds_cont,
    // i.e. multiple blocks of `a` which have the same sector on the leg to be scaled.
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    BlockInds a_block_inds = a_data->block_inds;
    BlockInds a_block_inds_cont =
      a_block_inds.take_columns(std::array<std::size_t, 1>{ static_cast<std::size_t>(leg) });
    if (leg != a->num_legs - 1) {
        auto sort = a_block_inds_cont.lexsort_indices();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(a_blocks.size());
        for (auto i : sort)
            sorted_blocks.push_back(a_blocks[i]);
        a_blocks = std::move(sorted_blocks);
        a_block_inds = a_block_inds.take(sort);
        a_block_inds_cont =
          a_block_inds.take_columns(std::array<std::size_t, 1>{ static_cast<std::size_t>(leg) });
    }
    Dtype common_dtype = dtype::common({ a->dtype, b->dtype });
    if (a_data->dtype != common_dtype)
        for (auto& blk : a_blocks)
            blk = block_backend->to_dtype(blk, common_dtype);
    if (b_data->dtype != common_dtype)
        for (auto& blk : b_blocks)
            blk = block_backend->to_dtype(blk, common_dtype);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    BlockInds b_cont = b_data->block_inds.take_columns(std::array<std::size_t, 1>{ 0 });
    BlockInds::iter_common_sorted(
      a_block_inds_cont,
      b_cont,
      /*a_strict=*/false,
      /*b_strict=*/true,
      [&](std::ptrdiff_t i, std::ptrdiff_t j) {
          res_blocks.push_back(block_backend->scale_axis(
            a_blocks[static_cast<std::size_t>(i)], b_blocks[static_cast<std::size_t>(j)], leg));
          auto row = a_block_inds.row(static_cast<std::size_t>(i));
          res_rows.emplace_back(row.begin(), row.end());
      });
    BlockInds res_block_inds =
      res_rows.empty() ? zeros_i64(0, a->num_legs) : BlockInds::from_rows(res_rows);
    return wrap(
      make_data(common_dtype, a_data->device, std::move(res_blocks), res_block_inds, false));
}

TensorBackend::DataPtr
AbelianBackend::split_legs(TensorCPtr a,
                           std::vector<int64> leg_idcs,
                           TensorProduct::Ptr new_codomain,
                           TensorProduct::Ptr new_domain)
{
    // --- hints from Python AbelianBackend.split_legs ---
    // = end - beg
    // shape (res_num_blocks, n_split)
    // generate new block_inds and figure out slices within old blocks to be extracted
    // splitting pipes in F style is done by splitting them in C style and permuting the axes
    // = i - k for indices below
    // index within pipes
    // i = index in old tensor
    // = a.legs[i]
    // = index where split legs begin in new tensor
    // = until where spaces go in new tensor
    // if the leg to be split is in the domain, the order of block_inds and of its
    // block_ind_map are opposite -> need to reverse
    // need to permute these shapes here to compensate the permute_axes on the blocks below
    // (only relevant for F style combining, i.e., dual pipes)
    // the actual loop to split the blocks
    // ---
    auto a_data = data_from_tensor(a);
    if (a_data->blocks.empty())
        return zero_data(new_codomain, new_domain, a_data->dtype, a_data->device);
    int64 n_split = static_cast<int64>(leg_idcs.size());
    py::list pipes;
    for (auto i : leg_idcs)
        pipes.append(py::cast(a).attr("get_leg_co_domain")(i));
    int64 res_num_legs = new_codomain->num_factors + new_domain->num_factors;
    auto old_blocks = a_data->blocks;
    auto const& old_block_inds = a_data->block_inds;
    py::array_t<int64> map_slices_beg = zeros_i64_array(
      { static_cast<py::ssize_t>(old_blocks.size()), static_cast<py::ssize_t>(n_split) });
    py::array_t<int64> map_slices_shape = zeros_i64_array(
      { static_cast<py::ssize_t>(old_blocks.size()), static_cast<py::ssize_t>(n_split) });
    for (py::ssize_t j = 0; j < n_split; ++j) {
        auto pipe = pipes[j].cast<AbelianLegPipe::Ptr>();
        auto block_inds_j =
          old_block_inds.column(static_cast<std::size_t>(leg_idcs[static_cast<std::size_t>(j)]));
        std::vector<int64> beg_col(block_inds_j.size());
        std::vector<int64> shape_col(block_inds_j.size());
        auto const& slices = pipe->block_ind_map_slices;
        for (std::size_t r = 0; r < block_inds_j.size(); ++r) {
            auto const bi = static_cast<std::size_t>(block_inds_j[r]);
            beg_col[r] = slices[bi];
            shape_col[r] = slices[bi + 1] - slices[bi];
        }
        map_slices_beg.attr("__setitem__")(py::make_tuple(py::ellipsis(), j),
                                           i64_vec_to_numpy(beg_col));
        map_slices_shape.attr("__setitem__")(py::make_tuple(py::ellipsis(), j),
                                             i64_vec_to_numpy(shape_col));
    }
    auto beg_b = map_slices_beg.unchecked<2>();
    auto shp_b = map_slices_shape.unchecked<2>();
    std::vector<int64> per(static_cast<std::size_t>(beg_b.shape(0)), 1);
    for (py::ssize_t r = 0; r < beg_b.shape(0); ++r) {
        for (py::ssize_t c = 0; c < n_split; ++c)
            per[static_cast<std::size_t>(r)] *= shp_b(r, c);
    }
    std::vector<int64> old_rows_vec;
    for (std::size_t i = 0; i < per.size(); ++i)
        for (int64 s = 0; s < per[i]; ++s)
            old_rows_vec.push_back(static_cast<int64>(i));
    py::array_t<int64> old_rows = i64_vec_to_numpy(old_rows_vec);
    py::ssize_t res_num_blocks = static_cast<py::ssize_t>(old_rows_vec.size());

    std::vector<std::vector<int64>> map_rows_all;
    for (py::ssize_t r = 0; r < beg_b.shape(0); ++r) {
        std::vector<int64> shape_v(static_cast<std::size_t>(n_split));
        for (py::ssize_t c = 0; c < n_split; ++c)
            shape_v[static_cast<std::size_t>(c)] = shp_b(r, c);
        auto grid = make_grid(shape_v, /*cstyle=*/true);
        for (auto& row : grid) {
            for (py::ssize_t c = 0; c < n_split; ++c)
                row[static_cast<std::size_t>(c)] += beg_b(r, c);
            map_rows_all.push_back(std::move(row));
        }
    }
    BlockInds map_rows_bi = BlockInds::from_rows(map_rows_all);
    py::array_t<int64> map_rows = block_inds_to_numpy(map_rows_bi);

    py::array_t<int64> new_block_inds =
      empty_i64_array({ res_num_blocks, static_cast<py::ssize_t>(res_num_legs) });
    py::array_t<int64> old_block_beg =
      zeros_i64_array({ res_num_blocks, static_cast<py::ssize_t>(a->num_legs) });
    py::array_t<int64> old_block_shapes =
      empty_i64_array({ res_num_blocks, static_cast<py::ssize_t>(a->num_legs) });
    py::list axes_perm_l;
    for (int64 ax = 0; ax < res_num_legs; ++ax)
        axes_perm_l.append(ax);
    std::vector<int64> axes_perm(static_cast<std::size_t>(res_num_legs));
    std::iota(axes_perm.begin(), axes_perm.end(), 0);
    int64 shift = 0;
    int64 jp = 0;
    int64 num_codomain = a->num_codomain_legs();
    int64 a_num_legs = a->num_legs;
    std::vector<bool> is_split(static_cast<std::size_t>(a_num_legs), false);
    for (auto li : leg_idcs)
        is_split[static_cast<std::size_t>(li)] = true;

    for (int64 i_leg = 0; i_leg < a_num_legs; ++i_leg) {
        if (is_split[static_cast<std::size_t>(i_leg)]) {
            bool in_domain = i_leg >= num_codomain;
            auto pipe = pipes[jp].cast<AbelianLegPipe::Ptr>();
            int64 k = i_leg + shift;
            int64 k2 = k + pipe->num_legs;
            if (pipe->combine_cstyle == in_domain) {
                std::reverse(axes_perm.begin() + k, axes_perm.begin() + k2);
            }
            auto map_rows_jp =
              asarray_i64_1d(map_rows.attr("__getitem__")(py::make_tuple(py::ellipsis(), jp)));
            auto mrb = map_rows_jp.unchecked<1>();
            std::vector<int64> map_row_idx(static_cast<std::size_t>(mrb.shape(0)));
            for (py::ssize_t r = 0; r < mrb.shape(0); ++r)
                map_row_idx[static_cast<std::size_t>(r)] = mrb(r);
            BlockInds block_ind_map = pipe->block_ind_map.take_i64(map_row_idx);
            if (in_domain) {
                // columns -2:1:-1 → leg columns reversed (excluding b0,b1 and J)
                std::vector<std::size_t> cols;
                cols.reserve(static_cast<std::size_t>(pipe->num_legs));
                for (int64 c = static_cast<int64>(pipe->num_legs) + 1; c >= 2; --c)
                    cols.push_back(static_cast<std::size_t>(c));
                new_block_inds.attr("__setitem__")(
                  py::make_tuple(py::ellipsis(), py::slice(k, k2, 1)),
                  block_inds_to_numpy(block_ind_map.take_columns(cols)));
            } else {
                std::vector<std::size_t> cols;
                cols.reserve(static_cast<std::size_t>(pipe->num_legs));
                for (std::size_t c = 2; c < 2 + static_cast<std::size_t>(pipe->num_legs); ++c)
                    cols.push_back(c);
                new_block_inds.attr("__setitem__")(
                  py::make_tuple(py::ellipsis(), py::slice(k, k2, 1)),
                  block_inds_to_numpy(block_ind_map.take_columns(cols)));
            }
            auto col0 = block_ind_map.column(0);
            auto col1 = block_ind_map.column(1);
            old_block_beg.attr("__setitem__")(py::make_tuple(py::ellipsis(), i_leg),
                                              i64_vec_to_numpy(col0));
            std::vector<int64> shapes(col0.size());
            for (std::size_t r = 0; r < col0.size(); ++r)
                shapes[r] = col1[r] - col0[r];
            old_block_shapes.attr("__setitem__")(py::make_tuple(py::ellipsis(), i_leg),
                                                 i64_vec_to_numpy(shapes));
            shift += pipe->num_legs - 1;
            ++jp;
        } else {
            auto col = old_block_inds.column(static_cast<std::size_t>(i_leg));
            auto old_rows_i = asarray_i64_1d(old_rows);
            auto orb = old_rows_i.unchecked<1>();
            std::vector<int64> nbi_vec(static_cast<std::size_t>(orb.shape(0)));
            for (py::ssize_t r = 0; r < orb.shape(0); ++r)
                nbi_vec[static_cast<std::size_t>(r)] = col[static_cast<std::size_t>(orb(r))];
            auto nbi = i64_vec_to_numpy(nbi_vec);
            new_block_inds.attr("__setitem__")(py::make_tuple(py::ellipsis(), i_leg + shift), nbi);
            old_block_shapes.attr("__setitem__")(py::make_tuple(py::ellipsis(), i_leg),
                                                 py::cast(a)
                                                   .attr("get_leg_co_domain")(i_leg)
                                                   .attr("multiplicities")
                                                   .attr("__getitem__")(nbi));
        }
    }

    py::array_t<int64> new_block_shapes =
      empty_i64_array({ res_num_blocks, static_cast<py::ssize_t>(res_num_legs) });
    auto legs = conventional_leg_order(new_codomain, new_domain);
    for (std::size_t li = 0; li < legs.size(); ++li) {
        new_block_shapes.attr("__setitem__")(
          py::make_tuple(py::ellipsis(), static_cast<py::ssize_t>(li)),
          py::cast(legs[li])
            .attr("multiplicities")
            .attr("__getitem__")(new_block_inds.attr("__getitem__")(
              py::make_tuple(py::ellipsis(), static_cast<py::ssize_t>(li)))));
    }
    new_block_shapes =
      new_block_shapes.attr("__getitem__")(py::make_tuple(py::ellipsis(), axes_perm));

    std::vector<BlockBackend::BlockPtr> new_blocks;
    new_blocks.reserve(static_cast<std::size_t>(res_num_blocks));
    auto old_rows_i = asarray_i64_1d(old_rows);
    auto orb = old_rows_i.unchecked<1>();
    for (py::ssize_t i = 0; i < res_num_blocks; ++i) {
        auto old_block = old_blocks[static_cast<std::size_t>(orb(i))];
        py::list slc_list;
        py::object beg_row = old_block_beg.attr("__getitem__")(i);
        py::object shp_row = old_block_shapes.attr("__getitem__")(i);
        for (int64 ax = 0; ax < a_num_legs; ++ax) {
            int64 b = beg_row.attr("__getitem__")(ax).cast<int64>();
            int64 s = shp_row.attr("__getitem__")(ax).cast<int64>();
            slc_list.append(py::slice(b, b + s, 1));
        }
        auto new_block = b_get(old_block, py::tuple(slc_list));
        auto shape_i = new_block_shapes.attr("__getitem__")(i).cast<std::vector<int64>>();
        new_blocks.push_back(block_backend->reshape(new_block, shape_i));
    }
    for (auto& blk : new_blocks)
        blk = block_backend->permute_axes(blk, axes_perm);
    return wrap(make_data(
      a_data->dtype, a_data->device, std::move(new_blocks), asarray_i64(new_block_inds), false));
}

TensorBackend::DataPtr
AbelianBackend::squeeze_legs(TensorCPtr a, std::vector<int64> idcs)
{
    auto a_data = data_from_tensor(a);
    int64 n_legs = a->num_legs;
    if (a_data->blocks.empty()) {
        return wrap(make_data(a_data->dtype,
                              a_data->device,
                              {},
                              zeros_i64(0, n_legs - static_cast<int64>(idcs.size())),
                              true));
    }
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->squeeze_axes(b, idcs));
    std::vector<bool> keep_mask(static_cast<std::size_t>(n_legs), true);
    for (auto i : idcs)
        keep_mask[static_cast<std::size_t>(i)] = false;
    BlockInds block_inds = a_data->block_inds.delete_columns(keep_mask);
    return wrap(make_data(a_data->dtype, a_data->device, std::move(blocks), block_inds, true));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::svd(SymmetricTensorCPtr a,
                    TensorProduct::Ptr new_co_domain,
                    std::optional<std::string> algorithm)
{
    // --- hints from Python AbelianBackend.svd ---
    // The issue here is that sector_decomposition of the (co)domain is sorted, but may be
    // dual_sorted for the single leg in the (co)domain. The block_inds do contain the indices
    // of the legs, i.e., either we (generically) cannot iterate over sorted arrays (= iterate
    // over legs) or we iterate over sorted arrays (= iterate over (co)domain) and then need an
    // additional step to find the correct indices.
    // We do the latter, i.e., assuming that sector_decomposition_where is efficient.
    // Additionally, the block_inds of u, s, vh are in general no longer lexsorted.
    // In the special case in which the sector_decomposition of all legs is sorted, it reduces
    // to the previous case, where we do not need to find any indices and the block_inds are
    // constructed in a lexsorted way.
    // we do not have a block for that sector.
    // => S_block == 0, dont even set it.
    // can choose arbitrary blocks for u and vh, as long as they are isometric / orthogonal
    // for all block_inds, the last column is sorted and duplicate-free,
    // thus the block_inds are np.lexsort( .T)-ed if the sector_order of
    // the corresponding leg is sorted
    // ---
    require_matrix_form(a, "svd");
    auto a_data = data_from_tensor(a);
    py::object new_leg = py::cast(new_co_domain->factors[0]);
    auto cod0 = py::cast(a->codomain).attr("__getitem__")(0);
    auto dom0 = py::cast(a->domain).attr("__getitem__")(0);
    auto a_blocks = a_data->blocks;
    auto a_block_inds = a_data->block_inds;
    std::vector<BlockBackend::BlockPtr> u_blocks, s_blocks, vh_blocks;
    py::list s_block_inds_list, u_block_inds, vh_block_inds;
    int64 i = 0;
    auto const& codom_secs = a->codomain->sector_decomposition;
    auto const& dom_secs = a->domain->sector_decomposition;
    int64 n_enum = 0;
    SectorArray::iter_common_sorted(
      codom_secs, dom_secs, true, true, [&](std::ptrdiff_t ji, std::ptrdiff_t ki) {
          int64 j = static_cast<int64>(ji);
          int64 k = static_cast<int64>(ki);
          int64 n = n_enum++;
          py::object sector = py::cast(codom_secs[static_cast<std::size_t>(j)]);
          if (cod0.attr("sector_order").cast<std::string>() != "sorted")
              j = cod0.attr("sector_decomposition_where")(sector).cast<int64>();
          if (dom0.attr("sector_order").cast<std::string>() != "sorted") {
              k = dom0.attr("sector_decomposition_where")(sector).cast<int64>();
              i = static_cast<int64>(a_block_inds.searchsorted_column(1, k));
          }
          if (new_leg.attr("sector_order").cast<std::string>() != "sorted")
              n = new_leg.attr("sector_decomposition_where")(sector).cast<int64>();

          if (i < static_cast<int64>(a_block_inds.nrows()) &&
              a_block_inds(static_cast<std::size_t>(i), 0) == j) {
              auto [u, s, vh] =
                block_backend->matrix_svd(a_blocks[static_cast<std::size_t>(i)], algorithm);
              u_blocks.push_back(u);
              s_blocks.push_back(s);
              vh_blocks.push_back(vh);
              s_block_inds_list.append(n);
              ++i;
          } else {
              int64 new_leg_dim = mults_of(new_leg)[static_cast<std::size_t>(n)];
              auto eye_u = block_backend->eye_matrix(
                mults_of(cod0)[static_cast<std::size_t>(j)], a->dtype, std::nullopt);
              u_blocks.push_back(
                b_get(eye_u,
                      py::make_tuple(py::slice(std::nullopt, std::nullopt, std::nullopt),
                                     py::slice(0, new_leg_dim, 1))));
              auto eye_v = block_backend->eye_matrix(
                mults_of(dom0)[static_cast<std::size_t>(k)], a->dtype, std::nullopt);
              vh_blocks.push_back(
                b_get(eye_v,
                      py::make_tuple(py::slice(0, new_leg_dim, 1),
                                     py::slice(std::nullopt, std::nullopt, std::nullopt))));
          }
          u_block_inds.append(py::make_tuple(j, n));
          vh_block_inds.append(py::make_tuple(n, k));
      });

    BlockInds s_bi;
    if (s_blocks.empty()) {
        s_bi = zeros_i64(0, 2);
    } else {
        {
            std::vector<std::vector<int64>> rows;
            rows.reserve(static_cast<std::size_t>(py::len(s_block_inds_list)));
            for (auto item : s_block_inds_list) {
                int64 v = py::reinterpret_borrow<py::object>(item).cast<int64>();
                rows.push_back({ v, v });
            }
            s_bi = BlockInds::from_rows(rows);
        }
    }
    BlockInds u_bi, vh_bi;
    if (u_blocks.empty()) {
        u_bi = vh_bi = zeros_i64(0, 2);
    } else {
        u_bi = block_inds_from_py_rows(u_block_inds);
        vh_bi = block_inds_from_py_rows(vh_block_inds);
    }
    bool u_sorted = new_leg.attr("sector_order").cast<std::string>() == "sorted";
    bool s_sorted = u_sorted;
    bool vh_sorted = dom0.attr("sector_order").cast<std::string>() == "sorted";
    Dtype a_dtype = a->dtype;
    return { wrap(make_data(a_dtype, a_data->device, std::move(u_blocks), u_bi, u_sorted)),
             wrap(make_data(
               dtype::to_real(a_dtype), a_data->device, std::move(s_blocks), s_bi, s_sorted)),
             wrap(make_data(a_dtype, a_data->device, std::move(vh_blocks), vh_bi, vh_sorted)) };
}

BlockBackend::BlockPtr
AbelianBackend::to_dense_block(TensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    // Tensor.shape is float64 (symmetry dims); block backends need int64 extents.
    std::vector<int64> shape;
    for (auto item : a->shape) {
        shape.push_back(static_cast<int64>(std::llround(item)));
    }
    auto res = block_backend->zeros(shape, a_data->dtype);
    auto legs = conventional_leg_order(a);
    auto const& bi = a_data->block_inds;
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        py::tuple slices(static_cast<py::ssize_t>(legs.size()));
        for (py::ssize_t c = 0; c < static_cast<py::ssize_t>(legs.size()); ++c) {
            slices[c] = slice_pair(py::cast(legs[static_cast<std::size_t>(c)])
                                     .attr("slices")
                                     .attr("__getitem__")(bi(static_cast<py::ssize_t>(i), c)));
        }
        b_set(res, slices, a_data->blocks[i]);
    }
    return res;
}

BlockBackend::Scalar
AbelianBackend::trace_full(SymmetricTensorCPtr a,
                           std::vector<int64> idcs1,
                           std::vector<int64> idcs2)
{
    // --- hints from Python AbelianBackend.trace_full ---
    // else: block is entirely off-diagonal and does not contribute to the trace
    // ---
    auto a_data = data_from_tensor(a);
    int64 K = a->num_codomain_legs();
    auto res = block_backend->as_scalar(dtype::zero_scalar(a_data->dtype), a_data->dtype);
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        bool on_diag = true;
        for (int64 c = 0; c < K; ++c) {
            if (bi(static_cast<py::ssize_t>(n), c) !=
                bi(static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(bi.ncols()) - 1 - c)) {
                on_diag = false;
                break;
            }
        }
        if (on_diag)
            res = res + block_backend->trace_full(a_data->blocks[n]);
    }
    return res;
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr, float64, float64>
AbelianBackend::truncate_singular_values(DiagonalTensorCPtr S,
                                         std::optional<int64> chi_max,
                                         int64 chi_min,
                                         float64 degeneracy_tol,
                                         float64 trunc_cut,
                                         std::optional<float64> svd_min,
                                         bool minimize_error)
{
    py::array S_np = block_backend->to_numpy(diagonal_tensor_to_block(S)).cast<py::array>();
    auto [keep, err, new_norm] = _truncate_singular_values_selection(
      S_np, py::none(), chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);
    auto keep_block = block_backend->as_block(keep, Dtype::Bool);
    auto [mask_data, small_leg] =
      mask_from_block(keep_block, py::cast(S->leg()).cast<Space::Ptr>());
    return { mask_data, small_leg, err, new_norm };
}

} // namespace lasyten
