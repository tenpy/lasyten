#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/backends/fusion_tree_mapping.h>
#include <lasyten/backends/fusion_tree_permute.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/backends/block_inds_numpy.h>
#include <lasyten/config.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/trees.h>
#include <lasyten/tools.h>
#include <lasyten/tools/lru_cache.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>
#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>

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

} // namespace

FusionTreeData::FusionTreeData(BlockInds block_inds_,
                               std::vector<BlockBackend::BlockPtr> blocks_,
                               Dtype dtype_,
                               std::string device_,
                               bool is_sorted)
  : block_inds(std::move(block_inds_))
  , blocks(std::move(blocks_))
  , dtype(dtype_)
  , device(std::move(device_))
{
    if (block_inds.ncols() != 2) {
        throw std::invalid_argument("FusionTreeData: block_inds must have shape (N, 2)");
    }
    if (block_inds.nrows() != blocks.size()) {
        throw std::invalid_argument("FusionTreeData: len(blocks) must equal block_inds.nrows()");
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
FusionTreeData::block_ind_from_coupled(Sector coupled, TensorProduct::Ptr domain) const
{
    auto domain_sector_ind = domain->sector_decomposition_where(coupled);
    if (!domain_sector_ind.has_value()) {
        return std::nullopt;
    }
    return block_ind_from_domain_sector_ind(*domain_sector_ind);
}

std::optional<int64>
FusionTreeData::block_ind_from_domain_sector_ind(int64 domain_sector_ind) const
{
    // Column 1 only — that axis is sorted by construction (lexsort last-key-first).
    auto ind = static_cast<int64>(block_inds.searchsorted_column(1, domain_sector_ind));
    auto n_rows = static_cast<int64>(block_inds.nrows());

    if (ind >= n_rows || block_inds(static_cast<std::size_t>(ind), 1) != domain_sector_ind) {
        return std::nullopt;
    }
    if (ind + 1 < n_rows &&
        block_inds(static_cast<std::size_t>(ind + 1), 1) == domain_sector_ind) {
        throw std::runtime_error(
          "FusionTreeData: duplicate domain sector index in block_inds[:, 1]");
    }
    return ind;
}

void
FusionTreeData::discard_zero_blocks(std::shared_ptr<BlockBackend> backend, float64 eps)
{
    std::vector<int64> keep;
    keep.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if ((backend->norm(blocks[i]) >= eps).as_bool()) {
            keep.push_back(static_cast<int64>(i));
        }
    }

    std::vector<BlockBackend::BlockPtr> kept_blocks;
    kept_blocks.reserve(keep.size());
    for (int64 i : keep) {
        kept_blocks.push_back(blocks[static_cast<std::size_t>(i)]);
    }
    blocks = std::move(kept_blocks);
    block_inds = block_inds.take_i64(keep);
}

void
FusionTreeData::save_hdf5(lasyten::hdf5::Saver& saver,
                          HighFive::Group& /*h5gr*/,
                          std::string subpath) const
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

FusionTreeData::Ptr
FusionTreeData::from_hdf5(lasyten::hdf5::Loader& loader,
                          HighFive::Group& h5gr,
                          std::string subpath)
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

    auto obj = std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), /*is_sorted=*/true);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

// FusionTreeBackend method implementations (append after FusionTreeData::from_hdf5)

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

[[nodiscard]] py::array_t<int64>
zeros_i64_array(std::vector<py::ssize_t> const& shape)
{
    py::array_t<int64> arr(shape);
    if (arr.size() > 0) {
        std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(), int64{ 0 });
    }
    return arr;
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

SectorArray
as_sector_array_from_list(py::list const& sectors)
{
    // Match Python ``as_sector_array([])`` → ``SectorArray.from_sector(Sector([]))``.
    if (sectors.empty()) {
        return SectorArray::from_sector(Sector::zeros(0));
    }
    std::vector<Sector> out;
    out.reserve(static_cast<std::size_t>(sectors.size()));
    for (auto h : sectors)
        out.push_back(h.cast<Sector>());
    return SectorArray(std::move(out));
}

void
iter_common_noncommon_sorted_1d(
  std::vector<int64> const& a,
  std::vector<int64> const& b,
  std::function<void(std::optional<std::ptrdiff_t>, std::optional<std::ptrdiff_t>)> const& yield)
{
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() || j < b.size()) {
        if (j >= b.size() || (i < a.size() && a[i] < b[j])) {
            yield(static_cast<std::ptrdiff_t>(i), std::nullopt);
            ++i;
        } else if (i >= a.size() || (j < b.size() && b[j] < a[i])) {
            yield(std::nullopt, static_cast<std::ptrdiff_t>(j));
            ++j;
        } else {
            yield(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
            ++i;
            ++j;
        }
    }
}

py::slice
slice_from_index_slice(IndexSlice slc)
{
    return py::slice(slc.start, slc.stop, 1);
}

/// Flatten two lists of per-leg slices into a single numpy index tuple ``(*j1, *j2)``.
/// ``py::make_tuple(*j1, *j2)`` does not unpack lists the way Python does.
py::tuple
concat_slice_lists(py::list const& j1, py::list const& j2)
{
    py::tuple out(j1.size() + j2.size());
    for (py::ssize_t i = 0; i < j1.size(); ++i)
        out[static_cast<std::size_t>(i)] = j1[i];
    for (py::ssize_t i = 0; i < j2.size(); ++i)
        out[static_cast<std::size_t>(j1.size() + i)] = j2[i];
    return out;
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

FusionTreeData::Ptr
make_data(Dtype dtype,
          std::string device,
          std::vector<BlockBackend::BlockPtr> blocks,
          BlockInds block_inds,
          bool is_sorted = false)
{
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), is_sorted);
}

int64
prod_int(std::vector<int64> const& v)
{
    int64 p = 1;
    for (int64 x : v)
        p *= x;
    return p;
}

std::vector<int64>
tp_mults(py::object tp)
{
    return tp.attr("multiplicities").cast<std::vector<int64>>();
}

std::vector<int64>
legs_flat_leg_permutation(py::object legs)
{
    std::vector<int64> out;
    int64 offset = 0;
    for (py::handle h : legs) {
        auto leg = h.cast<Leg::Ptr>();
        auto part_perm = leg._flat_leg_permutation(offset);
        out.insert(out.end(), part_perm.begin(), part_perm.end());
        offset += leg.num_flat_legs();
    }
    return out;
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

std::vector<int64>
mults_of(py::object leg)
{
    return leg.attr("multiplicities").cast<std::vector<int64>>();
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

std::vector<std::uint8_t>
flat_are_dual(TensorProduct::Ptr tp)
{
    std::vector<std::uint8_t> out;
    for (auto const& leg : tp->flat_legs())
        out.push_back(static_cast<std::uint8_t>(leg.is_dual()));
    return out;
}

} // namespace

TensorBackend::DataPtr
FusionTreeBackend::wrap(FusionTreeData::Ptr d)
{
    if (!d)
        throw std::invalid_argument("FusionTreeBackend::wrap: null");
    return d;
}

FusionTreeData::Ptr
FusionTreeBackend::unwrap(DataPtr d)
{
    if (!d)
        throw std::invalid_argument("FusionTreeBackend::unwrap: null DataPtr");
    auto* p = dynamic_cast<FusionTreeData*>(d.get());
    if (!p) {
        const auto& obj = *d;
        throw std::invalid_argument(std::format(
          "FusionTreeBackend::unwrap: expected FusionTreeData, got {}", typeid(obj).name()));
    }
    return std::static_pointer_cast<FusionTreeData>(d);
}

FusionTreeData::Ptr
FusionTreeBackend::data_from_tensor(TensorCPtr tensor)
{
    if (auto st = std::dynamic_pointer_cast<const SymmetricTensor>(tensor))
        return unwrap(st->data);
    if (auto m = std::dynamic_pointer_cast<const Mask>(tensor))
        return unwrap(m->data);
    throw std::invalid_argument(
      "FusionTreeBackend::data_from_tensor: expected SymmetricTensor or Mask");
}

FusionTreeBackend::FusionTreeBackend(std::shared_ptr<BlockBackend> block_backend_)
  : TensorBackend(std::move(block_backend_))
{
}

void
FusionTreeBackend::test_tensor_sanity(TensorCPtr a, bool is_diagonal)
{
    // --- hints from Python FusionTreeBackend.test_tensor_sanity ---
    // check device and dtype
    // check charge rule (matching coupled sectors)
    // ---
    TensorBackend::test_tensor_sanity(a, is_diagonal);
    py::object raw = py::cast(a).attr("data");
    FusionTreeData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a->device == data->device);
    assert(data->device == block_backend->as_device(data->device));
    assert(a->dtype == data->dtype);
    auto const& bi = data->block_inds;
    assert(bi.nrows() == data->blocks.size());
    assert(bi.ncols() == 2);
    assert(bi.all_ge(0));
    std::vector<int64> maxes = { a->codomain->num_sectors, a->domain->num_sectors };
    assert(bi.all_lt_per_column(maxes));
    assert(bi.is_lexsorted());
    for (std::size_t r = 0; r < bi.nrows(); ++r) {
        assert(a->codomain->sector_decomposition[static_cast<std::size_t>(bi(r, 0))] ==
               a->domain->sector_decomposition[static_cast<std::size_t>(bi(r, 1))]);
    }
    for (std::size_t i = 0; i < bi.nrows(); ++i) {
        int64 ic = bi(i, 0);
        int64 id = bi(i, 1);
        std::vector<int64> expect_shape = {
            tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(ic)],
            mults_of(py::cast(a->domain))[static_cast<std::size_t>(id)]
        };
        if (is_diagonal) {
            assert(expect_shape[0] == expect_shape[1]);
            expect_shape = { expect_shape[0] };
        }
        block_backend->test_block_sanity(data->blocks[i], expect_shape, a->dtype, a->device);
    }
}

void
FusionTreeBackend::test_mask_sanity(MaskCPtr a)
{
    // --- hints from Python FusionTreeBackend.test_mask_sanity ---
    // check device and dtype
    // block_inds
    // check charge rule (matching coupled sectors)
    // blocks
    // ---
    TensorBackend::test_mask_sanity(a);
    py::object raw = py::cast(a).attr("data");
    FusionTreeData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a->device == data->device);
    assert(data->dtype == Dtype::Bool);
    auto const& bi = data->block_inds;
    assert(bi.nrows() == data->blocks.size());
    assert(bi.ncols() == 2);
    bool is_projection = a->is_projection;
    // block_inds index TensorProduct (co)domain sectors, not ElementarySpace defining order.
    // For dual legs those orders differ, so use domain/codomain multiplicities (like Python).
    auto domain_mults = tp_mults(py::cast(a->domain));
    auto codomain_mults = tp_mults(py::cast(a->codomain));
    for (std::size_t i = 0; i < bi.nrows(); ++i) {
        int64 i_cod = bi(i, 0);
        int64 i_dom = bi(i, 1);
        int64 expect_len = is_projection ? domain_mults[static_cast<std::size_t>(i_dom)]
                                         : codomain_mults[static_cast<std::size_t>(i_cod)];
        int64 expect_sum = is_projection ? codomain_mults[static_cast<std::size_t>(i_cod)]
                                         : domain_mults[static_cast<std::size_t>(i_dom)];
        assert(expect_len > 0);
        assert(expect_sum > 0);
        block_backend->test_block_sanity(
          data->blocks[i], std::vector<int64>{ expect_len }, Dtype::Bool, data->device);
        assert(block_backend->sum_all(data->blocks[i]).as_int64() == expect_sum);
    }
}

TensorBackend::DataPtr
FusionTreeBackend::act_block_diagonal_square_matrix(SymmetricTensorCPtr a,
                                                    BlockUnaryFn block_method,
                                                    std::optional<DtypeMapFn> dtype_map)
{
    // --- hints from Python FusionTreeBackend.act_block_diagonal_square_matrix ---
    // square matrix => codomain == domain have the same sectors
    // ---
    auto a_data = data_from_tensor(a);
    auto block_inds_col0 = a_data->block_inds.column(0);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    int64 n = 0;
    int64 bi = block_inds_col0.empty() ? -1 : block_inds_col0[static_cast<std::size_t>(n)];
    int64 num_sectors = py::cast(a->codomain).attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        BlockBackend::BlockPtr block;
        if (bi == i) {
            block = a_data->blocks[static_cast<std::size_t>(n)];
            ++n;
            bi = (static_cast<std::size_t>(n) >= block_inds_col0.size())
                   ? -1
                   : block_inds_col0[static_cast<std::size_t>(n)];
        } else {
            block = block_backend->zeros(
              { tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i)],
                tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i)] },
              a->dtype);
        }
        res_blocks.push_back(block_method(block));
    }
    Dtype dt = dtype_map ? (*dtype_map)(a->dtype) : a->dtype;
    BlockInds res_block_inds = BlockInds::arange_diag(
      static_cast<std::size_t>(py::cast(a->domain).attr("num_sectors").cast<int64>()));
    return wrap(make_data(dt, a_data->device, std::move(res_blocks), res_block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::add_trivial_leg(TensorCPtr a,
                                   int64 legs_pos,
                                   bool add_to_domain,
                                   int64 co_domain_pos,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    // --- hints from Python FusionTreeBackend.add_trivial_leg ---
    // does not change blocks or coupled sectors at all.
    // ---
    return wrap(data_from_tensor(a));
}

bool
FusionTreeBackend::almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol)
{
    // --- hints from Python FusionTreeBackend.almost_equal ---
    // since the coupled sector must agree, it is enough to compare block_inds[:, 0]
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    bool ok = true;
    iter_common_noncommon_sorted_1d(
      a_data->block_inds.column(0),
      b_data->block_inds.column(0),
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

// ---------------------------------------------------------------------------------------------
// Cache of tree mappings ("plans") built by `apply_instructions`.
// The mapping depends only on the trees of the tensor -- the symmetry, sector lists and duality of
// the flat legs of codomain and domain, the coupled sector of every used block -- and on the
// instructions; it is applied to the data afterwards (`transform_tensor`, const). The key is made
// of exactly these inputs, so a hit returns the mapping the uncached code would have built.
// Least-recently-used entries are evicted when the estimated size exceeds ``tree_plan_cache_mb``.
namespace {

void
plan_key_put(std::string& k, std::int64_t v)
{
    k.append(reinterpret_cast<char const*>(&v), sizeof(v));
}

void
plan_key_put_sector(std::string& k, Sector const& s)
{
    k.push_back(static_cast<char>(s.len()));
    for (std::size_t i = 0; i < s.len(); ++i) {
        k.append(reinterpret_cast<char const*>(&s.q[i]), sizeof(int16_t));
    }
}

void
plan_key_put_product(std::string& k, TensorProduct::Ptr const& tp)
{
    auto const legs = tp->flat_legs();
    plan_key_put(k, static_cast<std::int64_t>(legs.size()));
    for (auto const& leg : legs) {
        plan_key_put(k, leg.is_dual() ? 1 : 0);
        auto const space = leg.as_space();
        plan_key_put(k, static_cast<std::int64_t>(space->sector_decomposition.size()));
        for (Sector const& s : space->sector_decomposition) {
            plan_key_put_sector(k, s);
        }
    }
}

std::string
plan_key(std::vector<Instruction> const& instructions,
         TensorProduct::Ptr const& codomain,
         TensorProduct::Ptr const& domain,
         BlockInds const& block_inds,
         bool mixes_codomain_domain)
{
    std::string k;
    auto const* symmetry_address = codomain->symmetry.get();
    k.append(reinterpret_cast<char const*>(&symmetry_address), sizeof(symmetry_address));
    plan_key_put(k, mixes_codomain_domain ? 1 : 2);
    plan_key_put(k,
                 static_cast<std::int64_t>(instructions.size())); // makes the key self-delimiting
    for (Instruction const& inst : instructions) {
        std::visit(
          [&](auto const& i) {
              using T = std::decay_t<decltype(i)>;
              if constexpr (std::is_same_v<T, BraidInstruction>) {
                  plan_key_put(k, 0);
                  plan_key_put(k, i.codomain ? 1 : 0);
                  plan_key_put(k, i.idx);
                  plan_key_put(k, i.overbraid ? 1 : 0);
              } else if constexpr (std::is_same_v<T, BendInstruction>) {
                  plan_key_put(k, 1);
                  plan_key_put(k, i.bend_down ? 1 : 0);
              } else {
                  plan_key_put(k, 2);
                  plan_key_put(k, i.codomain ? 1 : 0);
                  plan_key_put(k, static_cast<std::int64_t>(i.idcs.size()));
                  for (auto v : i.idcs) {
                      plan_key_put(k, v);
                  }
                  plan_key_put(k, i.overtwist ? 1 : 0);
              }
          },
          inst);
    }
    plan_key_put_product(k, codomain);
    plan_key_put_product(k, domain);
    plan_key_put(k, static_cast<std::int64_t>(block_inds.nrows()));
    for (std::size_t r = 0; r < block_inds.nrows(); ++r) {
        plan_key_put_sector(
          k, codomain->sector_decomposition[static_cast<std::size_t>(block_inds(r, 0))]);
    }
    return k;
}

/// A symmetry that a cached plan refers to, and how many strong references to it the plan itself
/// holds (its fusion trees own their symmetry), measured when the plan was built.
struct HeldSymmetry
{
    std::weak_ptr<Symmetry> symmetry;
    long refs = 0;
};

struct PlanEntry
{
    /// The symmetries of the codomain and of the domain the plan was built for (one entry if they
    /// are the same object); the first one is the one whose address is part of the cache key.
    std::vector<HeldSymmetry> held;
    std::shared_ptr<void const> mapping;
    std::size_t bytes = 0;
};
using PlanCache = ByteLRU<std::string, PlanEntry>;

PlanCache&
plan_cache()
{
    static PlanCache cache([] { return megabytes_to_bytes(get_config().tree_plan_cache_mb); },
                           [](std::string const& key, PlanEntry const& entry) {
                               constexpr std::size_t node_overhead =
                                 112; // list node + hash node + control block
                               return 2 * (sizeof(std::string) + key.capacity()) +
                                      sizeof(PlanEntry) + node_overhead + entry.bytes;
                           });
    return cache;
}

/// Remove the plans of symmetries that nothing but cached plans refer to any more.
///
/// The fusion trees of a cached mapping own their symmetry, so without this a symmetry that the
/// user has dropped (together with its open data files and packed index) would stay alive until
/// the plan is evicted or the cache is cleared. A symmetry is free when its strong count is
/// exactly the number of references that cached plans hold to it (``HeldSymmetry::refs``, summed
/// over all plans: the plans of the codomain's symmetry also refer to the domain's one and vice
/// versa); any other holder (a tensor, a space, a Python object) makes the count larger. A plan is
/// removed when all symmetries it refers to are free.
std::size_t
purge_orphaned_plans()
{
    auto& cache = plan_cache();
    std::unordered_map<Symmetry const*, long> internal;
    cache.for_each([&](std::string const&, PlanEntry const& entry) {
        for (auto const& held : entry.held) {
            if (auto const symmetry = held.symmetry.lock()) {
                internal[symmetry.get()] += held.refs;
            }
        }
    });
    return cache.erase_if([&](std::string const&, PlanEntry const& entry) {
        return std::ranges::all_of(entry.held, [&](HeldSymmetry const& held) {
            auto const symmetry = held.symmetry.lock();
            if (!symmetry) {
                return true;
            }
            long const external =
              symmetry.use_count() - 1 - internal[symmetry.get()]; // -1: `symmetry`
            return external <= 0;
        });
    });
}

/// Run `purge_orphaned_plans` now and then (after as many insertions as there are entries, at
/// least 256), so that the cost per insertion stays constant on average.
void
purge_orphaned_plans_now_and_then()
{
    static std::size_t inserts_since_purge = 0;
    static std::size_t threshold = 256;
    if (++inserts_since_purge < threshold) {
        return;
    }
    inserts_since_purge = 0;
    purge_orphaned_plans();
    threshold = std::max<std::size_t>(256, plan_cache().stats().entries);
}

/// The mapping for `key`: from the cache, or built by `make()` (returning a unique_ptr) and
/// cached. `symmetry` is the symmetry of the codomain (part of the key), `domain_symmetry` the one
/// of the domain; they may be different objects, and the mapping refers to both.
template<class Mapping, class Make>
std::shared_ptr<Mapping const>
cached_plan(std::string const& key,
            Symmetry::Ptr const& symmetry,
            Symmetry::Ptr const& domain_symmetry,
            Make&& make)
{
    auto& cache = plan_cache();
    if (auto const hit = cache.find(key);
        hit && !hit->held.empty() && hit->held.front().symmetry.lock() == symmetry) {
        return std::static_pointer_cast<Mapping const>(hit->mapping);
    }
    bool const same = symmetry.get() == domain_symmetry.get();
    long const refs_before = symmetry.use_count();
    long const domain_refs_before = same ? 0 : domain_symmetry.use_count();
    std::shared_ptr<Mapping const> mapping(make());
    PlanEntry entry;
    entry.held.push_back(
      { symmetry, symmetry.use_count() - refs_before }); // held by the new mapping
    if (!same) {
        entry.held.push_back(
          { domain_symmetry, domain_symmetry.use_count() - domain_refs_before });
    }
    entry.mapping = mapping;
    entry.bytes = mapping->approx_bytes();
    cache.insert(key, std::move(entry));
    purge_orphaned_plans_now_and_then();
    return mapping;
}

} // namespace

CacheStats
tree_plan_cache_stats()
{
    purge_orphaned_plans();
    return plan_cache().stats();
}

std::vector<std::vector<std::pair<long, long>>>
tree_plan_cache_debug()
{
    std::vector<std::vector<std::pair<long, long>>> out;
    plan_cache().for_each([&](std::string const&, PlanEntry const& entry) {
        auto& row = out.emplace_back();
        for (auto const& held : entry.held) {
            auto const symmetry = held.symmetry.lock();
            row.emplace_back(symmetry ? symmetry.use_count() - 1 : 0, held.refs);
        }
    });
    return out;
}

void
clear_tree_plan_cache()
{
    plan_cache().clear();
}

TensorBackend::DataPtr
FusionTreeBackend::apply_instructions(TensorCPtr tensor,
                                      py::object instructions,
                                      std::vector<int64> codomain_idcs,
                                      std::vector<int64> domain_idcs,
                                      TensorProduct::Ptr new_codomain,
                                      TensorProduct::Ptr new_domain,
                                      bool mixes_codomain_domain)
{
    auto t_data = data_from_tensor(tensor);
    auto codomain = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
    auto domain = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
    auto instructions_vec = instructions_from_python(instructions);

    auto const key =
      plan_key(instructions_vec, codomain, domain, t_data->block_inds, mixes_codomain_domain);
    FusionTreeData::Ptr res;
    if (mixes_codomain_domain) {
        auto mapping =
          cached_plan<TreePairMapping>(key, codomain->symmetry, domain->symmetry, [&] {
              return TreePairMapping::from_instructions(
                instructions_vec, codomain, domain, py::cast(t_data->block_inds));
          });
        res = mapping->transform_tensor(*t_data,
                                        codomain,
                                        domain,
                                        new_codomain,
                                        new_domain,
                                        codomain_idcs,
                                        domain_idcs,
                                        block_backend);
    } else {
        auto mapping =
          cached_plan<FactorizedTreeMapping>(key, codomain->symmetry, domain->symmetry, [&] {
              return FactorizedTreeMapping::from_instructions(
                instructions_vec, codomain, domain, py::cast(t_data->block_inds));
          });
        res = mapping->transform_tensor(*t_data,
                                        codomain,
                                        domain,
                                        new_codomain,
                                        new_domain,
                                        codomain_idcs,
                                        domain_idcs,
                                        block_backend);
    }
    res->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);
    return wrap(res);
}

TensorBackend::DataPtr
FusionTreeBackend::apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask)
{
    // --- hints from Python FusionTreeBackend.apply_mask_to_DiagonalTensor ---
    // append only for one leg, repeat later
    // ---
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    auto t_col = t_data->block_inds.column(0);
    auto m_col = m_data->block_inds.column(1);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_rows;
    iter_common_sorted_1d(t_col, m_col, [&](std::ptrdiff_t i, std::ptrdiff_t j) {
        res_blocks.push_back(block_backend->apply_mask(t_data->blocks[static_cast<std::size_t>(i)],
                                                       m_data->blocks[static_cast<std::size_t>(j)],
                                                       0));
        int64 v = m_data->block_inds(static_cast<std::size_t>(j), 0);
        res_rows.push_back({ v, v });
    });
    BlockInds res_block_inds = res_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(res_rows);
    return wrap(
      make_data(tensor->dtype, t_data->device, std::move(res_blocks), res_block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::combine_legs(TensorCPtr tensor,
                                std::vector<std::vector<int64>> leg_idcs_combine,
                                std::vector<Leg::Ptr> pipes,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain)
{
    return wrap(data_from_tensor(tensor));
}

TensorBackend::DataPtr
FusionTreeBackend::compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    Dtype res_dtype = dtype::common({ a->dtype, b->dtype });
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    if (a_data->dtype != res_dtype)
        for (auto& bl : a_blocks)
            bl = block_backend->to_dtype(bl, res_dtype);
    if (b_data->dtype != res_dtype)
        for (auto& bl : b_blocks)
            bl = block_backend->to_dtype(bl, res_dtype);
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<std::vector<int64>> block_inds_rows;
    if (a_data->blocks.size() > 0 && b_data->blocks.size() > 0) {
        iter_common_sorted_1d(
          a_data->block_inds.column(1),
          b_data->block_inds.column(0),
          [&](std::ptrdiff_t i, std::ptrdiff_t j) {
              blocks.push_back(block_backend->matrix_dot(a_blocks[static_cast<std::size_t>(i)],
                                                         b_blocks[static_cast<std::size_t>(j)]));
              block_inds_rows.push_back({ a_data->block_inds(static_cast<std::size_t>(i), 0),
                                          b_data->block_inds(static_cast<std::size_t>(j), 1) });
          });
    }
    BlockInds block_inds =
      block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
    return wrap(make_data(res_dtype, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::copy_data(TensorCPtr a, std::optional<std::string> device)
{
    // --- hints from Python FusionTreeBackend.copy_data ---
    // OPTIMIZE do we need to copy these?
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& block : a_data->blocks)
        blocks.push_back(block_backend->copy_block(block, device));
    std::string dev = device.has_value() ? block_backend->as_device(*device) : a_data->device;
    return wrap(
      make_data(a_data->dtype, std::move(dev), std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::dagger(TensorCPtr a)
{
    // --- hints from Python FusionTreeBackend.dagger ---
    // domain and codomain have swapped
    // ---
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->dagger(b));
    BlockInds block_inds = a_data->block_inds.reverse_columns();
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
FusionTreeBackend::data_item(DataPtr a)
{
    auto data = unwrap(a);
    if (data->blocks.size() > 1)
        throw std::runtime_error("Inconsistent data.");
    if (data->blocks.empty())
        return block_backend->as_scalar(dtype::zero_scalar(data->dtype), data->dtype);
    return block_backend->item(data->blocks[0]);
}

bool
FusionTreeBackend::diagonal_all(DiagonalTensorCPtr a)
{
    // --- hints from Python FusionTreeBackend.diagonal_all ---
    // there are missing blocks. -> they contain False -> all(a) == False
    // now it is enough to check the existing blocks
    // ---
    auto data = data_from_tensor(a);
    if (static_cast<int64>(data->blocks.size()) < nsec(py::cast(a->domain)))
        return false;
    for (auto const& b : data->blocks)
        if (!block_backend->all(b))
            return false;
    return true;
}

bool
FusionTreeBackend::diagonal_any(DiagonalTensorCPtr a)
{
    auto data = data_from_tensor(a);
    for (auto const& b : data->blocks)
        if (block_backend->any(b))
            return true;
    return false;
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                               DiagonalTensorCPtr b,
                                               BlockBinaryFn func,
                                               bool partial_zero_is_zero)
{
    // --- hints from Python FusionTreeBackend.diagonal_elementwise_binary ---
    // a_block_inds[:n_a] already visited
    // b_block_inds[:n_b] already visited
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    std::vector<BlockBackend::BlockPtr> blocks;
    BlockInds block_inds;
    if (partial_zero_is_zero) {
        std::vector<std::vector<int64>> rows;
        iter_common_sorted_1d(
          a_data->block_inds.column(0),
          b_data->block_inds.column(0),
          [&](std::ptrdiff_t i, std::ptrdiff_t j) {
              rows.push_back({ a_data->block_inds(static_cast<std::size_t>(i), 0),
                               a_data->block_inds(static_cast<std::size_t>(i), 1) });
              blocks.push_back(func(a_data->blocks[static_cast<std::size_t>(i)],
                                    b_data->blocks[static_cast<std::size_t>(j)]));
          });
        block_inds = rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(rows);
    } else {
        auto a_col0 = a_data->block_inds.column(0);
        auto b_col0 = b_data->block_inds.column(0);
        int64 n_a = 0;
        int64 bi_a = a_col0.empty() ? -1 : a_col0[static_cast<std::size_t>(n_a)];
        int64 n_b = 0;
        int64 bi_b = b_col0.empty() ? -1 : b_col0[static_cast<std::size_t>(n_b)];
        int64 num_sectors = py::cast(a->codomain).attr("num_sectors").cast<int64>();
        for (int64 i = 0; i < num_sectors; ++i) {
            BlockBackend::BlockPtr a_block;
            if (i == bi_a) {
                a_block = a_data->blocks[static_cast<std::size_t>(n_a++)];
                bi_a = static_cast<std::size_t>(n_a) >= a_col0.size()
                         ? -1
                         : a_col0[static_cast<std::size_t>(n_a)];
            } else {
                a_block = block_backend->zeros(
                  { mults_of(py::cast(a->domain))[static_cast<std::size_t>(i)] }, a->dtype);
            }
            BlockBackend::BlockPtr b_block;
            if (i == bi_b) {
                b_block = b_data->blocks[static_cast<std::size_t>(n_b++)];
                bi_b = static_cast<std::size_t>(n_b) >= b_col0.size()
                         ? -1
                         : b_col0[static_cast<std::size_t>(n_b)];
            } else {
                b_block = block_backend->zeros(
                  { mults_of(py::cast(a->domain))[static_cast<std::size_t>(i)] }, b->dtype);
            }
            blocks.push_back(func(a_block, b_block));
        }
        block_inds = BlockInds::arange_diag(static_cast<std::size_t>(num_sectors));
    }
    Dtype dt = blocks.empty()
                 ? block_backend->get_dtype(func(block_backend->ones_block({ 1 }, a->dtype),
                                                 block_backend->ones_block({ 1 }, b->dtype)))
                 : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                              BlockUnaryFn func,
                                              bool maps_zero_to_zero)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    BlockInds block_inds;
    if (maps_zero_to_zero) {
        for (auto const& b : a_data->blocks)
            blocks.push_back(func(b));
        block_inds = a_data->block_inds;
    } else {
        auto col0 = a_data->block_inds.column(0);
        int64 n = 0;
        int64 bi = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
        int64 num_sectors = py::cast(a->codomain).attr("num_sectors").cast<int64>();
        for (int64 i = 0; i < num_sectors; ++i) {
            BlockBackend::BlockPtr block;
            if (i == bi) {
                block = a_data->blocks[static_cast<std::size_t>(n++)];
                bi = static_cast<std::size_t>(n) >= col0.size()
                       ? -1
                       : col0[static_cast<std::size_t>(n)];
            } else {
                block = block_backend->zeros(
                  { tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i)] }, a->dtype);
            }
            blocks.push_back(func(block));
        }
        block_inds = BlockInds::arange_diag(static_cast<std::size_t>(num_sectors));
    }
    Dtype dt = blocks.empty()
                 ? block_backend->get_dtype(func(block_backend->ones_block({ 1 }, a->dtype)))
                 : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_from_block(BlockBackend::BlockPtr a,
                                       TensorProduct::Ptr co_domain,
                                       float64 tol)
{
    // --- hints from Python FusionTreeBackend.diagonal_from_block ---
    // OPTIMIZE (JU) this lookup is annoying, but currently needed because of potentially
    // different sorting order of the ``a.domain == TensorProduct(a.leg)``
    // versus the ``a.leg`` itself
    // project onto the identity on the coupled sector
    // ---
    py::object leg = py::cast(co_domain->factors[0]);
    Dtype dt = block_backend->get_dtype(a);
    BlockInds block_inds =
      BlockInds::arange_diag(static_cast<std::size_t>(co_domain->num_sectors));
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t idx = 0; idx < co_domain->sector_decomposition.size(); ++idx) {
        Sector coupled = co_domain->sector_decomposition[idx];
        int64 dim_c = co_domain->symmetry->sector_dim(coupled);
        auto j = leg.attr("sector_decomposition_where")(py::cast(coupled)).cast<int64>();
        auto slc = slice_pair(leg.attr("slices").attr("__getitem__")(j));
        auto entries = block_backend->reshape(b_get(a, py::make_tuple(slc)),
                                              { dim_c, co_domain->multiplicities[idx] });
        auto block = block_backend->sum(entries, 0);
        block = (*(block)) / block_backend->as_scalar(static_cast<float64>(dim_c));
        auto projected = block_backend->outer(block_backend->ones_block({ dim_c }, dt), block);
        if (block_backend->norm((*entries) - (*projected)).as_float64() >
            tol * block_backend->norm(entries).as_float64())
            throw std::invalid_argument("Block is not symmetric up to tolerance.");
        blocks.push_back(block);
    }
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                                   TensorProduct::Ptr co_domain)
{
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t coupled_idx = 0; coupled_idx < co_domain->sector_decomposition.size();
         ++coupled_idx) {
        blocks.push_back(
          func(std::vector<int64>{ co_domain->block_size(static_cast<int64>(coupled_idx)) },
               co_domain->sector_decomposition[coupled_idx]));
    }
    BlockInds block_inds =
      BlockInds::arange_diag(static_cast<std::size_t>(co_domain->num_sectors));
    BlockBackend::BlockPtr sample =
      blocks.empty() ? func(std::vector<int64>{ 1 }, co_domain->symmetry->trivial_sector)
                     : blocks[0];
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                                    std::optional<float64> tol)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->get_diagonal(b, tol));
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

BlockBackend::Scalar
FusionTreeBackend::diagonal_tensor_trace_full(DiagonalTensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto qdims = py::cast(a->domain).attr("sector_qdims").cast<std::vector<float64>>();
    auto res = block_backend->as_scalar(0.0, a->dtype);
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < bi.nrows(); ++n)
        res = res +
              block_backend->as_scalar(qdims[static_cast<std::size_t>(bi(n, 0))] *
                                         block_backend->sum_all(a_data->blocks[n]).as_complex128(),
                                       a->dtype);
    return res;
}

BlockBackend::BlockPtr
FusionTreeBackend::diagonal_tensor_to_block(DiagonalTensorCPtr a)
{
    // --- hints from Python FusionTreeBackend.diagonal_tensor_to_block ---
    // OPTIMIZE (JU) this lookup is annoying, but currently needed because of potentially
    // different sorting order of the ``a.domain == TensorProduct(a.leg)``
    // versus the ``a.leg`` itself
    // ---
    assert(py::cast(a->symmetry).attr("can_be_dropped").cast<bool>());
    auto a_data = data_from_tensor(a);
    auto res = block_backend->zeros({ py::cast(a->leg()).attr("dim").cast<int64>() }, a->dtype);
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        int64 bi_cod = bi(n, 0);
        Sector c = py::cast(a->codomain)
                     .attr("sector_decomposition")
                     .attr("__getitem__")(bi_cod)
                     .cast<Sector>();
        int64 dim_c =
          py::cast(a->codomain).attr("sector_dims").attr("__getitem__")(bi_cod).cast<int64>();
        auto entries = block_backend->reshape(
          block_backend->outer(block_backend->ones_block({ dim_c }, a->dtype), a_data->blocks[n]),
          std::vector<int64>{ -1 });
        auto j = py::cast(a->leg()).attr("sector_decomposition_where")(py::cast(c)).cast<int64>();
        b_set(res,
              py::make_tuple(slice_pair(py::cast(a->leg()).attr("slices").attr("__getitem__")(j))),
              entries);
    }
    return res;
}

std::string
FusionTreeBackend::get_device_from_data(DataPtr a)
{
    return block_backend->as_device(unwrap(a)->device);
}

Dtype
FusionTreeBackend::get_dtype_from_data(DataPtr a)
{
    return unwrap(a)->dtype;
}

bool
FusionTreeBackend::supports_symmetry(Symmetry::Ptr symmetry)
{
    // --- hints from Python FusionTreeBackend.supports_symmetry ---
    // supports all symmetries
    // ---
    return symmetry != nullptr;
}

TensorBackend::DataPtr
FusionTreeBackend::move_to_device(TensorCPtr a, std::string device)
{
    auto a_data = data_from_tensor(a);
    for (auto& b : a_data->blocks)
        b = block_backend->as_block(py::cast(b), std::nullopt, device);
    a_data->device = block_backend->as_device(device);
    return wrap(a_data);
}

TensorBackend::DataPtr
FusionTreeBackend::to_dtype(TensorCPtr a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::to_block_backend(DataPtr data,
                                    std::shared_ptr<BlockBackend> bb,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    auto d = unwrap(data);
    Dtype dt = dtype.has_value() ? *dtype : d->dtype;
    std::string dev = bb->as_device(device.has_value() ? *device : d->device);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : d->blocks)
        blocks.push_back(bb->as_block(py::cast(b), dt, dev));
    return wrap(make_data(dt, dev, std::move(blocks), d->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::full_data_from_diagonal_tensor(DiagonalTensorCPtr a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_diagonal(b));
    return wrap(make_data(a->dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::full_data_from_mask(MaskCPtr a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_mask(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::mask_dagger(MaskCPtr mask)
{
    // --- hints from Python FusionTreeBackend.mask_dagger ---
    // the legs swap between domain and codomain. need to swap the two columns of block_inds.
    // since both columns are unique and ascending, the resulting block_inds are still sorted.
    // ---
    auto data = data_from_tensor(mask);
    BlockInds block_inds = data->block_inds.reverse_columns();
    return wrap(make_data(mask->dtype, mask->device, data->blocks, block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::mask_to_diagonal(MaskCPtr a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    auto large_leg_bi =
      a->is_projection ? a_data->block_inds.column(1) : a_data->block_inds.column(0);
    BlockInds block_inds =
      BlockInds::column_stack(std::vector<std::span<const int64>>{ large_leg_bi, large_leg_bi });
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::split_legs(TensorCPtr a,
                              std::vector<int64> leg_idcs,
                              TensorProduct::Ptr new_codomain,
                              TensorProduct::Ptr new_domain)
{
    return wrap(data_from_tensor(a));
}

TensorBackend::DataPtr
FusionTreeBackend::squeeze_legs(TensorCPtr a, std::vector<int64> idcs)
{
    return wrap(data_from_tensor(a));
}

TensorBackend::DataPtr
FusionTreeBackend::from_dense_block_trivial_sector(BlockBackend::BlockPtr /*block*/,
                                                   Space::Ptr /*leg*/)
{
    throw NotImplemented("from_dense_block_trivial_sector not implemented");
}

TensorBackend::DataPtr
FusionTreeBackend::inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr /*vector*/,
                                                           Space::Ptr /*space*/,
                                                           ElementarySpace::Ptr /*charge_leg*/)
{
    throw NotImplemented("inv_part_from_dense_block_single_sector not implemented");
}

BlockBackend::BlockPtr
FusionTreeBackend::inv_part_to_dense_block_single_sector(SymmetricTensorCPtr tensor)
{
    throw NotImplemented("inv_part_to_dense_block_single_sector not implemented");
}

py::object
FusionTreeBackend::state_tensor_product(BlockBackend::BlockPtr /*state1*/,
                                        BlockBackend::BlockPtr /*state2*/,
                                        Leg::Ptr /*pipe*/)
{
    // --- hints from Python FusionTreeBackend.state_tensor_product ---
    // TODO clearly define what this should do in tensors.py first!
    // ---
    throw NotImplemented("state_tensor_product not implemented");
}

BlockBackend::BlockPtr
FusionTreeBackend::to_dense_block_trivial_sector(TensorCPtr tensor)
{
    throw NotImplemented("to_dense_block_trivial_sector not implemented");
}

TensorBackend::DataPtr
FusionTreeBackend::zero_diagonal_data(TensorProduct::Ptr /*co_domain*/,
                                      Dtype dtype,
                                      std::string device)
{
    return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
FusionTreeBackend::zero_mask_data(Space::Ptr /*large_leg*/, std::string device)
{
    return wrap(make_data(Dtype::Bool, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
FusionTreeBackend::zero_data(TensorProduct::Ptr codomain,
                             TensorProduct::Ptr domain,
                             Dtype dtype,
                             std::string device,
                             bool all_blocks)
{
    if (!all_blocks)
        return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
    std::vector<std::vector<int64>> block_inds_rows;
    std::vector<BlockBackend::BlockPtr> zero_blocks;
    for (std::size_t j = 0; j < domain->sector_decomposition.size(); ++j) {
        Sector coupled = domain->sector_decomposition[j];
        auto i = codomain->sector_decomposition_where(coupled);
        if (!i.has_value())
            continue;
        block_inds_rows.push_back({ *i, static_cast<int64>(j) });
        zero_blocks.push_back(block_backend->zeros(
          { codomain->block_size(*i), domain->block_size(static_cast<int64>(j)) }, dtype, device));
    }
    BlockInds block_inds =
      block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
    return wrap(make_data(dtype, std::move(device), std::move(zero_blocks), block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device)
{
    // --- hints from Python FusionTreeBackend.eye_data ---
    // Note: the identity has the same matrix elements in all ONB, so no need to consider
    // the basis perms.
    // ---
    std::vector<BlockBackend::BlockPtr> blocks;
    for (int64 c_idx = 0; c_idx < co_domain->num_sectors; ++c_idx)
        blocks.push_back(block_backend->eye_matrix(co_domain->block_size(c_idx), dtype, device));
    BlockInds block_inds =
      BlockInds::arange_diag(static_cast<std::size_t>(co_domain->num_sectors));
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::from_sector_block_func(SectorBlockFactoryFn func,
                                          TensorProduct::Ptr codomain,
                                          TensorProduct::Ptr domain)
{
    std::vector<std::vector<int64>> block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    SectorArray::iter_common_sorted(
      codomain->sector_decomposition,
      domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_p, std::ptrdiff_t j_p) {
          int64 i = static_cast<int64>(i_p);
          int64 j = static_cast<int64>(j_p);
          block_inds_rows.push_back({ i, j });
          Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
          blocks.push_back(func({ codomain->block_size(i), domain->block_size(j) }, coupled));
      });
    BlockBackend::BlockPtr sample =
      blocks.empty() ? func({ 1, 1 }, codomain->symmetry->trivial_sector) : blocks[0];
    BlockInds block_inds =
      block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::from_random_normal(TensorProduct::Ptr codomain,
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

BlockBackend::Scalar
FusionTreeBackend::inner(SymmetricTensorCPtr a, SymmetricTensorCPtr b, bool do_dagger)
{
    // --- hints from Python FusionTreeBackend.inner ---
    // need to match a.codomain == b.codomain
    // need to math a.codomain == b.domain
    // ---
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto qdims = py::cast(a->codomain).attr("sector_qdims").cast<std::vector<float64>>();
    auto a_col0 = a_data->block_inds.column(0);
    auto b_inds = do_dagger ? b_data->block_inds.column(0) : b_data->block_inds.column(1);
    auto res = block_backend->as_scalar(0.0, a->dtype);
    iter_common_sorted_1d(a_col0, b_inds, [&](std::ptrdiff_t i, std::ptrdiff_t j) {
        int64 bi = a_col0[static_cast<std::size_t>(i)];
        auto inn = block_backend->inner(a_data->blocks[static_cast<std::size_t>(i)],
                                        b_data->blocks[static_cast<std::size_t>(j)],
                                        do_dagger);
        res = res + block_backend->as_scalar(
                      qdims[static_cast<std::size_t>(bi)] * inn.as_complex128(), a->dtype);
    });
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::trace_full(SymmetricTensorCPtr a,
                              std::vector<int64> idcs1,
                              std::vector<int64> idcs2)
{
    auto a_data = data_from_tensor(a);
    auto qdims = py::cast(a->codomain).attr("sector_qdims").cast<std::vector<float64>>();
    auto res = block_backend->as_scalar(dtype::zero_scalar(a_data->dtype), a_data->dtype);
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        int64 bi_cod = bi(n, 0);
        res = res + block_backend->as_scalar(
                      qdims[static_cast<std::size_t>(bi_cod)] *
                        block_backend->trace_full(a_data->blocks[n]).as_complex128(),
                      a_data->dtype);
    }
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::norm(TensorCPtr a)
{
    // --- hints from Python FusionTreeBackend.norm ---
    // OPTIMIZE should we offer the square-norm instead?
    // ---
    auto a_data = data_from_tensor(a);
    auto qdims = py::cast(a->codomain).attr("sector_qdims").cast<std::vector<float64>>();
    float64 norm_sq = 0.;
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        auto nblock = block_backend->norm(a_data->blocks[n]);
        norm_sq +=
          qdims[static_cast<std::size_t>(bi(n, 0))] * nblock.as_float64() * nblock.as_float64();
    }
    return block_backend->as_scalar(std::sqrt(norm_sq), dtype::to_real(a->dtype));
}

TensorBackend::DataPtr
FusionTreeBackend::mul(BlockBackend::Scalar a, TensorCPtr b)
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
    Dtype dt = blocks.empty()
                 ? (dtype::is_real(a.dtype()) ? b_data->dtype : dtype::to_complex(b_data->dtype))
                 : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, b_data->device, std::move(blocks), b_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::linear_combination(BlockBackend::Scalar a,
                                      TensorCPtr v,
                                      BlockBackend::Scalar b,
                                      TensorCPtr w)
{
    auto v_data = data_from_tensor(v);
    auto w_data = data_from_tensor(w);
    Dtype common_dtype = dtype::common({ v->dtype, w->dtype });
    auto v_blocks = v_data->blocks;
    auto w_blocks = w_data->blocks;
    if (v_data->dtype != common_dtype)
        for (auto& T : v_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    if (w_data->dtype != common_dtype)
        for (auto& T : w_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    std::vector<std::vector<int64>> block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    iter_common_noncommon_sorted_1d(
      v_data->block_inds.column(0),
      w_data->block_inds.column(0),
      [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
          if (!i) {
              blocks.push_back(block_backend->mul(b, w_blocks[static_cast<std::size_t>(*j)]));
              block_inds_rows.push_back({ w_data->block_inds(static_cast<std::size_t>(*j), 0),
                                          w_data->block_inds(static_cast<std::size_t>(*j), 1) });
          } else if (!j) {
              blocks.push_back(block_backend->mul(a, v_blocks[static_cast<std::size_t>(*i)]));
              block_inds_rows.push_back({ v_data->block_inds(static_cast<std::size_t>(*i), 0),
                                          v_data->block_inds(static_cast<std::size_t>(*i), 1) });
          } else {
              blocks.push_back(
                block_backend->linear_combination(a,
                                                  v_blocks[static_cast<std::size_t>(*i)],
                                                  b,
                                                  w_blocks[static_cast<std::size_t>(*j)]));
              block_inds_rows.push_back({ v_data->block_inds(static_cast<std::size_t>(*i), 0),
                                          v_data->block_inds(static_cast<std::size_t>(*i), 1) });
          }
      });
    BlockInds block_inds =
      block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
    return wrap(make_data(common_dtype, v_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
FusionTreeBackend::get_element_diagonal(DiagonalTensorCPtr a, int64 idx)
{
    auto [sector_idx, idx_within] = parse_leg_index(a->leg(), idx);
    auto leg_es = std::dynamic_pointer_cast<ElementarySpace>(a->leg());
    if (!leg_es) {
        throw std::invalid_argument("get_element_diagonal requires ElementarySpace leg");
    }
    int64 multi = leg_es->multiplicities[static_cast<std::size_t>(sector_idx)];
    if (leg_es->is_dual) {
        Sector sector = leg_es->sector_decomposition[static_cast<std::size_t>(sector_idx)];
        auto where = a->domain->sector_decomposition_where(sector);
        if (!where) {
            throw std::runtime_error("get_element_diagonal: sector missing in domain");
        }
        sector_idx = *where;
    }
    auto block_idx = data_from_tensor(a)->block_ind_from_domain_sector_ind(sector_idx);
    if (!block_idx.has_value())
        return block_backend->as_scalar(dtype::zero_scalar(a->dtype), a->dtype);
    return block_backend->get_block_element(
      data_from_tensor(a)->blocks[static_cast<std::size_t>(*block_idx)], { idx_within % multi });
}

BlockBackend::Scalar
FusionTreeBackend::get_element_mask(MaskCPtr a, std::vector<int64> idcs)
{
    // --- hints from Python FusionTreeBackend.get_element_mask ---
    // domain leg index
    // ---
    auto legs = conventional_leg_order(a);
    std::vector<std::vector<int64>> pairs;
    pairs.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto [b, w] = parse_leg_index(legs[i], idcs[i]);
        pairs.push_back({ b, w });
    }
    BlockInds pos = BlockInds::from_rows(pairs);
    int64 sector_idx = pos(1, 0);
    Sector sector = py::cast(a->domain)
                      .attr("__getitem__")(0)
                      .attr("sector_decomposition")
                      .attr("__getitem__")(sector_idx)
                      .cast<Sector>();
    if (sector != py::cast(a->codomain)
                    .attr("__getitem__")(0)
                    .attr("sector_decomposition")
                    .attr("__getitem__")(pos(0, 0))
                    .cast<Sector>())
        return block_backend->as_scalar(false);
    if (py::cast(a->domain).attr("__getitem__")(0).attr("is_dual").cast<bool>())
        sector_idx =
          py::cast(a->domain).attr("sector_decomposition_where")(py::cast(sector)).cast<int64>();
    auto block_idx = data_from_tensor(a)->block_ind_from_domain_sector_ind(sector_idx);
    if (!block_idx.has_value())
        return block_backend->as_scalar(false);
    int64 small, large, multi;
    if (a->is_projection) {
        small = pos(0, 1);
        large = pos(1, 1);
        multi = mults_of(py::cast(a->small_leg()))[static_cast<std::size_t>(pos(0, 0))];
    } else {
        large = pos(0, 1);
        small = pos(1, 1);
        multi = mults_of(py::cast(a->small_leg()))[static_cast<std::size_t>(pos(1, 0))];
    }
    return block_backend->get_block_mask_element(
      data_from_tensor(a)->blocks[static_cast<std::size_t>(*block_idx)], large, small, multi);
}

TensorBackend::DataPtr
FusionTreeBackend::permute_legs(TensorCPtr a,
                                std::vector<int64> codomain_idcs,
                                std::vector<int64> domain_idcs,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain,
                                bool mixes_codomain_domain,
                                std::vector<std::optional<int64>> levels,
                                std::vector<std::optional<bool>> bend_right)
{
    // --- hints from Python FusionTreeBackend.permute_legs ---
    // mapping to flat indices for TreeMappingDict.from_permute_legs
    // OPTIMIZE rm check?
    // ---
    std::vector<std::optional<int64>> flat_levels;
    std::vector<std::optional<bool>> flat_bend_right;
    py::list codomain_pipe_inds;
    py::list domain_pipe_inds;
    int64 flat_index = 0;
    int64 num_codomain_flat_legs = a->num_codomain_flat_legs();
    py::list legs = py::cast(a->legs());
    for (py::ssize_t i = 0; i < py::len(legs); ++i) {
        py::object leg = legs[i];
        bool is_codomain = i < a->num_codomain_legs();
        // Match Python ``isinstance(leg, LegPipe)`` — not ``hasattr(num_flat_legs)``,
        // since ElementarySpace also exposes num_flat_legs.
        bool const is_pipe = py::isinstance<LegPipe>(leg);
        if (is_pipe) {
            int64 num = leg.attr("num_flat_legs").cast<int64>();
            py::list indices;
            for (int64 k = 0; k < num; ++k)
                indices.append(flat_index + k);
            if (is_codomain)
                codomain_pipe_inds.append(indices);
            else
                domain_pipe_inds.append(indices);
            flat_index += num;
            for (int64 k = 0; k < num; ++k) {
                flat_levels.push_back(levels[static_cast<std::size_t>(i)]);
                flat_bend_right.push_back(bend_right[static_cast<std::size_t>(i)]);
            }
        } else {
            py::list indices;
            indices.append(flat_index);
            if (is_codomain)
                codomain_pipe_inds.append(indices);
            else
                domain_pipe_inds.append(indices);
            ++flat_index;
            flat_levels.push_back(levels[static_cast<std::size_t>(i)]);
            flat_bend_right.push_back(bend_right[static_cast<std::size_t>(i)]);
        }
    }
    py::list leg_comb;
    for (py::handle x : codomain_pipe_inds)
        leg_comb.append(x);
    for (py::handle x : domain_pipe_inds)
        leg_comb.append(x);
    std::vector<int64> new_domain_idcs;
    for (int64 idx : domain_idcs) {
        py::list rev = leg_comb[static_cast<py::ssize_t>(idx)].cast<py::list>();
        for (py::ssize_t ri = py::len(rev) - 1; ri >= 0; --ri)
            new_domain_idcs.push_back(rev[ri].cast<int64>());
    }
    std::vector<int64> new_codomain_idcs;
    for (int64 idx : codomain_idcs) {
        for (py::handle k : leg_comb[static_cast<py::ssize_t>(idx)])
            new_codomain_idcs.push_back(k.cast<int64>());
    }
    int64 const num_domain_flat_legs = a->num_domain_flat_legs();
    bool const has_symmetric_braid =
      py::cast(a->symmetry).attr("has_symmetric_braid").cast<bool>();
    PermuteLegsInstructionEngine engine(num_codomain_flat_legs,
                                        num_domain_flat_legs,
                                        new_codomain_idcs,
                                        new_domain_idcs,
                                        flat_levels,
                                        flat_bend_right,
                                        has_symmetric_braid);
    auto instructions = engine.evaluate_instructions();
    py::list instructions_py;
    for (Instruction const& inst : instructions) {
        std::visit([&](auto const& i) { instructions_py.append(py::cast(i)); }, inst);
    }
    return apply_instructions(a,
                              instructions_py,
                              new_codomain_idcs,
                              new_domain_idcs,
                              new_codomain,
                              new_domain,
                              mixes_codomain_domain);
}

std::tuple<BlockBackend::BlockPtr, int64, int64>
FusionTreeBackend::_get_forest_block_contribution(BlockBackend::BlockPtr block,
                                                  Symmetry::Ptr sym,
                                                  TensorProduct::Ptr codomain,
                                                  TensorProduct::Ptr domain,
                                                  Sector coupled,
                                                  py::object a_sectors,
                                                  py::object b_sectors,
                                                  std::vector<int64> a_dims,
                                                  std::vector<int64> b_dims,
                                                  int64 tree_block_width,
                                                  int64 tree_block_height,
                                                  int64 i1_init,
                                                  int64 i2_init,
                                                  std::vector<int64> m_mults,
                                                  std::vector<int64> n_mults,
                                                  Dtype dtype) const
{
    // --- hints from Python FusionTreeBackend._get_forest_block_contribution ---
    // OPTIMIZE do one loop per vertex in the tree instead.
    // i1: start row index of the current tree block within the block
    // i2: start column index of the current tree block within the block
    // [a1,...,aJ,c]
    // [b1,...,bK,c]
    // [a1,...,aJ,b1,...,bK]
    // [M, N] -> [m1,...,mJ,n1,...,nK]
    // [{aj} {bk} {mj} {nk}]
    // reset to the left of the current forest-block
    // OPTIMIZE count loop iterations above instead?  (same in _add_forest_block_entries)
    // ---
    SectorArray a_sectors_arr = a_sectors.cast<SectorArray>();
    SectorArray b_sectors_arr = b_sectors.cast<SectorArray>();
    auto codomain_are_dual = flat_are_dual(codomain);
    auto domain_are_dual = flat_are_dual(domain);
    std::vector<int64> shape;
    for (int64 d : a_dims)
        shape.push_back(d);
    for (int64 d : b_dims)
        shape.push_back(d);
    for (int64 m : m_mults)
        shape.push_back(m);
    for (int64 n : n_mults)
        shape.push_back(n);
    auto entries = block_backend->zeros(shape, dtype);
    int64 i1 = i1_init;
    int64 i2 = i2_init;
    fusion_trees alpha_iter(sym, a_sectors_arr, coupled, codomain_are_dual);
    fusion_trees beta_iter(sym, b_sectors_arr, coupled, domain_are_dual);
    BlockBackend* bb = block_backend.get();
    for (auto const& alpha_tree : alpha_iter.all_trees()) {
        auto splitting_tree =
          block_backend->conj(alpha_tree.to_dense_block(bb, std::nullopt, true));
        for (auto const& beta_tree : beta_iter.all_trees()) {
            auto fusion_tree = beta_tree.to_dense_block(bb, std::nullopt, true);
            auto symmetry_data = block_backend->tdot(splitting_tree, fusion_tree, { -1 }, { -1 });
            py::slice idx1(i1, i1 + tree_block_height, 1);
            py::slice idx2(i2, i2 + tree_block_width, 1);
            auto degeneracy_data = b_get(block, py::make_tuple(idx1, idx2));
            degeneracy_data = block_backend->reshape(degeneracy_data, [&]() {
                std::vector<int64> s = m_mults;
                s.insert(s.end(), n_mults.begin(), n_mults.end());
                return s;
            }());
            entries = (*(entries)) + (*(block_backend->outer(symmetry_data, degeneracy_data)));
            i2 += tree_block_width;
        }
        i2 = i2_init;
        i1 += tree_block_height;
    }
    return { entries,
             static_cast<int64>(alpha_iter.size()),
             static_cast<int64>(beta_iter.size()) };
}

std::tuple<int64, int64>
FusionTreeBackend::_add_forest_block_entries(BlockBackend::BlockPtr block,
                                             BlockBackend::BlockPtr entries,
                                             Symmetry::Ptr sym,
                                             TensorProduct::Ptr codomain,
                                             TensorProduct::Ptr domain,
                                             Sector coupled,
                                             float64 dim_c,
                                             py::object a_sectors,
                                             py::object b_sectors,
                                             std::vector<int64> a_dims,
                                             std::vector<int64> b_dims,
                                             int64 tree_block_width,
                                             int64 tree_block_height,
                                             int64 i1_init,
                                             int64 i2_init,
                                             std::vector<int64> m_mults,
                                             std::vector<int64> n_mults) const
{
    // --- hints from Python FusionTreeBackend._add_forest_block_entries ---
    // used in tdot calls below
    // entries: [a1,...,aJ,b1,...,bK,m1,...,mJ,n1,...,nK]
    // [{bk}, {mj}, {nk}, c]
    // [{mj}, {nk}, c, c']
    // projected onto the identity on [c, c']
    // make sure we set in-range elements! otherwise item assignment silently does nothing.
    // move right by one tree-block
    // move down by one tree-block (we reset to the left at start of the loop)
    // ---
    SectorArray a_sectors_arr = a_sectors.cast<SectorArray>();
    SectorArray b_sectors_arr = b_sectors.cast<SectorArray>();
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    std::vector<int64> range_J;
    for (int64 i = 0; i < J; ++i)
        range_J.push_back(i);
    std::vector<int64> range_K;
    for (int64 i = 0; i < K; ++i)
        range_K.push_back(i);
    std::vector<int64> range_JK;
    for (int64 i = 0; i < J + K; ++i)
        range_JK.push_back(i);
    auto codomain_are_dual = flat_are_dual(codomain);
    auto domain_are_dual = flat_are_dual(domain);
    int64 i1 = i1_init;
    int64 i2 = i2_init;
    fusion_trees alpha_iter(sym, a_sectors_arr, coupled, codomain_are_dual);
    fusion_trees beta_iter(sym, b_sectors_arr, coupled, domain_are_dual);
    BlockBackend* bb = block_backend.get();
    for (auto const& alpha_tree : alpha_iter.all_trees()) {
        auto Y = alpha_tree.to_dense_block(bb, std::nullopt, true);
        auto Y_projected = block_backend->tdot(entries, Y, range_J, range_J);
        for (auto const& beta_tree : beta_iter.all_trees()) {
            auto X = block_backend->conj(beta_tree.to_dense_block(bb, std::nullopt, true));
            auto YX_projected = block_backend->tdot(Y_projected, X, range_K, range_K);
            auto tree_block = block_backend->trace_partial(YX_projected, { -2 }, { -1 }, range_JK);
            tree_block = (*(tree_block)) / block_backend->as_scalar(dim_c);
            auto ms_ns = block_backend->get_shape(tree_block);
            tree_block = block_backend->reshape(
              tree_block,
              { prod_int(std::vector<int64>(ms_ns.begin(), ms_ns.begin() + J)),
                prod_int(std::vector<int64>(ms_ns.begin() + J, ms_ns.end())) });
            py::slice idx1(i1, i1 + tree_block_height, 1);
            py::slice idx2(i2, i2 + tree_block_width, 1);
            b_set(block, py::make_tuple(idx1, idx2), tree_block);
            i2 += tree_block_width;
        }
        i2 = i2_init;
        i1 += tree_block_height;
    }
    return { static_cast<int64>(alpha_iter.size()), static_cast<int64>(beta_iter.size()) };
}

TensorBackend::DataPtr
FusionTreeBackend::from_dense_block(BlockBackend::BlockPtr a,
                                    TensorProduct::Ptr codomain,
                                    TensorProduct::Ptr domain,
                                    float64 tol)
{
    // --- hints from Python FusionTreeBackend.from_dense_block ---
    // we cannot simply use cstyles as argument in combine_legs since we potentially need to deal
    // with nested pipes with different cstyles -> choose to combine in C style and do axis
    // permutation explicitly use inverse permutation here s.t. it the dims agree with the flat
    // dims after permute_axes below convert to internal basis order, where the sectors are sorted
    // and contiguous [i1,...,iJ,jK,...,j1] -> [i1,...,iJ,j1,...,jK] main loop: iterate over
    // coupled sectors and construct the respective block. OPTIMIZE could be sth like np.empty
    // iterate over uncoupled sectors / forest-blocks within the block
    // start row index of the current forest block
    // start column index of the current forest block
    // [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
    // reshape to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
    // permute to [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...nK]
    // move down by one forest-block
    // reset to the top of the block
    // move right by one forest-block
    // since the symmetric and non-symmetric components of ``a = a_sym + a_rest`` are mutually
    // orthogonal, we have  ``norm(a) ** 2 = norm(a_sym) ** 2 + norm(a_rest) ** 2``.
    // thus ``abs_err = norm(a - a_sym) = norm(a_rest) = sqrt(norm(a) ** 2 - norm(a_sym) ** 2)``
    // ---
    if (codomain->has_pipes() || domain->has_pipes()) {
        // Match Python: split pipes on the dense block, then apply flat-leg permute.
        py::list idcs;
        py::list split_dims;
        py::list legs;
        for (auto const& f : codomain->factors)
            legs.append(py::cast(f));
        for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it)
            legs.append(py::cast(it->dual()));
        auto axes_perm = legs_flat_leg_permutation(legs);
        int64 perm_idx = 0;
        for (py::ssize_t n = 0; n < legs.size(); ++n) {
            py::object leg = legs[n];
            if (py::isinstance<LegPipe>(leg)) {
                idcs.append(n);
                py::list dims;
                for (auto const& s : leg.attr("flat_legs"))
                    dims.append(s.attr("dim"));
                std::vector<int64> perm(axes_perm.begin() + perm_idx,
                                        axes_perm.begin() + perm_idx +
                                          leg.attr("num_flat_legs").cast<int64>());
                int64 perm_min = *std::min_element(perm.begin(), perm.end());
                for (auto& p : perm)
                    p -= perm_min;
                auto inv = inverse_permutation(perm);
                py::list split_dim;
                for (int64 ih : inv)
                    split_dim.append(dims[ih]);
                split_dims.append(split_dim);
                perm_idx += leg.attr("num_flat_legs").cast<int64>();
            } else {
                perm_idx += 1;
            }
        }
        a = block_backend->split_legs(
          a, idcs.cast<std::vector<int64>>(), split_dims.cast<std::vector<std::vector<int64>>>());
        a = block_backend->permute_axes(a, axes_perm);
    }
    assert(codomain->symmetry->can_be_dropped());
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    int64 num_legs = J + K;
    std::vector<int64> perm_axes;
    for (int64 i = 0; i < J; ++i)
        perm_axes.push_back(i);
    for (int64 i = K - 1; i >= 0; --i)
        perm_axes.push_back(J + i);
    a = block_backend->permute_axes(a, perm_axes);
    Dtype dt =
      dtype::common({ block_backend->get_dtype(a),
                      codomain->symmetry->fusion_tensor_dtype.value_or(Dtype::Complex128) });
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    float64 norm_sq_projected = 0.;
    SectorArray::iter_common_sorted(
      codomain->sector_decomposition,
      domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_p, std::ptrdiff_t j_p) {
          int64 i = static_cast<int64>(i_p);
          int64 j = static_cast<int64>(j_p);
          Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
          int64 dim_c = codomain->symmetry->sector_dim(coupled);
          auto block =
            block_backend->zeros({ codomain->multiplicities[static_cast<std::size_t>(i)],
                                   domain->multiplicities[static_cast<std::size_t>(j)] },
                                 dt);
          int64 i1 = 0;
          int64 i2 = 0;
          for (auto const& b_item : domain->iter_uncoupled(true)) {
              auto b_dims = codomain->symmetry->batch_sector_dim(b_item.uncoupled);
              int64 tree_block_width = domain->tree_block_size(b_item.uncoupled);
              int64 forest_block_width = 0;
              for (auto const& a_item : codomain->iter_uncoupled(true)) {
                  auto a_dims = codomain->symmetry->batch_sector_dim(a_item.uncoupled);
                  int64 tree_block_height = codomain->tree_block_size(a_item.uncoupled);
                  py::list j1;
                  py::list j2;
                  for (auto const& slc : *a_item.slices)
                      j1.append(slice_from_index_slice(slc));
                  for (auto const& slc : *b_item.slices)
                      j2.append(slice_from_index_slice(slc));
                  auto entries = b_get(a, concat_slice_lists(j1, j2));
                  std::vector<int64> shape(2 * num_legs);
                  for (std::size_t si = 0; si < a_dims.size(); ++si) {
                      shape[2 * si] = a_dims[si];
                      shape[2 * si + 1] = a_item.multiplicities[si];
                  }
                  for (std::size_t si = 0; si < b_dims.size(); ++si) {
                      shape[2 * (J + si)] = b_dims[si];
                      shape[2 * (J + si) + 1] = b_item.multiplicities[si];
                  }
                  entries = block_backend->reshape(entries, shape);
                  std::vector<int64> pperm;
                  for (int64 k = 0; k < 2 * num_legs; k += 2)
                      pperm.push_back(k);
                  for (int64 k = 1; k < 2 * num_legs; k += 2)
                      pperm.push_back(k);
                  entries = block_backend->permute_axes(entries, pperm);
                  auto [num_alpha_trees, num_beta_trees] =
                    _add_forest_block_entries(block,
                                              entries,
                                              codomain->symmetry,
                                              codomain,
                                              domain,
                                              coupled,
                                              static_cast<float64>(dim_c),
                                              py::cast(a_item.uncoupled),
                                              py::cast(b_item.uncoupled),
                                              a_dims,
                                              b_dims,
                                              tree_block_width,
                                              tree_block_height,
                                              i1,
                                              i2,
                                              a_item.multiplicities,
                                              b_item.multiplicities);
                  int64 forest_block_height = num_alpha_trees * tree_block_height;
                  forest_block_width = num_beta_trees * tree_block_width;
                  i1 += forest_block_height;
              }
              i1 = 0;
              i2 += forest_block_width;
          }
          float64 block_norm = block_backend->norm(block, 2.).as_float64();
          if (block_norm <= 1e-14)
              return;
          block_inds_rows.append(py::make_tuple(i, j));
          blocks.push_back(block);
          norm_sq_projected += dim_c * block_norm * block_norm;
      });
    if (tol != 0.) {
        float64 a_norm_sq = block_backend->norm(a, 2.).as_float64();
        a_norm_sq *= a_norm_sq;
        if (a_norm_sq - norm_sq_projected > tol * tol * a_norm_sq)
            throw std::invalid_argument("Block is not symmetric up to tolerance.");
    }
    BlockInds block_inds;
    if (block_inds_rows.size() > 0) {
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(block_inds_rows.size()));
        for (py::handle h : block_inds_rows) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        block_inds = BlockInds::from_rows(rows);
    } else {
        block_inds = zeros_i64(0, 2);
    }
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds));
}

BlockBackend::BlockPtr
FusionTreeBackend::to_dense_block(TensorCPtr a)
{
    // --- hints from Python FusionTreeBackend.to_dense_block ---
    // first build it with the flattened legs, and if there are pipes, combine at the very end
    // build in internal basis order, is converted to public basis order in
    // SymmetricTensor.to_dense_block build in codomain/domain leg order first, then permute legs
    // in the end permute leg order [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1] need to cast to
    // a list here since numpy arrays leads errors in the torch block backend
    // ---
    assert(py::cast(a->symmetry).attr("can_be_dropped").cast<bool>());
    auto a_data = data_from_tensor(a);
    auto codomain = py::cast(a->codomain).cast<TensorProduct::Ptr>();
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    int64 num_legs = J + K;
    Dtype dt = dtype::common(
      { a_data->dtype, py::cast(a->symmetry).attr("fusion_tensor_dtype").cast<Dtype>() });
    std::vector<int64> shape;
    for (auto const& leg : codomain->flat_legs())
        shape.push_back(leg.dim());
    for (auto const& leg : domain->flat_legs())
        shape.push_back(leg.dim());
    auto res = block_backend->zeros(shape, dt);
    auto const& bi = a_data->block_inds;
    for (std::size_t row = 0; row < bi.nrows(); ++row) {
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi(row, 0))];
        int64 i1 = 0;
        int64 i2 = 0;
        for (auto const& b_item : domain->iter_uncoupled(true)) {
            auto b_dims =
              py::cast(a->symmetry).cast<Symmetry::Ptr>()->batch_sector_dim(b_item.uncoupled);
            int64 tree_block_width = domain->tree_block_size(b_item.uncoupled);
            int64 forest_b_width = 0;
            for (auto const& a_item : codomain->iter_uncoupled(true)) {
                auto a_dims =
                  py::cast(a->symmetry).cast<Symmetry::Ptr>()->batch_sector_dim(a_item.uncoupled);
                int64 tree_block_height = codomain->tree_block_size(a_item.uncoupled);
                auto [entries, num_alpha, num_beta] =
                  _get_forest_block_contribution(a_data->blocks[static_cast<std::size_t>(row)],
                                                 py::cast(a->symmetry).cast<Symmetry::Ptr>(),
                                                 codomain,
                                                 domain,
                                                 coupled,
                                                 py::cast(a_item.uncoupled),
                                                 py::cast(b_item.uncoupled),
                                                 a_dims,
                                                 b_dims,
                                                 tree_block_width,
                                                 tree_block_height,
                                                 i1,
                                                 i2,
                                                 a_item.multiplicities,
                                                 b_item.multiplicities,
                                                 dt);
                int64 forest_b_height = num_alpha * tree_block_height;
                forest_b_width = num_beta * tree_block_width;
                if (forest_b_height == 0 || forest_b_width == 0)
                    continue;
                // entries : [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
                // permute to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                // Python: [i + offset for i in range(num_legs) for offset in [0, num_legs]]
                std::vector<int64> pperm;
                pperm.reserve(static_cast<std::size_t>(2 * num_legs));
                for (int64 i = 0; i < num_legs; ++i) {
                    pperm.push_back(i);
                    pperm.push_back(i + num_legs);
                }
                entries = block_backend->permute_axes(entries, pperm);
                std::vector<int64> rshape;
                for (std::size_t si = 0; si < a_dims.size(); ++si)
                    rshape.push_back(a_dims[si] * a_item.multiplicities[si]);
                for (std::size_t si = 0; si < b_dims.size(); ++si)
                    rshape.push_back(b_dims[si] * b_item.multiplicities[si]);
                entries = block_backend->reshape(entries, rshape);
                py::list j1;
                py::list j2;
                for (auto const& slc : *a_item.slices)
                    j1.append(slice_from_index_slice(slc));
                for (auto const& slc : *b_item.slices)
                    j2.append(slice_from_index_slice(slc));
                b_set_add(res, concat_slice_lists(j1, j2), entries);
                i1 += forest_b_height;
            }
            i1 = 0;
            i2 += forest_b_width;
        }
    }
    // [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1]
    std::vector<int64> final_perm;
    for (int64 i = 0; i < J; ++i)
        final_perm.push_back(i);
    for (int64 i = K - 1; i >= 0; --i)
        final_perm.push_back(J + i);
    res = block_backend->permute_axes(res, final_perm);

    if (a->has_pipes()) {
        py::list legs = py::cast(a->legs());
        auto axes_perm_fwd = legs_flat_leg_permutation(legs);
        auto inv = inverse_permutation(axes_perm_fwd);
        std::vector<int64> axes_perm = inv;
        std::vector<std::vector<int64>> combine_axes;
        for (py::handle g : py::cast(a->codomain).attr("flat_legs_nesting")())
            combine_axes.push_back(g.cast<std::vector<int64>>());
        py::list dom_nesting = py::cast(a->domain).attr("flat_legs_nesting")();
        for (py::ssize_t gi = dom_nesting.size() - 1; gi >= 0; --gi) {
            auto group = dom_nesting[gi].cast<std::vector<int64>>();
            std::vector<int64> mapped;
            mapped.reserve(group.size());
            for (auto it = group.rbegin(); it != group.rend(); ++it)
                mapped.push_back(J + K - 1 - *it);
            combine_axes.push_back(std::move(mapped));
        }
        res = block_backend->permute_axes(res, axes_perm);
        res = block_backend->combine_legs(res, combine_axes);
    }
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                         BlockToScalarFn block_func,
                                         ScalarReduceFn func)
{
    auto data = data_from_tensor(tensor);
    std::vector<BlockBackend::Scalar> numbers;
    auto col0 = data->block_inds.column(0);
    int64 n = 0;
    int64 bi = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 num_sectors = py::cast(tensor->codomain).attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        BlockBackend::BlockPtr block;
        if (i == bi) {
            block = data->blocks[static_cast<std::size_t>(n++)];
            bi =
              static_cast<std::size_t>(n) >= col0.size() ? -1 : col0[static_cast<std::size_t>(n)];
        } else {
            block = block_backend->zeros(
              { mults_of(py::cast(tensor->codomain))[static_cast<std::size_t>(i)] },
              tensor->dtype);
        }
        numbers.push_back(block_func(block));
    }
    return func(numbers);
}

std::tuple<Space::Ptr, TensorBackend::DataPtr>
FusionTreeBackend::diagonal_transpose(DiagonalTensorCPtr tens)
{
    // --- hints from Python FusionTreeBackend.diagonal_transpose ---
    // result has block associated with coupled sector c that is given by the block of tens
    // with coupled sector dual(c).
    // since the TensorProduct.sector_decomposition is always sorted, those corresponding
    // sectors do not appear in the same order.
    // OPTIMIZE doing this sorting is duplicate work between here and forming tens.leg.dual
    // ---
    auto perm_py = py::cast(tens->symmetry)
                     .attr("dual_sectors")(py::cast(tens->domain).attr("sector_decomposition"))
                     .attr("lexsort_indices")();
    auto inv = inverse_permutation(perm_py.cast<std::vector<int64>>());
    auto tens_data = data_from_tensor(tens);
    // Map each stored sector index through the dual-sector inverse permutation.
    auto col0 = tens_data->block_inds.column(0);
    auto col1 = tens_data->block_inds.column(1);
    for (std::size_t i = 0; i < col0.size(); ++i) {
        col0[i] = inv[static_cast<std::size_t>(col0[i])];
        col1[i] = inv[static_cast<std::size_t>(col1[i])];
    }
    BlockInds block_inds =
      BlockInds::column_stack(std::vector<std::span<const int64>>{ col0, col1 });
    auto data = make_data(tens->dtype, tens_data->device, tens_data->blocks, block_inds, false);
    return { py::cast(tens->leg()).attr("dual").cast<Space::Ptr>(), wrap(data) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::eigh(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    // --- hints from Python FusionTreeBackend.eigh ---
    // there is not block for that sector. => eigenvalues are 0.
    // choose eigenvectors as standard basis vectors (eye matrix)
    // ---
    auto a_data = data_from_tensor(a);
    auto new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    std::vector<BlockBackend::BlockPtr> v_blocks;
    std::vector<BlockBackend::BlockPtr> w_blocks;
    auto col0 = a_data->block_inds.column(0);
    int64 n = 0;
    int64 bi = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 num_sectors = py::cast(a->codomain).attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        if (i == bi) {
            auto [vals, vects] =
              block_backend->eigh(a_data->blocks[static_cast<std::size_t>(n)], sort);
            v_blocks.push_back(vects);
            w_blocks.push_back(vals);
            ++n;
            bi =
              static_cast<std::size_t>(n) >= col0.size() ? -1 : col0[static_cast<std::size_t>(n)];
        } else {
            int64 block_size = tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i)];
            v_blocks.push_back(block_backend->eye_matrix(block_size, a->dtype));
        }
    }
    BlockInds v_block_inds = BlockInds::arange_diag(static_cast<std::size_t>(num_sectors));
    auto w_data = make_data(
      dtype::to_real(a->dtype), a_data->device, std::move(w_blocks), a_data->block_inds, true);
    auto v_data = make_data(a->dtype, a_data->device, std::move(v_blocks), v_block_inds);
    return { wrap(w_data), wrap(v_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::eig(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    auto a_data = data_from_tensor(a);
    auto new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    Dtype cdtype = dtype::to_complex(a->dtype);
    std::vector<BlockBackend::BlockPtr> v_blocks;
    std::vector<BlockBackend::BlockPtr> w_blocks;
    auto col0 = a_data->block_inds.column(0);
    int64 n = 0;
    int64 bi = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 num_sectors = py::cast(a->codomain).attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        if (i == bi) {
            auto [vals, vects] =
              block_backend->eig(a_data->blocks[static_cast<std::size_t>(n)], sort);
            v_blocks.push_back(vects);
            w_blocks.push_back(vals);
            ++n;
            bi =
              static_cast<std::size_t>(n) >= col0.size() ? -1 : col0[static_cast<std::size_t>(n)];
        } else {
            int64 block_size = tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i)];
            v_blocks.push_back(block_backend->eye_matrix(block_size, cdtype));
        }
    }
    BlockInds v_block_inds = BlockInds::arange_diag(static_cast<std::size_t>(num_sectors));
    auto w_data = make_data(cdtype, a_data->device, std::move(w_blocks), a_data->block_inds, true);
    auto v_data = make_data(cdtype, a_data->device, std::move(v_blocks), v_block_inds);
    return { wrap(w_data), wrap(v_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::eigvalsh(SymmetricTensorCPtr a,
                            bool new_leg_dual,
                            std::optional<std::string> sort)
{
    auto a_data = data_from_tensor(a);
    auto new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    w_blocks.reserve(a_data->blocks.size());
    for (auto const& block : a_data->blocks) {
        w_blocks.push_back(block_backend->eigvalsh(block, sort));
    }
    auto w_data = make_data(
      dtype::to_real(a->dtype), a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::eigvals(SymmetricTensorCPtr a,
                           bool new_leg_dual,
                           std::optional<std::string> sort)
{
    auto a_data = data_from_tensor(a);
    auto new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    Dtype cdtype = dtype::to_complex(a->dtype);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    w_blocks.reserve(a_data->blocks.size());
    for (auto const& block : a_data->blocks) {
        w_blocks.push_back(block_backend->eigvals(block, sort));
    }
    auto w_data = make_data(cdtype, a_data->device, std::move(w_blocks), a_data->block_inds, true);
    return { wrap(w_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::lq(SymmetricTensorCPtr a, TensorProduct::Ptr new_co_domain)
{
    auto a_data = data_from_tensor(a);
    py::list l_block_inds;
    py::list q_block_inds;
    std::vector<BlockBackend::BlockPtr> l_blocks;
    std::vector<BlockBackend::BlockPtr> q_blocks;
    auto col0 = a_data->block_inds.column(0);
    int64 n = 0;
    int64 bi_cod = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 i_new = 0;
    SectorArray::iter_common_sorted(
      a->codomain->sector_decomposition,
      a->domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_cod_p, std::ptrdiff_t i_dom_p) {
          int64 i_cod = static_cast<int64>(i_cod_p);
          int64 i_dom = static_cast<int64>(i_dom_p);
          q_block_inds.append(py::make_tuple(i_new, i_dom));
          if (bi_cod == i_cod) {
              auto [l, q] =
                block_backend->matrix_lq(a_data->blocks[static_cast<std::size_t>(n)], false);
              l_blocks.push_back(l);
              q_blocks.push_back(q);
              l_block_inds.append(py::make_tuple(i_cod, i_new));
              ++n;
              bi_cod = static_cast<std::size_t>(n) >= col0.size()
                         ? -1
                         : col0[static_cast<std::size_t>(n)];
          } else {
              int64 B_dom = tp_mults(py::cast(a->domain))[static_cast<std::size_t>(i_dom)];
              int64 B_new = new_co_domain->multiplicities[static_cast<std::size_t>(i_new)];
              q_blocks.push_back(b_get(
                block_backend->eye_matrix(B_dom, a->dtype),
                py::make_tuple(py::slice(0, B_new, 1), py::slice(std::nullopt, std::nullopt, 1))));
          }
          ++i_new;
      });
    auto rows_from_list = [](py::list const& lst) -> BlockInds {
        if (lst.size() == 0)
            return zeros_i64(0, 2);
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(lst.size()));
        for (py::handle h : lst) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        return BlockInds::from_rows(rows);
    };
    BlockInds l_bi = rows_from_list(l_block_inds);
    BlockInds q_bi = rows_from_list(q_block_inds);
    return { wrap(make_data(a->dtype, a_data->device, std::move(l_blocks), l_bi, true)),
             wrap(make_data(a->dtype, a_data->device, std::move(q_blocks), q_bi)) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::qr(SymmetricTensorCPtr a, TensorProduct::Ptr new_co_domain)
{
    // --- hints from Python FusionTreeBackend.qr ---
    // running index, indicating we have already processed a_blocks[:n]
    // there is no block for that sector. => r=0, no need to set it.
    // choose basis vectors for q as standard basis vectors (cols/rows of eye)
    // ---
    auto a_data = data_from_tensor(a);
    py::list q_block_inds;
    py::list r_block_inds;
    std::vector<BlockBackend::BlockPtr> q_blocks;
    std::vector<BlockBackend::BlockPtr> r_blocks;
    auto col0 = a_data->block_inds.column(0);
    int64 n = 0;
    int64 bi_cod = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 i_new = 0;
    SectorArray::iter_common_sorted(
      a->codomain->sector_decomposition,
      a->domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_cod_p, std::ptrdiff_t i_dom_p) {
          int64 i_cod = static_cast<int64>(i_cod_p);
          int64 i_dom = static_cast<int64>(i_dom_p);
          q_block_inds.append(py::make_tuple(i_cod, i_new));
          if (bi_cod == i_cod) {
              auto [q, r] =
                block_backend->matrix_qr(a_data->blocks[static_cast<std::size_t>(n)], false);
              q_blocks.push_back(q);
              r_blocks.push_back(r);
              r_block_inds.append(py::make_tuple(i_new, i_dom));
              ++n;
              bi_cod = static_cast<std::size_t>(n) >= col0.size()
                         ? -1
                         : col0[static_cast<std::size_t>(n)];
          } else {
              int64 B_cod = tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i_cod)];
              int64 B_new = new_co_domain->multiplicities[static_cast<std::size_t>(i_new)];
              q_blocks.push_back(b_get(
                block_backend->eye_matrix(B_cod, a->dtype),
                py::make_tuple(py::slice(std::nullopt, std::nullopt, 1), py::slice(0, B_new, 1))));
          }
          ++i_new;
      });
    auto rows_from_list = [](py::list const& lst) -> BlockInds {
        if (lst.size() == 0)
            return zeros_i64(0, 2);
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(lst.size()));
        for (py::handle h : lst) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        return BlockInds::from_rows(rows);
    };
    BlockInds q_bi = rows_from_list(q_block_inds);
    BlockInds r_bi = rows_from_list(r_block_inds);
    return { wrap(make_data(a->dtype, a_data->device, std::move(q_blocks), q_bi)),
             wrap(make_data(a->dtype, a_data->device, std::move(r_blocks), r_bi)) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::svd(SymmetricTensorCPtr a,
                       TensorProduct::Ptr new_co_domain,
                       std::optional<std::string> algorithm)
{
    // --- hints from Python FusionTreeBackend.svd ---
    // there is no block for that sector. => s=0, no need to set it.
    // choose basis vectors for u/vh as standard basis vectors (cols/rows of eye)
    // ---
    auto a_data = data_from_tensor(a);
    py::list u_block_inds;
    py::list s_block_inds;
    py::list vh_block_inds;
    std::vector<BlockBackend::BlockPtr> u_blocks;
    std::vector<BlockBackend::BlockPtr> s_blocks;
    std::vector<BlockBackend::BlockPtr> vh_blocks;
    auto col0 = a_data->block_inds.column(0);
    int64 n = 0;
    int64 bi_cod = col0.empty() ? -1 : col0[static_cast<std::size_t>(n)];
    int64 i_new = 0;
    SectorArray::iter_common_sorted(
      a->codomain->sector_decomposition,
      a->domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_cod_p, std::ptrdiff_t i_dom_p) {
          int64 i_cod = static_cast<int64>(i_cod_p);
          int64 i_dom = static_cast<int64>(i_dom_p);
          u_block_inds.append(py::make_tuple(i_cod, i_new));
          vh_block_inds.append(py::make_tuple(i_new, i_dom));
          if (bi_cod == i_cod) {
              auto [u, s, vh] =
                block_backend->matrix_svd(a_data->blocks[static_cast<std::size_t>(n)], algorithm);
              u_blocks.push_back(u);
              s_blocks.push_back(s);
              vh_blocks.push_back(vh);
              s_block_inds.append(i_new);
              ++n;
              bi_cod = static_cast<std::size_t>(n) >= col0.size()
                         ? -1
                         : col0[static_cast<std::size_t>(n)];
          } else {
              int64 B_cod = tp_mults(py::cast(a->codomain))[static_cast<std::size_t>(i_cod)];
              int64 B_dom = tp_mults(py::cast(a->domain))[static_cast<std::size_t>(i_dom)];
              int64 B_new = new_co_domain->multiplicities[static_cast<std::size_t>(i_new)];
              u_blocks.push_back(b_get(
                block_backend->eye_matrix(B_cod, a->dtype),
                py::make_tuple(py::slice(std::nullopt, std::nullopt, 1), py::slice(0, B_new, 1))));
              vh_blocks.push_back(b_get(
                block_backend->eye_matrix(B_dom, a->dtype),
                py::make_tuple(py::slice(0, B_new, 1), py::slice(std::nullopt, std::nullopt, 1))));
          }
          ++i_new;
      });
    auto mk = [](py::list const& lst) -> BlockInds {
        if (lst.size() == 0)
            return zeros_i64(0, 2);
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(lst.size()));
        for (py::handle h : lst) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        return BlockInds::from_rows(rows);
    };
    py::list s_rows;
    for (py::handle x : s_block_inds)
        s_rows.append(py::make_tuple(x, x));
    return { wrap(make_data(a->dtype, a_data->device, std::move(u_blocks), mk(u_block_inds))),
             wrap(make_data(
               dtype::to_real(a->dtype), a_data->device, std::move(s_blocks), mk(s_rows))),
             wrap(make_data(a->dtype, a_data->device, std::move(vh_blocks), mk(vh_block_inds))) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr, float64, float64>
FusionTreeBackend::truncate_singular_values(DiagonalTensorCPtr S,
                                            std::optional<int64> chi_max,
                                            int64 chi_min,
                                            float64 degeneracy_tol,
                                            float64 trunc_cut,
                                            std::optional<float64> svd_min,
                                            bool minimize_error)
{
    // --- hints from Python FusionTreeBackend.truncate_singular_values ---
    // build a numpy array of the singular values and a numpy array of the qdims
    // have already considered blocks[:i]
    // we have a block for that coupled sector
    // select which to keep
    // build the Mask
    // TODO not sure how to deal with the basis perm here...
    // but ideally the new leg of an SVD has no basis_perm anyway
    // all False. skip this block.
    // OPTIMIZE avoid computing it, if we reset it anyway
    // ---
    // Port of Python FusionTreeBackend.truncate_singular_values — builds a dense
    // S array from block multiplicities/qdims (works when can_be_dropped is False).
    auto S_data = data_from_tensor(S);
    auto domain = py::cast(S->domain);
    auto mults = domain.attr("multiplicities").cast<std::vector<int64>>();
    auto qdims_dom = domain.attr("sector_qdims").cast<std::vector<float64>>();
    int64 num_singular_values = 0;
    for (auto m : mults)
        num_singular_values += m;
    Dtype S_dtype = S->dtype;
    // Allocate via BlockBackend then to_numpy (no direct numpy import in this TU).
    auto S_block = block_backend->zeros({ num_singular_values }, S_dtype);
    py::array S_np = block_backend->to_numpy(S_block).cast<py::array>();
    py::array_t<float64> qdims_arr(num_singular_values);
    py::array qdims = qdims_arr;
    py::list slices;
    int64 stop = 0;
    int64 i = 0;
    auto const& bi = S_data->block_inds;
    int64 S_num_blocks = static_cast<int64>(bi.nrows());
    for (std::size_t j = 0; j < mults.size(); ++j) {
        int64 start = stop;
        stop += mults[j];
        py::slice slc(start, stop, 1);
        slices.append(slc);
        if (i < S_num_blocks && bi(static_cast<std::size_t>(i), 0) == static_cast<int64>(j)) {
            S_np.attr("__setitem__")(
              slc, block_backend->to_numpy(S_data->blocks[static_cast<std::size_t>(i)]));
            ++i;
        }
        qdims.attr("__setitem__")(slc, qdims_dom[j]);
    }
    auto [keep, err, new_norm] = _truncate_singular_values_selection(
      S_np, qdims, chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);

    if (!py::cast(S->leg()).attr("_basis_perm").is_none())
        throw NotImplemented("truncate_singular_values with basis_perm");

    py::list large_leg_block_inds;
    std::vector<BlockBackend::BlockPtr> mask_blocks;
    py::list small_leg_sectors;
    py::list small_leg_multiplicities;
    auto sector_decomp = domain.attr("sector_decomposition");
    for (py::ssize_t j = 0; j < py::len(slices); ++j) {
        py::object slc = slices[j];
        py::array block = keep.attr("__getitem__")(slc).cast<py::array>();
        auto keep_block = block_backend->block_from_numpy(block);
        if (!block_backend->any(keep_block))
            continue;
        large_leg_block_inds.append(j);
        mask_blocks.push_back(keep_block);
        small_leg_sectors.append(sector_decomp.attr("__getitem__")(j));
        small_leg_multiplicities.append(block_backend->sum_all(keep_block).as_int64());
    }
    BlockInds mask_block_inds;
    ElementarySpace::Ptr small_leg;
    bool is_dual = py::cast(S->leg()).attr("is_dual").cast<bool>();
    auto sym = py::cast(S->symmetry).cast<Symmetry::Ptr>();
    if (mask_blocks.empty()) {
        mask_block_inds = zeros_i64(0, 2);
        small_leg = ElementarySpace::from_sector_decomposition(
          sym, sym->empty_sector_array, std::vector<int64>{}, is_dual);
    } else {
        std::vector<int64> small_arange(static_cast<std::size_t>(py::len(small_leg_sectors)));
        std::iota(small_arange.begin(), small_arange.end(), 0);
        std::vector<int64> large_vec;
        large_vec.reserve(static_cast<std::size_t>(py::len(large_leg_block_inds)));
        for (py::handle h : large_leg_block_inds)
            large_vec.push_back(h.cast<int64>());
        mask_block_inds =
          BlockInds::column_stack(std::vector<std::span<const int64>>{ small_arange, large_vec });
        SectorArray sectors = as_sector_array_from_list(py::list(small_leg_sectors));
        std::vector<int64> mults_out;
        for (py::handle h : small_leg_multiplicities)
            mults_out.push_back(h.cast<int64>());
        small_leg =
          ElementarySpace::from_sector_decomposition(sym, std::move(sectors), mults_out, is_dual);
    }
    // Clear basis_perm like Python (new SVD legs should not carry one).
    py::cast(small_leg).attr("_basis_perm") = py::none();
    py::cast(small_leg).attr("_inverse_basis_perm") = py::none();
    auto mask_data =
      make_data(Dtype::Bool, S_data->device, std::move(mask_blocks), mask_block_inds, true);
    return { wrap(mask_data), small_leg, err, new_norm };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, true);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, false);
}

namespace {

/// `space` with its `LegPipe` factors replaced by their flat legs.
///
/// The fusion trees (and thus the block layout) of the data are w.r.t. the flat legs; the coupled
/// sectors are unchanged.
TensorProduct::Ptr
flatten_pipes(TensorProduct::Ptr const& space)
{
    if (!space->has_pipes())
        return space;
    return std::make_shared<TensorProduct>(
      space->flat_legs(), space->symmetry, space->sector_decomposition, space->multiplicities);
}

} // namespace

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::_mask_contract(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx, bool large_leg)
{
    // --- hints from Python FusionTreeBackend._mask_contract ---
    // sector decomposition changes, need to adjust res_block_inds
    // some coupled sectors may no longer be allowed if uncoupled sectors are projected out
    // -> the corresponding shape has a zero entry -> remove later using discard_zero_blocks
    // uncoupled sector not in mask
    // ---
    auto parsed = tensor->_parse_leg_idx(leg_idx);
    bool in_domain = std::get<0>(parsed);
    int64 co_domain_idx = std::get<1>(parsed);
    if (in_domain)
        assert(mask->is_projection != large_leg);
    else
        assert(mask->is_projection == large_leg);
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    TensorProduct::Ptr target_space;
    TensorProduct::Ptr iter_space;
    if (in_domain) {
        codomain = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
        auto spaces = codomain->factors;
        (void)spaces;
        domain = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
        // build target domain with swapped leg — use Python TensorProduct for now
        py::list sp = py::cast(tensor->domain).attr("factors");
        sp[co_domain_idx] = large_leg ? py::cast(mask->small_leg()) : py::cast(mask->large_leg());
        target_space = domain = py::cast<TensorProduct::Ptr>(
          py::type::of<TensorProduct>()(sp, py::arg("symmetry") = py::cast(tensor->symmetry)));
        iter_space = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
    } else {
        domain = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
        py::list sp = py::cast(tensor->codomain).attr("factors");
        sp[co_domain_idx] = large_leg ? py::cast(mask->small_leg()) : py::cast(mask->large_leg());
        target_space = codomain = py::cast<TensorProduct::Ptr>(
          py::type::of<TensorProduct>()(sp, py::arg("symmetry") = py::cast(tensor->symmetry)));
        iter_space = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
    }
    // pipes on the other legs are fine; the data is organized w.r.t. the flat legs
    auto const masked_flat_idcs = iter_space->flat_leg_idcs(co_domain_idx);
    if (masked_flat_idcs.size() != 1)
        throw NotImplemented("_mask_contract does not support masking a LegPipe");
    int64 const flat_idx = masked_flat_idcs.front();
    auto const flat_iter_space = flatten_pipes(iter_space);
    auto const flat_target_space = flatten_pipes(target_space);
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    int64 in_domain_int = in_domain ? 1 : 0;
    py::list coupled;
    auto const& t_bi = t_data->block_inds;
    for (std::size_t i = 0; i < t_bi.nrows(); ++i)
        coupled.append(
          py::cast(tensor->domain).attr("sector_decomposition").attr("__getitem__")(t_bi(i, 1)));
    SectorArray coupled_arr =
      coupled.size() > 0
        ? as_sector_array_from_list(py::list(coupled))
        : py::cast(tensor->symmetry).attr("empty_sector_array").cast<SectorArray>();
    bool same_decomp =
      iter_space->sector_decomposition.size() == target_space->sector_decomposition.size();
    std::vector<BlockBackend::BlockPtr> res_blocks;
    BlockInds res_block_inds = t_bi;
    for (std::size_t i = 0; i < t_bi.nrows(); ++i) {
        auto block = block_backend->zeros(
          { codomain->block_size(coupled_arr[i]), domain->block_size(coupled_arr[i]) },
          t_data->dtype);
        if (!same_decomp) {
            auto shape = block_backend->get_shape(block);
            if (shape[static_cast<std::size_t>(in_domain_int)] > 0) {
                auto where = target_space->sector_decomposition_where(coupled_arr[i]);
                if (where.has_value())
                    res_block_inds(i, static_cast<std::size_t>(in_domain_int)) = *where;
            }
        }
        res_blocks.push_back(block);
    }
    auto const& flat_legs = flat_iter_space->factors;
    for (auto const& fb : flat_iter_space->iter_forest_blocks(coupled_arr)) {
        auto unc = fb.uncoupled;
        auto dom_idx_mask =
          py::cast(mask->domain)
            .attr("sector_decomposition_where")(py::cast(unc[static_cast<std::size_t>(flat_idx)]));
        if (dom_idx_mask.is_none())
            continue;
        auto j = m_data->block_ind_from_domain_sector_ind(dom_idx_mask.cast<int64>());
        if (!j.has_value())
            continue;
        auto block_slice = in_domain
                             ? b_get(t_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                     py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                    slice_from_index_slice(fb.slice)))
                             : b_get(t_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                     py::make_tuple(slice_from_index_slice(fb.slice),
                                                    py::slice(std::nullopt, std::nullopt, 1)));
        std::vector<int64> intermediate_shape;
        intermediate_shape.reserve(unc.size() + 1);
        for (std::size_t li = 0; li < unc.size(); ++li)
            intermediate_shape.push_back(py::cast(flat_legs[li])
                                           .attr("sector_multiplicity")(py::cast(unc[li]))
                                           .cast<int64>());
        std::vector<int64> final_shape;
        if (in_domain) {
            intermediate_shape.insert(intermediate_shape.begin(), -1);
            final_shape = { block_backend->get_shape(block_slice)[0], -1 };
        } else {
            intermediate_shape.push_back(-1);
            final_shape = { -1, block_backend->get_shape(block_slice)[1] };
        }
        block_slice = block_backend->reshape(block_slice, intermediate_shape);
        int64 ax = in_domain_int + flat_idx;
        if (large_leg)
            block_slice = block_backend->apply_mask(
              block_slice, m_data->blocks[static_cast<std::size_t>(*j)], ax);
        else
            block_slice = block_backend->enlarge_leg(
              block_slice, m_data->blocks[static_cast<std::size_t>(*j)], ax);
        block_slice = block_backend->reshape(block_slice, final_shape);
        auto new_slc = flat_target_space->forest_block_slice(
          unc, coupled_arr[static_cast<std::size_t>(fb.coupled_idx)]);
        if (in_domain)
            b_set(res_blocks[static_cast<std::size_t>(fb.coupled_idx)],
                  py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                 slice_from_index_slice(new_slc)),
                  block_slice);
        else
            b_set(res_blocks[static_cast<std::size_t>(fb.coupled_idx)],
                  py::make_tuple(slice_from_index_slice(new_slc),
                                 py::slice(std::nullopt, std::nullopt, 1)),
                  block_slice);
    }
    auto res =
      make_data(tensor->dtype, t_data->device, std::move(res_blocks), res_block_inds, true);
    res->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);
    return { wrap(res), codomain, domain };
}

namespace {

void
tree_block_iter(
  TensorCPtr a,
  std::function<void(FusionTree const&, FusionTree const&, BlockBackend::BlockPtr const&)> const&
    fn)
{
    auto a_data = FusionTreeBackend::data_from_tensor(a);
    auto codomain = a->codomain;
    auto domain = a->domain;
    auto sym = a->symmetry;
    auto cod_are_dual = flat_are_dual(codomain);
    auto dom_are_dual = flat_are_dual(domain);
    auto const& bi = a_data->block_inds;
    for (std::size_t row = 0; row < bi.nrows(); ++row) {
        int64 bi_cod = bi(row, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        auto const& block = a_data->blocks[static_cast<std::size_t>(row)];
        int64 i1_forest = 0;
        int64 i2_forest = 0;
        for (auto const& b_item : domain->iter_uncoupled(false)) {
            int64 tree_block_width = prod_int(b_item.multiplicities);
            int64 forest_block_width = 0;
            for (auto const& a_item : codomain->iter_uncoupled(false)) {
                int64 tree_block_height = prod_int(a_item.multiplicities);
                int64 i1 = i1_forest;
                int64 i2 = i2_forest;
                fusion_trees alpha_iter(sym, a_item.uncoupled, coupled, cod_are_dual);
                fusion_trees beta_iter(sym, b_item.uncoupled, coupled, dom_are_dual);
                for (auto const& alpha_tree : alpha_iter.all_trees()) {
                    i2 = i2_forest;
                    for (auto const& beta_tree : beta_iter.all_trees()) {
                        py::slice idx1(i1, i1 + tree_block_height, 1);
                        py::slice idx2(i2, i2 + tree_block_width, 1);
                        fn(alpha_tree, beta_tree, b_get(block, py::make_tuple(idx1, idx2)));
                        i2 += tree_block_width;
                    }
                    i1 += tree_block_height;
                }
                forest_block_width = std::max(forest_block_width, i2 - i2_forest);
                i1_forest += (i1 - i1_forest);
            }
            i1_forest = 0;
            i2_forest += forest_block_width;
        }
    }
}

std::tuple<bool, complex128>
partial_trace_helper(FusionTree const& tree, std::vector<int64> const& idcs)
{
    // Native port of lasyten.backends.fusion_tree_backend._partial_trace_helper
    Symmetry::Ptr const& sym = tree.symmetry;
    complex128 b_symbols = 1.0;
    for (int64 idx : idcs) {
        auto const i = static_cast<std::size_t>(idx);
        if (tree.uncoupled[i] != sym->dual_sector(tree.uncoupled[i + 1])) {
            return { false, 0.0 };
        }
        Sector left_sec;
        if (idx == 0) {
            left_sec = sym->trivial_sector;
        } else if (idx == 1) {
            left_sec = tree.uncoupled[0];
        } else {
            left_sec = tree.inner_sectors[static_cast<std::size_t>(idx - 2)];
        }
        Sector center_sec =
          (idx == 0) ? tree.uncoupled[0] : tree.inner_sectors[static_cast<std::size_t>(idx - 1)];
        Sector right_sec = (static_cast<std::size_t>(idx) < tree.num_inner_edges)
                             ? tree.inner_sectors[i]
                             : tree.coupled;
        if (left_sec != right_sec) {
            return { false, 0.0 };
        }
        if (idx == 0) {
            // Match Python ``np.all(tree.multiplicities[:2] == [0, 0])``.
            // For a 2-leg tree, multiplicities has length 1; numpy broadcasts so only
            // multiplicities[0] must be 0. Requiring size >= 2 incorrectly rejects these.
            for (std::size_t k = 0; k < std::min<std::size_t>(2, tree.multiplicities.size());
                 ++k) {
                if (tree.multiplicities[k] != 0)
                    return { false, 0.0 };
            }
        }
        int64 mu = (idx == 0) ? 0 : tree.multiplicities[static_cast<std::size_t>(idx - 1)];
        int64 nu = tree.multiplicities[i];
        FusionSymbol B = sym->b_symbol(left_sec, tree.uncoupled[i], center_sec);
        b_symbols *=
          std::conj(B.get_complex(static_cast<std::size_t>(mu), static_cast<std::size_t>(nu)));
        if (tree.are_dual[i]) {
            b_symbols *= static_cast<complex128>(sym->frobenius_schur(tree.uncoupled[i]));
        }
    }
    return { true, b_symbols };
}

} // namespace

TensorBackend::DataPtr
FusionTreeBackend::outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    // --- hints from Python FusionTreeBackend.outer ---
    // idea: get the fusion trees in the combined (co)domain by inserting an identity
    // = summing over all fusion products of the coupled sectors of tensors a and b
    // OPTIMIZE new_codomain and new_domain are already computed in tensors.py -> reuse here
    // axes of new_tree_block after outer: (a.codomain, a.domain, b.codomain, b.domain)
    // ---
    auto a_data = data_from_tensor(a);
    auto new_codomain =
      TensorProduct::from_partial_products({ py::cast(a->codomain).cast<TensorProduct::Ptr>(),
                                             py::cast(b->codomain).cast<TensorProduct::Ptr>() });
    auto new_domain =
      TensorProduct::from_partial_products({ py::cast(a->domain).cast<TensorProduct::Ptr>(),
                                             py::cast(b->domain).cast<TensorProduct::Ptr>() });
    Dtype dtype = dtype::common({ a->dtype, b->dtype });
    auto new_data = unwrap(zero_data(new_codomain, new_domain, dtype, a_data->device, true));
    tree_block_iter(
      a,
      [&](FusionTree const& a_codom_tree,
          FusionTree const& a_dom_tree,
          BlockBackend::BlockPtr const& a_tree_block) {
          tree_block_iter(
            b,
            [&](FusionTree const& b_codom_tree,
                FusionTree const& b_dom_tree,
                BlockBackend::BlockPtr const& b_tree_block) {
                auto new_tree_block = block_backend->outer(a_tree_block, b_tree_block);
                new_tree_block = block_backend->permute_axes(new_tree_block, { 0, 2, 1, 3 });
                new_tree_block =
                  block_backend->combine_legs(new_tree_block, { { 0, 1 }, { 2, 3 } });
                auto new_codom_trees = a_codom_tree.outer(b_codom_tree);
                auto new_dom_trees = a_dom_tree.outer(b_dom_tree);
                for (auto const& [new_dom_tree, dom_amp] : new_dom_trees) {
                    auto dom_slc = new_domain->tree_block_slice(new_dom_tree);
                    auto block_idx =
                      new_data->block_ind_from_coupled(new_dom_tree.coupled, new_domain);
                    if (!block_idx.has_value())
                        continue;
                    for (auto const& [new_codom_tree, codom_amp] : new_codom_trees) {
                        if (new_codom_tree.coupled != new_dom_tree.coupled)
                            continue;
                        auto codom_slc = new_codomain->tree_block_slice(new_codom_tree);
                        auto factor =
                          block_backend->as_scalar(std::conj(codom_amp) * dom_amp, dtype);
                        auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                                         py::make_tuple(slice_from_index_slice(codom_slc),
                                                        slice_from_index_slice(dom_slc)));
                        b_set(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                              py::make_tuple(slice_from_index_slice(codom_slc),
                                             slice_from_index_slice(dom_slc)),
                              (*cur) + (*block_backend->mul(factor, new_tree_block)));
                    }
                }
            });
      });
    new_data->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);
    return wrap(new_data);
}

TensorBackend::DataPtr
FusionTreeBackend::partial_compose(SymmetricTensorCPtr a,
                                   SymmetricTensorCPtr b,
                                   int64 a_first_leg,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    // --- hints from Python FusionTreeBackend.partial_compose ---
    // OPTIMIZE ?
    // space used to iterate over dummy fusion trees containing the
    // relevant coupled sectors for the legs that are contracted
    // we do not want to recompute the transformations for the fusion trees, so cache them here
    // the new space (of b) only has sectors incompatible with the coupled sector of a
    // the corresponding block in b is trivial, but we only iterate over nontrivial blocks
    // (eff_space) since this is not the original (co)domain, the slices are different parts of the
    // multiplicities can be reused anyway reuse the tree transformations using that they only
    // depend on the vertex sectors the Ys in the domain of tensor a fit the Xs in the codomain of
    // tensor b
    // TODO check if the complex conjugation here is correct or if we need to do it as
    // currently done in the codomain; check this when we have a symmetry with complex
    // F symbols and verify that the incorrect way actually has test cases failing
    // ---
    Dtype dtype = dtype::common({ a->dtype, b->dtype });
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    if (a_data->blocks.empty() || b_data->blocks.empty())
        return wrap(make_data(dtype, a_data->device, {}, zeros_i64(0, 2), true));

    SectorArray eff_sectors =
      SectorArray::empty(py::cast(b->symmetry).cast<Symmetry::Ptr>()->sector_ind_len);
    auto const& b_bi = b_data->block_inds;
    for (std::size_t r = 0; r < b_bi.nrows(); ++r)
        eff_sectors.push_back(py::cast(b->domain)
                                .attr("sector_decomposition")
                                .attr("__getitem__")(b_bi(r, 1))
                                .cast<Sector>());
    auto eff_space = ElementarySpace::from_defining_sectors(
      py::cast(b->symmetry).cast<Symmetry::Ptr>(), eff_sectors);

    bool in_domain;
    int64 num_contr_legs;
    int64 leg_idx;
    TensorProduct::Ptr old_space;
    TensorProduct::Ptr new_space;
    TensorProduct::Ptr iter_space;
    int64 a_num_cod = a->num_codomain_legs();
    int64 a_num_legs = a->num_legs;
    if (a_first_leg < a_num_cod) {
        in_domain = false;
        num_contr_legs = b->num_domain_legs();
        leg_idx = a_first_leg;
        old_space = py::cast(a->codomain).cast<TensorProduct::Ptr>();
        new_space = new_codomain;
        py::list factors;
        auto cod = py::cast(a->codomain);
        for (int64 i = 0; i < a_first_leg; ++i)
            factors.append(cod.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = a_first_leg + num_contr_legs; i < a_num_cod; ++i)
            factors.append(cod.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<Leg::Ptr>>(),
                                                     py::cast(a->symmetry).cast<Symmetry::Ptr>());
    } else {
        in_domain = true;
        num_contr_legs = b->num_codomain_legs();
        leg_idx = a_num_legs - a_first_leg - num_contr_legs;
        old_space = py::cast(a->domain).cast<TensorProduct::Ptr>();
        new_space = new_domain;
        py::list factors;
        auto dom = py::cast(a->domain);
        int64 n_dom = dom.attr("num_factors").cast<int64>();
        for (int64 i = 0; i < leg_idx; ++i)
            factors.append(dom.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = leg_idx + num_contr_legs; i < n_dom; ++i)
            factors.append(dom.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<Leg::Ptr>>(),
                                                     py::cast(a->symmetry).cast<Symmetry::Ptr>());
    }

    // `leg_idx` counts the factors of the (co)domain, but the fusion trees run over the flat legs
    // (pipes are transparent). Index of the contracted leg among the flat legs of `iter_space`:
    int64 flat_leg_idx = 0;
    for (int64 i = 0; i < leg_idx; ++i)
        flat_leg_idx += iter_space->factors[static_cast<std::size_t>(i)].num_flat_legs();
    py::list new_block_inds_rows;
    std::vector<BlockBackend::BlockPtr> new_blocks;
    // Python builds a cache keyed by ``vertex_labels``, but the key expression is a generator
    // object so lookups never hit and insert_at is always recomputed. Caching by vertex labels
    // alone is incorrect when distinct dummy trees share those labels — always recompute.

    auto const& a_bi = a_data->block_inds;
    for (std::size_t block_ind = 0; block_ind < a_bi.nrows(); ++block_ind) {
        int64 i = a_bi(block_ind, 0);
        Sector coupled =
          py::cast(a->codomain).attr("sector_decomposition").attr("__getitem__")(i).cast<Sector>();
        auto new_i = new_codomain->sector_decomposition_where(coupled);
        auto new_j = new_domain->sector_decomposition_where(coupled);
        if (!new_i.has_value() || !new_j.has_value())
            continue;
        auto new_block = block_backend->zeros(
          { new_codomain->multiplicities[*new_i], new_domain->multiplicities[*new_j] },
          dtype,
          a->device);
        new_block_inds_rows.append(py::make_tuple(*new_i, *new_j));

        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& tb : iter_space->iter_tree_blocks(coupled_arr)) {
            Sector b_coupled = tb.tree.uncoupled[static_cast<std::size_t>(flat_leg_idx)];
            auto b_block_ind = b_data->block_ind_from_coupled(
              b_coupled, py::cast(b->domain).cast<TensorProduct::Ptr>());
            if (!b_block_ind.has_value())
                throw std::runtime_error("partial_compose: missing b block");
            std::vector<int64> dm = tb.multiplicities;
            int64 dm0 = prod_int(
              std::vector<int64>(dm.begin(), dm.begin() + static_cast<std::size_t>(flat_leg_idx)));
            int64 dm2 = prod_int(std::vector<int64>(
              dm.begin() + static_cast<std::size_t>(flat_leg_idx) + 1, dm.end()));
            std::vector<int64> dummy_mults = { dm0, 1, dm2 };

            auto b_cod = py::cast(b->codomain).cast<TensorProduct::Ptr>();
            auto b_dom = py::cast(b->domain).cast<TensorProduct::Ptr>();
            SectorArray b_coupled_arr = SectorArray::repeat(b_coupled, 1);
            for (auto const& xb : b_cod->iter_tree_blocks(b_coupled_arr)) {
                FusionTreeLinearCombination X_b_trafo = tb.tree.insert_at(flat_leg_idx, xb.tree);
                for (auto const& yb : b_dom->iter_tree_blocks(b_coupled_arr)) {
                    FusionTreeLinearCombination Y_b_trafo =
                      tb.tree.insert_at(flat_leg_idx, yb.tree);
                    auto b_tree_block =
                      b_get(b_data->blocks[static_cast<std::size_t>(*b_block_ind)],
                            py::make_tuple(slice_from_index_slice(xb.slice),
                                           slice_from_index_slice(yb.slice)));
                    auto b_shape = block_backend->get_shape(b_tree_block);

                    if (in_domain) {
                        // Domain of a matches codomain of b → old trees from X_b, new from Y_b.
                        for (auto const& [old_tree, amp_old_tree] : X_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest =
                              b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                    py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                   slice_from_index_slice(old_slc)));
                            a_forest = block_backend->reshape(
                              a_forest, { -1, dummy_mults[0], b_shape[0], dummy_mults[2] });
                            for (auto const& [new_tree, amp_new_tree] : Y_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution =
                                  block_backend->tdot(a_forest, b_tree_block, { 2 }, { 0 });
                                contribution =
                                  block_backend->permute_axes(contribution, { 0, 1, 3, 2 });
                                contribution = block_backend->reshape(
                                  contribution,
                                  { -1, dummy_mults[0] * b_shape[1] * dummy_mults[2] });
                                auto factor = block_backend->as_scalar(
                                  amp_old_tree * std::conj(amp_new_tree), dtype);
                                auto cur =
                                  b_get(new_block,
                                        py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                       slice_from_index_slice(new_slc)));
                                b_set(new_block,
                                      py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                     slice_from_index_slice(new_slc)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    } else {
                        for (auto const& [old_tree, amp_old_tree] : Y_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest =
                              b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                    py::make_tuple(slice_from_index_slice(old_slc),
                                                   py::slice(std::nullopt, std::nullopt, 1)));
                            a_forest = block_backend->reshape(
                              a_forest, { dummy_mults[0], b_shape[1], dummy_mults[2], -1 });
                            for (auto const& [new_tree, amp_new_tree] : X_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution =
                                  block_backend->tdot(a_forest, b_tree_block, { 1 }, { 1 });
                                contribution =
                                  block_backend->permute_axes(contribution, { 0, 3, 1, 2 });
                                contribution = block_backend->reshape(
                                  contribution,
                                  { dummy_mults[0] * b_shape[0] * dummy_mults[2], -1 });
                                auto factor = block_backend->as_scalar(
                                  std::conj(amp_old_tree) * amp_new_tree, dtype);
                                auto cur =
                                  b_get(new_block,
                                        py::make_tuple(slice_from_index_slice(new_slc),
                                                       py::slice(std::nullopt, std::nullopt, 1)));
                                b_set(new_block,
                                      py::make_tuple(slice_from_index_slice(new_slc),
                                                     py::slice(std::nullopt, std::nullopt, 1)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    }
                }
            }
        }
        new_blocks.push_back(new_block);
    }

    BlockInds block_inds;
    if (new_block_inds_rows.size() > 0) {
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(new_block_inds_rows.size()));
        for (py::handle h : new_block_inds_rows) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        block_inds = BlockInds::from_rows(rows);
    } else {
        block_inds = zeros_i64(0, 2);
    }
    auto res = make_data(dtype, a_data->device, std::move(new_blocks), block_inds, true);
    res->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);
    return wrap(res);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::partial_trace(SymmetricTensorCPtr tensor,
                                 std::vector<std::pair<int64, int64>> pairs,
                                 std::vector<std::optional<int64>> levels)
{
    // --- hints from Python FusionTreeBackend.partial_trace ---
    // step 1: permute legs such that the paired legs are next to each other
    // it does not matter which leg is moved; the partial trace implies that
    // the fusion channel of each pair is trivial. It is however crucial that
    // we keep the ordering within each pair.
    // OPTIMIZE decide if we want to optimize: There is in principle no need to
    // braid when tracing out two pairs of the form [1, 4] and [2, 3]
    // permute legs such that the ones with the smaller index do not move
    // leg at pair[1] is bent up
    // Build new codomain and domain
    // TODO (JU) this is duplicate code with tensors._permute_legs, but we cant import that
    // here (cyclic)
    // (co)domain has the same factor as before, only permuted -> can re-use sectors!
    // note: we bend to the right *by definition*
    // only consider coupled sectors in data that are consistent with co(domain) after tracing
    // OPTIMIZE use sorted properties to speed this up.
    // block indices
    // step 2: compute new entries: iterate over all trees in the untraced
    // spaces and construct the consistent trees in the traced spaces
    // need to get updated indices after permuting the legs
    // ---
    std::sort(pairs.begin(), pairs.end(), [](auto const& a, auto const& b) {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    });
    for (auto& p : pairs)
        if (p.first > p.second)
            std::swap(p.first, p.second);

    int64 N = tensor->num_legs;
    int64 num_codom_legs = tensor->num_codomain_legs();
    std::vector<int64> idcs1, idcs2;
    for (auto const& [i1, i2] : pairs) {
        idcs1.push_back(i1);
        idcs2.push_back(i2);
    }
    std::set<int64> traced(idcs1.begin(), idcs1.end());
    traced.insert(idcs2.begin(), idcs2.end());
    std::vector<int64> remaining;
    for (int64 n = 0; n < N; ++n)
        if (!traced.count(n))
            remaining.push_back(n);

    auto sym = py::cast(tensor->symmetry).cast<Symmetry::Ptr>();
    auto codomain_tp = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
    auto domain_tp = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
    py::list new_cod_factors, new_dom_factors;
    for (int64 n = 0; n < codomain_tp->num_factors; ++n) {
        if (traced.count(n))
            continue;
        new_cod_factors.append(py::cast(tensor->codomain).attr("__getitem__")(n));
    }
    for (int64 n = 0; n < domain_tp->num_factors; ++n) {
        if (traced.count(N - 1 - n))
            continue;
        new_dom_factors.append(py::cast(tensor->domain).attr("__getitem__")(n));
    }
    auto new_codomain =
      std::make_shared<TensorProduct>(new_cod_factors.cast<std::vector<Leg::Ptr>>(), sym);
    auto new_domain =
      std::make_shared<TensorProduct>(new_dom_factors.cast<std::vector<Leg::Ptr>>(), sym);

    py::list insert_idcs;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        int64 pos = 0;
        for (int64 r : remaining)
            if (r < pairs[i].first)
                ++pos;
        insert_idcs.append(pos + 2 * static_cast<int64>(i));
    }
    std::vector<int64> idcs = remaining;
    int64 codom_count = num_codom_legs;
    for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
        int64 idx = insert_idcs[pi].cast<int64>();
        idcs.insert(idcs.begin() + idx, pairs[pi].first);
        idcs.insert(idcs.begin() + idx + 1, pairs[pi].second);
        if (pairs[pi].first < num_codom_legs && pairs[pi].second >= num_codom_legs)
            ++codom_count;
    }
    int64 num_dom_legs = static_cast<int64>(idcs.size()) - codom_count;

    for (auto const& pair : pairs) {
        if (pair.first >= static_cast<int64>(levels.size()) ||
            pair.second >= static_cast<int64>(levels.size()))
            continue;
        auto const& lp0 = levels[static_cast<std::size_t>(pair.first)];
        auto const& lp1 = levels[static_cast<std::size_t>(pair.second)];
        if (!lp0.has_value() || !lp1.has_value())
            continue;
        for (int64 i = 0; i < N; ++i) {
            if (i == pair.first || i == pair.second)
                continue;
            if (i >= static_cast<int64>(levels.size()))
                continue;
            auto const& li = levels[static_cast<std::size_t>(i)];
            if (!li.has_value())
                continue;
            bool l1 = *li < *lp0;
            bool l2 = *li < *lp1;
            if (l1 != l2)
                throw std::invalid_argument("Inconsistent levels for partial_trace");
        }
    }

    std::vector<int64> codomain_idcs(idcs.begin(), idcs.begin() + codom_count);
    std::vector<int64> domain_idcs(idcs.begin() + codom_count, idcs.end());
    std::reverse(domain_idcs.begin(), domain_idcs.end());
    bool mixes = std::any_of(codomain_idcs.begin(),
                             codomain_idcs.end(),
                             [&](int64 i) { return i >= num_codom_legs; }) ||
                 std::any_of(domain_idcs.begin(), domain_idcs.end(), [&](int64 i) {
                     return i < num_codom_legs;
                 });

    TensorProduct::Ptr codom, dom;
    if (mixes) {
        py::list cf, df;
        for (int64 i : codomain_idcs)
            cf.append(py::cast(tensor).attr("_as_codomain_leg")(i));
        for (int64 i : domain_idcs)
            df.append(py::cast(tensor).attr("_as_domain_leg")(i));
        codom = std::make_shared<TensorProduct>(cf.cast<std::vector<Leg::Ptr>>(), sym);
        dom = std::make_shared<TensorProduct>(df.cast<std::vector<Leg::Ptr>>(), sym);
    } else {
        codom = codomain_tp->permuted(codomain_idcs);
        std::vector<int64> dom_perm;
        for (int64 i : domain_idcs)
            dom_perm.push_back(N - 1 - i);
        dom = domain_tp->permuted(dom_perm);
    }

    std::vector<std::optional<int64>> level_vec = levels;
    if (level_vec.size() < static_cast<std::size_t>(N))
        level_vec.resize(static_cast<std::size_t>(N), std::nullopt);
    std::vector<std::optional<bool>> bend(static_cast<std::size_t>(N), true);
    auto data_ptr =
      permute_legs(tensor, codomain_idcs, domain_idcs, codom, dom, mixes, level_vec, bend);
    auto data = unwrap(data_ptr);

    py::list coupled_sectors;
    auto const& d_bi = data->block_inds;
    for (std::size_t r = 0; r < d_bi.nrows(); ++r) {
        Sector sector = dom->sector_decomposition[static_cast<std::size_t>(d_bi(r, 1))];
        if (!new_domain->sector_decomposition_where(sector).has_value())
            continue;
        if (!new_codomain->sector_decomposition_where(sector).has_value())
            continue;
        coupled_sectors.append(py::cast(sector));
    }
    auto new_data = unwrap(zero_data(new_codomain, new_domain, data->dtype, data->device, true));
    SectorArray coupled_arr = coupled_sectors.size() > 0
                                ? as_sector_array_from_list(coupled_sectors)
                                : sym->empty_sector_array;

    std::vector<std::optional<int64>> old_inds, new_inds;
    for (py::handle h : coupled_sectors) {
        Sector c = h.cast<Sector>();
        old_inds.push_back(data->block_ind_from_coupled(c, dom));
        new_inds.push_back(new_data->block_ind_from_coupled(c, new_domain));
    }

    std::vector<int64> codom_unc_idcs, codom_inner_idcs, codom_multi_idcs, codom_tree_idcs;
    for (int64 i = 0; i < codom_count; ++i) {
        int64 const idx = idcs[static_cast<std::size_t>(i)];
        if (std::find(remaining.begin(), remaining.end(), idx) != remaining.end())
            codom_unc_idcs.push_back(i);
        if (std::find(idcs1.begin(), idcs1.end(), idx) != idcs1.end())
            codom_tree_idcs.push_back(i);
    }
    for (std::size_t k = 2; k < codom_unc_idcs.size(); ++k)
        codom_inner_idcs.push_back(codom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < codom_unc_idcs.size(); ++k)
        codom_multi_idcs.push_back(codom_unc_idcs[k] - 1);

    // Match Python: enumerate idcs[num_codom:] (not the reversed domain_idcs).
    std::vector<int64> dom_unc_idcs, dom_tree_idcs;
    for (int64 i = 0; i < num_dom_legs; ++i) {
        int64 const idx = idcs[static_cast<std::size_t>(codom_count + i)];
        if (std::find(remaining.begin(), remaining.end(), idx) != remaining.end())
            dom_unc_idcs.push_back(num_dom_legs - 1 - i);
        if (std::find(idcs2.begin(), idcs2.end(), idx) != idcs2.end())
            dom_tree_idcs.push_back(num_dom_legs - 1 - i);
    }
    std::reverse(dom_unc_idcs.begin(), dom_unc_idcs.end());
    std::reverse(dom_tree_idcs.begin(), dom_tree_idcs.end());
    std::vector<int64> dom_inner_idcs, dom_multi_idcs;
    for (std::size_t k = 2; k < dom_unc_idcs.size(); ++k)
        dom_inner_idcs.push_back(dom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < dom_unc_idcs.size(); ++k)
        dom_multi_idcs.push_back(dom_unc_idcs[k] - 1);

    // Python: tr_idcs = idcs[:num_codom] + idcs[num_codom:][::-1] == codomain_idcs + domain_idcs
    std::vector<int64> tr_idcs = codomain_idcs;
    tr_idcs.insert(tr_idcs.end(), domain_idcs.begin(), domain_idcs.end());
    std::vector<int64> tr_idcs1, tr_idcs2, remain_idcs;
    for (std::size_t i = 0; i < tr_idcs.size(); ++i) {
        if (std::find(idcs1.begin(), idcs1.end(), tr_idcs[i]) != idcs1.end())
            tr_idcs1.push_back(static_cast<int64>(i));
        if (std::find(idcs2.begin(), idcs2.end(), tr_idcs[i]) != idcs2.end())
            tr_idcs2.push_back(static_cast<int64>(i));
        if (std::find(remaining.begin(), remaining.end(), tr_idcs[i]) != remaining.end())
            remain_idcs.push_back(static_cast<int64>(i));
    }

    for (auto const& cod_tb : codom->iter_tree_blocks(coupled_arr)) {
        auto [on_diag, factor_codom] = partial_trace_helper(cod_tb.tree, codom_tree_idcs);
        if (!on_diag)
            continue;
        std::vector<std::uint8_t> ad(cod_tb.tree.are_dual.begin(), cod_tb.tree.are_dual.end());
        std::vector<std::uint8_t> sel_ad;
        SectorArray sel_unc = SectorArray::empty(cod_tb.tree.uncoupled.sector_ind_len());
        SectorArray sel_inner = SectorArray::empty(cod_tb.tree.inner_sectors.sector_ind_len());
        std::vector<int64> sel_mult;
        for (int64 ui : codom_unc_idcs) {
            sel_ad.push_back(cod_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
            sel_unc.push_back(cod_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
        }
        for (int64 ii : codom_inner_idcs)
            sel_inner.push_back(cod_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
        for (int64 mi : codom_multi_idcs)
            sel_mult.push_back(cod_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
        FusionTree new_codom_tree(sym, sel_unc, cod_tb.tree.coupled, sel_ad, sel_inner, sel_mult);
        auto new_codom_slc = new_codomain->tree_block_slice(new_codom_tree);
        auto old_ind = old_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        auto new_ind = new_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        if (!old_ind.has_value() || !new_ind.has_value())
            continue;

        for (auto const& dom_tb :
             dom->iter_tree_blocks(SectorArray::repeat(cod_tb.tree.coupled, 1))) {
            auto [on_diag2, factor_dom] = partial_trace_helper(dom_tb.tree, dom_tree_idcs);
            if (!on_diag2)
                continue;
            std::vector<int64> tmp_shape = cod_tb.multiplicities;
            tmp_shape.insert(
              tmp_shape.end(), dom_tb.multiplicities.begin(), dom_tb.multiplicities.end());
            std::vector<std::uint8_t> dad;
            SectorArray dunc = SectorArray::empty(dom_tb.tree.uncoupled.sector_ind_len());
            SectorArray dinn = SectorArray::empty(dom_tb.tree.inner_sectors.sector_ind_len());
            std::vector<int64> dmult;
            for (int64 ui : dom_unc_idcs) {
                dad.push_back(dom_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
                dunc.push_back(dom_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
            }
            for (int64 ii : dom_inner_idcs)
                dinn.push_back(dom_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
            for (int64 mi : dom_multi_idcs)
                dmult.push_back(dom_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
            FusionTree new_dom_tree(sym, dunc, dom_tb.tree.coupled, dad, dinn, dmult);
            auto new_dom_slc = new_domain->tree_block_slice(new_dom_tree);

            auto old_block = b_get(data->blocks[static_cast<std::size_t>(*old_ind)],
                                   py::make_tuple(slice_from_index_slice(cod_tb.slice),
                                                  slice_from_index_slice(dom_tb.slice)));
            old_block = block_backend->reshape(old_block, tmp_shape);
            auto contribution =
              block_backend->trace_partial(old_block, tr_idcs1, tr_idcs2, remain_idcs);
            contribution = block_backend->reshape(
              contribution,
              { new_codom_slc.stop - new_codom_slc.start, new_dom_slc.stop - new_dom_slc.start });
            auto factor =
              block_backend->as_scalar(factor_codom * std::conj(factor_dom), data->dtype);
            contribution = block_backend->mul(factor, contribution);
            auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                             py::make_tuple(slice_from_index_slice(new_codom_slc),
                                            slice_from_index_slice(new_dom_slc)));
            b_set(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                  py::make_tuple(slice_from_index_slice(new_codom_slc),
                                 slice_from_index_slice(new_dom_slc)),
                  (*cur) + (*contribution));
        }
    }
    new_data->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);

    if (remaining.empty()) {
        Dtype dt = tensor->dtype;
        // Dummy (1, 2) block_inds: FusionTreeData requires shape (N, 2); pybind
        // unwraps and returns block_backend.item for the full-trace case.
        if (new_data->blocks.empty()) {
            auto s = block_backend->as_scalar(dtype::zero_scalar(dt), dt);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 2), true)),
                     nullptr,
                     nullptr };
        }
        if (new_data->blocks.size() == 1) {
            auto s = block_backend->item(new_data->blocks[0]);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 2), true)),
                     nullptr,
                     nullptr };
        }
        throw std::runtime_error("partial_trace: multiple blocks for scalar result");
    }
    return { wrap(new_data), new_codomain, new_domain };
}

TensorBackend::DataPtr
FusionTreeBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype dtype,
  std::string device)
{
    // --- hints from Python FusionTreeBackend.from_tree_pairs ---
    // [m1,...,mJ,nK,...,n1] -> [m1,...,mJ,n1,...,nK]
    // [m1,...,mJ,n1,...,nK] -> [M, N]
    // check if we covered all keys in the dict
    // OPTIMIZE if the code works, we could remove this check
    // ---
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::set<std::pair<FusionTree, FusionTree>> pairs_done;
    SectorArray::iter_common_sorted(
      codomain->sector_decomposition,
      domain->sector_decomposition,
      true,
      true,
      [&](std::ptrdiff_t i_p, std::ptrdiff_t j_p) {
          int64 i = static_cast<int64>(i_p);
          int64 j = static_cast<int64>(j_p);
          Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
          auto block = block_backend->zeros(
            { codomain->multiplicities[i], domain->multiplicities[j] }, dtype, device);
          bool is_zero_block = true;
          SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
          for (auto const& xb : codomain->iter_tree_blocks(coupled_arr)) {
              for (auto const& yb : domain->iter_tree_blocks(coupled_arr)) {
                  std::pair<FusionTree, FusionTree> pr{ xb.tree, yb.tree };
                  auto it = trees.find(pr);
                  if (it == trees.end())
                      continue;
                  auto tree_block = it->second;
                  std::vector<int64> expect_shape = xb.multiplicities;
                  for (auto it2 = yb.multiplicities.rbegin(); it2 != yb.multiplicities.rend();
                       ++it2)
                      expect_shape.push_back(*it2);
                  assert(block_backend->get_shape(tree_block) == expect_shape);
                  std::vector<int64> perm;
                  for (int64 p = 0; p < J; ++p)
                      perm.push_back(p);
                  for (int64 p = J + K - 1; p >= J; --p)
                      perm.push_back(p);
                  tree_block = block_backend->permute_axes(tree_block, perm);
                  tree_block = block_backend->reshape(
                    tree_block, { prod_int(xb.multiplicities), prod_int(yb.multiplicities) });
                  b_set(block,
                        py::make_tuple(slice_from_index_slice(xb.slice),
                                       slice_from_index_slice(yb.slice)),
                        tree_block);
                  is_zero_block = false;
                  pairs_done.insert(pr);
              }
          }
          if (is_zero_block)
              return;
          block_inds_rows.append(py::make_tuple(i, j));
          blocks.push_back(block);
      });
    for (auto const& kv : trees)
        if (!pairs_done.count(kv.first))
            throw std::runtime_error("from_tree_pairs: uncovered tree pair");
    BlockInds block_inds;
    if (block_inds_rows.size() > 0) {
        std::vector<std::vector<int64>> rows;
        rows.reserve(static_cast<std::size_t>(block_inds_rows.size()));
        for (py::handle h : block_inds_rows) {
            auto t = h.cast<py::tuple>();
            rows.push_back({ t[0].cast<int64>(), t[1].cast<int64>() });
        }
        block_inds = BlockInds::from_rows(rows);
    } else {
        block_inds = zeros_i64(0, 2);
    }
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, true));
}

BlockBackend::Scalar
FusionTreeBackend::get_element(SymmetricTensorCPtr a, std::vector<int64> idcs)
{
    // --- hints from Python FusionTreeBackend.get_element ---
    // reverse domain idcs -> work in the non-conventional leg order [i1,...,iJ,j1,...,jK]
    // build the correct forest block to get the element
    // reshape to [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
    // ---
    warn("Accessing individual entries in the FusionTreeBackend is comparably expensive.");
    py::list flat_idcs;
    auto legs = a->legs();
    for (std::size_t li = 0; li < legs.size(); ++li) {
        std::vector<int64> dims;
        for (auto const& fl : legs[li].flat_legs())
            dims.push_back(static_cast<int64>(fl.dim()));
        auto strides = make_stride(dims, /*cstyle=*/true);
        int64 rem = idcs[li];
        std::vector<int64> unr(dims.size());
        for (std::size_t a = 0; a < dims.size(); ++a) {
            unr[a] = rem / strides[a];
            rem %= strides[a];
        }
        for (int64 u : unr)
            flat_idcs.append(u);
    }
    int64 num_cod_legs = a->num_codomain_flat_legs();
    int64 num_legs = a->num_flat_legs();
    py::list a_legs;
    for (auto const& l : a->codomain->flat_legs())
        a_legs.append(py::cast(l));
    for (auto const& l : a->domain->flat_legs())
        a_legs.append(py::cast(l));
    py::list rev_domain;
    for (py::ssize_t i = py::len(flat_idcs) - 1; i >= num_cod_legs; --i)
        rev_domain.append(flat_idcs[i]);
    py::list flat_ordered;
    for (py::ssize_t i = 0; i < num_cod_legs; ++i)
        flat_ordered.append(flat_idcs[i]);
    for (py::handle h : rev_domain)
        flat_ordered.append(h);
    py::list rows;
    std::vector<std::vector<int64>> pairs;
    pairs.reserve(static_cast<std::size_t>(py::len(a_legs)));
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i) {
        auto es = a_legs[i].cast<ElementarySpace::Ptr>();
        if (!es) {
            throw std::invalid_argument("get_element: expected ElementarySpace flat legs");
        }
        auto [b, w] = es->parse_index(flat_ordered[i].cast<int64>());
        pairs.push_back({ b, w });
    }
    BlockInds pos = BlockInds::from_rows(pairs);
    py::list uncoupled_list;
    for (std::size_t i = 0; i < pairs.size(); ++i)
        uncoupled_list.append(a_legs[static_cast<py::ssize_t>(i)]
                                .attr("sector_decomposition")
                                .attr("__getitem__")(pos(i, 0)));
    SectorArray uncoupled = as_sector_array_from_list(uncoupled_list);
    SectorArray codom_uncoupled = uncoupled.slice(0, static_cast<std::size_t>(num_cod_legs));
    SectorArray dom_uncoupled =
      uncoupled.slice(static_cast<std::size_t>(num_cod_legs), uncoupled.size());
    std::vector<int64> mults, dims_vec;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        int64 si = pos(i, 0);
        mults.push_back(
          mults_of(a_legs[static_cast<py::ssize_t>(i)])[static_cast<std::size_t>(si)]);
    }
    auto sym = py::cast(a->symmetry).cast<Symmetry::Ptr>();
    auto batch_dims = sym->batch_sector_dim(uncoupled);
    std::vector<int64> codom_dims(batch_dims.begin(), batch_dims.begin() + num_cod_legs);
    std::vector<int64> dom_dims(batch_dims.begin() + num_cod_legs, batch_dims.end());
    std::vector<int64> codom_mults(mults.begin(), mults.begin() + num_cod_legs);
    std::vector<int64> dom_mults(mults.begin() + num_cod_legs, mults.end());
    std::vector<int64> shape;
    for (std::size_t i = 0; i < batch_dims.size(); ++i)
        shape.push_back(batch_dims[i] * mults[i]);
    auto a_data = data_from_tensor(a);
    Dtype dt =
      dtype::common({ a_data->dtype, sym->fusion_tensor_dtype.value_or(Dtype::Complex128) });
    auto forest_block = block_backend->zeros(shape, dt, a->device);
    auto codomain = py::cast(a->codomain).cast<TensorProduct::Ptr>();
    auto domain = py::cast(a->domain).cast<TensorProduct::Ptr>();
    int64 tree_block_height = codomain->tree_block_size(codom_uncoupled);
    int64 tree_block_width = domain->tree_block_size(dom_uncoupled);
    auto const& bi = a_data->block_inds;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        int64 bi_cod = bi(n, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        int64 i1 = codomain->forest_block_slice(codom_uncoupled, coupled).start;
        int64 i2 = domain->forest_block_slice(dom_uncoupled, coupled).start;
        auto [entries, _, __] =
          _get_forest_block_contribution(a_data->blocks[static_cast<std::size_t>(n)],
                                         sym,
                                         codomain,
                                         domain,
                                         coupled,
                                         py::cast(codom_uncoupled),
                                         py::cast(dom_uncoupled),
                                         codom_dims,
                                         dom_dims,
                                         tree_block_width,
                                         tree_block_height,
                                         i1,
                                         i2,
                                         codom_mults,
                                         dom_mults,
                                         dt);
        std::vector<int64> perm;
        for (int64 i = 0; i < num_legs; ++i) {
            perm.push_back(i);
            perm.push_back(i + num_legs);
        }
        entries = block_backend->permute_axes(entries, perm);
        entries = block_backend->reshape(entries, shape);
        forest_block = (*forest_block) + (*entries);
    }
    std::vector<int64> idx_within;
    idx_within.reserve(pos.nrows());
    for (std::size_t i = 0; i < pos.nrows(); ++i)
        idx_within.push_back(pos(i, 1));
    return block_backend->get_block_element(forest_block, idx_within);
}

TensorBackend::DataPtr
FusionTreeBackend::from_grid(std::vector<std::vector<py::object>> grid,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain,
                             std::vector<std::vector<int64>> left_mult_slices,
                             std::vector<std::vector<int64>> right_mult_slices,
                             Dtype dtype,
                             std::string device)
{
    // --- hints from Python FusionTreeBackend.from_grid ---
    // idea: iterate over forest blocks of the operators in the grid; forest blocks are always
    // contiguous except in the case where contributions to the same forest block comes from
    // different operators -> reshape such that the legs along which we stack are isolated
    // goal: reshape co_domain part such that it has 3 axes:
    // for the trees, for the multiplicity of codomain[0] / domain[-1], for all other
    // multiplicities only the first space is different only the last space is different
    // ---
    auto data = unwrap(zero_data(new_codomain, new_domain, dtype, device, true));
    auto new_codomain_legs = new_codomain->flat_legs();
    auto new_domain_legs = new_domain->flat_legs();
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid[i].size(); ++j) {
            py::object op = grid[i][j];
            if (op.is_none())
                continue;
            auto op_t = op.cast<TensorCPtr>();
            auto op_data = data_from_tensor(op_t);
            auto op_codomain = op_t->codomain;
            auto op_domain = op_t->domain;
            py::list op_coupled_list;
            auto const& op_bi = op_data->block_inds;
            for (std::size_t r = 0; r < op_bi.nrows(); ++r)
                op_coupled_list.append(
                  op_domain->sector_decomposition[static_cast<std::size_t>(op_bi(r, 1))]);
            SectorArray op_coupled = as_sector_array_from_list(op_coupled_list);
            for (auto const& fb : op_codomain->iter_forest_blocks(op_coupled)) {
                Sector coupled = op_coupled[static_cast<std::size_t>(fb.coupled_idx)];
                auto block_idx = data->block_ind_from_coupled(coupled, new_domain);
                if (!block_idx.has_value())
                    continue;
                std::vector<int64> op_codom_shape;
                for (std::size_t l = 0; l < fb.uncoupled.size(); ++l)
                    op_codom_shape.push_back(
                      py::cast(op_codomain->flat_legs()[l])
                        .attr("sector_multiplicity")(py::cast(fb.uncoupled[l]))
                        .cast<int64>());
                int64 op_codom0 = op_codom_shape.empty() ? 1 : op_codom_shape[0];
                int64 op_codom_rest =
                  prod_int(std::vector<int64>(op_codom_shape.begin() + 1, op_codom_shape.end()));
                op_codom_shape = { op_codom0, op_codom_rest };
                int64 op_num_codom_trees =
                  (fb.slice.stop - fb.slice.start) / std::max<int64>(1, prod_int(op_codom_shape));
                std::vector<int64> codom_shape = { py::cast(new_codomain_legs[0])
                                                     .attr("sector_multiplicity")(
                                                       py::cast(fb.uncoupled[0]))
                                                     .cast<int64>(),
                                                   op_codom_rest };
                auto codom_slc = new_codomain->forest_block_slice(fb.uncoupled, coupled);
                int64 num_codom_trees =
                  (codom_slc.stop - codom_slc.start) / std::max<int64>(1, prod_int(codom_shape));
                auto codom_leg_idx =
                  py::cast(new_codomain_legs[0])
                    .attr("sector_decomposition_where")(py::cast(fb.uncoupled[0]))
                    .cast<int64>();
                py::slice codom_leg_slc(
                  left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i],
                  left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i + 1],
                  1);
                for (auto const& fb_dom :
                     op_domain->iter_forest_blocks(SectorArray::repeat(coupled, 1))) {
                    std::vector<int64> op_dom_shape;
                    for (std::size_t l = 0; l < fb_dom.uncoupled.size(); ++l)
                        op_dom_shape.push_back(
                          py::cast(op_domain->flat_legs()[l])
                            .attr("sector_multiplicity")(py::cast(fb_dom.uncoupled[l]))
                            .cast<int64>());
                    int64 op_dom_last = op_dom_shape.empty() ? 1 : op_dom_shape.back();
                    int64 op_dom_rest =
                      prod_int(std::vector<int64>(op_dom_shape.begin(), op_dom_shape.end() - 1));
                    op_dom_shape = { op_dom_rest, op_dom_last };
                    int64 op_num_dom_trees = (fb_dom.slice.stop - fb_dom.slice.start) /
                                             std::max<int64>(1, prod_int(op_dom_shape));
                    std::vector<int64> op_new_shape = { op_num_codom_trees, op_codom0,
                                                        op_codom_rest,      op_num_dom_trees,
                                                        op_dom_rest,        op_dom_last };
                    std::vector<int64> dom_shape = {
                        op_dom_rest,
                        py::cast(new_domain_legs.back())
                          .attr("sector_multiplicity")(
                            py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                          .cast<int64>()
                    };
                    auto dom_slc = new_domain->forest_block_slice(fb_dom.uncoupled, coupled);
                    int64 num_dom_trees =
                      (dom_slc.stop - dom_slc.start) / std::max<int64>(1, prod_int(dom_shape));
                    std::vector<int64> new_shape = { num_codom_trees, codom_shape[0],
                                                     codom_shape[1],  num_dom_trees,
                                                     dom_shape[0],    dom_shape[1] };
                    auto dom_leg_idx = py::cast(new_domain_legs.back())
                                         .attr("sector_decomposition_where")(
                                           py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                                         .cast<int64>();
                    py::slice dom_leg_slc(
                      right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j],
                      right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j + 1],
                      1);
                    auto op_block =
                      b_get(op_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                            py::make_tuple(slice_from_index_slice(fb.slice),
                                           slice_from_index_slice(fb_dom.slice)));
                    op_block = block_backend->reshape(op_block, op_new_shape);
                    auto block = block_backend->copy_block(
                      b_get(data->blocks[static_cast<std::size_t>(*block_idx)],
                            py::make_tuple(slice_from_index_slice(codom_slc),
                                           slice_from_index_slice(dom_slc))),
                      device);
                    auto final_shape = block_backend->get_shape(block);
                    block = block_backend->reshape(block, new_shape);
                    auto cur = b_get(block,
                                     py::make_tuple(py::slice(0, std::nullopt, 1),
                                                    codom_leg_slc,
                                                    py::slice(0, std::nullopt, 1),
                                                    py::slice(0, std::nullopt, 1),
                                                    py::slice(0, std::nullopt, 1),
                                                    dom_leg_slc));
                    b_set(block,
                          py::make_tuple(py::slice(0, std::nullopt, 1),
                                         codom_leg_slc,
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         dom_leg_slc),
                          (*cur) + (*op_block));
                    block = block_backend->reshape(block, final_shape);
                    b_set(data->blocks[static_cast<std::size_t>(*block_idx)],
                          py::make_tuple(slice_from_index_slice(codom_slc),
                                         slice_from_index_slice(dom_slc)),
                          block);
                }
            }
        }
    }
    data->discard_zero_blocks(block_backend, get_config().fusion_tree_eps);
    return wrap(data);
}

TensorBackend::DataPtr
FusionTreeBackend::scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg)
{
    // --- hints from Python FusionTreeBackend.scale_axis ---
    // 1 if in_domain, 0 else
    // [:, 1] must be equal
    // special case where it is essentially compose.
    // use flattened tensor product -> need to shift co_domain_idx
    // potential coupled sectors
    // mapping between index in coupled sectors and index in blocks
    // add -1 for reshaping to take care of multiple trees within the same forest
    // + 1 for axis comes from adding -1 to the reshaping
    // ---
    Dtype dtype = dtype::common({ a->dtype, b->dtype });
    auto parsed = a->_parse_leg_idx(leg);
    bool in_domain = std::get<0>(parsed);
    int64 co_domain_idx = std::get<1>(parsed);
    int64 ax_a = in_domain ? 1 : 0;
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto const& a_bi = a_data->block_inds;
    auto b_col0 = b_data->block_inds.column(0);
    int64 num_cod = a->num_codomain_legs();
    int64 num_dom = py::cast(a->domain).attr("num_factors").cast<int64>();
    if ((in_domain && num_dom == 1) || (!in_domain && num_cod == 1)) {
        std::vector<std::vector<int64>> block_inds_rows;
        std::vector<BlockBackend::BlockPtr> blocks;
        auto a_contr = in_domain ? a_bi.column(1) : a_bi.column(0);
        if (a_bi.nrows() > 0 && b_data->block_inds.nrows() > 0) {
            iter_common_sorted_1d(a_contr, b_col0, [&](std::ptrdiff_t n_a, std::ptrdiff_t n_b) {
                blocks.push_back(
                  block_backend->scale_axis(a_data->blocks[static_cast<std::size_t>(n_a)],
                                            b_data->blocks[static_cast<std::size_t>(n_b)],
                                            ax_a));
                // Use the diagonal's *sector* index, not the row index n_b —
                // they differ when the diagonal omits zero sectors.
                int64 b_sector = b_data->block_inds(static_cast<std::size_t>(n_b), 0);
                if (in_domain)
                    block_inds_rows.push_back(
                      { a_bi(static_cast<std::size_t>(n_a), 0), b_sector });
                else
                    block_inds_rows.push_back(
                      { b_sector, a_bi(static_cast<std::size_t>(n_a), 1) });
            });
        }
        BlockInds block_inds =
          block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
        return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
    }
    TensorProduct::Ptr iter_space = in_domain ? py::cast(a->domain).cast<TensorProduct::Ptr>()
                                              : py::cast(a->codomain).cast<TensorProduct::Ptr>();
    if (a->has_pipes()) {
        co_domain_idx = iter_space->flat_leg_idcs(co_domain_idx).front();
        iter_space = flatten_pipes(iter_space);
    }
    py::list coupled_list;
    for (std::size_t r = 0; r < a_bi.nrows(); ++r)
        coupled_list.append(
          py::cast(a->codomain).attr("sector_decomposition").attr("__getitem__")(a_bi(r, 0)));
    SectorArray coupled_sectors = as_sector_array_from_list(coupled_list);
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<std::vector<int64>> block_inds_rows;
    std::map<int64, std::size_t> ind_mapping;
    std::set<int64> domain_inds_seen;
    for (auto const& fb : iter_space->iter_forest_blocks(coupled_sectors)) {
        int64 ind = a_bi(static_cast<std::size_t>(fb.coupled_idx), 1);
        auto ind_b =
          b_data->block_ind_from_coupled(fb.uncoupled[static_cast<std::size_t>(co_domain_idx)],
                                         py::cast(b->domain).cast<TensorProduct::Ptr>());
        if (!ind_b.has_value())
            continue;
        if (!domain_inds_seen.count(ind)) {
            domain_inds_seen.insert(ind);
            ind_mapping[fb.coupled_idx] = blocks.size();
            block_inds_rows.push_back({ a_bi(static_cast<std::size_t>(fb.coupled_idx), 0), ind });
            blocks.push_back(block_backend->zeros(
              block_backend->get_shape(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)]),
              dtype));
        }
        std::vector<int64> reshape;
        for (std::size_t li = 0; li < fb.uncoupled.size(); ++li)
            reshape.push_back(py::cast(iter_space->flat_legs()[li])
                                .attr("sector_multiplicity")(py::cast(fb.uncoupled[li]))
                                .cast<int64>());
        BlockBackend::BlockPtr forest;
        std::vector<int64> initial_shape;
        py::object slcs0, slcs1;
        if (in_domain) {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                          slice_from_index_slice(fb.slice)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { initial_shape[0], -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            forest = block_backend->reshape(forest, rshape);
            slcs0 = py::slice(0, initial_shape[0], 1);
            slcs1 = slice_from_index_slice(fb.slice);
        } else {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(slice_from_index_slice(fb.slice),
                                          py::slice(std::nullopt, std::nullopt, 1)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            rshape.push_back(initial_shape[1]);
            forest = block_backend->reshape(forest, rshape);
            slcs0 = slice_from_index_slice(fb.slice);
            slcs1 = py::slice(0, initial_shape[1], 1);
        }
        forest = block_backend->scale_axis(
          forest, b_data->blocks[static_cast<std::size_t>(*ind_b)], ax_a + co_domain_idx + 1);
        forest = block_backend->reshape(forest, initial_shape);
        b_set(blocks[ind_mapping.at(fb.coupled_idx)], py::make_tuple(slcs0, slcs1), forest);
    }
    BlockInds block_inds =
      block_inds_rows.empty() ? zeros_i64(0, 2) : BlockInds::from_rows(block_inds_rows);
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::diagonal_to_mask(DiagonalTensorCPtr tens)
{
    // --- hints from Python FusionTreeBackend.diagonal_to_mask ---
    // block_inds are w.r.t. TensorProducts, not the legs
    // -> maybe need to do additional sorting if leg is dual
    // get the defining sector
    // ---
    // Match Python FusionTreeBackend.diagonal_to_mask: defining sectors + ElementarySpace ctor.
    auto tens_data = data_from_tensor(tens);
    // Keep TensorProduct alive while reading Space members (sector_decomposition is
    // exposed via def_readwrite; chained attr() on a temporary is UAF).
    auto codomain = py::cast(tens->codomain).cast<TensorProduct::Ptr>();
    py::object large_leg = py::cast(tens->leg());
    auto large_es = large_leg.cast<ElementarySpace::Ptr>();
    bool is_dual = large_es->is_dual;
    bool is_sorted = !is_dual;
    bool have_basis_perm = large_es->has_custom_basis_perm();
    py::object basis_perm =
      have_basis_perm ? large_leg.attr("_basis_perm") : py::object(py::none());
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> codom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto const& bi = tens_data->block_inds;
    auto sym = py::cast(tens->symmetry).cast<Symmetry::Ptr>();
    auto const& codom_secs = codomain->sector_decomposition;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        auto const& diag_block = tens_data->blocks[n];
        if (!block_backend->any(diag_block))
            continue;
        int64 bii = bi(n, 0);
        Sector coupled = codom_secs[static_cast<std::size_t>(bii)];
        Sector defining = is_dual ? sym->dual_sector(coupled) : coupled;
        blocks.push_back(diag_block);
        codom_block_inds.push_back(bii);
        sectors_vec.push_back(defining);
        multiplicities.push_back(block_backend->sum_all(diag_block).as_int64());
        if (have_basis_perm) {
            int64 dim = sym->sector_dim(defining);
            auto tiled = block_backend->tile(diag_block, dim);
            py::array mask = block_backend->to_numpy(tiled, py_bool_dtype());
            int64 slice_bi = bii;
            if (is_dual) {
                auto j = large_leg.attr("sector_decomposition_where")(py::cast(coupled));
                if (j.is_none())
                    throw std::runtime_error("diagonal_to_mask: coupled sector not in large_leg");
                slice_bi = j.cast<int64>();
            }
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(slice_bi));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    BlockInds block_inds;
    ElementarySpace::Ptr small_leg;
    if (blocks.empty()) {
        block_inds = zeros_i64(0, 2);
        small_leg = std::make_shared<ElementarySpace>(
          sym, sym->empty_sector_array, std::vector<int64>{}, is_dual, std::nullopt);
    } else {
        SectorArray sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
        if (!is_sorted) {
            auto perm = sectors.lexsort_indices();
            sectors = sectors.take(perm);
            std::vector<int64> new_mults;
            new_mults.reserve(multiplicities.size());
            py::list new_ranks;
            for (auto pi : perm) {
                new_mults.push_back(multiplicities[pi]);
                if (have_basis_perm)
                    new_ranks.append(basis_perm_ranks[static_cast<py::ssize_t>(pi)]);
            }
            multiplicities = std::move(new_mults);
            if (have_basis_perm)
                basis_perm_ranks = std::move(new_ranks);
        }
        if (have_basis_perm) {
            auto ranked = rank_concat(basis_perm_ranks);
            basis_perm_opt = ranked;
        }
        // Like Python: do not reorder blocks/codom_block_inds with the defining lexsort.
        std::vector<int64> arange(blocks.size());
        std::iota(arange.begin(), arange.end(), 0);
        block_inds =
          BlockInds::column_stack(std::vector<std::span<const int64>>{ arange, codom_block_inds });
        small_leg = std::make_shared<ElementarySpace>(
          sym, std::move(sectors), multiplicities, is_dual, std::move(basis_perm_opt));
    }
    auto data = make_data(Dtype::Bool, tens_data->device, std::move(blocks), block_inds, true);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_binary_operand(MaskCPtr mask1, MaskCPtr mask2, BlockBinaryFn func)
{
    // --- hints from Python FusionTreeBackend.mask_binary_operand ---
    // -> maybe need to do additional sorting and searching if leg is dual
    // next block of mask1 to process; iterating like this only works if is_sorted.
    // its block_ind for the large leg.
    // do this here for both masks
    // mask1 has no further blocks
    // mask2 has no further blocks
    // ---
    py::object large_leg = py::cast(mask1->large_leg());
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask1_data = data_from_tensor(mask1);
    auto mask2_data = data_from_tensor(mask2);
    auto const& m1 = mask1_data->block_inds;
    auto const& m2 = mask2_data->block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i1 = 0, i2 = 0;
    int64 b1_i1 = m1.nrows() == 0 ? -1 : m1(0, 1);
    int64 b2_i2 = m2.nrows() == 0 ? -1 : m2(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = py::cast(mask1->symmetry).cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block1, block2;
        bool block1_found = false, block2_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b1_i1) {
            block1 = mask1_data->blocks[static_cast<std::size_t>(i1)];
            ++i1;
            b1_i1 = static_cast<std::size_t>(i1) >= m1.nrows()
                      ? -1
                      : m1(static_cast<std::size_t>(i1), 1);
            block1_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition")
                                .attr("__getitem__")(sector_idx)
                                .cast<Sector>();
            auto dom_idx = py::cast(mask1->domain).attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask1_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block1 = mask1_data->blocks[static_cast<std::size_t>(*idx)];
                    block1_found = true;
                }
            }
        }
        if (!block1_found)
            block1 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        if (is_sorted && static_cast<int64>(sector_idx) == b2_i2) {
            block2 = mask2_data->blocks[static_cast<std::size_t>(i2)];
            ++i2;
            b2_i2 = static_cast<std::size_t>(i2) >= m2.nrows()
                      ? -1
                      : m2(static_cast<std::size_t>(i2), 1);
            block2_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition")
                                .attr("__getitem__")(sector_idx)
                                .cast<Sector>();
            auto dom_idx = py::cast(mask2->domain).attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask2_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block2 = mask2_data->blocks[static_cast<std::size_t>(*idx)];
                    block2_found = true;
                }
            }
        }
        if (!block2_found)
            block2 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(block1, block2);
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            auto tiled = block_backend->tile(new_block, dim);
            py::array mask = block_backend->to_numpy(tiled, py_bool_dtype());
            auto slc = slice_pair(
              large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    SectorArray sectors = sym->empty_sector_array;
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
        std::iota(arange.begin(), arange.end(), 0);
        block_inds =
          BlockInds::column_stack(std::vector<std::span<const int64>>{ arange, dom_block_inds });
    }
    auto data = make_data(Dtype::Bool, mask1->device, std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(sym,
                                                       std::move(sectors),
                                                       multiplicities,
                                                       large_leg.attr("is_dual").cast<bool>(),
                                                       basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    // --- hints from Python FusionTreeBackend.mask_from_block ---
    // block_inds are w.r.t. TensorProducts, not the legs
    // -> maybe need to do additional sorting and searching if leg is dual
    // ---
    py::object large_leg_obj = py::cast(large_leg);
    py::object basis_perm = large_leg_obj.attr("_basis_perm");
    bool is_sorted = !large_leg_obj.attr("is_dual").cast<bool>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg_obj.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg_obj.attr("slices");
    py::object domain;
    if (!is_sorted) {
        auto perm = large_leg_obj.attr("sector_decomposition").attr("lexsort_indices")();
        auto sorted_duals =
          large_leg_obj.attr("sector_decomposition").attr("__getitem__")(perm).cast<SectorArray>();
        auto multis = large_leg_obj.attr("multiplicities")
                        .attr("__getitem__")(perm)
                        .cast<std::vector<int64>>();
        auto large_es = std::dynamic_pointer_cast<ElementarySpace>(large_leg);
        if (!large_es) {
            throw std::invalid_argument("mask_from_block: large_leg must be an ElementarySpace");
        }
        auto leg_ptr = Leg{ large_es };
        domain = py::cast(std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ leg_ptr },
                                                          large_leg->symmetry,
                                                          std::move(sorted_duals),
                                                          std::move(multis)));
    }
    for (std::size_t bi_large = 0; bi_large < defining.size(); ++bi_large) {
        auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(bi_large)));
        auto block = b_get(a, slc);
        int64 mult = block_backend->sum_all(block).as_int64();
        if (mult == 0)
            continue;
        Sector sector = defining[bi_large];
        int64 bi_out = static_cast<int64>(bi_large);
        if (!is_sorted) {
            Sector dual_sector = large_leg->symmetry->dual_sector(sector);
            bi_out =
              domain.attr("sector_decomposition_where")(py::cast(dual_sector)).cast<int64>();
        }
        dom_block_inds.push_back(bi_out);
        sectors_vec.push_back(sector);
        int64 dim = large_leg->symmetry->sector_dim(sector);
        int64 stop = block_backend->get_shape(block)[0] / dim;
        blocks.push_back(b_get(block, py::slice(0, stop, 1)));
        multiplicities.push_back(mult / dim);
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(block).cast<py::array>();
            basis_perm_ranks.append(
              large_leg_obj.attr("basis_perm").attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    SectorArray sectors = large_leg->symmetry->empty_sector_array;
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
        if (!is_sorted) {
            // Match Python: only reorder dom_block_inds and blocks (not sectors/multiplicities).
            std::vector<std::size_t> perm(dom_block_inds.size());
            std::iota(perm.begin(), perm.end(), std::size_t{ 0 });
            std::stable_sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
                return dom_block_inds[i] < dom_block_inds[j];
            });
            py::list new_dom, new_blocks;
            for (std::size_t pi : perm) {
                new_dom.append(dom_block_inds[pi]);
                new_blocks.append(blocks[pi]);
            }
            dom_block_inds.clear();
            for (py::handle h : new_dom)
                dom_block_inds.push_back(h.cast<int64>());
            blocks.clear();
            for (py::handle h : new_blocks)
                blocks.push_back(h.cast<BlockBackend::BlockPtr>());
        }
        std::vector<int64> arange(sectors.size());
        std::iota(arange.begin(), arange.end(), 0);
        block_inds =
          BlockInds::column_stack(std::vector<std::span<const int64>>{ arange, dom_block_inds });
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
FusionTreeBackend::mask_to_block(MaskCPtr a)
{
    auto a_data = data_from_tensor(a);
    auto large_leg = py::cast(a->large_leg());
    auto res = block_backend->zeros({ large_leg.attr("dim").cast<int64>() }, Dtype::Bool);
    bool is_projection = a->is_projection;
    auto const& bi = a_data->block_inds;
    auto co_dom = is_projection ? py::cast(a->domain) : py::cast(a->codomain);
    for (std::size_t i = 0; i < bi.nrows(); ++i) {
        int64 bi_large = is_projection ? bi(i, 1) : bi(i, 0);
        Sector sector =
          co_dom.attr("sector_decomposition").attr("__getitem__")(bi_large).cast<Sector>();
        int64 dim = co_dom.attr("symmetry").attr("sector_dim")(sector).cast<int64>();
        if (large_leg.attr("is_dual").cast<bool>())
            bi_large = large_leg.attr("sector_decomposition_where")(sector).cast<int64>();
        auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bi_large));
        b_set(res, slc, block_backend->tile(a_data->blocks[static_cast<std::size_t>(i)], dim));
    }
    return res;
}

std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
FusionTreeBackend::mask_transpose(MaskCPtr tens)
{
    // --- hints from Python FusionTreeBackend.mask_transpose ---
    // similar implementation to diagonal_transpose
    // ---
    auto data = data_from_tensor(tens);
    auto sym = py::cast(tens->symmetry).cast<Symmetry::Ptr>();
    auto perm_dom =
      sym->dual_sectors(py::cast(tens->domain).attr("sector_decomposition").cast<SectorArray>())
        .lexsort_indices();
    auto perm_codom =
      sym->dual_sectors(py::cast(tens->codomain).attr("sector_decomposition").cast<SectorArray>())
        .lexsort_indices();
    std::vector<int64> perm_dom_i64(perm_dom.begin(), perm_dom.end());
    std::vector<int64> perm_codom_i64(perm_codom.begin(), perm_codom.end());
    auto inv_dom = inverse_permutation(perm_dom_i64);
    auto inv_codom = inverse_permutation(perm_codom_i64);
    auto col1 = data->block_inds.column(1);
    auto col0 = data->block_inds.column(0);
    for (std::size_t i = 0; i < col1.size(); ++i) {
        col1[i] = inv_dom[static_cast<std::size_t>(col1[i])];
        col0[i] = inv_codom[static_cast<std::size_t>(col0[i])];
    }
    // Transpose: (codomain_idx, domain_idx) -> (mapped_domain, mapped_codomain)
    BlockInds block_inds =
      BlockInds::column_stack(std::vector<std::span<const int64>>{ col1, col0 });
    auto out = make_data(tens->dtype, data->device, data->blocks, block_inds, false);
    return { py::cast(tens->codomain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             py::cast(tens->domain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             wrap(out) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_unary_operand(MaskCPtr mask, BlockUnaryFn func)
{
    // --- hints from Python FusionTreeBackend.mask_unary_operand ---
    // next block of mask to process; iterating like this only works if is_sorted.
    // mask has no further blocks
    // ---
    py::object large_leg = py::cast(mask->large_leg());
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask_data = data_from_tensor(mask);
    auto const& mbuf = mask_data->block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i = 0;
    int64 b_i = mbuf.nrows() == 0 ? -1 : mbuf(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = py::cast(mask->symmetry).cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block;
        bool block_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b_i) {
            block = mask_data->blocks[static_cast<std::size_t>(i)];
            ++i;
            b_i = static_cast<std::size_t>(i) >= mbuf.nrows()
                    ? -1
                    : mbuf(static_cast<std::size_t>(i), 1);
            block_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition")
                                .attr("__getitem__")(sector_idx)
                                .cast<Sector>();
            auto idx = mask_data->block_ind_from_coupled(
              dual_sec, py::cast(mask->domain).cast<TensorProduct::Ptr>());
            if (idx.has_value()) {
                block = mask_data->blocks[static_cast<std::size_t>(*idx)];
                block_found = true;
            }
        }
        if (!block_found)
            block = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(block);
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            auto tiled = block_backend->tile(new_block, dim);
            py::array mask_np = block_backend->to_numpy(tiled, py_bool_dtype());
            auto slc = slice_pair(
              large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(
              basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask_np));
        }
    }
    SectorArray sectors = sym->empty_sector_array;
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
        std::iota(arange.begin(), arange.end(), 0);
        block_inds =
          BlockInds::column_stack(std::vector<std::span<const int64>>{ arange, dom_block_inds });
    }
    auto data = make_data(Dtype::Bool, mask->device, std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(sym,
                                                       std::move(sectors),
                                                       multiplicities,
                                                       large_leg.attr("is_dual").cast<bool>(),
                                                       basis_perm_opt);
    return { wrap(data), small_leg };
}

} // namespace lasyten

// =============================================================================
// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)
// =============================================================================
// --- _tree_block_iter ---
// reset to the left of the current forest block
// move right by one tree block
// move down by one tree block
// --- PermuteLegsInstructionEngine.__init__ ---
// should have raised in tensors.py already
// stateful attributes (need to be kept up to date!):
// --- FactorizedTreeMapping._transform_splitting_trees ---
// note: we first add all contributions to the new rows, and then do the
// axes permutation only once to the result.
// --- _partial_trace_helper ---
// this must be the case if there is only one way to fuse to the trivial sector
// =============================================================================
