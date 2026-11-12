#include <lasyten/tensors/mask.h>

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tools.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

[[nodiscard]] py::object
py_bool_dtype()
{
    return py::reinterpret_borrow<py::object>((PyObject*)&PyBool_Type);
}

ElementarySpace::Ptr
as_elementary_space(Space::Ptr obj)
{
    if (std::dynamic_pointer_cast<LegPipe>(obj)) {
        throw std::invalid_argument("Mask is not defined on LegPipes.");
    }
    auto es = std::dynamic_pointer_cast<ElementarySpace>(obj);
    if (!es) {
        throw std::invalid_argument("Expected ElementarySpace.");
    }
    return es;
}

BlockBackend::LegCPtr
as_leg_cptr(Space::Ptr const& space)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(space);
    if (!es) {
        throw std::invalid_argument("Expected ElementarySpace for Mask leg.");
    }
    return std::static_pointer_cast<Leg const>(es);
}

/// Invert each bool entry of a block (replaces ``operator.invert`` via numpy).
BlockUnaryFn
adapt_block_bool_invert(std::shared_ptr<BlockBackend> bb)
{
    return [bb](BlockBackend::BlockPtr const& block) {
        py::array arr = bb->to_numpy(block, py_bool_dtype());
        py::array_t<bool, py::array::c_style | py::array::forcecast> flat =
          py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(arr).reshape(
            { -1 });
        py::array_t<bool> out(flat.size());
        auto src = flat.unchecked<1>();
        auto dst = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < src.shape(0); ++i) {
            dst(i) = !src(i);
        }
        auto info = arr.request();
        std::vector<py::ssize_t> shape(static_cast<std::size_t>(info.ndim));
        for (py::ssize_t d = 0; d < info.ndim; ++d) {
            shape[static_cast<std::size_t>(d)] = info.shape[d];
        }
        return bb->as_block(out.reshape(shape), Dtype::Bool, block->device());
    };
}

int64
space_dim(Space const& space)
{
    return static_cast<int64>(space.Space::dim);
}

int64
sum_multiplicities(Space const& space)
{
    return std::accumulate(space.multiplicities.begin(), space.multiplicities.end(), int64{ 0 });
}

bool
basis_perm_trivial(ElementarySpace const& leg)
{
    if (!leg.has_custom_basis_perm()) {
        return true;
    }
    auto const& perm = leg.basis_perm();
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64>(i)) {
            return false;
        }
    }
    return true;
}

TensorBackend::Ptr
resolve_backend(TensorBackend::Ptr backend, Space::Ptr const& space)
{
    if (!backend) {
        return get_backend(space->symmetry);
    }
    return backend;
}

Mask::Ptr
make_mask(TensorBackend::DataPtr data,
          Space::Ptr space_in,
          Space::Ptr space_out,
          bool is_projection,
          TensorBackend::Ptr backend,
          std::optional<LegLabelList> labels)
{
    as_elementary_space(space_in);
    as_elementary_space(space_out);
    auto [codomain, domain, backend_tp, symmetry] = Tensor::_init_parse_args(
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_out) }),
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_in) }),
      std::move(backend));
    auto labs = Tensor::_init_parse_labels(std::move(labels), codomain, domain);
    auto device_s = backend_tp->get_device_from_data(data);
    return std::make_shared<Mask>(std::move(data),
                                  std::move(space_in),
                                  std::move(space_out),
                                  is_projection,
                                  std::move(backend_tp),
                                  std::move(symmetry),
                                  std::move(labs),
                                  std::move(device_s));
}

} // namespace

/// @cond
std::vector<Dtype> Mask::_forbidden_dtypes = {
    Dtype::Float32,
    Dtype::Float64,
    Dtype::Complex64,
    Dtype::Complex128,
};
/// @endcond

Mask::Mask(TensorBackend::DataPtr data_in,
           Space::Ptr space_in,
           Space::Ptr space_out,
           bool is_projection_in,
           TensorBackend::Ptr backend_in,
           Symmetry::Ptr symmetry_in,
           LegLabelList labels_in,
           std::string device_in)
  : Tensor(std::make_shared<TensorProduct>(
             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_out) }),
           std::make_shared<TensorProduct>(
             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_in) }),
           std::move(backend_in),
           std::move(symmetry_in),
           std::move(labels_in),
           Dtype::Bool,
           std::move(device_in))
  , is_projection(is_projection_in)
  , data(std::move(data_in))
{
    assert(backend->is_correct_data_type(data));
    if (py::isinstance<LegPipe>(py::cast(space_in)) ||
        py::isinstance<LegPipe>(py::cast(space_out))) {
        throw std::invalid_argument("Mask is not defined on LegPipes.");
    }
    if (!std::dynamic_pointer_cast<ElementarySpace>(space_in) ||
        !std::dynamic_pointer_cast<ElementarySpace>(space_out)) {
        throw std::invalid_argument("Expected ElementarySpace.");
    }
    if (is_projection) {
        if (!(space_dim(*space_in) >= space_dim(*space_out))) {
            throw std::invalid_argument("projection mask requires the incoming space to be at "
                                        "least as large as the outgoing space");
        }
        if (!space_out->is_subspace_of(*space_in)) {
            throw std::invalid_argument("projection mask requires the outgoing space to be a "
                                        "subspace of the incoming space");
        }
    } else {
        if (!(space_dim(*space_in) <= space_dim(*space_out))) {
            throw std::invalid_argument("inclusion mask requires the outgoing space to be at "
                                        "least as large as the incoming space");
        }
        if (!space_in->is_subspace_of(*space_out)) {
            throw std::invalid_argument(
              "inclusion mask requires the incoming space to be a subspace of the outgoing space");
        }
    }
    auto es_out = std::dynamic_pointer_cast<ElementarySpace>(space_out);
    auto es_in = std::dynamic_pointer_cast<ElementarySpace>(space_in);
    if (!es_out || !es_in || es_out->is_dual != es_in->is_dual) {
        throw std::invalid_argument("mask legs must have matching duality");
    }
}

std::vector<Dtype> const&
Mask::forbidden_dtypes() const
{
    return _forbidden_dtypes;
}

std::string
Mask::ascii_diagram_type_name() const
{
    return "Mask";
}

std::string
Mask::class_name() const
{
    return "Mask";
}

ElementarySpace::Ptr
Mask::large_leg() const
{
    if (is_projection) {
        return std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
    }
    return std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
}

ElementarySpace::Ptr
Mask::small_leg() const
{
    if (is_projection) {
        return std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
    }
    return std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
}

void
Mask::test_sanity() const
{
    // --- hints from Python Mask.test_sanity ---
    // check consistency of the basis perm of the small leg.
    // this is consistent.
    // check if ranks is sorted
    // ---
    Tensor::test_sanity();
    backend->test_mask_sanity(std::static_pointer_cast<Mask const>(shared_from_this()));
    assert(codomain->num_factors == 1 && domain->num_factors == 1);
    assert(std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]));
    assert(std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]));
    auto large = large_leg();
    auto small = small_leg();
    assert(large->is_dual == small->is_dual);
    assert(small->is_subspace_of(*large));
    assert(dtype == Dtype::Bool);
    assert(device == backend->get_device_from_data(data));

    // check consistency of the basis perm of the small leg.
    if (!large->has_custom_basis_perm()) {
        if (!small->has_custom_basis_perm()) {
            // consistent
        } else {
            auto const& actual = small->basis_perm();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (actual[i] != static_cast<int64>(i)) {
                    throw std::logic_error("Mask.test_sanity: small_leg.basis_perm inconsistent "
                                           "with trivial large_leg");
                }
            }
        }
    } else {
        auto mask_block =
          backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this()));
        auto mask_np = backend->block_backend->to_numpy(mask_block, py_bool_dtype());
        auto mask_bool =
          py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(mask_np);
        auto mbuf = mask_bool.unchecked<1>();
        auto const& pi_1 = large->basis_perm();
        auto const& pi_2_inv = small->inverse_basis_perm();
        std::vector<int64> kept_ranks;
        kept_ranks.reserve(static_cast<std::size_t>(mbuf.shape(0)));
        for (py::ssize_t i = 0; i < mbuf.shape(0); ++i) {
            if (mbuf(i)) {
                kept_ranks.push_back(pi_1[static_cast<std::size_t>(i)]);
            }
        }
        std::vector<int64> ranks(kept_ranks.size());
        for (std::size_t i = 0; i < kept_ranks.size(); ++i) {
            ranks[i] = kept_ranks[static_cast<std::size_t>(pi_2_inv[i])];
        }
        for (std::size_t i = 1; i < ranks.size(); ++i) {
            if (!(ranks[i] > ranks[i - 1])) {
                throw std::logic_error("Mask.test_sanity: kept basis ranks are not sorted");
            }
        }
    }
}

Mask::Ptr
Mask::from_eye(Space::Ptr leg,
               bool is_projection_flag,
               TensorBackend::Ptr backend,
               std::optional<LegLabelList> labels,
               std::optional<std::string> device)
{
    auto diag = DiagonalTensor::from_eye(std::move(leg), backend, labels, Dtype::Bool, device);
    auto res = from_DiagonalTensor(diag);
    if (!is_projection_flag) {
        return std::static_pointer_cast<Mask>(res->dagger());
    }
    return res;
}

Mask::Ptr
Mask::from_block_mask(BlockBackend::BlockPtr block_mask,
                      Space::Ptr large_leg,
                      TensorBackend::Ptr backend,
                      std::optional<LegLabelList> labels,
                      std::optional<std::string> device)
{
    if (!large_leg->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}",
                      large_leg->symmetry->repr()));
    }
    backend = resolve_backend(std::move(backend), large_leg);
    auto block = backend->block_backend->as_block(py::cast(block_mask), Dtype::Bool, device);
    block =
      backend->block_backend->apply_basis_perm(block, { as_leg_cptr(large_leg) }, /*inv=*/false);
    auto [data_out, small_leg] = backend->mask_from_block(block, large_leg);
    return make_mask(data_out, large_leg, small_leg, true, backend, std::move(labels));
}

Mask::Ptr
Mask::from_DiagonalTensor(DiagonalTensorCPtr diag)
{
    if (!diag) {
        throw std::invalid_argument("diag must be non-null");
    }
    if (diag->dtype != Dtype::Bool) {
        throw std::invalid_argument("Mask.from_DiagonalTensor requires bool dtype");
    }
    auto [data_out, small_leg] = diag->backend->diagonal_to_mask(diag);
    return std::make_shared<Mask>(data_out,
                                  as_space(diag->domain->factors[0]),
                                  small_leg,
                                  true,
                                  diag->backend,
                                  diag->symmetry,
                                  diag->labels.as_list(),
                                  diag->device);
}

Mask::Ptr
Mask::from_indices(py::object indices,
                   Space::Ptr large_leg,
                   TensorBackend::Ptr backend,
                   std::optional<LegLabelList> labels,
                   std::optional<std::string> device)
{
    backend = resolve_backend(std::move(backend), large_leg);
    auto bb = backend->block_backend;
    auto dim = space_dim(*large_leg);
    auto block = bb->zeros({ dim }, Dtype::Bool, device);
    auto true_s = bb->as_scalar(true);
    auto idx = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(indices);
    if (!idx) {
        throw std::invalid_argument("Mask.from_indices: indices must be array-like of int");
    }
    auto info = idx.request();
    auto const* ptr = static_cast<int64 const*>(info.ptr);
    auto const n = static_cast<std::size_t>(info.size);
    for (std::size_t i = 0; i < n; ++i) {
        block->set_item(ptr[i], true_s);
    }
    return from_block_mask(
      block, std::move(large_leg), std::move(backend), std::move(labels), device);
}

Mask::Ptr
Mask::from_random(Space::Ptr large_leg_in,
                  Space::Ptr small_leg_in,
                  TensorBackend::Ptr backend,
                  float64 p_keep,
                  int64 min_keep,
                  std::optional<LegLabelList> labels,
                  std::optional<std::string> device,
                  py::object np_random)
{
    // --- hints from Python Mask.from_random ---
    // diagonal entries are uniform in [-1, 1].
    // explicitly constructing the small_leg with exactly min_keep sectors kept is
    // quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
    // first, try a heuristic
    // step halfway towards 100%
    // ---
    auto large_leg = as_elementary_space(std::move(large_leg_in));
    backend = resolve_backend(std::move(backend), large_leg);

    if (!small_leg_in) {
        if (!(0. <= p_keep && p_keep <= 1.)) {
            throw std::invalid_argument(std::format("p_keep must be in [0, 1], got {}", p_keep));
        }
        auto diag =
          DiagonalTensor::from_random_uniform(large_leg, backend, labels, Dtype::Float32, device);
        float64 cutoff = 2. * p_keep - 1.; // diagonal entries are uniform in [-1, 1].
        auto res =
          from_DiagonalTensor(py::cast(diag).attr("__lt__")(cutoff).cast<DiagonalTensorCPtr>());

        if (sum_multiplicities(*res->small_leg()) >= min_keep) {
            return res;
        }

        int64 large_leg_sector_num = sum_multiplicities(*large_leg);
        if (min_keep > large_leg_sector_num) {
            throw std::invalid_argument(
              std::format("min_keep={} cannot be fulfilled for a leg with {} sectors",
                          min_keep,
                          large_leg_sector_num));
        }
        if (min_keep == large_leg_sector_num) {
            return from_eye(large_leg, /*is_projection=*/true, backend, labels, device);
        }
        // explicitly constructing the small_leg with exactly min_keep sectors kept is
        // quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
        // first, try a heuristic
        p_keep = std::ceil(1.05 * static_cast<float64>(min_keep) /
                           static_cast<float64>(large_leg_sector_num));
        if (p_keep > 1.) {
            p_keep = 1.;
        }
        res = from_DiagonalTensor(
          py::cast(diag).attr("__lt__")(2. * p_keep - 1.).cast<DiagonalTensorCPtr>());
        for (int i = 0; i < 20; ++i) {
            if (sum_multiplicities(*res->small_leg()) >= min_keep) {
                return res;
            }
            p_keep = 0.5 * (p_keep + 1.); // step halfway towards 100%
            res = from_DiagonalTensor(
              py::cast(diag).attr("__lt__")(2. * p_keep - 1.).cast<DiagonalTensorCPtr>());
        }
        throw std::runtime_error("Could not fulfill min_keep");
    }

    auto small_leg = as_elementary_space(std::move(small_leg_in));
    if (!small_leg->is_subspace_of(*large_leg)) {
        throw std::invalid_argument("small_leg must be a subspace of the large leg.");
    }

    if ((!basis_perm_trivial(*large_leg)) || (!basis_perm_trivial(*small_leg))) {
        throw NotImplemented(
          "Generating random Masks with non-trivial, fixed basis_perm is hard and hopefully never "
          "needed.");
    }

    auto small_leg_cap = small_leg;
    auto np_random_cap = np_random;
    auto bb = backend->block_backend;
    SectorBlockFactoryFn func = [small_leg_cap, np_random_cap, bb, device](
                                  std::vector<int64> const& shape, Sector const& coupled) {
        int64 num_keep = small_leg_cap->sector_multiplicity(coupled);
        auto block = bb->zeros(shape, Dtype::Bool, device);
        auto true_s = bb->as_scalar(true);
        std::vector<int64> which;
        which.reserve(static_cast<std::size_t>(num_keep));
        if (!np_random_cap.is_none()) {
            auto choice = np_random_cap.attr("choice")(
              shape[0], py::arg("size") = num_keep, py::arg("replace") = false);
            auto arr =
              py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(choice);
            auto buf = arr.unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                which.push_back(buf(i));
            }
        } else {
            // Native sample without replacement (Fisher–Yates partial shuffle).
            std::vector<int64> pool(static_cast<std::size_t>(shape[0]));
            std::iota(pool.begin(), pool.end(), int64{ 0 });
            thread_local std::mt19937_64 rng{ std::random_device{}() };
            for (int64 k = 0; k < num_keep; ++k) {
                std::uniform_int_distribution<int64> dist(k, shape[0] - 1);
                std::swap(pool[static_cast<std::size_t>(k)],
                          pool[static_cast<std::size_t>(dist(rng))]);
                which.push_back(pool[static_cast<std::size_t>(k)]);
            }
        }
        for (int64 idx : which) {
            block->set_item(idx, true_s);
        }
        return block;
    };

    auto diag = DiagonalTensor::from_sector_block_func(
      std::move(func), large_leg, backend, labels, Dtype::Bool, device);
    auto res = from_DiagonalTensor(diag);
    assert(static_cast<Space const&>(*res->small_leg()) == static_cast<Space const&>(*small_leg));
    return res;
}

Mask::Ptr
Mask::from_zero(Space::Ptr large_leg,
                TensorBackend::Ptr backend,
                std::optional<LegLabelList> labels,
                std::optional<std::string> device)
{
    backend = resolve_backend(std::move(backend), large_leg);
    auto device_s = backend->block_backend->as_device(device);
    auto data_out = backend->zero_mask_data(large_leg, device_s);
    bool is_dual = false;
    if (auto es = std::dynamic_pointer_cast<ElementarySpace>(large_leg)) {
        is_dual = es->is_dual;
    }
    auto small_leg = ElementarySpace::from_null_space(large_leg->symmetry, is_dual);
    return make_mask(data_out, std::move(large_leg), small_leg, true, backend, std::move(labels));
}

Tensor::Ptr
Mask::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    throw std::invalid_argument(
      "Mask requires Dtype.bool; use as_DiagonalTensor() or as_SymmetricTensor() "
      "for conversion to other tensor classes");
}

DiagonalTensor::Ptr
Mask::as_DiagonalTensor(Dtype out_dtype)
{
    return std::make_shared<DiagonalTensor>(
      backend->mask_to_diagonal(std::static_pointer_cast<Mask const>(shared_from_this()),
                                out_dtype),
      large_leg(),
      backend,
      symmetry,
      labels.as_list());
}

SymmetricTensorPtr
Mask::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> warning)
{
    // --- hints from Python Mask.as_SymmetricTensor ---
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    return as_SymmetricTensor(guarantee_copy, std::move(warning), Dtype::Complex128);
}

SymmetricTensorPtr
Mask::as_SymmetricTensor(bool /*guarantee_copy*/,
                         std::optional<std::string> warning,
                         Dtype out_dtype)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    if (!is_projection) {
        // OPTIMIZE how hard is it to deal with inclusions in the backend?
        auto proj = std::static_pointer_cast<Mask>(dagger());
        auto sym = proj->as_SymmetricTensor(false, std::nullopt, out_dtype);
        auto dag = lasyten::dagger(sym);
        auto out = std::dynamic_pointer_cast<SymmetricTensor>(dag);
        if (!out) {
            throw std::runtime_error(
              "Mask::as_SymmetricTensor: expected SymmetricTensor after dagger");
        }
        return out;
    }
    auto new_data = backend->full_data_from_mask(
      std::static_pointer_cast<Mask const>(shared_from_this()), out_dtype);
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels.as_list());
}

Mask::Ptr
Mask::_binary_operand(bool other, BlockBinaryFn func, std::string const& /*operand*/)
{
    auto bb = backend->block_backend;
    auto other_block = std::const_pointer_cast<BlockBackend::Block>(bb->as_scalar(other)._block());
    return _unary_operand([func, other_block](BlockBackend::BlockPtr const& block) {
        return func(block, other_block);
    });
}

Mask::Ptr
Mask::_binary_operand(MaskCPtr other, BlockBinaryFn func, std::string const& operand)
{
    // --- hints from Python Mask._binary_operand ---
    // remaining case: other is Mask
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    if (is_projection != other->is_projection) {
        throw std::invalid_argument("Mismatching is_projection.");
    }
    if (!is_projection) {
        // OPTIMIZE how hard is it to deal with inclusions in the backend?
        auto self_proj = std::static_pointer_cast<Mask>(dagger());
        auto other_proj = std::dynamic_pointer_cast<Mask>(other->dagger());
        auto res_projection = self_proj->_binary_operand(other_proj, func, operand);
        return std::static_pointer_cast<Mask>(res_projection->dagger());
    }

    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    if (!(static_cast<Space const&>(*domain) == static_cast<Space const&>(*other->domain))) {
        throw std::invalid_argument("Incompatible domain.");
    }
    auto [data_out, small] = same->mask_binary_operand(
      std::static_pointer_cast<Mask const>(shared_from_this()), other, std::move(func));
    auto labs = _get_matching_labels(labels.as_list(), other->labels.as_list());
    return make_mask(data_out, large_leg(), small, is_projection, same, labs);
}

Mask::Ptr
Mask::_unary_operand(BlockUnaryFn func)
{
    // --- hints from Python Mask._unary_operand ---
    // operate on the respective projection
    // OPTIMIZE: how hard is it to deal with inclusion Masks in the backends?
    // ---
    // operate on the respective projection
    if (!is_projection) {
        // OPTIMIZE: how hard is it to deal with inclusion Masks in the backends?
        auto proj = std::static_pointer_cast<Mask>(dagger());
        return std::static_pointer_cast<Mask>(proj->_unary_operand(func)->dagger());
    }

    auto [data_out, small] =
      backend->mask_unary_operand(std::static_pointer_cast<Mask const>(shared_from_this()), func);
    return make_mask(data_out, large_leg(), small, true, backend, labels.as_list());
}

Tensor::Ptr
Mask::copy(bool deep, std::optional<std::string> device_opt, std::optional<Dtype> dtype_opt)
{
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        // Python: as_dtype then maybe move — Mask.as_dtype only allows bool
        return as_dtype(*dtype_opt);
    }
    TensorBackend::DataPtr new_data;
    if (deep) {
        std::optional<std::string> device_arg = device_opt;
        new_data = backend->copy_data(shared_from_this(), device_arg);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(shared_from_this(), *device_opt);
    } else {
        new_data = data;
    }
    auto space_in = is_projection ? large_leg() : small_leg();
    auto space_out = is_projection ? small_leg() : large_leg();
    // domain is space_in, codomain is space_out
    space_in = std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
    space_out = std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
    return std::make_shared<Mask>(new_data,
                                  space_in,
                                  space_out,
                                  is_projection,
                                  backend,
                                  symmetry,
                                  labels.as_list(),
                                  backend->get_device_from_data(new_data));
}

Tensor::Ptr
Mask::dagger() const
{
    auto labs = labels.as_list();
    LegLabelList dual_rev;
    dual_rev.reserve(labs.size());
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_rev.push_back(_dual_leg_label(*it));
    }
    auto new_data = backend->mask_dagger(std::static_pointer_cast<Mask const>(shared_from_this()));
    return std::make_shared<Mask>(new_data,
                                  as_space(codomain->factors[0]),
                                  as_space(domain->factors[0]),
                                  !is_projection,
                                  backend,
                                  symmetry,
                                  std::move(dual_rev),
                                  device);
}

BlockBackend::Scalar
Mask::_get_item(std::vector<int64> const& idx)
{
    return backend->get_element_mask(std::static_pointer_cast<Mask const>(shared_from_this()),
                                     idx);
}

Mask::Ptr
Mask::logical_not()
{
    return orthogonal_complement();
}

void
Mask::move_to_device(std::string device_in)
{
    data = backend->move_to_device(shared_from_this(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Mask::Ptr
Mask::orthogonal_complement()
{
    return _unary_operand(adapt_block_bool_invert(backend->block_backend));
}

bool
Mask::all() const
{
    // --- hints from Python Mask.all ---
    // assuming subspace, it is enough to check that the total sector number is the same.
    // ---
    // assuming subspace, it is enough to check that the total sector number is the same.
    return sum_multiplicities(*small_leg()) == sum_multiplicities(*large_leg());
}

bool
Mask::any() const
{
    return space_dim(*small_leg()) > 0;
}

BlockBackend::BlockPtr
Mask::as_block_mask()
{
    auto res = backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this()));
    return backend->block_backend->apply_basis_perm(
      res, { as_leg_cptr(large_leg()) }, /*inv=*/true);
}

py::array
Mask::as_numpy_mask()
{
    return backend->block_backend->to_numpy(as_block_mask(), py_bool_dtype());
}

Tensor::Ptr
Mask::to_backend(TensorBackend::Ptr new_backend,
                 std::optional<Dtype> dtype_opt,
                 std::optional<std::string> device_opt)
{
    // --- hints from Python Mask.to_backend ---
    // similar to DiagonalTensor, we can just go via dense mask, with some exceptions.
    // these exceptions only occurr for FusionTreeBackend -> FusionTreeBackend, and that allows
    // a simple implementation directly
    // mask_from_block assumes projection mask -> swap block_inds for inclusion
    // ---
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }

    if (dtype_opt.has_value() && *dtype_opt != Dtype::Bool) {
        throw std::invalid_argument("Mask requires Dtype.bool");
    }

    // similar to DiagonalTensor, we can just go via dense mask, with some exceptions.
    // these exceptions only occurr for FusionTreeBackend -> FusionTreeBackend, and that allows
    // a simple implementation directly

    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<FusionTreeBackend>(backend) &&
        std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        new_data =
          backend->to_block_backend(data, new_backend->block_backend, Dtype::Bool, device_s);
    } else {
        auto old_mask =
          backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this()));
        auto new_mask =
          new_backend->block_backend->as_block(py::cast(old_mask), Dtype::Bool, device_s);
        auto [data_out, unused_small] = new_backend->mask_from_block(new_mask, large_leg());
        (void)unused_small;
        new_data = std::move(data_out);
        if (std::dynamic_pointer_cast<AbelianBackend>(new_backend) && !is_projection) {
            // mask_from_block assumes projection mask -> swap block_inds for inclusion
            auto abd = std::dynamic_pointer_cast<AbelianBackendData>(new_data);
            assert(abd);
            int64 cols[] = { 1, 0 };
            abd->block_inds = abd->block_inds.take_columns_i64(cols);
        }
    }
    return std::make_shared<Mask>(new_data,
                                  as_space(domain->factors[0]),
                                  as_space(codomain->factors[0]),
                                  is_projection,
                                  new_backend,
                                  symmetry,
                                  labels.as_list(),
                                  device_s);
}

BlockBackend::BlockPtr
Mask::to_dense_block(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                     std::optional<Dtype> dtype_opt,
                     bool understood_braiding)
{
    // --- hints from Python Mask.to_dense_block ---
    // for Mask, defining via numpy is actually easier, to use numpy indexing
    // ---
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of to_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    assert(shape.size() == 2);
    auto m = static_cast<int64>(shape[0]);
    auto n = static_cast<int64>(shape[1]);
    Dtype dt = dtype_opt.value_or(Dtype::Bool);
    auto bb = backend->block_backend;
    auto res = bb->zeros({ m, n }, dt);
    auto one = (dt == Dtype::Bool) ? bb->as_scalar(true) : bb->as_scalar(py::int_(1), dt);

    auto mask_block = as_block_mask();
    auto mask_np = bb->to_numpy(mask_block, py_bool_dtype());
    auto mask_bool = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(mask_np);
    auto info = mask_bool.request();
    auto const* ptr = static_cast<bool const*>(info.ptr);
    std::vector<int64> kept;
    for (py::ssize_t i = 0; i < info.size; ++i) {
        if (ptr[i]) {
            kept.push_back(static_cast<int64>(i));
        }
    }
    if (is_projection) {
        if (static_cast<int64>(kept.size()) != m) {
            throw std::logic_error("Mask.to_dense_block: projection mask length mismatch");
        }
        for (int64 i = 0; i < m; ++i) {
            res->set_item(std::vector<int64>{ i, kept[static_cast<std::size_t>(i)] }, one);
        }
    } else {
        if (static_cast<int64>(kept.size()) != n) {
            throw std::logic_error("Mask.to_dense_block: inclusion mask length mismatch");
        }
        for (int64 j = 0; j < n; ++j) {
            res->set_item(std::vector<int64>{ kept[static_cast<std::size_t>(j)], j }, one);
        }
    }
    if (leg_order.has_value()) {
        auto idcs = labels.indices(*leg_order);
        res = bb->permute_axes(res, idcs);
    }
    return res;
}

py::array
Mask::to_numpy(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
               py::object numpy_dtype,
               bool understood_braiding)
{
    // --- hints from Python Mask.to_numpy ---
    // sets the appropriate dtype. e.g. sets ``True`` for bool.
    // ---
    std::optional<Dtype> dt;
    if (!numpy_dtype.is_none()) {
        dt = dtype::from_numpy_dtype(numpy_dtype);
    }
    auto block = to_dense_block(std::move(leg_order), dt, understood_braiding);
    std::optional<py::object> np_dt =
      numpy_dtype.is_none() ? std::nullopt : std::optional<py::object>{ numpy_dtype };
    return backend->block_backend->to_numpy(block, np_dt).cast<py::array>();
}

void
Mask::save_hdf5(lasyten::hdf5::Saver& saver,
                HighFive::Group& h5gr,
                std::string const& subpath) const
{
    hdf5_export::save_tensor_product(saver, subpath + "domain", domain);
    hdf5_export::save_tensor_product(saver, subpath + "codomain", codomain);
    hdf5_export::save_tensor_backend(saver, subpath + "backend", backend);
    hdf5_export::save_tensor_backend_data(saver, subpath + "data", data);
    hdf5_export::save_symmetry(saver, subpath + "symmetry", symmetry);
    hdf5_io::h5_set_attr(h5gr.getId(), "dtype", dtype::repr(dtype));
    hdf5_io::h5_set_attr(h5gr.getId(), "num_legs", static_cast<std::int64_t>(num_legs));
    hdf5_export::save_f64_vector(saver, subpath + "shape", shape);
    hdf5_io::h5_set_attr(h5gr.getId(), "is_projection", is_projection);
    hdf5_export::save_leg_labels(saver, subpath + "labels", labels.as_list());
}

Mask::Ptr
Mask::from_hdf5(lasyten::hdf5::Loader& loader, HighFive::Group& h5gr, std::string const& subpath)
{
    auto domain_tp = hdf5_export::load_tensor_product(loader, subpath + "domain");
    auto codomain_tp = hdf5_export::load_tensor_product(loader, subpath + "codomain");
    auto symmetry_in = hdf5_export::load_symmetry(loader, subpath + "symmetry");
    auto backend_in = hdf5_export::load_tensor_backend(loader, subpath + "backend");
    auto data_in = hdf5_export::load_tensor_backend_data(loader, subpath + "data");
    auto shape_in = hdf5_export::load_f64_vector(loader, subpath + "shape");

    bool proj = true;
    auto proj_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "is_projection");
    if (proj_attr.has_value()) {
        proj = *proj_attr != 0;
    } else {
        auto space_in = as_space(domain_tp->factors[0]);
        auto space_out = as_space(codomain_tp->factors[0]);
        proj = space_dim(*space_in) >= space_dim(*space_out);
    }

    LegLabelList labels_in(2, "");
    if (hdf5_io::h5_contains(loader.root(), subpath + "labels")) {
        labels_in = hdf5_export::load_leg_labels(loader, subpath + "labels");
        if (labels_in.empty()) {
            labels_in.assign(2, "");
        }
    }

    auto device_in = backend_in->get_device_from_data(data_in);
    auto obj = std::make_shared<Mask>(data_in,
                                      as_space(domain_tp->factors[0]),
                                      as_space(codomain_tp->factors[0]),
                                      proj,
                                      backend_in,
                                      symmetry_in,
                                      std::move(labels_in),
                                      device_in);
    obj->shape = std::move(shape_in);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

} // namespace lasyten
