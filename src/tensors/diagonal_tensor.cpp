#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/block_backend/torch.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/tools.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <format>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

Space::Ptr
as_space_leg(Space::Ptr leg)
{
    if (!leg) {
        throw std::invalid_argument("Must specify the leg.");
    }
    if (std::dynamic_pointer_cast<LegPipe>(leg)) {
        throw std::invalid_argument("DiagonalTensor is not defined on LegPipes.");
    }
    return leg;
}

TensorProduct::Ptr
product_of_leg(Space::Ptr const& leg)
{
    auto sp = as_space_leg(leg);
    return std::make_shared<TensorProduct>(
      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(sp) });
}

BlockBackend::LegCPtr
as_leg_cptr(Space::Ptr const& leg)
{
    return std::dynamic_pointer_cast<Leg const>(as_space_leg(leg));
}

} // namespace

std::vector<Dtype> DiagonalTensor::_forbidden_dtypes = {};

DiagonalTensor::DiagonalTensor(TensorBackend::DataPtr data_in,
                               Space::Ptr leg_in,
                               TensorBackend::Ptr backend_in,
                               Symmetry::Ptr symmetry_in,
                               LegLabelList labels_in)
  : SymmetricTensor(std::move(data_in),
                    std::make_shared<TensorProduct>(
                      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                    std::make_shared<TensorProduct>(
                      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                    std::move(backend_in),
                    std::move(symmetry_in),
                    std::move(labels_in),
                    /*check_complex_dtype=*/false)
{
    if (std::dynamic_pointer_cast<LegPipe>(leg_in)) {
        throw std::invalid_argument("DiagonalTensor is not defined on LegPipes.");
    }
}

std::vector<Dtype> const&
DiagonalTensor::forbidden_dtypes() const
{
    return _forbidden_dtypes;
}

void
DiagonalTensor::test_sanity() const
{
    SymmetricTensor::test_sanity();
    assert(domain->operator==(*codomain));
    assert(domain->num_factors == 1);
}

void
DiagonalTensor::verify_dtype() const
{
    // --- hints from Python DiagonalTensor.verify_dtype ---
    // for diagonal tensors, we always allow real dtypes
    // ---
    // for diagonal tensors, we always allow real dtypes
}

std::string
DiagonalTensor::ascii_diagram_type_name() const
{
    return "Diag";
}

std::string
DiagonalTensor::class_name() const
{
    return "DiagonalTensor";
}

Space::Ptr
DiagonalTensor::leg() const
{
    return as_space(codomain->factors[0]);
}

DiagonalTensor::Ptr
DiagonalTensor::from_block_func(BlockFactoryFn func,
                                Space::Ptr leg,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabelList> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device)
{
    // --- hints from Python DiagonalTensor.from_block_func ---
    // use same backend function as from_sector_block_func, so we include the coupled arg
    // but just ignore it.
    // ---
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    (void)dtype;
    (void)device;

    SectorBlockFactoryFn block_func = [func](std::vector<int64> const& shape,
                                             Sector const& /*coupled*/) { return func(shape); };

    auto data = backend_tp->diagonal_from_sector_block_func(block_func, co_domain);
    auto res = std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
    res->test_sanity();
    return res;
}

DiagonalTensor::Ptr
DiagonalTensor::from_dense_block(BlockBackend::BlockPtr block,
                                 Space::Ptr leg,
                                 TensorBackend::Ptr backend,
                                 std::optional<LegLabelList> labels,
                                 std::optional<Dtype> dtype,
                                 float64 tol,
                                 std::optional<std::string> device,
                                 bool understood_braiding)
{
    auto leg_sp = as_space_leg(std::move(leg));
    if (!leg_sp->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}",
                      leg_sp->symmetry->repr()));
    }
    if (!leg_sp->symmetry->has_symmetric_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of from_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    if (!backend) {
        backend = get_backend(leg_sp->symmetry);
    }
    auto block_ptr = backend->block_backend->as_block(py::cast(block), dtype, device);
    auto diag = backend->block_backend->get_diagonal(block_ptr, 1e-10);
    return from_diag_block(diag, leg_sp, backend, std::move(labels), dtype, device, tol);
}

DiagonalTensor::Ptr
DiagonalTensor::from_diag_block(BlockBackend::BlockPtr diag,
                                Space::Ptr leg,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabelList> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device,
                                float64 tol)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    auto diag_ptr = backend_tp->block_backend->as_block(py::cast(diag), dtype, device);
    diag_ptr = backend_tp->block_backend->apply_basis_perm(diag_ptr, { as_leg_cptr(leg_sp) });
    auto data = backend_tp->diagonal_from_block(diag_ptr, co_domain, tol);
    return std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
}

DiagonalTensor::Ptr
DiagonalTensor::from_eye(Space::Ptr leg,
                         TensorBackend::Ptr backend,
                         std::optional<LegLabelList> labels,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)co_domain;
    (void)unused_domain;
    (void)symmetry;
    auto bb = backend_tp->block_backend;
    BlockFactoryFn ones = [bb, dtype, device](std::vector<int64> const& shape) {
        return bb->ones_block(shape, dtype, device);
    };
    return from_block_func(
      std::move(ones), std::move(leg_sp), std::move(backend_tp), std::move(labels), dtype, device);
}

DiagonalTensor::Ptr
DiagonalTensor::from_random_normal(Space::Ptr leg,
                                   TensorCPtr mean,
                                   float64 sigma,
                                   TensorBackend::Ptr backend,
                                   std::optional<LegLabelList> labels,
                                   Dtype dtype,
                                   std::optional<std::string> device)
{
    if (!dtype::is_complex(dtype)) {
        throw std::invalid_argument("from_random_normal requires a complex dtype");
    }
    if (!(sigma > 0.0)) {
        throw std::invalid_argument(std::format("sigma must be positive, got {}", sigma));
    }
    if (mean) {
        auto mean_diag = std::dynamic_pointer_cast<DiagonalTensor const>(mean);
        Space::Ptr mean_leg = mean_diag ? mean_diag->leg() : as_space(mean->codomain->factors[0]);
        if (!leg) {
            leg = mean_leg;
        } else {
            if (!(*mean_leg == *leg)) {
                throw std::invalid_argument("mean.leg must match the given leg");
            }
        }
        if (!backend) {
            backend = mean->backend;
        } else {
            if (mean->backend != backend) {
                throw std::invalid_argument("mean.backend must match the given backend");
            }
        }
        if (!labels.has_value()) {
            labels = mean->labels.as_list();
        }
        if (!device.has_value()) {
            device = mean->device;
        }
    } else {
        if (!leg) {
            throw std::invalid_argument("Must specify the leg if mean is not given.");
        }
        auto tp = product_of_leg(leg);
        auto [co_domain, unused_domain, backend_tp, symmetry] =
          _init_parse_args(tp, tp, std::move(backend));
        (void)co_domain;
        (void)unused_domain;
        (void)symmetry;
        backend = std::move(backend_tp);
        if (!device.has_value()) {
            device = backend->block_backend->default_device;
        }
    }

    auto bb = backend->block_backend;
    BlockFactoryFn randn = [bb, dtype, sigma](std::vector<int64> const& shape) {
        return bb->random_normal(shape, dtype, sigma, std::nullopt);
    };
    auto with_zero_mean =
      from_block_func(std::move(randn), leg, backend, std::move(labels), dtype, device);

    if (mean) {
        auto one = backend->block_backend->as_scalar(1.0);
        auto new_data = backend->linear_combination(one, mean, one, with_zero_mean);
        return std::make_shared<DiagonalTensor>(new_data,
                                                with_zero_mean->leg(),
                                                backend,
                                                with_zero_mean->symmetry,
                                                with_zero_mean->labels.as_list());
    }
    return with_zero_mean;
}

DiagonalTensor::Ptr
DiagonalTensor::from_random_uniform(Space::Ptr leg,
                                    TensorBackend::Ptr backend,
                                    std::optional<LegLabelList> labels,
                                    Dtype dtype,
                                    std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)co_domain;
    (void)unused_domain;
    (void)symmetry;
    auto bb = backend_tp->block_backend;
    BlockFactoryFn func = [bb, dtype, device](std::vector<int64> const& shape) {
        return bb->random_uniform(shape, dtype, device);
    };
    return from_block_func(
      std::move(func), std::move(leg_sp), std::move(backend_tp), std::move(labels), dtype, device);
}

DiagonalTensor::Ptr
DiagonalTensor::from_sector_block_func(SectorBlockFactoryFn func,
                                       Space::Ptr leg,
                                       TensorBackend::Ptr backend,
                                       std::optional<LegLabelList> labels,
                                       std::optional<Dtype> dtype,
                                       std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, unused_symm] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)unused_symm;
    (void)dtype;
    (void)device;

    auto data = backend_tp->diagonal_from_sector_block_func(std::move(func), co_domain);
    auto res = std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
    res->test_sanity();
    return res;
}

DiagonalTensor::Ptr
DiagonalTensor::from_tensor(SymmetricTensorCPtr tens, std::optional<float64> tol)
{
    if (tens->num_legs != 2) {
        throw std::invalid_argument("from_tensor requires a two-leg tensor");
    }
    if (!(tens->domain->operator==(*tens->codomain))) {
        throw std::invalid_argument("from_tensor requires matching domain and codomain");
    }
    auto data = tens->backend->diagonal_tensor_from_full_tensor(tens, tol);
    auto leg = as_space(tens->codomain->factors[0]);
    return std::make_shared<DiagonalTensor>(
      data, leg, tens->backend, tens->symmetry, tens->labels.as_list());
}

DiagonalTensor::Ptr
DiagonalTensor::from_zero(Space::Ptr leg,
                          TensorBackend::Ptr backend,
                          std::optional<LegLabelList> labels,
                          Dtype dtype,
                          std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->zero_diagonal_data(co_domain, dtype, device_s);
    return std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
}

Tensor::Ptr
DiagonalTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto new_data = backend->to_dtype(shared_from_this(), new_dtype);
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels.as_list());
}

SymmetricTensorPtr
DiagonalTensor::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    auto new_data = backend->full_data_from_diagonal_tensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels.as_list());
}

DiagonalTensor::Ptr
DiagonalTensor::as_DiagonalTensor(bool guarantee_copy, std::optional<std::string> /*warning*/)
{
    if (guarantee_copy) {
        return std::static_pointer_cast<DiagonalTensor>(copy());
    }
    return std::static_pointer_cast<DiagonalTensor>(shared_from_this());
}

DiagonalTensor::Ptr
DiagonalTensor::_binary_operand(BlockBackend::Scalar other,
                                BlockBinaryFn func,
                                std::string const& /*operand*/,
                                bool right)
{
    auto other_block = std::const_pointer_cast<BlockBackend::Block>(other._block());
    TensorBackend::DataPtr new_data;
    if (right) {
        new_data = backend->diagonal_elementwise_unary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          [func, other_block](BlockBackend::BlockPtr const& block) {
              return func(other_block, block);
          },
          /*maps_zero_to_zero=*/false);
    } else {
        new_data = backend->diagonal_elementwise_unary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          [func, other_block](BlockBackend::BlockPtr const& block) {
              return func(block, other_block);
          },
          /*maps_zero_to_zero=*/false);
    }
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels.as_list());
}

DiagonalTensor::Ptr
DiagonalTensor::_binary_operand(DiagonalTensorCPtr other,
                                BlockBinaryFn func,
                                std::string const& /*operand*/,
                                bool right)
{
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    if (!(*leg() == *other->leg())) {
        throw std::invalid_argument("Incompatible legs!");
    }
    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    TensorBackend::DataPtr new_data;
    if (right) {
        new_data = same->diagonal_elementwise_binary(
          other,
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          func,
          /*partial_zero_is_zero=*/false);
    } else {
        new_data = same->diagonal_elementwise_binary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          other,
          func,
          /*partial_zero_is_zero=*/false);
    }
    auto out_labels = _get_matching_labels(labels.as_list(), other->labels.as_list());
    return std::make_shared<DiagonalTensor>(new_data, leg(), same, symmetry, out_labels);
}

Tensor::Ptr
DiagonalTensor::copy(bool deep,
                     std::optional<std::string> device_opt,
                     std::optional<Dtype> dtype_opt)
{
    TensorBackend::DataPtr new_data;
    // Match Python: dtype change effectively always takes as_dtype branch
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        return as_dtype(*dtype_opt);
    }
    if (deep) {
        new_data = backend->copy_data(shared_from_this(), device_opt);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(shared_from_this(), *device_opt);
    } else {
        new_data = data;
    }
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels.as_list());
}

DiagonalTensorPtr
DiagonalTensor::diagonal(bool /*check_offdiagonal*/) const
{
    return std::const_pointer_cast<DiagonalTensor>(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

BlockBackend::BlockPtr
DiagonalTensor::diagonal_as_block(std::optional<Dtype> dtype_opt)
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto res = backend->diagonal_tensor_to_block(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
    res = backend->block_backend->apply_basis_perm(res, { codomain->factors[0] }, /*inv=*/true);
    if (dtype_opt.has_value()) {
        res = backend->block_backend->to_dtype(res, *dtype_opt);
    }
    return res;
}

py::array
DiagonalTensor::diagonal_as_numpy(py::object numpy_dtype)
{
    std::optional<Dtype> dt;
    if (!numpy_dtype.is_none()) {
        dt = dtype::from_numpy_dtype(numpy_dtype);
    }
    auto block = diagonal_as_block(dt);
    std::optional<py::object> np_dt =
      numpy_dtype.is_none() ? std::nullopt : std::optional<py::object>{ numpy_dtype };
    return py::reinterpret_borrow<py::array>(backend->block_backend->to_numpy(block, np_dt));
}

DiagonalTensor::Ptr
DiagonalTensor::elementwise_almost_equal(DiagonalTensorCPtr other, float64 rtol, float64 atol)
{
    // --- hints from Python DiagonalTensor.elementwise_almost_equal ---
    // no (Scalar + Block) operation defined, so requires explicit casting
    // ---
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    auto ones = from_eye(leg(), backend, labels.as_list(), dtype::to_real(dtype), device);
    auto self_m = std::static_pointer_cast<DiagonalTensor>(shared_from_this());
    BlockBinaryFn sub = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) - (*b);
    };
    BlockBinaryFn mul = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) * (*b);
    };
    BlockBinaryFn add = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) + (*b);
    };
    BlockBinaryFn le = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) <= (*b);
    };
    auto diff = self_m->_binary_operand(other, sub, "sub");
    auto left = diff->abs();
    auto bb = backend->block_backend;
    auto atol_ones = ones->_binary_operand(bb->as_scalar(atol), mul, "mul");
    auto rtol_abs = self_m->abs()->_binary_operand(bb->as_scalar(rtol), mul, "mul");
    auto right = atol_ones->_binary_operand(rtol_abs, add, "add");
    return left->_binary_operand(right, le, "le");
}

DiagonalTensor::Ptr
DiagonalTensor::_elementwise_binary(DiagonalTensorCPtr other,
                                    BlockBinaryFn func,
                                    bool partial_zero_is_zero)
{
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    if (!(*leg() == *other->leg())) {
        throw std::invalid_argument("Incompatible legs");
    }
    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    auto data_out = same->diagonal_elementwise_binary(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      other,
      std::move(func),
      partial_zero_is_zero);
    auto labs = _get_matching_labels(labels.as_list(), other->labels.as_list());
    return std::make_shared<DiagonalTensor>(data_out, leg(), same, symmetry, labs);
}

DiagonalTensor::Ptr
DiagonalTensor::_elementwise_unary(BlockUnaryFn func, bool maps_zero_to_zero)
{
    auto data_out = backend->diagonal_elementwise_unary(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      std::move(func),
      maps_zero_to_zero);
    return std::make_shared<DiagonalTensor>(data_out, leg(), backend, symmetry, labels.as_list());
}

BlockBackend::Scalar
DiagonalTensor::_get_item(std::vector<int64> const& idx)
{
    if (idx.size() != 2) {
        throw std::invalid_argument(std::format("expected 2 indices, got {}", idx.size()));
    }
    int64 i1 = idx[0];
    int64 i2 = idx[1];
    if (i1 != i2) {
        return backend->block_backend->as_scalar(dtype::zero_scalar(dtype), dtype);
    }
    return backend->get_element_diagonal(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()), i1);
}

bool
DiagonalTensor::all() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("all is not defined for dtype {}", dtype::repr(dtype)));
    }
    return backend->diagonal_all(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

bool
DiagonalTensor::any() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("any is not defined for dtype {}", dtype::repr(dtype)));
    }
    return backend->diagonal_any(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

BlockBackend::Scalar
DiagonalTensor::max() const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument("max is only defined for real dtypes");
    }
    auto bb = backend->block_backend;
    return backend->reduce_DiagonalTensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      [bb](BlockBackend::BlockPtr const& block) { return bb->max(block); },
      [](std::vector<BlockBackend::Scalar> const& xs) {
          if (xs.empty()) {
              throw std::invalid_argument("max of empty sequence");
          }
          BlockBackend::Scalar best = xs[0];
          float64 best_v = best.as_float64();
          for (std::size_t i = 1; i < xs.size(); ++i) {
              float64 v = xs[i].as_float64();
              if (v > best_v) {
                  best = xs[i];
                  best_v = v;
              }
          }
          return best;
      });
}

BlockBackend::Scalar
DiagonalTensor::min() const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument("min is only defined for real dtypes");
    }
    auto bb = backend->block_backend;
    return backend->reduce_DiagonalTensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      [bb](BlockBackend::BlockPtr const& block) { return bb->min(block); },
      [](std::vector<BlockBackend::Scalar> const& xs) {
          if (xs.empty()) {
              throw std::invalid_argument("min of empty sequence");
          }
          BlockBackend::Scalar best = xs[0];
          float64 best_v = best.as_float64();
          for (std::size_t i = 1; i < xs.size(); ++i) {
              float64 v = xs[i].as_float64();
              if (v < best_v) {
                  best = xs[i];
                  best_v = v;
              }
          }
          return best;
      });
}

std::pair<std::shared_ptr<Mask>, BlockBackend::Scalar>
DiagonalTensor::sector_argmin(std::optional<Sector> s) const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument(
          std::format("sector_argmin is not defined for dtype {}", dtype::repr(dtype)));
    }

    auto space = leg();
    bool abelian_backend = static_cast<bool>(std::dynamic_pointer_cast<AbelianBackend>(backend));
    auto const& sectors =
      abelian_backend ? space->sector_decomposition : codomain->sector_decomposition;
    std::optional<std::size_t> restricted_sector;
    if (s) {
        auto sector_idx = abelian_backend ? space->sector_decomposition_where(*s)
                                          : codomain->sector_decomposition_where(*s);
        if (!sector_idx) {
            throw std::invalid_argument("Sector is not in the leg of the DiagonalTensor");
        }
        restricted_sector = static_cast<std::size_t>(*sector_idx);
    }

    auto bb = backend->block_backend;
    std::vector<int64> local_argmins;
    std::optional<std::size_t> winning_sector;
    int64 winning_local_argmin = -1;
    auto minimum = backend->reduce_DiagonalTensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      [bb, &local_argmins](BlockBackend::BlockPtr const& block) {
          auto idcs = bb->argmin(block);
          assert(idcs.size() == 1);
          local_argmins.push_back(idcs[0]);
          return bb->min(block);
      },
      [restricted_sector, &winning_sector, &winning_local_argmin, &local_argmins](
        std::vector<BlockBackend::Scalar> const& numbers) {
          std::optional<std::size_t> best;
          for (std::size_t i = 0; i < numbers.size(); ++i) {
              if (restricted_sector && i != *restricted_sector) {
                  continue;
              }
              if (!best || (numbers[i] < numbers[*best]).as_bool()) {
                  best = i;
              }
          }
          if (!best) {
              throw std::invalid_argument("Cannot compute sector_argmin of an empty leg");
          }
          assert(local_argmins.size() == numbers.size());
          winning_sector = best;
          winning_local_argmin = local_argmins[*best];
          return numbers[*best];
      });

    if (!winning_sector && local_argmins.size() == 1) {
        winning_sector = 0;
        winning_local_argmin = local_argmins[0];
    }
    assert(winning_sector);
    auto selected_sector = sectors[*winning_sector];
    auto selected_local_argmin = winning_local_argmin;
    auto device_s = device;
    auto mask_labels = labels.as_list();
    mask_labels.resize(1);
    mask_labels.push_back(_dual_leg_label(mask_labels[0]));
    auto diagonal = from_sector_block_func(
      [bb, selected_sector, selected_local_argmin, device_s](std::vector<int64> const& shape,
                                                             Sector const& coupled) {
          if (coupled == selected_sector) {
              py::array_t<bool> arr(static_cast<py::ssize_t>(shape[0]));
              auto buf = arr.mutable_unchecked<1>();
              for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                  buf(i) = false;
              }
              buf(static_cast<py::ssize_t>(selected_local_argmin)) = true;
              return bb->as_block(arr, Dtype::Bool, device_s);
          }
          return bb->zeros(shape, Dtype::Bool, device_s);
      },
      space,
      backend,
      mask_labels,
      Dtype::Bool,
      device);
    return { Mask::from_DiagonalTensor(diagonal), minimum };
}

int64
DiagonalTensor::argmin(std::optional<Sector> s) const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument(
          std::format("argmin is not defined for dtype {}", dtype::repr(dtype)));
    }
    auto public_diag = const_cast<DiagonalTensor*>(this)->diagonal_as_block();
    auto bb = backend->block_backend;
    if (!s) {
        auto idcs = bb->argmin(public_diag);
        assert(idcs.size() == 1);
        return idcs[0];
    }
    auto space = leg();
    auto sector_idx = space->sector_decomposition_where(*s);
    if (!sector_idx) {
        throw std::invalid_argument("Sector is not in the leg of the DiagonalTensor");
    }
    if (!space->slices) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto const& sl = (*space->slices)[static_cast<std::size_t>(*sector_idx)];
    int64 const start = sl[0];
    int64 const stop = sl[1];
    if (stop <= start) {
        throw std::invalid_argument("Cannot compute argmin of an empty sector");
    }
    auto es = std::dynamic_pointer_cast<ElementarySpace>(space);
    if (!es) {
        throw std::invalid_argument("DiagonalTensor.argmin requires an ElementarySpace leg");
    }
    auto const perm = es->basis_perm();
    std::vector<int64> pub_idcs;
    pub_idcs.reserve(static_cast<std::size_t>(stop - start));
    for (int64 i = start; i < stop; ++i) {
        pub_idcs.push_back(perm[static_cast<std::size_t>(i)]);
    }
    std::sort(pub_idcs.begin(), pub_idcs.end());
    auto gathered = (*public_diag)[{ pub_idcs }];
    auto local = bb->argmin(gathered);
    assert(local.size() == 1);
    return pub_idcs[static_cast<std::size_t>(local[0])];
}

DiagonalTensor::Ptr
DiagonalTensor::abs() const
{
    auto bb = backend->block_backend;
    return const_cast<DiagonalTensor*>(this)->_elementwise_unary(
      [bb](BlockBackend::BlockPtr const& block) { return bb->abs(block); },
      /*maps_zero_to_zero=*/true);
}

void
DiagonalTensor::move_to_device(std::string device_in)
{
    data = backend->move_to_device(shared_from_this(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
DiagonalTensor::to_backend(TensorBackend::Ptr new_backend,
                           std::optional<Dtype> dtype_opt,
                           std::optional<std::string> device_opt)
{
    // --- hints from Python DiagonalTensor.to_backend ---
    // In most cases, we can just go via a single block for the diagonal
    // exceptions:
    // - for non-abelian symmetries this is inefficient (needs to expand sectors into multiplets)
    // - for symmetries that can not be dropped, this is not possible
    // Both of these exceptions can only ocurr if both backends are FusionTreeBackend, which is
    // then also simple OPTIMIZE for abelian <-> fusion tree, this might be slightly inefficient.
    // I think the blocks should be the same already, so we could get away without first
    // concatenating all of them and them splitting them back up
    // ---
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }

    // In most cases, we can just go via a single block for the diagonal
    // exceptions:
    //   - for non-abelian symmetries this is inefficient (needs to expand sectors into multiplets)
    //   - for symmetries that can not be dropped, this is not possible
    // Both of these exceptions can only ocurr if both backends are FusionTreeBackend, which is
    // then also simple

    // OPTIMIZE
    //   for abelian <-> fusion tree, this might be slightly inefficient.
    //   I think the blocks should be the same already, so we could get away without first
    //   concatenating all of them and them splitting them back up

    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<FusionTreeBackend>(backend) &&
        std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        new_data =
          backend->to_block_backend(data, new_backend->block_backend, dtype_opt, device_s);
    } else {
        auto old_diag = backend->diagonal_tensor_to_block(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
        auto new_diag =
          new_backend->block_backend->as_block(py::cast(old_diag), dtype_opt, device_s);
        new_data = new_backend->diagonal_from_block(new_diag, codomain, 0.);
    }
    return std::make_shared<DiagonalTensor>(
      new_data, leg(), new_backend, symmetry, labels.as_list());
}

BlockBackend::BlockPtr
DiagonalTensor::to_dense_block(
  std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
  std::optional<Dtype> dtype_opt,
  bool understood_braiding)
{
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
    auto diag = diagonal_as_block(dtype_opt);
    auto res = backend->block_backend->block_from_diagonal(diag);
    if (leg_order.has_value()) {
        res = backend->block_backend->permute_axes(res, labels.indices(*leg_order));
    }
    return res;
}

void
DiagonalTensor::save_hdf5(lasyten::hdf5::Saver& saver,
                          HighFive::Group& h5gr,
                          std::string const& subpath) const
{
    /// Export DiagonalTensor to hdf5 such that it can be re-imported with from_hdf5
    SymmetricTensor::save_hdf5(saver, h5gr, subpath);
}

DiagonalTensor::Ptr
DiagonalTensor::from_hdf5(lasyten::hdf5::Loader& loader,
                          HighFive::Group& h5gr,
                          std::string const& subpath)
{
    auto domain = hdf5_export::load_tensor_product(loader, subpath + "domain");
    auto codomain = hdf5_export::load_tensor_product(loader, subpath + "codomain");
    auto symmetry = hdf5_export::load_symmetry(loader, subpath + "symmetry");
    auto backend = hdf5_export::load_tensor_backend(loader, subpath + "backend");
    auto data = hdf5_export::load_tensor_backend_data(loader, subpath + "data");
    if (hdf5_io::h5_contains(loader.root(), subpath + "device")) {
        hid_t id = loader.open(subpath + "device");
        (void)loader.load_string(id);
        H5Idec_ref(id);
    }
    auto dt = hdf5_export::load_dtype_string(loader, subpath + "dtype");
    auto labels = hdf5_export::load_leg_labels(loader, subpath + "labels");
    int64 nlegs = codomain->num_factors + domain->num_factors;
    if (labels.empty() && nlegs > 0) {
        labels.assign(static_cast<std::size_t>(nlegs), "");
    }

    auto obj = std::make_shared<DiagonalTensor>(
      data, as_space(codomain->factors[0]), backend, symmetry, std::move(labels));
    obj->dtype = dt;
    obj->device = backend->get_device_from_data(data);
    if (dynamic_cast<TorchBlockBackend*>(backend->block_backend.get()) != nullptr) {
        obj->device = TorchBlockBackend::device_for_hdf5_load(
          obj->device, backend->block_backend->default_device);
        if (auto* abd = dynamic_cast<AbelianBackendData*>(data.get())) {
            abd->device = obj->device;
        } else if (auto* ftd = dynamic_cast<FusionTreeData*>(data.get())) {
            ftd->device = obj->device;
        }
    }
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

void
Identity::unsupported_factory(char const* name)
{
    throw std::invalid_argument(std::format("{} is not supported for Identity", name));
}

Identity::Identity(Space::Ptr leg_in,
                   TensorBackend::Ptr backend_in,
                   Symmetry::Ptr symmetry_in,
                   LegLabelList labels_in,
                   Dtype dtype_in,
                   std::string device_in)
  : DiagonalTensor(
      // Do not std::move(leg_in)/backend_in in this mem-initializer list: argument
      // evaluation order is unspecified, and eye_data still needs both.
      backend_in->eye_data(std::make_shared<TensorProduct>(
                             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                           dtype_in,
                           device_in),
      leg_in,
      backend_in,
      std::move(symmetry_in),
      std::move(labels_in))
{
}

void
Identity::test_sanity() const
{
    Tensor::test_sanity();
    verify_dtype();
}

std::string
Identity::class_name() const
{
    return "Identity";
}

Identity::Ptr
Identity::from_eye(Space::Ptr leg,
                   TensorBackend::Ptr backend,
                   std::optional<LegLabelList> labels,
                   Dtype dtype,
                   std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    auto dt = _parse_default_dtype(dtype, symmetry);
    if (!dt.has_value()) {
        dt = Dtype::Float64;
    }
    std::string device_s =
      device.has_value() ? *device : backend_tp->block_backend->default_device;
    auto labs =
      _init_parse_labels(std::move(labels), co_domain, co_domain, /*is_endomorphism=*/true);
    return std::make_shared<Identity>(
      leg_sp, backend_tp, symmetry, std::move(labs), *dt, std::move(device_s));
}

Identity::Ptr
Identity::from_hdf5(lasyten::hdf5::Loader& loader,
                    HighFive::Group& h5gr,
                    std::string const& subpath)
{
    auto domain = hdf5_export::load_tensor_product(loader, subpath + "domain");
    (void)hdf5_export::load_tensor_product(loader, subpath + "codomain");
    auto symmetry = hdf5_export::load_symmetry(loader, subpath + "symmetry");
    auto backend = hdf5_export::load_tensor_backend(loader, subpath + "backend");
    (void)hdf5_export::load_tensor_backend_data(loader, subpath + "data");
    hid_t device_id = loader.open(subpath + "device");
    auto device = loader.load_string(device_id);
    H5Idec_ref(device_id);
    auto dt = hdf5_export::load_dtype_string(loader, subpath + "dtype");
    auto labels = hdf5_export::load_leg_labels(loader, subpath + "labels");
    if (labels.empty()) {
        labels.assign(2, "");
    }

    if (dynamic_cast<TorchBlockBackend*>(backend->block_backend.get()) != nullptr) {
        device =
          TorchBlockBackend::device_for_hdf5_load(device, backend->block_backend->default_device);
    } else {
        device = backend->block_backend->as_device(device);
    }

    auto obj = std::make_shared<Identity>(
      as_space(domain->factors[0]), backend, symmetry, std::move(labels), dt, std::move(device));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

Tensor::Ptr
Identity::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    return std::make_shared<Identity>(
      leg(), backend, symmetry, labels.as_list(), new_dtype, device);
}

SymmetricTensorPtr
Identity::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    return SymmetricTensor::from_eye(codomain, backend, labels.as_list(), dtype, device);
}

DiagonalTensor::Ptr
Identity::as_DiagonalTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    return DiagonalTensor::from_eye(leg(), backend, labels.as_list(), dtype, device);
}

DiagonalTensor::Ptr
Identity::_binary_operand(BlockBackend::Scalar other,
                          BlockBinaryFn func,
                          std::string const& operand,
                          bool right)
{
    return as_DiagonalTensor()->_binary_operand(other, std::move(func), operand, right);
}

DiagonalTensor::Ptr
Identity::_binary_operand(DiagonalTensorCPtr other,
                          BlockBinaryFn func,
                          std::string const& operand,
                          bool right)
{
    return as_DiagonalTensor()->_binary_operand(std::move(other), std::move(func), operand, right);
}

Tensor::Ptr
Identity::copy(bool /*deep*/,
               std::optional<std::string> /*device_opt*/,
               std::optional<Dtype> dtype_opt)
{
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        // Python: ``return self.as_dtype(dtype, device=device)`` — as_dtype ignores device.
        return as_dtype(*dtype_opt);
    }
    return shared_from_this();
}

DiagonalTensorPtr
Identity::diagonal(bool /*check_offdiagonal*/) const
{
    return const_cast<Identity*>(this)->as_DiagonalTensor();
}

BlockBackend::BlockPtr
Identity::diagonal_as_block(std::optional<Dtype> dtype_opt)
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    return backend->block_backend->ones_block(
      { static_cast<int64>(leg()->dim) }, dtype_opt.value_or(dtype), device);
}

py::array
Identity::diagonal_as_numpy(py::object numpy_dtype)
{
    if (numpy_dtype.is_none()) {
        numpy_dtype = dtype::to_numpy_dtype(dtype);
    }
    auto block =
      backend->block_backend->ones_block({ static_cast<int64>(leg()->dim) }, dtype, device);
    return backend->block_backend->to_numpy(block, numpy_dtype);
}

DiagonalTensor::Ptr
Identity::elementwise_almost_equal(DiagonalTensorCPtr other, float64 rtol, float64 atol)
{
    return as_DiagonalTensor()->elementwise_almost_equal(std::move(other), rtol, atol);
}

DiagonalTensor::Ptr
Identity::_elementwise_unary(BlockUnaryFn func, bool maps_zero_to_zero)
{
    return as_DiagonalTensor()->_elementwise_unary(std::move(func), maps_zero_to_zero);
}

DiagonalTensor::Ptr
Identity::_elementwise_binary(DiagonalTensorCPtr other,
                              BlockBinaryFn func,
                              bool partial_zero_is_zero)
{
    return as_DiagonalTensor()->_elementwise_binary(
      std::move(other), std::move(func), partial_zero_is_zero);
}

BlockBackend::Scalar
Identity::_get_item(std::vector<int64> const& idx)
{
    if (idx.size() != 2) {
        throw std::invalid_argument(std::format("expected 2 indices, got {}", idx.size()));
    }
    if (idx[0] != idx[1]) {
        return backend->block_backend->as_scalar(dtype::zero_scalar(dtype), dtype);
    }
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

bool
Identity::all() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("all is not defined for dtype {}", dtype::repr(dtype)));
    }
    return true;
}

bool
Identity::any() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("any is not defined for dtype {}", dtype::repr(dtype)));
    }
    return leg()->dim > 0;
}

BlockBackend::Scalar
Identity::max() const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument("max is only defined for real dtypes");
    }
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

BlockBackend::Scalar
Identity::min() const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument("min is only defined for real dtypes");
    }
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

int64
Identity::argmin(std::optional<Sector> /*s*/) const
{
    throw std::invalid_argument("argmin is not supported for Identity");
}

DiagonalTensor::Ptr
Identity::abs() const
{
    return std::const_pointer_cast<Identity>(
      std::static_pointer_cast<Identity const>(shared_from_this()));
}

void
Identity::move_to_device(std::string device_in)
{
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
Identity::to_backend(TensorBackend::Ptr new_backend,
                     std::optional<Dtype> dtype_opt,
                     std::optional<std::string> device_opt)
{
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }
    Dtype dt = dtype_opt.value_or(dtype);
    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    return std::make_shared<Identity>(
      leg(), new_backend, symmetry, labels.as_list(), dt, device_s);
}

BlockBackend::BlockPtr
Identity::to_dense_block(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                         std::optional<Dtype> dtype_opt,
                         bool understood_braiding)
{
    return as_DiagonalTensor()->to_dense_block(leg_order, dtype_opt, understood_braiding);
}

} // namespace lasyten
