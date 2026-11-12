#include <lasyten/tensors/charged_tensor.h>

#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tools.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

ChargedTensor::ChargedTensor(SymmetricTensor::Ptr inv, BlockBackend::BlockPtr charged_state_in)
  : Tensor(
      inv->codomain,
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>(inv->domain->factors.begin() + 1, inv->domain->factors.end()),
        inv->symmetry),
      inv->backend,
      inv->symmetry,
      [&]() {
          auto labs = inv->labels.as_list();
          if (labs.empty()) {
              throw std::invalid_argument(
                "ChargedTensor invariant_part must have a charge-leg label");
          }
          return LegLabelList(labs.begin(), labs.end() - 1);
      }(),
      inv->dtype,
      inv->device)
  , invariant_part(std::move(inv))
  , charged_state(std::move(charged_state_in))
  // Match Python: keep the domain factor as-is (ElementarySpace or LegPipe).
  , charge_leg(invariant_part->domain->factors[0])
{
    if (invariant_part->domain->num_factors <= 0) {
        throw std::invalid_argument(
          "ChargedTensor invariant_part must have a charge leg in the domain");
    }
    auto labs = invariant_part->labels.as_list();
    if (labs.empty() || labs.back() != _CHARGE_LEG_LABEL) {
        throw std::invalid_argument(
          std::format("ChargedTensor invariant_part last label must be '{}'", _CHARGE_LEG_LABEL));
    }
    if (!supports_symmetry(invariant_part->symmetry)) {
        throw SymmetryError(std::format("ChargedTensor is not well-defined for symmetry {}.",
                                        invariant_part->symmetry->repr()));
    }
    if (!charged_state) {
        throw std::invalid_argument(
          "ChargedTensor requires a charged_state. Use HiddenLegTensor to hide legs without a "
          "state.");
    }
    if (!invariant_part->symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("charged_state can not be specified for symmetry {}",
                                        invariant_part->symmetry->repr()));
    }
    charged_state = invariant_part->backend->block_backend->as_block(
      py::cast(charged_state), invariant_part->dtype, invariant_part->device);
    invariant_part->allow_charge_leg_label = true;
    reject_exclamation_in_labels(labels.as_list(), "ChargedTensor");

    labels.set_hooks(
      [this](LegLabels& /*self*/, LegLabelList const& proposed) {
          reject_exclamation_in_labels(proposed, "ChargedTensor");
      },
      [this](LegLabels& self) {
          auto inv_labs = self.as_list();
          inv_labs.emplace_back(_CHARGE_LEG_LABEL);
          invariant_part->allow_charge_leg_label = true;
          invariant_part->labels.set(std::move(inv_labs));
      });
}

void
ChargedTensor::test_sanity() const
{
    Tensor::test_sanity();
    auto inv_labs = invariant_part->labels.as_list();
    assert(labels.as_list() == LegLabelList(inv_labs.begin(), inv_labs.end() - 1));
    invariant_part->allow_charge_leg_label = true;
    invariant_part->test_sanity();
    assert(invariant_part->device == device);
    assert(charged_state);
    backend->block_backend->test_block_sanity(
      charged_state,
      std::vector<int64>{ static_cast<int64>(charge_leg->dim) },
      std::nullopt,
      device);
    reject_exclamation_in_labels(labels.as_list(), "ChargedTensor");
}

std::string
ChargedTensor::ascii_diagram_type_name() const
{
    return "Chrg";
}

std::string
ChargedTensor::class_name() const
{
    return "ChargedTensor";
}

bool
ChargedTensor::supports_symmetry(Symmetry::Ptr const& symmetry)
{
    // charged_state is always required and needs a droppable symmetry.
    return symmetry->can_be_dropped() && symmetry->has_symmetric_braid();
}

std::tuple<TensorProduct::Ptr, Space::Ptr>
ChargedTensor::_parse_inv_domain(TensorProduct::Ptr domain,
                                 std::variant<ElementarySpace::Ptr, Sector> charge)
{
    if (!domain) {
        throw std::invalid_argument("domain must be parsed before constructing ChargedTensor");
    }
    Space::Ptr charge_leg_sp;
    if (auto const* es = std::get_if<ElementarySpace::Ptr>(&charge)) {
        charge_leg_sp = *es;
    } else {
        Sector const& sec = std::get<Sector>(charge);
        charge_leg_sp =
          std::make_shared<ElementarySpace>(domain->symmetry, SectorArray::from_sector(sec));
    }
    return { domain->left_multiply(std::dynamic_pointer_cast<Leg>(charge_leg_sp)), charge_leg_sp };
}

std::tuple<LegLabelList, LegLabelList>
ChargedTensor::_parse_inv_labels(std::optional<LegLabelList> labels,
                                 TensorProduct::Ptr const& codomain,
                                 TensorProduct::Ptr const& domain)
{
    auto labs = _init_parse_labels(std::move(labels), codomain, domain);
    auto inv_labels = labs;
    inv_labels.emplace_back(std::string(_CHARGE_LEG_LABEL));
    return { labs, inv_labels };
}

ChargedTensor::Ptr
ChargedTensor::from_block_func(BlockFactoryFn func,
                               std::variant<ElementarySpace::Ptr, Sector> charge,
                               TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain,
                               BlockBackend::BlockPtr charged_state,
                               TensorBackend::Ptr backend,
                               std::optional<LegLabelList> labels,
                               std::optional<Dtype> dtype,
                               std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (!charged_state) {
            device_s = backend_tp->block_backend->default_device;
        } else {
            device_s = backend_tp->block_backend->get_device(charged_state);
        }
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, std::move(charge));
    (void)charge_leg_sp;
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    auto data = backend_tp->from_sector_block_func(
      [func = std::move(func)](std::vector<int64> const& shape, Sector const& /*coupled*/) {
          return func(shape);
      },
      codomain_tp,
      inv_domain);
    auto inv = std::make_shared<SymmetricTensor>(
      data, codomain_tp, inv_domain, backend_tp, symmetry, std::move(inv_labels));
    inv->allow_charge_leg_label = true;
    inv->test_sanity();
    return std::make_shared<ChargedTensor>(inv, charged_state);
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block(BlockBackend::BlockPtr block,
                                TensorProduct::Ptr codomain,
                                TensorProduct::Ptr domain,
                                std::optional<std::variant<ElementarySpace::Ptr, Sector>> charge,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabelList> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device,
                                float64 tol,
                                bool understood_braiding)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto block_ptr = backend_tp->block_backend->as_block(py::cast(block), dtype, device);
    if (!charge.has_value()) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge=None");
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, *charge);
    if (charge_leg_sp->Space::dim != 1.) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge_leg.dim != 1");
    }
    auto inv_part =
      SymmetricTensor::from_dense_block(backend_tp->block_backend->add_axis(block_ptr, -1),
                                        codomain_tp,
                                        inv_domain,
                                        backend_tp,
                                        inv_labels,
                                        std::nullopt,
                                        std::nullopt,
                                        tol,
                                        understood_braiding);
    auto cs = backend_tp->block_backend->as_block(
      py::cast(std::vector<int64>{ 1 }), inv_part->dtype, inv_part->device);
    return std::make_shared<ChargedTensor>(inv_part, cs);
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                              Leg::Ptr space,
                                              Sector sector,
                                              TensorBackend::Ptr backend,
                                              std::optional<std::string> label,
                                              std::optional<std::string> device)
{
    if (!space) {
        throw std::invalid_argument("space must be specified");
    }
    if (!backend) {
        backend = get_backend(space->symmetry);
    }
    if (space->symmetry->qdim(sector) > 1.) {
        throw NotImplemented("from_dense_block_single_sector does not support higher-dim sectors");
    }
    auto charge_leg =
      std::make_shared<ElementarySpace>(space->symmetry, SectorArray::from_sector(sector));
    auto space_as_space = as_space(space);
    auto data =
      backend->inv_part_from_dense_block_single_sector(vector, space_as_space, charge_leg);
    auto codomain =
      std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ space }, space->symmetry);
    auto inv_domain =
      std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ charge_leg }, space->symmetry);
    LegLabelList inv_labels{ label.value_or(""), std::string(_CHARGE_LEG_LABEL) };
    auto inv_part = std::make_shared<SymmetricTensor>(
      data, codomain, inv_domain, backend, space->symmetry, std::move(inv_labels));
    auto charged_state =
      backend->block_backend->as_block(py::cast(std::vector<int64>{ 1 }), inv_part->dtype, device);
    return std::make_shared<ChargedTensor>(inv_part, charged_state);
}

std::variant<ChargedTensor::Ptr, BlockBackend::Scalar>
ChargedTensor::from_invariant_part(SymmetricTensor::Ptr inv, BlockBackend::BlockPtr charged_state)
{
    // --- hints from Python ChargedTensor.from_invariant_part ---
    // OPTIMIZE ?
    // ---
    if (!inv) {
        throw std::invalid_argument("invariant_part must be specified");
    }
    if (!charged_state) {
        throw std::invalid_argument("ChargedTensor.from_invariant_part requires a charged_state. "
                                    "Use HiddenLegTensor to hide legs without a state.");
    }
    if (inv->num_legs == 1) {
        // OPTIMIZE ?
        auto inv_block =
          inv->to_dense_block(std::nullopt, std::nullopt, /*understood_braiding=*/true);
        auto state =
          inv->backend->block_backend->as_block(py::cast(charged_state), inv->dtype, inv->device);
        return inv->backend->block_backend->inner(inv_block, state, /*do_dagger=*/false);
    }
    inv->allow_charge_leg_label = true;
    return std::make_shared<ChargedTensor>(std::move(inv), std::move(charged_state));
}

std::variant<ChargedTensor::Ptr, BlockBackend::Scalar>
ChargedTensor::from_two_charge_legs(SymmetricTensor::Ptr invariant_part,
                                    BlockBackend::BlockPtr state1,
                                    BlockBackend::BlockPtr state2)
{
    // Uses combine_legs free function.
    auto labs = invariant_part->labels.as_list();
    if (labs.size() < 2) {
        throw std::invalid_argument("from_two_charge_legs requires at least two labels");
    }
    if (!labs[labs.size() - 1].starts_with(_CHARGE_LEG_LABEL) ||
        !labs[labs.size() - 2].starts_with(_CHARGE_LEG_LABEL)) {
        throw std::invalid_argument(
          std::format("from_two_charge_legs requires the last two labels to start with '{}'",
                      _CHARGE_LEG_LABEL));
    }
    if (!state1 || !state2) {
        throw std::invalid_argument("from_two_charge_legs requires both charged states. "
                                    "Use HiddenLegTensor to hide legs without a state.");
    }
    invariant_part->allow_charge_leg_label = true;
    auto combined = combine_legs(invariant_part, { { int64{ -2 }, int64{ -1 } } });
    auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(combined);
    if (!inv_sym) {
        throw std::runtime_error(
          "from_two_charge_legs: expected SymmetricTensor after combine_legs");
    }
    inv_sym->labels[-1] = _CHARGE_LEG_LABEL;
    auto pipe = std::dynamic_pointer_cast<LegPipe>(inv_sym->domain->factors[0]);
    if (!pipe) {
        throw std::runtime_error("from_two_charge_legs: expected LegPipe charge leg");
    }
    auto state = invariant_part->backend->state_tensor_product(state1, state2, pipe)
                   .cast<BlockBackend::BlockPtr>();
    return from_invariant_part(inv_sym, state);
}

ChargedTensor::Ptr
ChargedTensor::from_zero(TensorProduct::Ptr codomain,
                         TensorProduct::Ptr domain,
                         std::variant<ElementarySpace::Ptr, Sector> charge,
                         BlockBackend::BlockPtr charged_state,
                         TensorBackend::Ptr backend,
                         std::optional<LegLabelList> labels,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (!charged_state) {
            throw std::invalid_argument("ChargedTensor.from_zero requires a charged_state");
        }
        device_s = backend_tp->block_backend->get_device(charged_state);
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, std::move(charge));
    (void)charge_leg_sp;
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    auto inv_part =
      SymmetricTensor::from_zero(codomain_tp, inv_domain, backend_tp, inv_labels, dtype, device_s);
    inv_part->allow_charge_leg_label = true;
    return std::make_shared<ChargedTensor>(inv_part, charged_state);
}

Tensor::Ptr
ChargedTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(invariant_part->as_dtype(new_dtype));
    assert(inv);
    return std::make_shared<ChargedTensor>(inv, charged_state);
}

SymmetricTensorPtr
ChargedTensor::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    // LegPipe charge legs (from combine_legs) expose Space APIs via as_Space().
    Space::Ptr charge_space = as_space(charge_leg);
    if (charge_space->sector_decomposition.size() != 1 ||
        charge_space->sector_decomposition[0] != symmetry->trivial_sector) {
        throw SymmetryError("Not a symmetric tensor");
    }
    if (charge_leg->dim == 1.) {
        auto squeezed = squeeze_legs(invariant_part, std::vector<LegRef>{ int64{ -1 } });
        auto scale = backend->block_backend->item(charged_state);
        auto scaled = std::dynamic_pointer_cast<SymmetricTensor>(squeezed->scaled(scale));
        if (!scaled) {
            throw std::runtime_error("as_SymmetricTensor: expected SymmetricTensor after scaling");
        }
        return scaled;
    }
    // charge_leg.dual (Python wrote charged_state.dual — treat as charge_leg.dual)
    auto dual = charge_leg->dual_leg();
    auto state_codomain = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ dual });
    auto state = SymmetricTensor::from_dense_block(
      charged_state,
      state_codomain,
      nullptr,
      backend,
      LegLabelList{ _dual_leg_label(std::string(_CHARGE_LEG_LABEL)) },
      dtype,
      std::nullopt,
      1e-6,
      /*understood_braiding=*/true);
    state->allow_charge_leg_label = true;
    auto contracted = tdot(state,
                           invariant_part,
                           std::vector<LegRef>{ int64{ 0 } },
                           std::vector<LegRef>{ int64{ -1 } });
    auto* tens = std::get_if<TensorPtr>(&contracted);
    if (!tens) {
        throw std::runtime_error("as_SymmetricTensor: unexpected scalar from tdot");
    }
    auto bent = bend_legs(*tens, num_codomain_legs());
    auto out = std::dynamic_pointer_cast<SymmetricTensor>(bent);
    if (!out) {
        throw std::runtime_error("as_SymmetricTensor: expected SymmetricTensor after bend_legs");
    }
    return out;
}

Tensor::Ptr
ChargedTensor::copy(bool deep,
                    std::optional<std::string> device_opt,
                    std::optional<Dtype> dtype_opt)
{
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(
      invariant_part->copy(deep, device_opt, dtype_opt));
    assert(inv);
    inv->allow_charge_leg_label = true;
    BlockBackend::BlockPtr cs = charged_state;
    if ((device_opt.has_value() && !deep) || (dtype_opt.has_value() && *dtype_opt != dtype)) {
        cs = backend->block_backend->as_block(py::cast(cs), dtype_opt, device_opt);
    }
    if (deep) {
        cs = backend->block_backend->copy_block(cs, device_opt);
    }
    return std::make_shared<ChargedTensor>(inv, cs);
}

Tensor::Ptr
ChargedTensor::dagger() const
{
    // Match free-function dagger(ChargedTensor); dagger the invariant part in C++.
    auto labs = invariant_part->labels.as_list();
    LegLabelList dual_rev;
    dual_rev.reserve(labs.size());
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_rev.push_back(_dual_leg_label(*it));
    }
    auto inv_data = backend->dagger(invariant_part);
    auto inv_sym = std::make_shared<SymmetricTensor>(inv_data,
                                                     invariant_part->domain,
                                                     invariant_part->codomain,
                                                     backend,
                                                     symmetry,
                                                     std::move(dual_rev));
    inv_sym->labels[0] = _CHARGE_LEG_LABEL;
    auto moved = move_leg(inv_sym, 0, std::nullopt, 0, std::nullopt, BendRight{ true });
    auto moved_sym = std::dynamic_pointer_cast<SymmetricTensor>(moved);
    if (!moved_sym) {
        throw std::runtime_error("ChargedTensor::dagger: expected SymmetricTensor after move_leg");
    }
    BlockBackend::BlockPtr cs = backend->block_backend->conj(charged_state);
    return std::get<ChargedTensor::Ptr>(from_invariant_part(moved_sym, cs));
}

BlockBackend::Scalar
ChargedTensor::_get_item(std::vector<int64> const& idx)
{
    // --- hints from Python ChargedTensor._get_item ---
    // should do sth smarter...
    // ---
    if (charged_state->shape()[0] > 10) {
        throw NotImplemented("ChargedTensor::_get_item for large charged_state");
    }
    auto bb = backend->block_backend;
    auto acc = bb->as_scalar(dtype::zero_scalar(dtype), dtype);
    int64 n_charge = charged_state->shape()[0];
    for (int64 n = 0; n < n_charge; ++n) {
        auto a = charged_state->get_item(n);
        std::vector<int64> inv_idx = idx;
        inv_idx.push_back(n);
        auto term = a * invariant_part->_get_item(inv_idx);
        acc = acc + term;
    }
    return acc;
}

void
ChargedTensor::move_to_device(std::string device_in)
{
    invariant_part->move_to_device(device_in);
    device = invariant_part->device;
    charged_state =
      backend->block_backend->as_block(py::cast(charged_state), std::nullopt, device);
}

std::vector<std::string>
ChargedTensor::_repr_header_lines(std::string const& indent, bool use_symm_str) const
{
    auto linewidth = get_config().print_linewidth;
    auto lines = Tensor::_repr_header_lines(indent, use_symm_str);
    Space::Ptr charge_space = as_space(charge_leg);
    lines.push_back(
      std::format("{}* Charge Leg: dim={} sectors={}",
                  indent,
                  std::round(charge_leg->dim * 1000.) / 1000.,
                  py::str(py::cast(charge_space->sector_decomposition)).cast<std::string>()));
    std::string start = indent + "* Charged State: ";
    auto state_lines = backend->block_backend->_block_repr_lines(
      charged_state, indent + "  ", linewidth - static_cast<int64>(start.size()), 1);
    lines.push_back(start + state_lines[0]);
    return lines;
}

Tensor::Ptr
ChargedTensor::to_backend(TensorBackend::Ptr new_backend,
                          std::optional<Dtype> dtype_opt,
                          std::optional<std::string> device_opt)
{
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }
    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });

    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(
      invariant_part->to_backend(new_backend, dtype_opt, device_s));
    assert(inv);
    BlockBackend::BlockPtr cs;
    if (charged_state) {
        cs = new_backend->block_backend->as_block(py::cast(charged_state), dtype_opt, device_s);
    }
    return std::make_shared<ChargedTensor>(inv, cs);
}

BlockBackend::BlockPtr
ChargedTensor::to_dense_block(
  std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
  std::optional<Dtype> dtype_opt,
  bool understood_braiding)
{
    auto inv_block = invariant_part->to_dense_block(std::nullopt, dtype_opt, understood_braiding);
    auto block = backend->block_backend->tdot(inv_block, charged_state, { -1 }, { 0 });
    if (dtype_opt.has_value()) {
        block = backend->block_backend->to_dtype(block, *dtype_opt);
    }
    if (leg_order.has_value()) {
        block = backend->block_backend->permute_axes(block, labels.indices(*leg_order));
    }
    return block;
}

BlockBackend::BlockPtr
ChargedTensor::to_dense_block_single_sector()
{
    if (num_legs > 1) {
        throw std::invalid_argument("Expected a single leg");
    }
    Space::Ptr charge_space = as_space(charge_leg);
    if (charge_space->num_sectors != 1 || charge_space->multiplicities[0] != 1) {
        throw std::invalid_argument("Not a single sector.");
    }
    auto sector_dims = charge_space->sector_dims;
    if (sector_dims.has_value() && (*sector_dims)[0] > 1) {
        throw NotImplemented("to_dense_block_single_sector does not support higher-dim sectors");
    }
    auto block = backend->inv_part_to_dense_block_single_sector(invariant_part);
    return backend->block_backend->item(charged_state) * *block;
}

void
ChargedTensor::save_hdf5(lasyten::hdf5::Saver& saver,
                         HighFive::Group& h5gr,
                         std::string const& subpath) const
{
    saver.save_instance(subpath + "invariant_part",
                        hdf5_export::kModule,
                        "SymmetricTensor",
                        invariant_part.get(),
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            invariant_part->save_hdf5(s, g, sub);
                        });
    hdf5_export::save_block(saver, subpath + "charged_state", charged_state);
    hdf5_io::h5_set_attr(h5gr.getId(), "has_charged_state", true);
    hdf5_io::h5_set_attr(h5gr.getId(), "dtype", dtype::repr(dtype));
    hdf5_io::h5_set_attr(h5gr.getId(), "num_legs", static_cast<std::int64_t>(num_legs));
    hdf5_export::save_f64_vector(saver, subpath + "shape", shape);
    hdf5_export::save_leg_labels(saver, subpath + "labels", labels.as_list());
}

ChargedTensor::Ptr
ChargedTensor::from_hdf5(lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string const& subpath)
{
    auto inv = hdf5_export::load_instance<SymmetricTensor>(
      loader, subpath + "invariant_part", [&](HighFive::Group& g, std::string const& sub) {
          return SymmetricTensor::from_hdf5(loader, g, sub);
      });
    inv->allow_charge_leg_label = true;
    auto has_cs_attr = hdf5_io::h5_get_attr_int64(h5gr.getId(), "has_charged_state");
    bool has_cs = !has_cs_attr.has_value() || *has_cs_attr != 0;
    if (!has_cs) {
        throw std::invalid_argument(
          "HDF5 ChargedTensor without charged_state is no longer supported. "
          "Re-save or convert to HiddenLegTensor.");
    }
    auto cs = hdf5_export::load_block(loader, subpath + "charged_state");
    auto obj = std::make_shared<ChargedTensor>(inv, cs);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

} // namespace lasyten
