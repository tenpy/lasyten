#include <lasyten/tensors/hidden_leg_tensor.h>

#include <lasyten/tools.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <format>
#include <hdf5_io/h5_ops.h>
#include <lasyten/tools/hdf5.h>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace lasyten {

namespace {

[[nodiscard]] bool
is_dual_pair(std::string const& a, std::string const& b)
{
    return _dual_leg_label(a) == b;
}

void
check_public_labels_no_bang(LegLabelList const& labs)
{
    for (auto const& lab : labs) {
        if (!lab.empty() && !HiddenLegTensor::is_hidden_leg_label(lab) &&
            label_contains_exclamation(lab)) {
            throw std::invalid_argument(std::format(
              "Public label '{}' must not contain '{}'", lab, HiddenLegTensor::HIDDEN_PREFIX));
        }
    }
}

} // namespace

bool
label_contains_exclamation(std::string const& label)
{
    return label.find(HiddenLegTensor::HIDDEN_PREFIX) != std::string::npos;
}

void
reject_exclamation_in_labels(LegLabelList const& labels, std::string const& context)
{
    for (auto const& lab : labels) {
        if (label_contains_exclamation(lab)) {
            throw std::invalid_argument(
              std::format("{}: leg labels must not contain '{}'; got '{}'. "
                          "Use HiddenLegTensor to hide legs, or ChargedTensor for a charge leg.",
                          context,
                          HiddenLegTensor::HIDDEN_PREFIX,
                          lab));
        }
    }
}

bool
HiddenLegTensor::is_charge_temp_label(std::string const& label)
{
    // ChargedTensor invariant-part markers and short-lived labels during compose/inner:
    // "!", "!1"/"!2", "!A"/"!B", and their duals "!A*"/"!B*".
    if (label.empty() || label[0] != HIDDEN_PREFIX) {
        return false;
    }
    if (label.size() == 1) {
        return true;
    }
    std::string_view suffix(label.begin() + 1, label.end());
    if (std::all_of(
          suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return true;
    }
    if (suffix.size() == 1 && std::isupper(static_cast<unsigned char>(suffix[0])) != 0) {
        return true;
    }
    if (suffix.size() == 2 && std::isupper(static_cast<unsigned char>(suffix[0])) != 0 &&
        suffix[1] == '*') {
        return true;
    }
    return false;
}

bool
HiddenLegTensor::is_hidden_leg_label(std::string const& label)
{
    // Hidden labels are '!' plus a nonempty name (e.g. '!p', '!charge*', '!(p.q)').
    if (label.size() <= 1 || label[0] != HIDDEN_PREFIX) {
        return false;
    }
    return !is_charge_temp_label(label);
}

bool
HiddenLegTensor::has_hidden_leg_labels(LegLabelList const& labels)
{
    return std::ranges::any_of(labels, [](auto const& l) { return is_hidden_leg_label(l); });
}

std::string
HiddenLegTensor::strip_hidden_prefix(std::string const& label)
{
    if (!is_hidden_leg_label(label)) {
        return label;
    }
    return label.substr(1);
}

std::string
HiddenLegTensor::add_hidden_prefix(std::string const& label)
{
    if (label.empty()) {
        throw std::invalid_argument("Cannot hide an empty label");
    }
    if (label[0] == HIDDEN_PREFIX) {
        throw std::invalid_argument(
          std::format("Label '{}' already starts with '{}'", label, HIDDEN_PREFIX));
    }
    if (label.find(HIDDEN_PREFIX) != std::string::npos) {
        throw std::invalid_argument(std::format(
          "Label '{}' must not contain '{}' except as a hidden prefix", label, HIDDEN_PREFIX));
    }
    return std::string(1, HIDDEN_PREFIX) + label;
}

void
HiddenLegTensor::validate_no_dual_hidden_pair(LegLabelList const& labels)
{
    std::vector<std::string> hidden;
    for (auto const& lab : labels) {
        if (is_hidden_leg_label(lab)) {
            hidden.push_back(lab);
        }
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        for (std::size_t j = i + 1; j < hidden.size(); ++j) {
            if (hidden[i] == hidden[j]) {
                throw std::invalid_argument(
                  std::format("Duplicate hidden leg label '{}'", hidden[i]));
            }
            if (is_dual_pair(hidden[i], hidden[j])) {
                throw std::invalid_argument(
                  std::format("HiddenLegTensor must not contain a dual pair of hidden labels "
                              "('{}' and '{}')",
                              hidden[i],
                              hidden[j]));
            }
        }
    }
}

HiddenLegTensor::HiddenLegTensor(SymmetricTensor::Ptr tensor)
  : SymmetricTensor(tensor->data,
                    tensor->codomain,
                    tensor->domain,
                    tensor->backend,
                    tensor->symmetry,
                    tensor->labels.as_list(),
                    /*check_complex_dtype=*/false)
{
    if (!has_hidden_leg_labels(labels.as_list())) {
        throw std::invalid_argument(
          "HiddenLegTensor adopting constructor requires at least one '!'-prefixed label");
    }
    validate_no_dual_hidden_pair(labels.as_list());
    check_public_labels_no_bang(labels.as_list());

    labels.set_hooks(
      [](LegLabels& /*self*/, LegLabelList const& proposed) {
          if (!has_hidden_leg_labels(proposed)) {
              throw std::invalid_argument(
                "HiddenLegTensor must keep at least one '!'-prefixed hidden leg label");
          }
          validate_no_dual_hidden_pair(proposed);
          check_public_labels_no_bang(proposed);
      },
      [](LegLabels& /*self*/) {});
}

HiddenLegTensor::Ptr
HiddenLegTensor::from_tensor(Tensor::Ptr tensor,
                             std::vector<std::variant<int64, std::string>> which_legs)
{
    if (which_legs.empty()) {
        throw std::invalid_argument("HiddenLegTensor requires at least one leg to hide");
    }
    auto sym = tensor->as_SymmetricTensor();
    if (auto existing = std::dynamic_pointer_cast<HiddenLegTensor>(sym)) {
        sym = existing->unhide_legs();
    }
    auto idcs = sym->labels.indices(which_legs);
    std::unordered_set<int64> hide_set(idcs.begin(), idcs.end());
    if (hide_set.size() != idcs.size()) {
        throw std::invalid_argument("Duplicate legs in which_legs for HiddenLegTensor");
    }
    auto labs = sym->labels.as_list();
    for (auto idx : idcs) {
        auto& lab = labs[static_cast<std::size_t>(idx)];
        if (lab.empty()) {
            throw std::invalid_argument(std::format("Cannot hide unlabeled leg at index {}", idx));
        }
        lab = add_hidden_prefix(lab);
    }
    validate_no_dual_hidden_pair(labs);
    check_public_labels_no_bang(labs);
    auto with_labels = std::make_shared<SymmetricTensor>(
      sym->data, sym->codomain, sym->domain, sym->backend, sym->symmetry, std::move(labs));
    return std::make_shared<HiddenLegTensor>(std::move(with_labels));
}

void
HiddenLegTensor::test_sanity() const
{
    SymmetricTensor::test_sanity();
    assert(has_hidden_leg_labels(labels.as_list()));
    validate_no_dual_hidden_pair(labels.as_list());
    check_public_labels_no_bang(labels.as_list());
}

std::string
HiddenLegTensor::ascii_diagram_type_name() const
{
    return "Hide";
}

std::string
HiddenLegTensor::class_name() const
{
    return "HiddenLegTensor";
}

std::vector<int64>
HiddenLegTensor::hidden_leg_idcs() const
{
    std::vector<int64> out;
    auto labs = labels.as_list();
    for (int64 i = 0; i < static_cast<int64>(labs.size()); ++i) {
        if (is_hidden_leg_label(labs[static_cast<std::size_t>(i)])) {
            out.push_back(i);
        }
    }
    return out;
}

std::vector<int64>
HiddenLegTensor::public_leg_idcs() const
{
    std::vector<int64> out;
    auto labs = labels.as_list();
    for (int64 i = 0; i < static_cast<int64>(labs.size()); ++i) {
        if (!is_hidden_leg_label(labs[static_cast<std::size_t>(i)])) {
            out.push_back(i);
        }
    }
    return out;
}

SymmetricTensorPtr
HiddenLegTensor::unhide_legs() const
{
    auto labs = labels.as_list();
    for (auto& lab : labs) {
        lab = strip_hidden_prefix(lab);
    }
    reject_exclamation_in_labels(labs, "HiddenLegTensor::unhide_legs");
    return std::make_shared<SymmetricTensor>(
      data, codomain, domain, backend, symmetry, std::move(labs), /*check_complex_dtype=*/false);
}

SymmetricTensorPtr
HiddenLegTensor::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    auto plain = unhide_legs();
    if (guarantee_copy) {
        return std::dynamic_pointer_cast<SymmetricTensor>(plain->copy(/*deep=*/true));
    }
    return plain;
}

Tensor::Ptr
HiddenLegTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto base = std::dynamic_pointer_cast<SymmetricTensor>(SymmetricTensor::as_dtype(new_dtype));
    assert(base);
    return std::make_shared<HiddenLegTensor>(std::move(base));
}

Tensor::Ptr
HiddenLegTensor::copy(bool deep,
                      std::optional<std::string> device_opt,
                      std::optional<Dtype> dtype_opt)
{
    auto base = std::dynamic_pointer_cast<SymmetricTensor>(
      SymmetricTensor::copy(deep, device_opt, dtype_opt));
    assert(base);
    return std::make_shared<HiddenLegTensor>(std::move(base));
}

Tensor::Ptr
HiddenLegTensor::dagger() const
{
    auto self = std::dynamic_pointer_cast<Tensor const>(shared_from_this());
    auto new_data = backend->dagger(self);
    LegLabelList dual_labs;
    auto labs = labels.as_list();
    dual_labs.reserve(labs.size());
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_labs.push_back(_dual_leg_label(*it));
    }
    auto base = std::make_shared<SymmetricTensor>(std::move(new_data),
                                                  domain,
                                                  codomain,
                                                  backend,
                                                  symmetry,
                                                  std::move(dual_labs),
                                                  /*check_complex_dtype=*/false);
    return std::make_shared<HiddenLegTensor>(std::move(base));
}

Tensor::Ptr
HiddenLegTensor::to_backend(TensorBackend::Ptr new_backend,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device_opt)
{
    auto base = std::dynamic_pointer_cast<SymmetricTensor>(
      SymmetricTensor::to_backend(std::move(new_backend), dtype_opt, device_opt));
    assert(base);
    return std::make_shared<HiddenLegTensor>(std::move(base));
}

TensorPtr
HiddenLegTensor::maybe_wrap(SymmetricTensor::Ptr tensor)
{
    if (!tensor) {
        return nullptr;
    }
    if (std::dynamic_pointer_cast<HiddenLegTensor>(tensor)) {
        return tensor;
    }
    if (has_hidden_leg_labels(tensor->labels.as_list())) {
        return std::make_shared<HiddenLegTensor>(std::move(tensor));
    }
    return tensor;
}

void
HiddenLegTensor::save_hdf5(lasyten::hdf5::Saver& saver,
                           HighFive::Group& h5gr,
                           std::string const& subpath) const
{
    SymmetricTensor::save_hdf5(saver, h5gr, subpath);
    hdf5_io::h5_set_attr(h5gr.getId(), "is_hidden_leg_tensor", true);
}

HiddenLegTensor::Ptr
HiddenLegTensor::from_hdf5(lasyten::hdf5::Loader& loader,
                           HighFive::Group& h5gr,
                           std::string const& subpath)
{
    auto sym = SymmetricTensor::from_hdf5(loader, h5gr, subpath);
    auto obj = std::make_shared<HiddenLegTensor>(std::move(sym));
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

} // namespace lasyten
