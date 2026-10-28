#include <lasyten/tensors/decompositions.h>
#include <lasyten/tensors/ops_algebra.h>

#include <lasyten/backends/no_symmetry.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/helpers.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_elementwise.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>
#include <lasyten/tools/misc.h>
#include <lasyten/tools/warn.h>

#include <lasyten/tensors/ops_legs.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>

namespace lasyten {

namespace {

char const* _USE_PERMUTE_LEGS_ERR_MSG =
  "Legs can not be permuted automatically. Explicitly use permute_legs()";

[[nodiscard]] MaskCPtr
as_Mask(TensorCPtr t)
{
    return std::dynamic_pointer_cast<Mask const>(t);
}
[[nodiscard]] DiagonalTensorCPtr
as_Diagonal(TensorCPtr t)
{
    return std::dynamic_pointer_cast<DiagonalTensor const>(t);
}
[[nodiscard]] IdentityCPtr
as_Identity(TensorCPtr t)
{
    return std::dynamic_pointer_cast<Identity const>(t);
}
[[nodiscard]] SymmetricTensorCPtr
as_Symmetric(TensorCPtr t)
{
    return std::dynamic_pointer_cast<SymmetricTensor const>(t);
}
[[nodiscard]] ChargedTensorCPtr
as_Charged(TensorCPtr t)
{
    return std::dynamic_pointer_cast<ChargedTensor const>(t);
}
[[nodiscard]] HiddenLegTensorCPtr
as_Hidden(TensorCPtr t)
{
    return std::dynamic_pointer_cast<HiddenLegTensor const>(t);
}

[[nodiscard]] TensorPtr
maybe_wrap_hidden(TensorPtr result, bool wrap_if_hidden_labels)
{
    if (!wrap_if_hidden_labels || !result) {
        return result;
    }
    if (as_Hidden(result) || as_Charged(result)) {
        return result;
    }
    if (as_Diagonal(result) || as_Mask(result) || as_Identity(result)) {
        if (HiddenLegTensor::has_hidden_leg_labels(result->labels.as_list())) {
            throw std::runtime_error(
              "Internal error: DiagonalTensor/Mask/Identity with hidden labels");
        }
        return result;
    }
    if (auto sym = std::dynamic_pointer_cast<SymmetricTensor>(result)) {
        return HiddenLegTensor::maybe_wrap(std::move(sym));
    }
    return result;
}

[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar>
maybe_wrap_hidden_variant(std::variant<TensorPtr, BlockBackend::Scalar> v, bool wrap)
{
    if (std::holds_alternative<BlockBackend::Scalar>(v)) {
        return v;
    }
    return maybe_wrap_hidden(std::get<TensorPtr>(std::move(v)), wrap);
}

void
require_no_remaining_hidden(TensorCPtr tensor, char const* op)
{
    if (!as_Hidden(tensor)) {
        return;
    }
    throw std::invalid_argument(std::format(
      "{} requires that no hidden legs remain. Unmatched hidden labels: use partial_trace "
      "or contract them with a dual HiddenLegTensor first.",
      op));
}

void
reject_hidden_leg_arguments(TensorCPtr tensor, std::vector<int64> const& leg_idcs, char const* op)
{
    if (!as_Hidden(tensor)) {
        return;
    }
    auto const& labs = tensor->labels.as_list();
    for (auto idx : leg_idcs) {
        if (idx < 0) {
            idx += tensor->num_legs;
        }
        if (idx < 0 || idx >= tensor->num_legs) {
            continue;
        }
        if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(idx)])) {
            throw std::invalid_argument(std::format(
              "{}: cannot specify hidden leg '{}' (index {}) in arguments. "
              "Hidden legs are handled implicitly.",
              op,
              labs[static_cast<std::size_t>(idx)].empty() ? "?"
                                                          : labs[static_cast<std::size_t>(idx)],
              idx));
        }
    }
}

[[nodiscard]] std::vector<std::pair<int64, int64>>
implicit_hidden_contraction_pairs(TensorCPtr tensor1, TensorCPtr tensor2)
{
    std::vector<std::pair<int64, int64>> pairs;
    auto h1 = as_Hidden(tensor1);
    auto h2 = as_Hidden(tensor2);
    if (!h1 || !h2) {
        return pairs;
    }
    auto const& labs1 = tensor1->labels.as_list();
    auto const& labs2 = tensor2->labels.as_list();
    std::vector<std::pair<int64, std::string>> hidden1;
    std::vector<std::pair<int64, std::string>> hidden2;
    for (int64 i = 0; i < tensor1->num_legs; ++i) {
        if (HiddenLegTensor::is_hidden_leg_label(labs1[static_cast<std::size_t>(i)])) {
            hidden1.emplace_back(i, labs1[static_cast<std::size_t>(i)]);
        }
    }
    for (int64 i = 0; i < tensor2->num_legs; ++i) {
        if (HiddenLegTensor::is_hidden_leg_label(labs2[static_cast<std::size_t>(i)])) {
            hidden2.emplace_back(i, labs2[static_cast<std::size_t>(i)]);
        }
    }
    std::vector<bool> used2(hidden2.size(), false);
    for (auto const& [i1, lab1] : hidden1) {
        auto dual1 = _dual_leg_label(lab1);
        for (std::size_t j = 0; j < hidden2.size(); ++j) {
            if (used2[j]) {
                continue;
            }
            auto const& [i2, lab2] = hidden2[j];
            if (lab1 == lab2) {
                throw std::invalid_argument(std::format(
                  "Cannot contract HiddenLegTensors with equal hidden label '{}' "
                  "(both or neither starred). Dual pairs like '!a' with '!a*' are contracted "
                  "implicitly.",
                  lab1));
            }
            if (dual1 == lab2) {
                pairs.emplace_back(i1, i2);
                used2[j] = true;
                break;
            }
        }
    }
    return pairs;
}

void
check_spaces_tp(TensorProduct::Ptr const& a, TensorProduct::Ptr const& b, bool expect_equal = true)
{
    // TensorProduct::factors are Legs (ElementarySpace is both Leg+Space; LegPipe is Leg-only).
    // Never dynamic_cast factors to Space — that yields null for LegPipe and segfaults.
    _check_compatible_legs(a->factors, b->factors, expect_equal);
}

void
check_spaces_two(TensorProduct::Ptr a1,
                 TensorProduct::Ptr a2,
                 TensorProduct::Ptr b1,
                 TensorProduct::Ptr b2,
                 bool expect_equal = true)
{
    check_spaces_tp(a1, b1, expect_equal);
    check_spaces_tp(a2, b2, expect_equal);
}

void
check_leg_vectors(std::vector<Leg::Ptr> const& a,
                  std::vector<Leg::Ptr> const& b,
                  bool expect_equal = true)
{
    _check_compatible_legs(a, b, expect_equal);
}

[[nodiscard]] std::vector<LegRef>
leg_refs_from_ints(std::vector<int64> const& idcs)
{
    std::vector<LegRef> out;
    out.reserve(idcs.size());
    for (auto i : idcs) {
        out.emplace_back(i);
    }
    return out;
}

std::string
relabel_one(std::string lab, std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (lab.empty() || !relabel.has_value()) {
        return lab;
    }
    auto it = relabel->find(lab);
    if (it != relabel->end()) {
        return it->second;
    }
    return lab;
}

LegLabelList
apply_relabel(LegLabelList labels,
              std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (!relabel.has_value()) {
        return labels;
    }
    for (auto& lab : labels) {
        lab = relabel_one(lab, relabel);
    }
    return labels;
}

LegLabelList
dual_labels_reversed(LegLabelList const& labs)
{
    LegLabelList dual_labs;
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_labs.push_back(_dual_leg_label(*it));
    }
    return dual_labs;
}

[[nodiscard]] TensorPtr
from_compose_sym(std::variant<SymmetricTensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit(
      [](auto const& x) -> TensorPtr {
          if constexpr (std::is_same_v<std::decay_t<decltype(x)>, BlockBackend::Scalar>) {
              throw std::logic_error("from_compose_sym: unexpected scalar in tensor-only context");
          } else {
              return x;
          }
      },
      v);
}

[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar>
from_compose_sym_variant(std::variant<SymmetricTensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit(
      [](auto const& x) -> std::variant<TensorPtr, BlockBackend::Scalar> { return x; }, v);
}

[[nodiscard]] Mask::Ptr
make_mask_native(TensorBackend::DataPtr data,
                 Space::Ptr space_in,
                 Space::Ptr space_out,
                 bool is_projection,
                 TensorBackend::Ptr backend,
                 LegLabelList labels)
{
    return std::make_shared<Mask>(std::move(data),
                                  std::move(space_in),
                                  std::move(space_out),
                                  is_projection,
                                  std::move(backend),
                                  space_in->symmetry,
                                  std::move(labels),
                                  backend->get_device_from_data(data));
}

[[nodiscard]] SymmetricTensorPtr
make_symmetric_native(TensorBackend::DataPtr data,
                      TensorProduct::Ptr codomain,
                      TensorProduct::Ptr domain,
                      TensorBackend::Ptr backend,
                      LegLabelList labels)
{
    return std::make_shared<SymmetricTensor>(std::move(data),
                                             std::move(codomain),
                                             std::move(domain),
                                             std::move(backend),
                                             codomain->symmetry,
                                             std::move(labels));
}

[[nodiscard]] DiagonalTensorPtr
make_diagonal_native(TensorBackend::DataPtr data,
                     Space::Ptr leg,
                     TensorBackend::Ptr backend,
                     LegLabelList labels)
{
    return std::make_shared<DiagonalTensor>(
      std::move(data), std::move(leg), std::move(backend), leg->symmetry, std::move(labels));
}

[[nodiscard]] ChargedTensorPtr
make_charged_native(SymmetricTensorPtr inv_part, BlockBackend::BlockPtr charged_state)
{
    return std::make_shared<ChargedTensor>(std::move(inv_part), std::move(charged_state));
}

[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar>
from_invariant_part_variant(SymmetricTensorPtr inv_part, BlockBackend::BlockPtr charged_state)
{
    auto v = ChargedTensor::from_invariant_part(std::move(inv_part), std::move(charged_state));
    return std::visit(
      [](auto&& x) -> std::variant<TensorPtr, BlockBackend::Scalar> {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, BlockBackend::Scalar>) {
              return std::forward<decltype(x)>(x);
          } else {
              return std::static_pointer_cast<Tensor>(std::forward<decltype(x)>(x));
          }
      },
      std::move(v));
}

[[nodiscard]] ChargedTensorPtr
from_invariant_part_native(SymmetricTensorPtr inv_part, BlockBackend::BlockPtr charged_state)
{
    auto v = from_invariant_part_variant(std::move(inv_part), std::move(charged_state));
    if (std::holds_alternative<BlockBackend::Scalar>(v)) {
        throw std::invalid_argument("from_invariant_part returned scalar unexpectedly");
    }
    return std::dynamic_pointer_cast<ChargedTensor>(std::get<TensorPtr>(std::move(v)));
}

[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar>
from_two_charge_legs_variant(SymmetricTensorPtr inv_part,
                             BlockBackend::BlockPtr state1,
                             BlockBackend::BlockPtr state2)
{
    auto v = ChargedTensor::from_two_charge_legs(
      std::move(inv_part), std::move(state1), std::move(state2));
    return std::visit(
      [](auto&& x) -> std::variant<TensorPtr, BlockBackend::Scalar> {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, BlockBackend::Scalar>) {
              return std::forward<decltype(x)>(x);
          } else {
              return std::static_pointer_cast<Tensor>(std::forward<decltype(x)>(x));
          }
      },
      std::move(v));
}

[[nodiscard]] ChargedTensorPtr
from_two_charge_legs_native(SymmetricTensorPtr inv_part,
                            BlockBackend::BlockPtr state1,
                            BlockBackend::BlockPtr state2)
{
    auto v =
      from_two_charge_legs_variant(std::move(inv_part), std::move(state1), std::move(state2));
    if (std::holds_alternative<BlockBackend::Scalar>(v)) {
        throw std::invalid_argument("from_two_charge_legs returned scalar unexpectedly");
    }
    return std::dynamic_pointer_cast<ChargedTensor>(std::get<TensorPtr>(std::move(v)));
}

[[nodiscard]] TensorPtr
move_leg_wrap(TensorPtr tensor,
              LegRef which,
              std::optional<int64> codomain_pos = std::nullopt,
              std::optional<int64> domain_pos = std::nullopt,
              std::optional<BendRight> bend_right = std::nullopt)
{
    return move_leg(std::move(tensor),
                    std::move(which),
                    codomain_pos,
                    domain_pos,
                    std::nullopt,
                    std::move(bend_right));
}

[[nodiscard]] TensorPtr
permute_legs_wrap(TensorPtr tensor,
                  std::optional<std::vector<LegRef>> codomain = std::nullopt,
                  std::optional<std::vector<LegRef>> domain = std::nullopt,
                  std::optional<BendRight> bend_right = std::nullopt)
{
    return permute_legs(std::move(tensor),
                        std::move(codomain),
                        std::move(domain),
                        std::nullopt,
                        std::move(bend_right));
}

[[nodiscard]] TensorPtr
bend_legs_wrap(TensorPtr tensor,
               std::optional<int64> num_codomain_legs = std::nullopt,
               std::optional<int64> num_domain_legs = std::nullopt)
{
    return bend_legs(std::move(tensor), num_codomain_legs, num_domain_legs);
}

std::map<std::string, std::string>
relabel_or_empty(std::optional<std::map<std::string, std::string>> const& relabel)
{
    return relabel.value_or(std::map<std::string, std::string>{});
}

[[noreturn]] void
rethrow_permute_legs_err()
{
    throw SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG);
}

[[noreturn]] void
handle_permute_legs_symmetry_error()
{
    try {
        throw;
    } catch (SymmetryError const&) {
        rethrow_permute_legs_err();
    }
}

bool
legs_equal(Leg::Ptr const& a, Leg::Ptr const& b)
{
    return a && b && (*a == *b);
}

bool
mask_almost_equal(MaskCPtr m1, MaskCPtr m2)
{
    BlockBinaryFn eq_fn = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) == (*b);
    };
    auto m1_nc = std::const_pointer_cast<Mask>(m1);
    return m1_nc->_binary_operand(m2, std::move(eq_fn), "==")->all();
}

bool
all_multiplicities_one(TensorProduct::Ptr const& tp)
{
    return std::ranges::all_of(tp->multiplicities, [](int64 m) { return m == 1; });
}

char const*
charge_leg_label()
{
    return ChargedTensor::_CHARGE_LEG_LABEL;
}

[[nodiscard]] LegLabelList
labels_from_tensor(TensorCPtr t)
{
    return t->labels.as_list();
}

[[nodiscard]] LegLabelList
identity_labels(LegLabelList labels)
{
    return labels;
}

[[nodiscard]] BlockBackend::Scalar
scalar_return(BlockBackend::Scalar s)
{
    return s;
}

[[nodiscard]] BlockBackend::Scalar
backend_item(TensorCPtr t)
{
    return t->backend->item(t);
}

[[nodiscard]] IdentityPtr
make_identity_native(Space::Ptr leg, TensorBackend::Ptr backend, LegLabelList labels)
{
    auto dt = SymmetricTensor::_parse_default_dtype(std::nullopt, leg->symmetry);
    if (!dt.has_value()) {
        dt = Dtype::Float64;
    }
    std::string device_s = backend->block_backend->default_device;
    return std::make_shared<Identity>(std::move(leg),
                                      std::move(backend),
                                      leg->symmetry,
                                      std::move(labels),
                                      *dt,
                                      std::move(device_s));
}

LegLabelList
nested_flat_labels(LegLabelList codomain_labels, LegLabelList domain_labels)
{
    LegLabelList res = std::move(codomain_labels);
    for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
        res.push_back(*it);
    }
    return res;
}

[[nodiscard]] Space::Ptr
tp_factor_space(TensorProduct::Ptr const& tp, std::size_t i)
{
    return std::dynamic_pointer_cast<Space>(tp->factors.at(i));
}

[[nodiscard]] TensorProduct::Ptr
splice_tp_factors(TensorProduct::Ptr const& base,
                  std::size_t first,
                  std::size_t last_exclusive,
                  TensorProduct::Ptr const& insert_tp)
{
    std::vector<Leg::Ptr> factors = base->factors;
    factors.erase(factors.begin() + static_cast<std::ptrdiff_t>(first),
                  factors.begin() + static_cast<std::ptrdiff_t>(last_exclusive));
    factors.insert(factors.begin() + static_cast<std::ptrdiff_t>(first),
                   insert_tp->factors.begin(),
                   insert_tp->factors.end());
    return std::make_shared<TensorProduct>(std::move(factors), base->symmetry);
}

[[nodiscard]] int64
leg_ref_index(TensorCPtr tensor, LegRef const& leg)
{
    return std::visit([&](auto const& x) -> int64 { return tensor->labels.index(x); }, leg);
}

[[nodiscard]] std::optional<int64>
levels_min(LevelsSpec const& levels)
{
    std::optional<int64> min_v;
    for (auto const& lv : levels) {
        if (!lv.has_value()) {
            continue;
        }
        if (!min_v.has_value() || *lv < *min_v) {
            min_v = *lv;
        }
    }
    return min_v;
}

} // namespace

// Forward declarations (mutual recursion).
TensorPtr dagger(TensorCPtr tensor);
TensorPtr transpose(TensorCPtr tensor);
TensorPtr scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v);
TensorPtr linear_combination(BlockBackend::Scalar const& a,
                             TensorCPtr v,
                             BlockBackend::Scalar const& b,
                             TensorCPtr w);
TensorPtr scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg);
std::variant<TensorPtr, BlockBackend::Scalar> compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1,
  std::optional<std::map<std::string, std::string>> relabel2);
TensorPtr partial_compose(TensorCPtr tensor1,
                          TensorCPtr tensor2,
                          LegRef tensor1_first_leg,
                          std::optional<std::map<std::string, std::string>> relabel1,
                          std::optional<std::map<std::string, std::string>> relabel2);
TensorPtr outer(TensorCPtr tensor1,
                TensorCPtr tensor2,
                std::optional<std::map<std::string, std::string>> relabel1,
                std::optional<std::map<std::string, std::string>> relabel2);
std::variant<TensorPtr, BlockBackend::Scalar> partial_trace(TensorCPtr tensor,
                                                            std::vector<std::vector<LegRef>> pairs,
                                                            std::optional<LevelsSpec> levels);
BlockBackend::Scalar trace(TensorCPtr tensor);
BlockBackend::Scalar inner(TensorCPtr A, TensorCPtr B, bool do_dagger);
std::variant<TensorPtr, BlockBackend::Scalar> tdot(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::vector<LegRef> legs1,
  std::vector<LegRef> legs2,
  std::optional<std::map<std::string, std::string>> relabel1,
  std::optional<std::map<std::string, std::string>> relabel2);

std::string
get_same_device(std::vector<TensorCPtr> const& tensors, std::string const& error_msg)
{
    if (tensors.empty()) {
        throw std::invalid_argument("Need at least one tensor");
    }
    std::string device = tensors[0]->device;
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i]->device != device) {
            throw std::invalid_argument(error_msg);
        }
    }
    return device;
}

bool
almost_equal(TensorCPtr tensor_1,
             TensorCPtr tensor_2,
             float64 rtol,
             float64 atol,
             bool allow_different_types)
{
    check_same_legs(tensor_1, tensor_2);
    (void)get_same_device({ tensor_1, tensor_2 });

    if (auto m1 = as_Mask(tensor_1)) {
        if (auto m2 = as_Mask(tensor_2)) {
            return mask_almost_equal(m1, m2);
        }
        if (allow_different_types) {
            if (as_Diagonal(tensor_2)) {
                auto m1_nc = std::const_pointer_cast<Mask>(m1);
                return almost_equal(m1_nc->as_DiagonalTensor(), tensor_2, rtol, atol);
            }
            if (as_Symmetric(tensor_2) || as_Charged(tensor_2)) {
                auto m1_nc = std::const_pointer_cast<Mask>(m1);
                return almost_equal(m1_nc->as_SymmetricTensor(), tensor_2, rtol, atol);
            }
        }
    }

    if (auto d1 = as_Diagonal(tensor_1)) {
        if (auto m2 = as_Mask(tensor_2)) {
            if (allow_different_types) {
                auto m2_nc = std::const_pointer_cast<Mask>(m2);
                return almost_equal(tensor_1, m2_nc->as_DiagonalTensor(), rtol, atol);
            }
        }
        if (auto d2 = as_Diagonal(tensor_2)) {
            auto d1_nc = std::const_pointer_cast<DiagonalTensor>(d1);
            return d1_nc->elementwise_almost_equal(d2, rtol, atol)->all();
        }
        if (allow_different_types && (as_Symmetric(tensor_2) || as_Charged(tensor_2))) {
            auto d1_nc = std::const_pointer_cast<DiagonalTensor>(d1);
            return almost_equal(d1_nc->as_SymmetricTensor(), tensor_2, rtol, atol);
        }
    }

    if (auto s1 = as_Symmetric(tensor_1)) {
        if (allow_different_types && (as_Mask(tensor_2) || as_Diagonal(tensor_2))) {
            if (auto m2 = as_Mask(tensor_2)) {
                auto m2_nc = std::const_pointer_cast<Mask>(m2);
                return almost_equal(tensor_1, m2_nc->as_SymmetricTensor(), rtol, atol);
            }
            if (auto d2 = as_Diagonal(tensor_2)) {
                auto d2_nc = std::const_pointer_cast<DiagonalTensor>(d2);
                return almost_equal(tensor_1, d2_nc->as_SymmetricTensor(), rtol, atol);
            }
        }
        if (as_Symmetric(tensor_2)) {
            auto backend = get_same_backend({ tensor_1, tensor_2 });
            return backend->almost_equal(tensor_1, tensor_2, rtol, atol);
        }
        if (allow_different_types && as_Charged(tensor_2)) {
            try {
                auto c2 = as_Charged(tensor_2);
                return almost_equal(tensor_1, c2->invariant_part, rtol, atol);
            } catch (SymmetryError const&) {
                throw NotImplemented("almost_equal");
            }
        }
    }

    if (auto c1 = as_Charged(tensor_1)) {
        if (allow_different_types && (as_Mask(tensor_2) || as_Diagonal(tensor_2))) {
            if (auto m2 = as_Mask(tensor_2)) {
                auto m2_nc = std::const_pointer_cast<Mask>(m2);
                return almost_equal(tensor_1, m2_nc->as_SymmetricTensor(), rtol, atol);
            }
            if (auto d2 = as_Diagonal(tensor_2)) {
                auto d2_nc = std::const_pointer_cast<DiagonalTensor>(d2);
                return almost_equal(tensor_1, d2_nc->as_SymmetricTensor(), rtol, atol);
            }
        }
        if (as_Symmetric(tensor_2)) {
            return almost_equal(tensor_2, tensor_1, rtol, atol);
        }
        if (auto c2 = as_Charged(tensor_2)) {
            if (!(*c1->charge_leg == *c2->charge_leg)) {
                throw std::invalid_argument("Mismatched charge_leg");
            }
            auto backend = get_same_backend({ tensor_1, tensor_2 });
            auto charge_space = std::dynamic_pointer_cast<Space const>(c1->charge_leg);
            if (charge_space && charge_space->dim == 1.) {
                auto bb = backend->block_backend;
                auto s2 = bb->item(c2->charged_state);
                auto s1 = bb->item(c1->charged_state);
                return almost_equal(
                  scalar_multiply(s2, std::static_pointer_cast<Tensor const>(c1->invariant_part)),
                  scalar_multiply(s1, std::static_pointer_cast<Tensor const>(c2->invariant_part)),
                  rtol,
                  atol);
            }
            throw NotImplemented("almost_equal");
        }
    }

    throw std::invalid_argument(std::format(
      "Incompatible types: {} and {}", tensor_1->class_name(), tensor_2->class_name()));
}

TensorPtr
apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg)
{
    (void)get_same_device({ tensor, mask });
    auto [in_domain, co_idx, leg_idx] = tensor->_parse_leg_idx(leg);
    (void)co_idx;
    MaskPtr mask_mut = std::const_pointer_cast<Mask>(mask);
    if (!mask_mut->is_projection) {
        throw std::invalid_argument("mask must be a projection");
    }
    if (in_domain) {
        mask_mut = std::dynamic_pointer_cast<Mask>(transpose(mask_mut));
    }
    return _compose_with_Mask(tensor, mask_mut, leg_idx);
}

TensorPtr
enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg)
{
    (void)get_same_device({ tensor, mask });
    auto [in_domain, co_idx, leg_idx] = tensor->_parse_leg_idx(leg);
    (void)co_idx;
    MaskPtr mask_mut = std::const_pointer_cast<Mask>(mask);
    if (mask_mut->is_projection) {
        throw std::invalid_argument("enlarge_leg requires a non-projection mask");
    }
    if (in_domain) {
        mask_mut = std::dynamic_pointer_cast<Mask>(transpose(mask_mut));
    }
    return _compose_with_Mask(tensor, mask_mut, leg_idx);
}

[[nodiscard]] Space::Ptr
space_factor0(TensorProduct::Ptr const& tp)
{
    return std::dynamic_pointer_cast<Space>(tp->factors.at(0));
}

[[nodiscard]] TensorPtr
shallow_copy_labels(TensorCPtr t, LegLabelList labels)
{
    TensorPtr mut = std::const_pointer_cast<Tensor>(t);
    TensorPtr res = mut->copy(/*deep=*/false);
    res->labels.set(std::move(labels));
    return res;
}

void
check_spaces_tensors(TensorProduct::Ptr a1,
                     TensorProduct::Ptr a2,
                     TensorProduct::Ptr b1,
                     TensorProduct::Ptr b2,
                     bool expect_equal = true)
{
    check_spaces_tp(a1, b1, expect_equal);
    check_spaces_tp(a2, b2, expect_equal);
}

TensorPtr
dagger(TensorCPtr tensor)
{
    if (auto m = as_Mask(tensor)) {
        auto backend = m->backend;
        auto data = backend->mask_dagger(m);
        LegLabelList dual_labs = dual_labels_reversed(m->labels.as_list());
        return make_mask_native(std::move(data),
                                space_factor0(m->codomain),
                                space_factor0(m->domain),
                                !m->is_projection,
                                backend,
                                std::move(dual_labs));
    }
    if (as_Identity(tensor)) {
        return std::const_pointer_cast<Tensor>(tensor);
    }
    if (auto d = as_Diagonal(tensor)) {
        LegLabelList dual_labs = dual_labels_reversed(d->labels.as_list());
        if (d->dtype == Dtype::Bool) {
            return shallow_copy_labels(d, std::move(dual_labs));
        }
        DiagonalTensorPtr res = complex_conj(std::const_pointer_cast<DiagonalTensor>(d));
        res->labels.set(std::move(dual_labs));
        return res;
    }
    if (auto h = as_Hidden(tensor)) {
        return h->dagger();
    }
    if (auto s = as_Symmetric(tensor)) {
        auto backend = s->backend;
        auto data = backend->dagger(s);
        LegLabelList dual_labs = dual_labels_reversed(s->labels.as_list());
        return make_symmetric_native(
          std::move(data), s->domain, s->codomain, backend, std::move(dual_labs));
    }
    if (auto c = as_Charged(tensor)) {
        SymmetricTensorPtr inv_part =
          std::dynamic_pointer_cast<SymmetricTensor>(dagger(c->invariant_part));
        inv_part->labels[0] = charge_leg_label();
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          move_leg_wrap(inv_part, 0, std::nullopt, 0, BendRight{ true }));
        auto charged_state = c->backend->block_backend->conj(c->charged_state);
        return make_charged_native(std::move(inv_part), std::move(charged_state));
    }
    throw std::invalid_argument("Invalid type for tensor. Expected a Tensor subtype");
}

TensorPtr
transpose(TensorCPtr tensor)
{
    LegLabelList labels;
    {
        LegLabelList domain_labels = tensor->domain_labels();
        LegLabelList codomain_labels = tensor->codomain_labels();
        for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
            labels.push_back(*it);
        }
        labels.insert(labels.end(), codomain_labels.begin(), codomain_labels.end());
    }

    if (auto m = as_Mask(tensor)) {
        auto backend = m->backend;
        auto [space_in, space_out, data] = backend->mask_transpose(m);
        return make_mask_native(std::move(data),
                                std::move(space_in),
                                std::move(space_out),
                                !m->is_projection,
                                backend,
                                labels);
    }
    if (auto id = as_Identity(tensor)) {
        return make_identity_native(id->leg()->dual(), id->backend, labels);
    }
    if (auto d = as_Diagonal(tensor)) {
        auto backend = d->backend;
        auto [dual_leg, data] = backend->diagonal_transpose(d);
        return make_diagonal_native(std::move(data), std::move(dual_leg), backend, labels);
    }
    if (auto s = as_Symmetric(tensor)) {
        int64 const n_cod = s->num_codomain_legs();
        int64 const n_dom = s->num_domain_legs();
        int64 const n_legs = s->num_legs;
        std::vector<int64> codomain;
        for (int64 i = n_cod; i < n_legs; ++i) {
            codomain.push_back(i);
        }
        std::vector<int64> domain;
        for (int64 i = n_cod - 1; i >= 0; --i) {
            domain.push_back(i);
        }
        std::vector<std::optional<bool>> bend_opts(static_cast<std::size_t>(n_cod), false);
        bend_opts.insert(bend_opts.end(), static_cast<std::size_t>(n_dom), true);
        return permute_legs_wrap(std::const_pointer_cast<SymmetricTensor>(s),
                                 leg_refs_from_ints(codomain),
                                 leg_refs_from_ints(domain),
                                 BendRight{ std::move(bend_opts) });
    }
    if (auto c = as_Charged(tensor)) {
        if (!c->symmetry->has_trivial_braid()) {
            throw SymmetryError(
              "transpose is not defined for ChargedTensors with fermionic symmetries. "
              "This is because there is no way to recover the ChargedTensor format in such a "
              "way that transposing twice gives back the original tensor. "
              "Use permute_legs instead");
        }
        SymmetricTensorPtr inv_part =
          std::dynamic_pointer_cast<SymmetricTensor>(transpose(c->invariant_part));
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          move_leg_wrap(inv_part, charge_leg_label(), std::nullopt, 0));
        return make_charged_native(std::move(inv_part), c->charged_state);
    }
    throw std::invalid_argument("Invalid type for tensor.");
}

TensorPtr
on_device(TensorCPtr tensor, std::string device, bool copy)
{
    if (copy) {
        return std::const_pointer_cast<Tensor>(tensor)->copy(/*deep=*/true, /*device=*/device);
    }
    auto mut = std::const_pointer_cast<Tensor>(tensor);
    mut->move_to_device(device);
    return mut;
}

bool
is_scalar(TensorCPtr obj)
{
    if (obj->domain->num_sectors != 1) {
        return false;
    }
    if (obj->codomain->num_sectors != 1) {
        return false;
    }
    if (obj->domain->sector_decomposition != obj->codomain->sector_decomposition) {
        return false;
    }
    if (!all_multiplicities_one(obj->domain)) {
        return false;
    }
    if (!all_multiplicities_one(obj->codomain)) {
        return false;
    }
    return true;
}

BlockBackend::Scalar
item(TensorCPtr tensor)
{
    if (!is_scalar(tensor)) {
        throw std::invalid_argument("Not a scalar");
    }
    if (auto m = as_Mask(tensor)) {
        return m->backend->block_backend->as_scalar(m->any());
    }
    if (auto id = as_Identity(tensor)) {
        auto bb = id->backend->block_backend;
        if (id->dtype == Dtype::Bool) {
            return bb->as_scalar(true);
        }
        if (id->dtype == Dtype::Int64) {
            return bb->as_scalar(static_cast<int64>(1));
        }
        return bb->as_scalar(static_cast<float64>(1.), id->dtype);
    }
    if (as_Hidden(tensor)) {
        require_no_remaining_hidden(tensor, "item");
    }
    if (auto c = as_Charged(tensor)) {
        auto backend = c->backend;
        auto bb = backend->block_backend;
        auto inv_block = c->invariant_part->to_dense_block(
          std::nullopt, std::nullopt, /*understood_braiding=*/true);
        auto res = bb->tdot(c->charged_state, inv_block, { 0 }, { -1 });
        return bb->item(res);
    }
    if (as_Diagonal(tensor) || as_Symmetric(tensor)) {
        return backend_item(tensor);
    }
    throw std::invalid_argument("Invalid type for tensor.");
}

BlockBackend::Scalar
norm(TensorCPtr tensor)
{
    if (auto m = as_Mask(tensor)) {
        auto bb = m->backend->block_backend;
        auto const& small = m->small_leg();
        return bb->as_scalar(std::sqrt(static_cast<float64>(small->Space::dim)));
    }
    if (auto id = as_Identity(tensor)) {
        auto bb = id->backend->block_backend;
        return bb->as_scalar(std::sqrt(static_cast<float64>(id->leg()->Space::dim)));
    }
    if (as_Diagonal(tensor) || as_Symmetric(tensor)) {
        return tensor->backend->norm(tensor);
    }
    if (auto c = as_Charged(tensor)) {
        auto backend = c->backend;
        if (c->charge_leg->dim == 1.) {
            auto factor = backend->block_backend->item(c->charged_state).abs();
            return factor * backend->norm(c->invariant_part);
        }
        warn("Converting ChargedTensor to dense block for `norm`");
        auto c_mut = std::const_pointer_cast<ChargedTensor>(c);
        auto block =
          c_mut->to_dense_block(std::nullopt, std::nullopt, /*understood_braiding=*/true);
        return backend->block_backend->norm(block, 2);
    }
    throw std::invalid_argument("Invalid type for tensor.");
}

TensorPtr
pinv(TensorCPtr tensor, float64 cutoff)
{
    if (as_Identity(tensor)) {
        return std::const_pointer_cast<Tensor>(tensor);
    }
    if (auto d = as_Diagonal(tensor)) {
        return cutoff_inverse(std::const_pointer_cast<DiagonalTensor>(d), cutoff);
    }
    auto [U, S, Vh, err, renormalize] = truncated_svd(tensor,
                                                      /*new_labels=*/std::nullopt,
                                                      /*new_leg_dual=*/false,
                                                      /*charge_leg_top=*/true,
                                                      /*algorithm=*/std::nullopt,
                                                      /*normalize_to=*/std::nullopt,
                                                      /*chi_max=*/std::nullopt,
                                                      /*chi_min=*/1,
                                                      /*degeneracy_tol=*/0.,
                                                      /*trunc_cut=*/0.,
                                                      /*svd_min=*/cutoff);
    (void)err;
    (void)renormalize;
    auto mid = compose(U, cutoff_inverse(S, cutoff));
    TensorPtr t_mid = std::visit(
      [](auto&& x) -> TensorPtr {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, BlockBackend::Scalar>) {
              throw std::logic_error("pinv: unexpected scalar in compose chain");
          } else {
              return x;
          }
      },
      mid);
    auto fin = compose(t_mid, Vh);
    TensorPtr t_fin = std::visit(
      [](auto&& x) -> TensorPtr {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, BlockBackend::Scalar>) {
              throw std::logic_error("pinv: unexpected scalar in compose chain");
          } else {
              return x;
          }
      },
      fin);
    return dagger(t_fin);
}

TensorPtr
scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v)
{
    if (auto d = as_Diagonal(v)) {
        auto d_mut = std::const_pointer_cast<DiagonalTensor>(d);
        BlockUnaryFn func = [a](BlockBackend::BlockPtr const& block) { return a * (*block); };
        return d_mut->_elementwise_unary(std::move(func), /*maps_zero_to_zero=*/true);
    }
    if (auto m = as_Mask(v)) {
        char const* msg = "Converting to SymmetricTensor for scalar multiplication. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        warn(msg);
        v = std::const_pointer_cast<Tensor>(std::static_pointer_cast<Tensor const>(m))
              ->as_SymmetricTensor(/*guarantee_copy=*/false, /*warning=*/std::nullopt);
    }
    if (auto c = as_Charged(v)) {
        auto charged_state = c->backend->block_backend->mul(a, c->charged_state);
        return make_charged_native(c->invariant_part, std::move(charged_state));
    }
    if (auto h = as_Hidden(v)) {
        auto backend = h->backend;
        auto data = backend->mul(a, h);
        return maybe_wrap_hidden(
          make_symmetric_native(
            std::move(data), h->codomain, h->domain, backend, h->labels.as_list()),
          true);
    }
    auto s = as_Symmetric(v);
    if (!s) {
        throw std::invalid_argument("scalar_multiply: unsupported tensor type");
    }
    auto backend = s->backend;
    auto data = backend->mul(a, s);
    return make_symmetric_native(
      std::move(data), s->codomain, s->domain, backend, s->labels.as_list());
}

TensorPtr
linear_combination(BlockBackend::Scalar const& a,
                   TensorCPtr v,
                   BlockBackend::Scalar const& b,
                   TensorCPtr w)
{
    (void)get_same_device({ v, w });
    check_spaces_tensors(v->codomain, v->domain, w->codomain, w->domain);

    if (auto vd = as_Diagonal(v)) {
        if (auto wd = as_Diagonal(w)) {
            BlockBinaryFn func = [a, b](BlockBackend::BlockPtr const& bv,
                                        BlockBackend::BlockPtr const& bw) {
                auto left = a * (*bv);
                auto right = b * (*bw);
                return (*left) + (*right);
            };
            auto v_mut = std::const_pointer_cast<DiagonalTensor>(vd);
            return v_mut->_binary_operand(wd, std::move(func), "linear_combination");
        }
    }
    if (auto vc = as_Charged(v)) {
        if (auto wc = as_Charged(w)) {
            if (!(*vc->charge_leg == *wc->charge_leg)) {
                throw std::invalid_argument(
                  "Can not add ChargedTensors with different dummy legs");
            }
            if (vc->charge_leg->dim == 1.) {
                auto bb = vc->backend->block_backend;
                auto factor = bb->item(wc->charged_state) / bb->item(vc->charged_state);
                SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
                  linear_combination(a,
                                     std::static_pointer_cast<Tensor const>(vc->invariant_part),
                                     b * factor,
                                     std::static_pointer_cast<Tensor const>(wc->invariant_part)));
                return make_charged_native(std::move(inv_part), vc->charged_state);
            }
            throw NotImplemented("linear_combination");
        }
        throw std::invalid_argument("Can not add ChargedTensor and non-charged tensor.");
    }
    if (as_Charged(w)) {
        throw std::invalid_argument("Can not add ChargedTensor and non-charged tensor.");
    }
    if (auto hv = as_Hidden(v)) {
        if (auto hw = as_Hidden(w)) {
            auto hv_mut = std::const_pointer_cast<HiddenLegTensor>(hv);
            auto hw_mut = std::const_pointer_cast<HiddenLegTensor>(hw);
            SymmetricTensorPtr res = std::dynamic_pointer_cast<SymmetricTensor>(linear_combination(
              a,
              std::static_pointer_cast<Tensor const>(hv_mut->as_SymmetricTensor()),
              b,
              std::static_pointer_cast<Tensor const>(hw_mut->as_SymmetricTensor())));
            res->labels.set(hv->labels.as_list());
            return maybe_wrap_hidden(res, true);
        }
    }

    // DiagonalTensor/Mask/Identity inherit SymmetricTensor, so as_Symmetric succeeds on them
    // even though their blocks are 1D. Always expand those subclasses to full Symmetric data.
    bool needs_expand = as_Diagonal(v) || as_Diagonal(w) || as_Mask(v) || as_Mask(w);
    SymmetricTensorCPtr vs = as_Symmetric(v);
    SymmetricTensorCPtr ws = as_Symmetric(w);
    if (!vs || !ws || needs_expand) {
        vs = as_Symmetric(std::const_pointer_cast<Tensor>(v)->as_SymmetricTensor());
        ws = as_Symmetric(std::const_pointer_cast<Tensor>(w)->as_SymmetricTensor());
    }
    auto backend = get_same_backend({ v, w });
    auto data = backend->linear_combination(a, vs, b, ws);
    LegLabelList labels = _get_matching_labels(vs->labels.as_list(), ws->labels.as_list());
    return make_symmetric_native(
      std::move(data), vs->codomain, vs->domain, backend, std::move(labels));
}

TensorPtr
scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg)
{
    (void)get_same_device({ tensor, diag });
    if (as_Identity(diag)) {
        return std::const_pointer_cast<Tensor>(tensor);
    }

    auto [in_domain, co_domain_idx, leg_idx] = tensor->_parse_leg_idx(leg);
    Leg::Ptr tens_leg = in_domain
                          ? tensor->domain->factors.at(static_cast<std::size_t>(co_domain_idx))
                          : tensor->codomain->factors.at(static_cast<std::size_t>(co_domain_idx));
    if (!tensor->symmetry->is_equivalent_to(*diag->symmetry)) {
        throw SymmetryError("scale_axis requires equivalent symmetries");
    }
    DiagonalTensorCPtr diag_use = diag;
    auto diag_leg = std::dynamic_pointer_cast<Leg>(diag->leg());
    if (legs_equal(tens_leg, diag_leg)) {
        // pass
    } else if (legs_equal(tens_leg, std::dynamic_pointer_cast<Leg>(diag->leg()->dual_space()))) {
        diag_use = std::dynamic_pointer_cast<DiagonalTensor const>(transpose(diag));
    } else {
        throw std::invalid_argument("Incompatible legs");
    }

    if (auto dt = as_Diagonal(tensor)) {
        auto dt_mut = std::const_pointer_cast<DiagonalTensor>(dt);
        BlockBinaryFn mul_fn = [](BlockBackend::BlockPtr const& x,
                                  BlockBackend::BlockPtr const& y) { return (*x) * (*y); };
        auto prod = dt_mut->_binary_operand(
          std::const_pointer_cast<DiagonalTensor>(diag_use), std::move(mul_fn), "*");
        prod->labels.set(tensor->labels.as_list());
        return prod;
    }
    if (auto m = as_Mask(tensor)) {
        TensorPtr res;
        if (leg_idx == 0) {
            res = std::get<TensorPtr>(compose(diag_use, m));
        } else {
            res = std::get<TensorPtr>(compose(m, diag_use));
        }
        res->labels.set(tensor->labels.as_list());
        return res;
    }
    if (auto c = as_Charged(tensor)) {
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(scale_axis(
          std::static_pointer_cast<Tensor const>(c->invariant_part), diag_use, leg_idx));
        return make_charged_native(std::move(inv_part), c->charged_state);
    }
    auto backend = get_same_backend({ tensor, diag_use });
    auto data = backend->scale_axis(tensor, diag_use, leg_idx);
    // the labels of a HiddenLegTensor keep the hidden prefix -> wrap the result again
    return HiddenLegTensor::maybe_wrap(make_symmetric_native(
      std::move(data), tensor->codomain, tensor->domain, backend, tensor->labels.as_list()));
}

[[nodiscard]] TensorPtr
tensor_only_compose(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    if (std::holds_alternative<BlockBackend::Scalar>(v)) {
        throw std::logic_error("Expected tensor from compose");
    }
    return std::get<TensorPtr>(v);
}

std::variant<TensorPtr, BlockBackend::Scalar>
compose(TensorCPtr tensor1,
        TensorCPtr tensor2,
        std::optional<std::map<std::string, std::string>> relabel1,
        std::optional<std::map<std::string, std::string>> relabel2)
{
    (void)get_same_device({ tensor1, tensor2 });
    check_spaces_tp(tensor1->domain, tensor2->codomain);

    LegLabelList codomain_labels = apply_relabel(tensor1->codomain_labels(), relabel1);
    LegLabelList domain_labels = apply_relabel(tensor2->domain_labels(), relabel2);
    LegLabelList res_labels = nested_flat_labels(codomain_labels, domain_labels);

    if (auto m1 = as_Mask(tensor1)) {
        TensorPtr res = _compose_with_Mask(tensor2, std::const_pointer_cast<Mask>(m1), 0);
        res->labels[0] = m1->labels[0];
        return maybe_wrap_hidden(std::move(res), true);
    }
    if (auto m2 = as_Mask(tensor2)) {
        TensorPtr res = _compose_with_Mask(tensor1, std::const_pointer_cast<Mask>(m2), -1);
        res->labels[-1] = m2->labels[1];
        return maybe_wrap_hidden(std::move(res), true);
    }
    if (as_Identity(tensor1)) {
        return maybe_wrap_hidden(shallow_copy_labels(tensor2, res_labels), true);
    }
    if (as_Identity(tensor2)) {
        return maybe_wrap_hidden(shallow_copy_labels(tensor1, res_labels), true);
    }
    if (as_Diagonal(tensor1)) {
        TensorPtr res =
          scale_axis(tensor2,
                     std::dynamic_pointer_cast<DiagonalTensor const>(as_Diagonal(tensor1)),
                     int64{ 0 });
        res->labels.set(res_labels);
        return maybe_wrap_hidden(std::move(res), true);
    }
    if (as_Diagonal(tensor2)) {
        TensorPtr res =
          scale_axis(tensor1,
                     std::dynamic_pointer_cast<DiagonalTensor const>(as_Diagonal(tensor2)),
                     int64{ -1 });
        res->labels.set(res_labels);
        return maybe_wrap_hidden(std::move(res), true);
    }
    if (as_Charged(tensor1)) {
        TensorPtr res =
          partial_compose(tensor1, tensor2, tensor1->num_codomain_legs(), relabel1, relabel2);
        return res;
    }
    if (auto c2 = as_Charged(tensor2)) {
        auto comp = compose(
          tensor1, std::static_pointer_cast<Tensor const>(c2->invariant_part), relabel1, relabel2);
        if (std::holds_alternative<BlockBackend::Scalar>(comp)) {
            return comp;
        }
        return make_charged_native(
          std::dynamic_pointer_cast<SymmetricTensor>(std::get<TensorPtr>(comp)),
          c2->charged_state);
    }
    return maybe_wrap_hidden_variant(
      from_compose_sym_variant(_compose_SymmetricTensors(
        as_Symmetric(tensor1), as_Symmetric(tensor2), relabel1, relabel2)),
      true);
}

TensorPtr
partial_compose(TensorCPtr tensor1,
                TensorCPtr tensor2,
                LegRef tensor1_first_leg,
                std::optional<std::map<std::string, std::string>> relabel1,
                std::optional<std::map<std::string, std::string>> relabel2)
{
    if (auto c1 = as_Charged(tensor1)) {
        if (auto c2 = as_Charged(tensor2)) {
            std::string c = charge_leg_label();
            std::string c1l = c + "1";
            std::string c2l = c + "2";
            auto r1 = relabel_or_empty(relabel1);
            auto r2 = relabel_or_empty(relabel2);
            r1[c] = c1l;
            r2[c] = c2l;
            SymmetricTensorPtr inv_part = c2->invariant_part;
            int64 t1_first = leg_ref_index(tensor1, tensor1_first_leg);
            if (t1_first < tensor1->num_codomain_legs()) {
                inv_part = std::dynamic_pointer_cast<SymmetricTensor>(move_leg_wrap(
                  inv_part, c, c2->num_codomain_legs() - 1, std::nullopt, BendRight{ true }));
            }
            inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              partial_compose(std::static_pointer_cast<Tensor const>(c1->invariant_part),
                              std::static_pointer_cast<Tensor const>(inv_part),
                              tensor1_first_leg,
                              r1,
                              r2));
            inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              move_leg_wrap(inv_part, c2l, std::nullopt, 1, BendRight{ true }));
            return from_two_charge_legs_native(inv_part, c1->charged_state, c2->charged_state);
        }
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          partial_compose(std::static_pointer_cast<Tensor const>(c1->invariant_part),
                          tensor2,
                          tensor1_first_leg,
                          relabel1,
                          relabel2));
        return from_invariant_part_native(inv_part, c1->charged_state);
    }
    if (auto c2 = as_Charged(tensor2)) {
        SymmetricTensorPtr inv_part = c2->invariant_part;
        int64 t1_first = leg_ref_index(tensor1, tensor1_first_leg);
        if (t1_first < tensor1->num_codomain_legs()) {
            inv_part =
              std::dynamic_pointer_cast<SymmetricTensor>(move_leg_wrap(inv_part,
                                                                       charge_leg_label(),
                                                                       c2->num_codomain_legs() - 1,
                                                                       std::nullopt,
                                                                       BendRight{ true }));
        }
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          partial_compose(tensor1,
                          std::static_pointer_cast<Tensor const>(inv_part),
                          tensor1_first_leg,
                          relabel1,
                          relabel2));
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          move_leg_wrap(inv_part, charge_leg_label(), std::nullopt, 0, BendRight{ true }));
        return from_invariant_part_native(inv_part, c2->charged_state);
    }

    (void)get_same_device({ tensor1, tensor2 });
    int64 t1_first = leg_ref_index(tensor1, tensor1_first_leg);

    LegLabelList codomain_labels = apply_relabel(tensor1->codomain_labels(), relabel1);
    LegLabelList domain_labels = apply_relabel(tensor1->domain_labels(), relabel1);

    char const* leg_msg = "Not all legs to be contracted are in the (co)domain";
    char const* compose_msg = "Use compose for contracting the full (co)domain";
    char const* contract_msg = "Use compose or outer when no legs are to be contracted";

    TensorProduct::Ptr new_codomain;
    TensorProduct::Ptr new_domain;
    int64 const num_codomain_legs = tensor1->num_codomain_legs();

    if (t1_first < num_codomain_legs) {
        int64 const num_legs = tensor2->num_domain_legs();
        int64 const t1_last = t1_first + num_legs - 1;
        if (!(num_legs > 0)) {
            throw std::runtime_error(contract_msg);
        }
        if (!(t1_last < num_codomain_legs)) {
            throw std::runtime_error(leg_msg);
        }
        if (!(num_legs < num_codomain_legs)) {
            throw std::runtime_error(compose_msg);
        }
        std::vector<Leg::Ptr> factors1(tensor1->codomain->factors.begin() + t1_first,
                                       tensor1->codomain->factors.begin() + t1_last + 1);
        check_leg_vectors(factors1, tensor2->domain->factors);
        LegLabelList tensor2_labels = apply_relabel(tensor2->codomain_labels(), relabel2);
        codomain_labels.erase(codomain_labels.begin() + t1_first,
                              codomain_labels.begin() + t1_last + 1);
        codomain_labels.insert(
          codomain_labels.begin() + t1_first, tensor2_labels.begin(), tensor2_labels.end());

        std::vector<Leg::Ptr> new_factors = tensor1->codomain->factors;
        new_factors.erase(new_factors.begin() + t1_first, new_factors.begin() + t1_last + 1);
        new_factors.insert(new_factors.begin() + t1_first,
                           tensor2->codomain->factors.begin(),
                           tensor2->codomain->factors.end());
        new_codomain = std::make_shared<TensorProduct>(std::move(new_factors), tensor1->symmetry);
        new_domain = tensor1->domain;
    } else {
        int64 const num_legs = tensor2->num_codomain_legs();
        int64 const t1_last = t1_first + num_legs - 1;
        int64 const num_legs_t1 = tensor1->num_legs;
        int64 const num_domain_legs = tensor1->num_domain_legs();
        if (!(num_legs > 0)) {
            throw std::runtime_error(contract_msg);
        }
        if (!(t1_last < num_legs_t1)) {
            throw std::runtime_error(leg_msg);
        }
        if (!(num_legs < num_domain_legs)) {
            throw std::runtime_error(compose_msg);
        }
        int64 const domain_first_leg = num_legs_t1 - 1 - t1_last;
        int64 const domain_last_leg = num_legs_t1 - 1 - t1_first;
        std::vector<Leg::Ptr> factors1(tensor1->domain->factors.begin() + domain_first_leg,
                                       tensor1->domain->factors.begin() + domain_last_leg + 1);
        check_leg_vectors(factors1, tensor2->codomain->factors);
        LegLabelList tensor2_labels = apply_relabel(tensor2->domain_labels(), relabel2);
        domain_labels.erase(domain_labels.begin() + domain_first_leg,
                            domain_labels.begin() + domain_last_leg + 1);
        domain_labels.insert(
          domain_labels.begin() + domain_first_leg, tensor2_labels.begin(), tensor2_labels.end());

        new_codomain = tensor1->codomain;
        std::vector<Leg::Ptr> new_dom_factors = tensor1->domain->factors;
        new_dom_factors.erase(new_dom_factors.begin() + domain_first_leg,
                              new_dom_factors.begin() + domain_last_leg + 1);
        new_dom_factors.insert(new_dom_factors.begin() + domain_first_leg,
                               tensor2->domain->factors.begin(),
                               tensor2->domain->factors.end());
        new_domain =
          std::make_shared<TensorProduct>(std::move(new_dom_factors), tensor1->symmetry);
    }

    LegLabelList res_labels = codomain_labels;
    for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
        res_labels.push_back(*it);
    }
    {
        std::vector<std::string> named;
        for (auto const& lab : res_labels) {
            if (!lab.empty()) {
                named.push_back(lab);
            }
        }
        if (!duplicate_entries(named).empty()) {
            throw std::runtime_error("duplicate labels");
        }
    }

    if (as_Identity(tensor1)) {
        return maybe_wrap_hidden(shallow_copy_labels(tensor2, res_labels), true);
    }
    if (as_Identity(tensor2)) {
        return maybe_wrap_hidden(shallow_copy_labels(tensor1, res_labels), true);
    }
    if (auto m2 = as_Mask(tensor2)) {
        TensorPtr res = _compose_with_Mask(tensor1, std::const_pointer_cast<Mask>(m2), t1_first);
        res->labels.set(res_labels);
        return maybe_wrap_hidden(std::move(res), true);
    }
    if (auto d2 = as_Diagonal(tensor2)) {
        TensorPtr res = scale_axis(tensor1, d2, t1_first);
        res->labels.set(res_labels);
        return maybe_wrap_hidden(std::move(res), true);
    }

    auto backend = get_same_backend({ tensor1, tensor2 });
    auto data = backend->partial_compose(
      as_Symmetric(tensor1), as_Symmetric(tensor2), t1_first, new_codomain, new_domain);
    return maybe_wrap_hidden(
      make_symmetric_native(std::move(data), new_codomain, new_domain, backend, res_labels), true);
}

TensorPtr
outer(TensorCPtr tensor1,
      TensorCPtr tensor2,
      std::optional<std::map<std::string, std::string>> relabel1,
      std::optional<std::map<std::string, std::string>> relabel2)
{
    (void)get_same_device({ tensor1, tensor2 });
    if (!tensor1->symmetry->is_equivalent_to(*tensor2->symmetry)) {
        throw SymmetryError("outer requires equivalent symmetries");
    }

    TensorCPtr t1 = tensor1;
    TensorCPtr t2 = tensor2;
    if (as_Mask(t1) || as_Diagonal(t1)) {
        char const* msg =
          "Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to "
          "suppress the warning.";
        warn(msg);
        t1 = std::const_pointer_cast<Tensor>(t1)->as_SymmetricTensor();
    }
    if (as_Mask(t2) || as_Diagonal(t2)) {
        char const* msg =
          "Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to "
          "suppress the warning.";
        warn(msg);
        t2 = std::const_pointer_cast<Tensor>(t2)->as_SymmetricTensor();
    }

    if (auto c1 = as_Charged(t1)) {
        if (auto c2 = as_Charged(t2)) {
            std::string bang = charge_leg_label();
            auto r1 = relabel_or_empty(relabel1);
            auto r2 = relabel_or_empty(relabel2);
            r1[bang] = bang + "1";
            r2[bang] = bang + "2";
            SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              outer(std::static_pointer_cast<Tensor const>(c1->invariant_part),
                    std::static_pointer_cast<Tensor const>(c2->invariant_part),
                    r1,
                    r2));
            inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              move_leg_wrap(inv_part, bang + "2", std::nullopt, 1));
            return from_two_charge_legs_native(inv_part, c1->charged_state, c2->charged_state);
        }
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(outer(
          std::static_pointer_cast<Tensor const>(c1->invariant_part), t2, relabel1, relabel2));
        return make_charged_native(std::move(inv_part), c1->charged_state);
    }
    if (auto c2 = as_Charged(t2)) {
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(outer(
          t1, std::static_pointer_cast<Tensor const>(c2->invariant_part), relabel1, relabel2));
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          move_leg_wrap(inv_part, t1->num_codomain_legs() + c2->num_legs, std::nullopt, 0));
        return make_charged_native(std::move(inv_part), c2->charged_state);
    }
    if ((as_Hidden(t1) && as_Charged(t2)) || (as_Charged(t1) && as_Hidden(t2))) {
        throw std::invalid_argument(
          "Cannot outer ChargedTensor with HiddenLegTensor. Unhide or convert first.");
    }
    if (as_Hidden(t1) || as_Hidden(t2)) {
        auto pairs = implicit_hidden_contraction_pairs(t1, t2);
        if (!pairs.empty()) {
            std::vector<LegRef> legs1;
            std::vector<LegRef> legs2;
            for (auto const& [i1, i2] : pairs) {
                legs1.emplace_back(i1);
                legs2.emplace_back(i2);
            }
            return std::get<TensorPtr>(tdot(t1, t2, legs1, legs2, relabel1, relabel2));
        }
    }

    auto s1 = as_Symmetric(t1);
    auto s2 = as_Symmetric(t2);
    auto backend = get_same_backend({ t1, t2 });
    auto data = backend->outer(s1, s2);
    auto codomain = TensorProduct::from_partial_products({ s1->codomain, s2->codomain });
    auto domain = TensorProduct::from_partial_products({ s1->domain, s2->domain });

    LegLabelList codomain_labels = apply_relabel(s1->codomain_labels(), relabel1);
    LegLabelList domain_labels = apply_relabel(s1->domain_labels(), relabel1);
    {
        auto c2 = apply_relabel(s2->codomain_labels(), relabel2);
        auto d2 = apply_relabel(s2->domain_labels(), relabel2);
        codomain_labels.insert(codomain_labels.end(), c2.begin(), c2.end());
        domain_labels.insert(domain_labels.end(), d2.begin(), d2.end());
    }
    LegLabelList flat = nested_flat_labels(codomain_labels, domain_labels);
    return maybe_wrap_hidden(
      make_symmetric_native(std::move(data), codomain, domain, backend, flat),
      as_Hidden(t1) || as_Hidden(t2));
}

std::variant<TensorPtr, BlockBackend::Scalar>
partial_trace(TensorCPtr tensor,
              std::vector<std::vector<LegRef>> pairs,
              std::optional<LevelsSpec> levels)
{
    std::vector<std::pair<int64, int64>> parsed_pairs;
    parsed_pairs.reserve(pairs.size());
    std::vector<int64> traced_idcs;
    for (auto const& pair : pairs) {
        if (pair.size() != 2) {
            throw std::invalid_argument("Each pair must have two legs");
        }
        int64 i1 = leg_ref_index(tensor, pair.at(0));
        int64 i2 = leg_ref_index(tensor, pair.at(1));
        parsed_pairs.emplace_back(i1, i2);
        traced_idcs.push_back(i1);
        traced_idcs.push_back(i2);
    }
    if (!duplicate_entries(traced_idcs).empty()) {
        throw std::invalid_argument("Pairs may not contain duplicates.");
    }
    {
        std::vector<Leg::Ptr> as_cod;
        std::vector<Leg::Ptr> as_dom;
        for (auto const& [i1, i2] : parsed_pairs) {
            as_cod.push_back(tensor->_as_codomain_leg(i1));
            as_dom.push_back(tensor->_as_domain_leg(i2));
        }
        check_leg_vectors(as_cod, as_dom);
    }

    if (pairs.empty()) {
        return std::const_pointer_cast<Tensor>(tensor);
    }
    if (as_Diagonal(tensor) || as_Mask(tensor)) {
        return trace(tensor);
    }
    if (auto c = as_Charged(tensor)) {
        LevelsSpec levels_use = levels.value_or(LevelsSpec{});
        if (levels.has_value()) {
            auto min_v = levels_min(*levels);
            if (min_v.has_value()) {
                levels_use = *levels;
                levels_use.push_back(*min_v - 1);
            }
        }
        auto inv_res = partial_trace(
          std::static_pointer_cast<Tensor const>(c->invariant_part), pairs, levels_use);
        if (std::holds_alternative<BlockBackend::Scalar>(inv_res)) {
            throw std::logic_error("partial_trace charged invariant scalar unexpected");
        }
        TensorPtr inv_part = std::get<TensorPtr>(inv_res);
        if (inv_part->num_legs == 1) {
            auto bb = c->backend->block_backend;
            auto inv_block = std::dynamic_pointer_cast<SymmetricTensor>(inv_part)->to_dense_block(
              std::nullopt, std::nullopt, true);
            auto res = bb->tdot(inv_block, c->charged_state, { 0 }, { 0 });
            return bb->item(res);
        }
        return make_charged_native(std::dynamic_pointer_cast<SymmetricTensor>(inv_part),
                                   c->charged_state);
    }
    if (as_Hidden(tensor)) {
        std::vector<int64> all_traced;
        for (auto const& [i1, i2] : parsed_pairs) {
            all_traced.push_back(i1);
            all_traced.push_back(i2);
        }
        reject_hidden_leg_arguments(tensor, all_traced, "partial_trace");
    }
    auto s = as_Symmetric(tensor);
    if (!s) {
        throw std::invalid_argument(
          std::format("Unexpected tensor type: {}", tensor->class_name()));
    }

    LevelsSpec levels_vec;
    if (!levels.has_value()) {
        levels_vec.assign(static_cast<std::size_t>(s->num_legs), std::nullopt);
    } else {
        levels_vec = *levels;
    }

    auto backend = s->backend;
    TensorBackend::DataPtr data;
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    try {
        auto traced = backend->partial_trace(s, parsed_pairs, levels_vec);
        data = std::move(std::get<0>(traced));
        codomain = std::move(std::get<1>(traced));
        domain = std::move(std::get<2>(traced));
    } catch (...) {
        handle_permute_legs_symmetry_error();
    }

    if (s->num_legs == static_cast<int64>(traced_idcs.size())) {
        return backend->data_item(std::move(data));
    }
    std::unordered_set<int64> traced_set;
    for (auto const& [i1, i2] : parsed_pairs) {
        traced_set.insert(i1);
        traced_set.insert(i2);
    }
    LegLabelList labels;
    LegLabelList all_labels = s->labels.as_list();
    for (std::size_t n = 0; n < all_labels.size(); ++n) {
        if (!traced_set.contains(static_cast<int64>(n))) {
            labels.push_back(all_labels[n]);
        }
    }
    return maybe_wrap_hidden(
      make_symmetric_native(std::move(data), codomain, domain, backend, labels),
      static_cast<bool>(as_Hidden(tensor)));
}

std::variant<TensorPtr, BlockBackend::Scalar>
tdot(TensorCPtr tensor1,
     TensorCPtr tensor2,
     std::vector<LegRef> legs1,
     std::vector<LegRef> legs2,
     std::optional<std::map<std::string, std::string>> relabel1,
     std::optional<std::map<std::string, std::string>> relabel2)
{
    (void)get_same_device({ tensor1, tensor2 });

    std::vector<int64> legs1_v = tensor1->labels.indices(legs1);
    std::vector<int64> legs2_v = tensor2->labels.indices(legs2);
    if (!duplicate_entries(legs1_v).empty() || !duplicate_entries(legs2_v).empty()) {
        throw std::invalid_argument("Duplicate leg entries.");
    }
    if (legs1_v.size() != legs2_v.size()) {
        throw std::invalid_argument("legs1 and legs2 must have the same length");
    }
    {
        std::vector<Leg::Ptr> as_dom;
        std::vector<Leg::Ptr> as_cod;
        for (std::size_t i = 0; i < legs1_v.size(); ++i) {
            as_dom.push_back(tensor1->_as_domain_leg(legs1_v[i]));
            as_cod.push_back(tensor2->_as_codomain_leg(legs2_v[i]));
        }
        check_leg_vectors(as_dom, as_cod);
    }

    bool do_relabel =
      (relabel1.has_value() && !relabel1->empty()) || (relabel2.has_value() && !relabel2->empty());
    if (do_relabel) {
        auto hidden_pairs = implicit_hidden_contraction_pairs(tensor1, tensor2);
        std::unordered_set<int64> skip1(legs1_v.begin(), legs1_v.end());
        std::unordered_set<int64> skip2(legs2_v.begin(), legs2_v.end());
        for (auto const& [i1, i2] : hidden_pairs) {
            skip1.insert(i1);
            skip2.insert(i2);
        }
        LegLabelList codomain_labels;
        LegLabelList all1 = tensor1->labels.as_list();
        for (std::size_t n = 0; n < all1.size(); ++n) {
            if (!skip1.contains(static_cast<int64>(n))) {
                codomain_labels.push_back(relabel_one(all1[n], relabel1));
            }
        }
        LegLabelList domain_labels;
        LegLabelList all2 = tensor2->labels.as_list();
        for (std::size_t n = 0; n < all2.size(); ++n) {
            if (!skip2.contains(static_cast<int64>(n))) {
                domain_labels.push_back(relabel_one(all2[n], relabel2));
            }
        }
        auto res = tdot(tensor1, tensor2, legs1, legs2);
        if (std::holds_alternative<BlockBackend::Scalar>(res)) {
            return res;
        }
        LegLabelList flat = codomain_labels;
        flat.insert(flat.end(), domain_labels.begin(), domain_labels.end());
        std::get<TensorPtr>(res)->labels.set(flat);
        return res;
    }

    int64 num_contr = static_cast<int64>(legs1_v.size());
    TensorPtr work1 = std::const_pointer_cast<Tensor>(tensor1);
    TensorPtr work2 = std::const_pointer_cast<Tensor>(tensor2);
    bool const wrap_hidden = static_cast<bool>(as_Hidden(tensor1) || as_Hidden(tensor2));

    auto require_tensor = [](std::variant<TensorPtr, BlockBackend::Scalar> v,
                             char const* ctx) -> TensorPtr {
        if (!std::holds_alternative<TensorPtr>(v)) {
            throw std::logic_error(std::format("{}: expected tensor, got scalar", ctx));
        }
        return std::get<TensorPtr>(std::move(v));
    };

    // Deal with Masks: either return or reduce to SymmetricTensor
    if (auto m1 = as_Mask(work1)) {
        if (num_contr == 0) {
            work1 = std::const_pointer_cast<Mask>(m1)->as_SymmetricTensor();
        } else if (num_contr == 1) {
            bool t1_in_domain = legs1_v[0] == 1;
            bool t2_in_domain = legs2_v[0] >= work2->num_codomain_legs();
            MaskCPtr mask_use = m1;
            if (t2_in_domain == t1_in_domain) {
                mask_use = std::dynamic_pointer_cast<Mask const>(transpose(m1));
            }
            TensorPtr res = _compose_with_Mask(work2, mask_use, legs2_v[0]);
            res->labels[legs2_v[0]] = m1->labels[static_cast<int64>(1 - legs1_v[0])];
            try {
                return permute_legs_wrap(res, leg_refs_from_ints(legs1_v));
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            bool is_proj = m1->is_projection;
            auto which_is_large = static_cast<std::size_t>(
              std::find(legs1_v.begin(), legs1_v.end(), is_proj ? 1 : 0) - legs1_v.begin());
            bool t1_in_domain = is_proj;
            bool t2_in_domain = legs2_v[which_is_large] >= work2->num_codomain_legs();
            MaskCPtr mask_use = m1;
            if (t1_in_domain == t2_in_domain) {
                mask_use = std::dynamic_pointer_cast<Mask const>(transpose(m1));
            }
            TensorPtr res = _compose_with_Mask(work2, mask_use, legs2_v[which_is_large]);
            auto traced = partial_trace(res, { { legs2_v[0], legs2_v[1] } });
            if (work2->num_legs == 2) {
                return traced;
            }
            return bend_legs_wrap(require_tensor(std::move(traced), "tdot Mask num_contr=2"), 0);
        }
    }
    if (auto m2 = as_Mask(work2)) {
        if (num_contr == 0) {
            work2 = std::const_pointer_cast<Mask>(m2)->as_SymmetricTensor();
        } else if (num_contr == 1) {
            bool t1_in_domain = legs1_v[0] >= work1->num_codomain_legs();
            bool t2_in_domain = legs2_v[0] == 1;
            MaskCPtr mask_use = m2;
            if (t1_in_domain == t2_in_domain) {
                mask_use = std::dynamic_pointer_cast<Mask const>(transpose(m2));
            }
            TensorPtr res = _compose_with_Mask(work1, mask_use, legs1_v[0]);
            res->labels[legs1_v[0]] = m2->labels[static_cast<int64>(1 - legs2_v[0])];
            try {
                return permute_legs_wrap(res, std::nullopt, leg_refs_from_ints(legs2_v));
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            bool is_proj = m2->is_projection;
            auto which_is_large = static_cast<std::size_t>(
              std::find(legs2_v.begin(), legs2_v.end(), is_proj ? 1 : 0) - legs2_v.begin());
            bool t1_in_domain = legs1_v[which_is_large] >= work1->num_codomain_legs();
            bool t2_in_domain = is_proj;
            MaskCPtr mask_use = m2;
            if (t1_in_domain == t2_in_domain) {
                mask_use = std::dynamic_pointer_cast<Mask const>(transpose(m2));
            }
            TensorPtr res = _compose_with_Mask(work1, mask_use, legs1_v[which_is_large]);
            auto traced = partial_trace(res, { { legs1_v[0], legs1_v[1] } });
            if (work1->num_legs == 2) {
                return traced;
            }
            return bend_legs_wrap(
              require_tensor(std::move(traced), "tdot Mask2 num_contr=2"), std::nullopt, 0);
        }
    }

    if (auto id1 = as_Identity(work1)) {
        if (num_contr == 1) {
            TensorPtr res = permute_legs_wrap(work2, leg_refs_from_ints(legs2_v));
            res->labels[0] = id1->labels[static_cast<int64>(1 - legs1_v[0])];
            return res;
        }
        if (num_contr == 2) {
            auto traced = partial_trace(work2, { { legs2_v[0], legs2_v[1] } });
            return bend_legs_wrap(require_tensor(std::move(traced), "tdot Identity"), 0);
        }
        work1 = std::const_pointer_cast<Identity>(id1)->as_DiagonalTensor();
    }

    if (auto id2 = as_Identity(work2)) {
        if (num_contr == 1) {
            // Match Python (computes permute then ignores it):
            (void)permute_legs_wrap(work1, std::nullopt, leg_refs_from_ints(legs1_v));
            TensorPtr res = work1->copy(/*deep=*/false);
            res->labels[legs1_v[0]] = id2->labels[static_cast<int64>(1 - legs2_v[0])];
            return res;
        }
        if (num_contr == 2) {
            auto traced = partial_trace(work1, { { legs1_v[0], legs1_v[1] } });
            return bend_legs_wrap(
              require_tensor(std::move(traced), "tdot Identity2"), std::nullopt, 0);
        }
        work2 = std::const_pointer_cast<Identity>(id2)->as_DiagonalTensor();
    }

    // Deal with DiagonalTensor (Identity already reduced above when num_contr==0)
    if (auto d1 = as_Diagonal(work1)) {
        if (num_contr == 0) {
            work1 = std::const_pointer_cast<DiagonalTensor>(d1)->as_SymmetricTensor();
        } else if (num_contr == 1) {
            TensorPtr res = scale_axis(work2, d1, legs2_v[0]);
            res->labels[legs2_v[0]] = d1->labels[static_cast<int64>(1 - legs1_v[0])];
            try {
                return permute_legs_wrap(res, leg_refs_from_ints(legs1_v));
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            TensorPtr res = scale_axis(work2, d1, legs2_v[0]);
            auto traced = partial_trace(res, { { legs2_v[0], legs2_v[1] } });
            if (work2->num_legs == 2) {
                return traced;
            }
            return bend_legs_wrap(require_tensor(std::move(traced), "tdot Diagonal"), 0);
        }
    }
    if (auto d2 = as_Diagonal(work2)) {
        if (num_contr == 0) {
            work2 = std::const_pointer_cast<DiagonalTensor>(d2)->as_SymmetricTensor();
        } else if (num_contr == 1) {
            TensorPtr res = scale_axis(work1, d2, legs1_v[0]);
            res->labels[legs1_v[0]] = d2->labels[static_cast<int64>(1 - legs2_v[0])];
            try {
                return permute_legs_wrap(res, std::nullopt, leg_refs_from_ints(legs1_v));
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            TensorPtr res = scale_axis(work1, d2, legs1_v[0]);
            auto traced = partial_trace(res, { { legs1_v[0], legs1_v[1] } });
            if (work1->num_legs == 2) {
                return traced;
            }
            return bend_legs_wrap(
              require_tensor(std::move(traced), "tdot Diagonal2"), std::nullopt, 0);
        }
    }

    // Deal with ChargedTensor / HiddenLegTensor
    if ((as_Charged(work1) && as_Hidden(work2)) || (as_Hidden(work1) && as_Charged(work2))) {
        throw std::invalid_argument(
          "Cannot tdot ChargedTensor with HiddenLegTensor. Unhide or convert first.");
    }
    if (as_Hidden(work1) || as_Hidden(work2)) {
        reject_hidden_leg_arguments(work1, legs1_v, "tdot");
        reject_hidden_leg_arguments(work2, legs2_v, "tdot");
        auto pairs = implicit_hidden_contraction_pairs(work1, work2);
        for (auto const& [i1, i2] : pairs) {
            legs1_v.push_back(i1);
            legs2_v.push_back(i2);
        }
        legs1.clear();
        legs2.clear();
        for (auto i : legs1_v) {
            legs1.emplace_back(i);
        }
        for (auto i : legs2_v) {
            legs2.emplace_back(i);
        }
        num_contr = static_cast<int64>(legs1_v.size());
        (void)num_contr;
    }

    if (auto c1 = as_Charged(work1)) {
        if (auto c2 = as_Charged(work2)) {
            std::string c = charge_leg_label();
            auto inv_v = tdot(std::static_pointer_cast<Tensor const>(c1->invariant_part),
                              std::static_pointer_cast<Tensor const>(c2->invariant_part),
                              legs1,
                              legs2,
                              std::map<std::string, std::string>{ { c, c + "1" } },
                              std::map<std::string, std::string>{ { c, c + "2" } });
            SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              require_tensor(std::move(inv_v), "tdot charged×charged"));
            inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              move_leg_wrap(inv_part, c + "1", std::nullopt, 0));
            // Full contraction of physical legs can leave only the charge leg → scalar.
            return from_two_charge_legs_variant(inv_part, c1->charged_state, c2->charged_state);
        }
        auto inv_v =
          tdot(std::static_pointer_cast<Tensor const>(c1->invariant_part), work2, legs1, legs2);
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          require_tensor(std::move(inv_v), "tdot charged×sym"));
        inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          move_leg_wrap(inv_part, charge_leg_label(), std::nullopt, 0));
        return from_invariant_part_variant(inv_part, c1->charged_state);
    }
    if (auto c2 = as_Charged(work2)) {
        auto inv_v =
          tdot(work1, std::static_pointer_cast<Tensor const>(c2->invariant_part), legs1, legs2);
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          require_tensor(std::move(inv_v), "tdot sym×charged"));
        return from_invariant_part_variant(inv_part, c2->charged_state);
    }

    // Remaining case: both are SymmetricTensor (including HiddenLegTensor)
    std::optional<BendRight> bend_opt =
      wrap_hidden ? std::optional<BendRight>(BendRight{ true }) : std::nullopt;
    TensorPtr t1;
    TensorPtr t2;
    try {
        t1 = permute_legs_wrap(work1, std::nullopt, leg_refs_from_ints(legs1_v), bend_opt);
        t2 = permute_legs_wrap(work2, leg_refs_from_ints(legs2_v), std::nullopt, bend_opt);
    } catch (...) {
        handle_permute_legs_symmetry_error();
    }
    return maybe_wrap_hidden_variant(
      from_compose_sym_variant(_compose_SymmetricTensors(as_Symmetric(t1), as_Symmetric(t2))),
      wrap_hidden);
}

[[nodiscard]] BlockBackend::Scalar
scalar_from_compose(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    if (std::holds_alternative<BlockBackend::Scalar>(v)) {
        return std::get<BlockBackend::Scalar>(v);
    }
    return trace(std::get<TensorPtr>(v));
}

BlockBackend::Scalar
trace(TensorCPtr tensor)
{
    if (as_Hidden(tensor)) {
        require_no_remaining_hidden(tensor, "trace");
    }
    check_spaces_tp(tensor->domain, tensor->codomain);
    if (auto id = as_Identity(tensor)) {
        return id->backend->block_backend->as_scalar(static_cast<float64>(id->leg()->Space::dim));
    }
    if (auto d = as_Diagonal(tensor)) {
        return d->backend->diagonal_tensor_trace_full(d);
    }
    if (auto c = as_Charged(tensor)) {
        int64 const N = c->num_legs;
        int64 const n_cod = c->num_codomain_legs();
        std::vector<std::vector<LegRef>> pairs;
        pairs.reserve(static_cast<std::size_t>(n_cod));
        for (int64 n = 0; n < n_cod; ++n) {
            pairs.push_back({ n, N - 1 - n });
        }
        auto res = partial_trace(c, pairs);
        if (std::holds_alternative<BlockBackend::Scalar>(res)) {
            return std::get<BlockBackend::Scalar>(res);
        }
        throw std::logic_error("trace(ChargedTensor): expected scalar from partial_trace");
    }
    return as_Symmetric(tensor)->backend->trace_full(as_Symmetric(tensor), {}, {});
}

BlockBackend::Scalar
inner(TensorCPtr A, TensorCPtr B, bool do_dagger)
{
    (void)get_same_device({ A, B });
    if (do_dagger) {
        check_spaces_tensors(A->codomain, A->domain, B->codomain, B->domain);
    } else {
        check_spaces_tensors(A->codomain, A->domain, B->domain, B->codomain);
    }

    if (as_Identity(A)) {
        return trace(B);
    }
    if (as_Identity(B)) {
        if (do_dagger) {
            return trace(A).conj();
        }
        return trace(A);
    }
    if (as_Diagonal(A) || as_Mask(A)) {
        if (do_dagger) {
            return scalar_from_compose(compose(dagger(A), B));
        }
        return scalar_from_compose(compose(A, B));
    }
    if (as_Diagonal(B) || as_Mask(B)) {
        if (do_dagger) {
            return scalar_from_compose(compose(dagger(B), A)).conj();
        }
        return scalar_from_compose(compose(A, B));
    }

    auto backend = get_same_backend({ A, B });

    if (as_Hidden(A) || as_Hidden(B)) {
        TensorCPtr left = do_dagger ? dagger(A) : A;
        (void)implicit_hidden_contraction_pairs(left, B);
        return backend->inner(as_Symmetric(left), as_Symmetric(B), /*do_dagger=*/false);
    }

    if (as_Charged(A) && as_Charged(B)) {
        auto bb = backend->block_backend;
        auto ac = as_Charged(A);
        auto bc = as_Charged(B);
        if (do_dagger) {
            SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              from_compose_sym(_compose_SymmetricTensors(
                std::dynamic_pointer_cast<SymmetricTensor const>(
                  bend_legs_wrap(dagger(ac->invariant_part), 1, std::nullopt)),
                std::dynamic_pointer_cast<SymmetricTensor const>(
                  bend_legs_wrap(bc->invariant_part, std::nullopt, 1)))));
            auto inv_b = inv_part->to_dense_block(std::nullopt, std::nullopt, true);
            auto tmp = bb->tdot(inv_b, bc->charged_state, { 1 }, { 0 });
            auto res = bb->tdot(bb->conj(ac->charged_state), tmp, { 0 }, { 0 });
            return bb->item(res);
        }
        int64 const n_legs = ac->num_legs;
        std::vector<int64> rev_legs;
        for (int64 i = n_legs - 1; i >= 0; --i) {
            rev_legs.push_back(i);
        }
        std::vector<std::optional<bool>> bend_opts(static_cast<std::size_t>(n_legs), true);
        bend_opts.push_back(false);
        SymmetricTensorPtr A_inv = std::dynamic_pointer_cast<SymmetricTensor>(
          permute_legs_wrap(ac->invariant_part,
                            leg_refs_from_ints({ -1 }),
                            leg_refs_from_ints(rev_legs),
                            BendRight{ std::move(bend_opts) }));
        std::vector<int64> fwd_legs(static_cast<std::size_t>(n_legs));
        std::iota(fwd_legs.begin(), fwd_legs.end(), 0);
        SymmetricTensorPtr B_inv = std::dynamic_pointer_cast<SymmetricTensor>(
          permute_legs_wrap(bc->invariant_part,
                            leg_refs_from_ints(fwd_legs),
                            leg_refs_from_ints({ -1 }),
                            BendRight{ true }));
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(from_compose_sym(
          _compose_SymmetricTensors(A_inv,
                                    B_inv,
                                    std::map<std::string, std::string>{ { "!", "!A" } },
                                    std::map<std::string, std::string>{ { "!", "!B" } })));
        auto inv_b = inv_part->to_dense_block(std::nullopt, std::nullopt, true);
        auto res = bb->tdot(inv_b, bc->charged_state, { 1 }, { 0 });
        res = bb->tdot(ac->charged_state, res, { 0 }, { 0 });
        return bb->item(res);
    }

    if (as_Charged(A)) {
        if (do_dagger) {
            return inner(B, A, true).conj();
        }
        return inner(B, A, false);
    }

    if (auto bc = as_Charged(B)) {
        auto bb = backend->block_backend;
        auto charge_space = std::dynamic_pointer_cast<Space const>(bc->charge_leg);
        if (charge_space &&
            charge_space->sector_multiplicity(charge_space->symmetry->trivial_sector) == 0) {
            Dtype dt = dtype::common({ A->dtype, B->dtype });
            return bb->as_scalar(static_cast<float64>(0.), dt);
        }
        int64 const nA = A->num_legs;
        std::vector<LegRef> legsA;
        std::vector<LegRef> legsB;
        legsA.reserve(static_cast<std::size_t>(nA));
        legsB.reserve(static_cast<std::size_t>(nA));
        for (int64 i = 0; i < nA; ++i) {
            legsA.emplace_back(i);
            legsB.emplace_back(nA - 1 - i);
        }
        if (do_dagger) {
            auto inv_v = tdot(dagger(A), bc->invariant_part, legsA, legsB);
            SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
              std::holds_alternative<TensorPtr>(inv_v)
                ? std::get<TensorPtr>(inv_v)
                : throw std::logic_error("inner tdot scalar"));
            auto B_state = bb->conj(bc->charged_state);
            auto res = bb->tdot(inv_part->to_dense_block(), B_state, { 0 }, { 0 });
            return bb->item(res);
        }
        auto inv_v = tdot(A, bc->invariant_part, legsA, legsB);
        SymmetricTensorPtr inv_part = std::dynamic_pointer_cast<SymmetricTensor>(
          std::holds_alternative<TensorPtr>(inv_v) ? std::get<TensorPtr>(inv_v)
                                                   : throw std::logic_error("inner tdot scalar"));
        auto res = bb->tdot(inv_part->to_dense_block(), bc->charged_state, { 0 }, { 0 });
        return bb->item(res);
    }

    return backend->inner(as_Symmetric(A), as_Symmetric(B), do_dagger);
}

BlockBackend::Scalar
inner(VectorLikeCPtr A, VectorLikeCPtr B, bool do_dagger)
{
    if (!A || !B) {
        throw std::invalid_argument("inner() requires non-null VectorLike arguments");
    }
    if (auto ta = std::dynamic_pointer_cast<Tensor const>(A)) {
        if (auto tb = std::dynamic_pointer_cast<Tensor const>(B)) {
            return inner(ta, tb, do_dagger);
        }
    }
    return A->vector_inner(std::move(B), do_dagger);
}

BlockBackend::Scalar
norm(VectorLikeCPtr vec)
{
    if (!vec) {
        throw std::invalid_argument("norm() requires a non-null VectorLike");
    }
    if (auto t = std::dynamic_pointer_cast<Tensor const>(vec)) {
        return norm(t);
    }
    return vec->vector_norm();
}

VectorLikePtr
linear_combination(BlockBackend::Scalar const& a,
                   VectorLikeCPtr v,
                   BlockBackend::Scalar const& b,
                   VectorLikeCPtr w)
{
    if (!v || !w) {
        throw std::invalid_argument("linear_combination() requires non-null VectorLike arguments");
    }
    if (auto tv = std::dynamic_pointer_cast<Tensor const>(v)) {
        auto tw = std::dynamic_pointer_cast<Tensor const>(w);
        if (!tw) {
            throw std::invalid_argument(
              "linear_combination: mixed Tensor / non-Tensor VectorLike arguments");
        }
        return linear_combination(a, std::move(tv), b, std::move(tw));
    }
    return v->axpy(a, w->scaled(b));
}

VectorLikePtr
scalar_multiply(BlockBackend::Scalar const& a, VectorLikeCPtr v)
{
    if (!v) {
        throw std::invalid_argument("scalar_multiply() requires a non-null VectorLike");
    }
    if (auto t = std::dynamic_pointer_cast<Tensor const>(v)) {
        return scalar_multiply(a, std::move(t));
    }
    return v->scaled(a);
}
} // namespace lasyten
