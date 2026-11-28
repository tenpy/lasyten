#include <lasyten/tensors/ops_legs.h>

#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/trees.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/constructors.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/planar.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>
#include <lasyten/tools/misc.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

[[nodiscard]] DiagonalTensorCPtr
as_Diagonal(TensorCPtr t)
{
    return std::dynamic_pointer_cast<DiagonalTensor const>(t);
}
[[nodiscard]] MaskCPtr
as_Mask(TensorCPtr t)
{
    return std::dynamic_pointer_cast<Mask const>(t);
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

[[nodiscard]] ChargedTensorPtr
make_charged_native(SymmetricTensorPtr inv_part, BlockBackend::BlockPtr charged_state)
{
    return std::make_shared<ChargedTensor>(std::move(inv_part), std::move(charged_state));
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
    if (as_Diagonal(result) || as_Mask(result)) {
        return result;
    }
    if (auto sym = std::dynamic_pointer_cast<SymmetricTensor>(result)) {
        return HiddenLegTensor::maybe_wrap(std::move(sym));
    }
    return result;
}

[[nodiscard]] TensorPtr
as_tensor_ptr(TensorCPtr t)
{
    return std::const_pointer_cast<Tensor>(t);
}

[[nodiscard]] bool
contains_int(std::vector<int64> const& v, int64 x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

/// One-level factors of a pipe (`LegPipe` or `AbelianLegPipe`).
[[nodiscard]] std::vector<Leg::Ptr>
pipe_legs(Leg::Ptr const& pipe)
{
    if (auto alp = std::dynamic_pointer_cast<AbelianLegPipe>(pipe.as_elementary_space())) {
        std::vector<Leg::Ptr> out;
        out.reserve(alp->legs.size());
        for (auto const& leg : alp->legs) {
            out.push_back(Leg{ leg });
        }
        return out;
    }
    if (auto lp = pipe.as_leg_pipe()) {
        return lp->legs;
    }
    throw std::invalid_argument("expected a pipe (LegPipe or AbelianLegPipe)");
}

[[nodiscard]] int64
pipe_num_legs(Leg::Ptr const& pipe)
{
    if (auto alp = std::dynamic_pointer_cast<AbelianLegPipe>(pipe.as_elementary_space())) {
        return alp->num_legs;
    }
    if (auto lp = pipe.as_leg_pipe()) {
        return lp->num_legs;
    }
    throw std::invalid_argument("expected a pipe (LegPipe or AbelianLegPipe)");
}

[[nodiscard]] ElementarySpace::Ptr
strip_abelian_leg_pipe(AbelianLegPipe::Ptr const& alp)
{
    std::optional<std::vector<int64>> bperm;
    if (alp->has_custom_basis_perm()) {
        bperm = alp->basis_perm();
    }
    return std::make_shared<ElementarySpace>(
      alp->symmetry, alp->defining_sectors, alp->multiplicities, alp->is_dual, std::move(bperm));
}

[[nodiscard]] std::vector<int64>
resolve_leg_refs(TensorCPtr const& tensor, std::vector<LegRef> const& refs)
{
    std::vector<std::variant<int64, std::string>> keys(refs.begin(), refs.end());
    return tensor->labels.indices(keys);
}

[[nodiscard]] std::vector<int64>
resolve_leg_refs_opt(TensorCPtr const& tensor, std::optional<std::vector<LegRef>> const& refs)
{
    if (!refs.has_value()) {
        return {};
    }
    return resolve_leg_refs(tensor, *refs);
}

[[nodiscard]] TensorProduct::Ptr
make_tensor_product(std::vector<Leg::Ptr> factors,
                    Symmetry::Ptr symmetry,
                    std::optional<SectorArray> sectors = std::nullopt,
                    std::optional<std::vector<int64>> mults = std::nullopt)
{
    return std::make_shared<TensorProduct>(
      std::move(factors), std::move(symmetry), std::move(sectors), std::move(mults));
}

[[nodiscard]] LegLabelList
flat_labels(LegLabelList const& codomain_labels, LegLabelList const& domain_labels)
{
    // Flat leg labels are ``codomain_labels + reversed(domain_labels)``, matching
    // ``parse_tensor_init_labels`` / ``nested_flat_labels`` and Python SymmetricTensor init.
    LegLabelList out = codomain_labels;
    out.insert(out.end(), domain_labels.rbegin(), domain_labels.rend());
    return out;
}

[[nodiscard]] LevelsSpec
normalize_levels(std::optional<LevelsSpec> const& levels, int64 num_legs)
{
    if (!levels.has_value()) {
        return LevelsSpec(static_cast<std::size_t>(num_legs), std::nullopt);
    }
    if (static_cast<int64>(levels->size()) != num_legs) {
        throw std::invalid_argument(
          std::format("expected {} levels, got {}", num_legs, levels->size()));
    }
    return *levels;
}

[[nodiscard]] std::vector<std::optional<bool>>
normalize_bend_right(std::optional<BendRight> const& bend_right, int64 num_legs)
{
    std::vector<std::optional<bool>> bend_right_v;
    bend_right_v.reserve(static_cast<std::size_t>(num_legs));
    if (!bend_right.has_value()) {
        bend_right_v.assign(static_cast<std::size_t>(num_legs), std::nullopt);
        return bend_right_v;
    }
    return std::visit(
      [&](auto const& spec) -> std::vector<std::optional<bool>> {
          using T = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<T, bool>) {
              return std::vector<std::optional<bool>>(static_cast<std::size_t>(num_legs), spec);
          } else {
              if (static_cast<int64>(spec.size()) != num_legs) {
                  throw std::invalid_argument(
                    std::format("expected {} bend_right entries, got {}", num_legs, spec.size()));
              }
              return spec;
          }
      },
      *bend_right);
}

[[nodiscard]] std::vector<bool>
normalize_pipe_dualities(std::optional<PipeDualities> const& pipe_dualities, std::size_t n)
{
    if (!pipe_dualities.has_value()) {
        return std::vector<bool>(n, false);
    }
    return std::visit(
      [&](auto const& spec) -> std::vector<bool> {
          using T = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<T, bool>) {
              return std::vector<bool>(n, spec);
          } else {
              if (spec.size() != n) {
                  throw std::invalid_argument(
                    std::format("expected {} pipe_dualities entries, got {}", n, spec.size()));
              }
              return spec;
          }
      },
      *pipe_dualities);
}

[[nodiscard]] std::string
format_optional_labels(LegLabelList const& labs)
{
    std::string out = "[";
    for (std::size_t i = 0; i < labs.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        if (!labs[i].empty()) {
            out += '\'';
            out += labs[i];
            out += '\'';
        } else {
            out += "None";
        }
    }
    out += ']';
    return out;
}

} // namespace

TensorPtr
bend_legs(TensorCPtr tensor,
          std::optional<int64> num_codomain_legs,
          std::optional<int64> num_domain_legs)
{
    if (!num_codomain_legs.has_value() && !num_domain_legs.has_value()) {
        throw std::invalid_argument("Must specify either num_codomain_legs or num_domain_legs");
    }
    int64 num_legs = tensor->num_legs;
    int64 n_cod;
    int64 n_dom;
    if (!num_domain_legs.has_value()) {
        n_cod = *num_codomain_legs;
        n_dom = num_legs - n_cod;
    } else if (!num_codomain_legs.has_value()) {
        n_dom = *num_domain_legs;
        n_cod = num_legs - n_dom;
    } else {
        n_cod = *num_codomain_legs;
        n_dom = *num_domain_legs;
        if (n_cod + n_dom != num_legs) {
            throw std::invalid_argument(
              std::format("num_codomain_legs ({}) + num_domain_legs ({}) must equal num_legs ({})",
                          n_cod,
                          n_dom,
                          num_legs));
        }
        (void)n_dom;
    }

    std::vector<LegRef> codomain;
    codomain.reserve(static_cast<std::size_t>(n_cod));
    for (int64 i = 0; i < n_cod; ++i) {
        codomain.push_back(i);
    }
    std::vector<LegRef> domain;
    domain.reserve(static_cast<std::size_t>(num_legs - n_cod));
    for (int64 i = num_legs - 1; i >= n_cod; --i) {
        domain.push_back(i);
    }
    return permute_legs(
      std::move(tensor), std::move(codomain), std::move(domain), std::nullopt, BendRight{ true });
}

void
check_same_legs(TensorCPtr t1, TensorCPtr t2)
{
    // --- hints from Python check_same_legs ---
    // either l1 is None or l1 not in l2.labels
    // ---
    if (!t1 || !t2) {
        throw std::invalid_argument("check_same_legs() requires non-null Tensor arguments");
    }
    if (!t1->symmetry->is_equivalent_to(*t2->symmetry)) {
        throw std::invalid_argument("Incompatible symmetries");
    }
    bool incompatible_labels = false;
    LegLabelList const labels1 = t1->labels.as_list();
    for (std::size_t n = 0; n < labels1.size(); ++n) {
        if (labels1[n].empty()) {
            // either l1 is None or l1 not in l2.labels
            continue;
        }
        if (!t2->labels.has(labels1[n])) {
            continue;
        }
        if (t2->labels.index(labels1[n]) != static_cast<int64>(n)) {
            incompatible_labels = true;
            break;
        }
    }
    Space const& dom1 = *t1->domain;
    Space const& dom2 = *t2->domain;
    Space const& cod1 = *t1->codomain;
    Space const& cod2 = *t2->codomain;
    bool same_legs = (dom1 == dom2) && (cod1 == cod2);
    if (!same_legs) {
        std::string msg = "Incompatible legs. ";
        if (incompatible_labels) {
            msg += std::format("Should you permute_legs first? t1.labels={}  t2.labels={}",
                               format_optional_labels(t1->labels.as_list()),
                               format_optional_labels(t2->labels.as_list()));
        }
        throw std::invalid_argument(msg);
    }
    if (incompatible_labels) {
        warn("Compatible legs with permuted labels detected. Double check your leg order!",
             /*stack_level=*/3);
    }
}

TensorPtr
permute_legs(TensorCPtr tensor,
             std::optional<std::vector<LegRef>> codomain,
             std::optional<std::vector<LegRef>> domain,
             std::optional<LevelsSpec> levels,
             std::optional<BendRight> bend_right)
{
    if (!codomain.has_value() && !domain.has_value()) {
        throw std::invalid_argument("Need to specify either domain or codomain.");
    }

    std::vector<int64> domain_v;
    std::vector<int64> codomain_v;
    int64 num_legs = tensor->num_legs;
    int64 num_codomain_legs = tensor->num_codomain_legs();

    if (!codomain.has_value()) {
        domain_v = resolve_leg_refs(tensor, *domain);
        for (int64 n = 0; n < num_legs; ++n) {
            if (!contains_int(domain_v, n)) {
                codomain_v.push_back(n);
            }
        }
    } else if (!domain.has_value()) {
        codomain_v = resolve_leg_refs(tensor, *codomain);
        for (int64 n = num_legs - 1; n >= 0; --n) {
            if (!contains_int(codomain_v, n)) {
                domain_v.push_back(n);
            }
        }
    } else {
        domain_v = resolve_leg_refs(tensor, *domain);
        codomain_v = resolve_leg_refs(tensor, *codomain);
        std::vector<int64> specified_legs = domain_v;
        specified_legs.insert(specified_legs.end(), codomain_v.begin(), codomain_v.end());
        auto duplicates = duplicate_entries(specified_legs);
        if (!duplicates.empty()) {
            std::string joined;
            bool first = true;
            for (auto d : duplicates) {
                if (!first) {
                    joined += ", ";
                }
                first = false;
                joined += std::to_string(d);
            }
            throw std::invalid_argument(
              std::format("Duplicate entries. By leg index: {}", joined));
        }
        std::vector<int64> missing;
        for (int64 n = 0; n < num_legs; ++n) {
            if (!contains_int(specified_legs, n)) {
                missing.push_back(n);
            }
        }
        if (!missing.empty()) {
            if (as_Hidden(tensor)) {
                auto labs = tensor->labels.as_list();
                std::vector<int64> public_missing;
                for (auto m : missing) {
                    if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(m)])) {
                        if (m < num_codomain_legs) {
                            auto it = codomain_v.begin();
                            while (it != codomain_v.end() && *it < m) {
                                ++it;
                            }
                            codomain_v.insert(it, m);
                        } else {
                            auto it = domain_v.begin();
                            while (it != domain_v.end() && *it > m) {
                                ++it;
                            }
                            domain_v.insert(it, m);
                        }
                    } else {
                        public_missing.push_back(m);
                    }
                }
                if (!public_missing.empty()) {
                    std::string joined;
                    bool first = true;
                    for (auto m : public_missing) {
                        if (!first) {
                            joined += ", ";
                        }
                        first = false;
                        joined += std::to_string(m);
                    }
                    throw std::invalid_argument(
                      std::format("Missing public legs. By leg index: {}", joined));
                }
            } else {
                std::string joined;
                bool first = true;
                for (auto m : missing) {
                    if (!first) {
                        joined += ", ";
                    }
                    first = false;
                    joined += std::to_string(m);
                }
                throw std::invalid_argument(std::format("Missing legs. By leg index: {}", joined));
            }
        }
    }

    bool unchanged = true;
    if (static_cast<int64>(codomain_v.size()) != num_codomain_legs) {
        unchanged = false;
    } else {
        for (int64 i = 0; i < num_codomain_legs; ++i) {
            if (codomain_v[static_cast<std::size_t>(i)] != i) {
                unchanged = false;
                break;
            }
        }
        if (unchanged) {
            int64 expect = num_legs - 1;
            for (auto d : domain_v) {
                if (d != expect) {
                    unchanged = false;
                    break;
                }
                --expect;
            }
        }
    }
    if (unchanged) {
        return as_tensor_ptr(tensor);
    }

    auto levels_v = normalize_levels(levels, num_legs);

    std::vector<int64> legs_bending_down;
    for (auto i : domain_v) {
        if (i < num_codomain_legs) {
            legs_bending_down.push_back(i);
        }
    }
    std::vector<int64> legs_bending_up;
    for (auto i : codomain_v) {
        if (i >= num_codomain_legs) {
            legs_bending_up.push_back(i);
        }
    }
    std::vector<int64> bending_legs = legs_bending_down;
    bending_legs.insert(bending_legs.end(), legs_bending_up.begin(), legs_bending_up.end());

    auto bend_right_v = normalize_bend_right(bend_right, num_legs);

    if (tensor->symmetry->has_trivial_braid()) {
        bend_right_v.assign(static_cast<std::size_t>(num_legs), true);
    } else {
        for (auto l : bending_legs) {
            if (!bend_right_v[static_cast<std::size_t>(l)].has_value()) {
                throw SymmetryError("Need to specify bend_right!");
            }
        }
    }

    if (as_Diagonal(tensor) || as_Mask(tensor)) {
        if (codomain_v == std::vector<int64>{ 0 } && domain_v == std::vector<int64>{ 1 }) {
            return as_tensor_ptr(tensor);
        }
        if (codomain_v == std::vector<int64>{ 1 } && domain_v == std::vector<int64>{ 0 }) {
            bool trivial_braid = tensor->symmetry->has_trivial_braid();
            bool opposite_bends = bend_right_v[0].has_value() && bend_right_v[1].has_value() &&
                                  (*bend_right_v[0] != *bend_right_v[1]);
            if (trivial_braid || opposite_bends) {
                return transpose(tensor);
            }
        }
        char const* msg = "Converting to SymmetricTensor for permuting legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = as_tensor_ptr(tensor)->as_SymmetricTensor(false, std::string(msg));
    }

    if (auto charged = as_Charged(tensor)) {
        std::vector<LegRef> domain_with_charge;
        domain_with_charge.push_back(-1);
        for (auto d : domain_v) {
            domain_with_charge.push_back(d);
        }
        LevelsSpec levels_ext = levels_v;
        levels_ext.push_back(std::nullopt);
        std::vector<std::optional<bool>> bend_ext = bend_right_v;
        bend_ext.push_back(std::nullopt);
        std::vector<LegRef> cod_refs;
        for (auto c : codomain_v) {
            cod_refs.push_back(c);
        }
        auto inv_part = permute_legs(charged->invariant_part,
                                     std::move(cod_refs),
                                     std::move(domain_with_charge),
                                     std::move(levels_ext),
                                     BendRight{ std::move(bend_ext) });
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("permute_legs: expected SymmetricTensor invariant_part");
        }
        return make_charged_native(std::move(inv_sym), charged->charged_state);
    }

    if (as_Hidden(tensor)) {
        auto labs = tensor->labels.as_list();
        int64 min_public = 0;
        bool any_public_level = false;
        for (int64 i = 0; i < num_legs; ++i) {
            if (!HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(i)]) &&
                levels_v[static_cast<std::size_t>(i)].has_value()) {
                if (!any_public_level || *levels_v[static_cast<std::size_t>(i)] < min_public) {
                    min_public = *levels_v[static_cast<std::size_t>(i)];
                }
                any_public_level = true;
            }
        }
        if (!any_public_level) {
            min_public = 0;
        }
        for (int64 i = 0; i < num_legs; ++i) {
            if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(i)]) &&
                !levels_v[static_cast<std::size_t>(i)].has_value()) {
                levels_v[static_cast<std::size_t>(i)] = min_public - 1;
            }
        }
    }

    TensorProduct::Ptr new_codomain;
    TensorProduct::Ptr new_domain;
    if (!bending_legs.empty()) {
        std::vector<Leg::Ptr> cod_spaces;
        cod_spaces.reserve(codomain_v.size());
        for (auto i : codomain_v) {
            cod_spaces.push_back(tensor->_as_codomain_leg(i));
        }
        std::vector<Leg::Ptr> dom_spaces;
        dom_spaces.reserve(domain_v.size());
        for (auto i : domain_v) {
            dom_spaces.push_back(tensor->_as_domain_leg(i));
        }
        new_codomain = make_tensor_product(std::move(cod_spaces), tensor->symmetry);
        new_domain = make_tensor_product(std::move(dom_spaces), tensor->symmetry);
    } else {
        new_codomain = tensor->codomain->permuted(codomain_v);
        std::vector<int64> dom_perm;
        dom_perm.reserve(domain_v.size());
        for (auto i : domain_v) {
            dom_perm.push_back(num_legs - 1 - i);
        }
        new_domain = tensor->domain->permuted(dom_perm);
    }

    auto backend = tensor->backend;
    auto data = backend->permute_legs(tensor,
                                      codomain_v,
                                      domain_v,
                                      new_codomain,
                                      new_domain,
                                      /*mixes_codomain_domain=*/!bending_legs.empty(),
                                      levels_v,
                                      bend_right_v);

    LegLabelList all_labels = tensor->labels.as_list();
    LegLabelList cod_labels;
    LegLabelList dom_labels;
    for (auto n : codomain_v) {
        cod_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    for (auto n : domain_v) {
        dom_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    auto flat = flat_labels(cod_labels, dom_labels);
    auto res = make_symmetric_native(
      std::move(data), std::move(new_codomain), std::move(new_domain), backend, std::move(flat));
    bool wrap = as_Hidden(tensor) && (HiddenLegTensor::has_hidden_leg_labels(cod_labels) ||
                                      HiddenLegTensor::has_hidden_leg_labels(dom_labels));
    return maybe_wrap_hidden(std::move(res), wrap);
}

TensorPtr
move_leg(TensorCPtr tensor,
         LegRef which_leg,
         std::optional<int64> codomain_pos,
         std::optional<int64> domain_pos,
         std::optional<LevelsSpec> levels,
         std::optional<BendRight> bend_right)
{
    auto const [from_domain, unused_co, leg_idx] = tensor->_parse_leg_idx(
      std::visit([](auto const& k) -> std::variant<int64, std::string> { return k; }, which_leg));
    (void)unused_co;
    int64 num_codomain_legs = tensor->num_codomain_legs();
    int64 num_legs = tensor->num_legs;

    std::vector<LegRef> new_codomain;
    std::vector<LegRef> new_domain;
    if (from_domain) {
        for (int64 n = 0; n < num_codomain_legs; ++n) {
            new_codomain.push_back(n);
        }
        for (int64 n = num_legs - 1; n >= num_codomain_legs; --n) {
            if (n != leg_idx) {
                new_domain.push_back(n);
            }
        }
    } else {
        for (int64 n = 0; n < num_codomain_legs; ++n) {
            if (n != leg_idx) {
                new_codomain.push_back(n);
            }
        }
        for (int64 n = num_legs - 1; n >= num_codomain_legs; --n) {
            new_domain.push_back(n);
        }
    }

    if (codomain_pos.has_value()) {
        if (domain_pos.has_value()) {
            throw std::invalid_argument("Can not specify both codomain_pos and domain_pos.");
        }
        int64 pos = to_valid_idx(*codomain_pos, static_cast<int64>(new_codomain.size()) + 1);
        new_codomain.insert(new_codomain.begin() + pos, leg_idx);
    } else if (domain_pos.has_value()) {
        int64 pos = to_valid_idx(*domain_pos, static_cast<int64>(new_domain.size()) + 1);
        new_domain.insert(new_domain.begin() + pos, leg_idx);
    } else {
        throw std::invalid_argument("Need to specify either codomain_pos or domain_pos.");
    }

    return permute_legs(std::move(tensor),
                        std::move(new_codomain),
                        std::move(new_domain),
                        std::move(levels),
                        std::move(bend_right));
}

TensorPtr
combine_legs(TensorCPtr tensor,
             std::vector<std::vector<LegRef>> which_legs,
             std::optional<PipeDualities> pipe_dualities,
             std::optional<std::vector<Leg::Ptr>> pipes,
             std::optional<LevelsSpec> levels)
{
    if (as_Diagonal(tensor) || as_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for combine_legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = as_tensor_ptr(tensor)->as_SymmetricTensor(false, std::string(msg));
    }

    std::vector<std::vector<int64>> which_legs_v;
    which_legs_v.reserve(which_legs.size());
    for (auto const& group : which_legs) {
        which_legs_v.push_back(resolve_leg_refs(tensor, group));
    }

    if (auto charged = as_Charged(tensor)) {
        std::optional<LevelsSpec> levels_for_inv = levels;
        if (levels.has_value() && !levels->empty()) {
            LevelsSpec levels_list = *levels;
            int64 min_level = levels_list[0].value_or(0);
            for (auto const& item : levels_list) {
                if (item.has_value()) {
                    min_level = std::min(min_level, *item);
                }
            }
            levels_list.push_back(min_level - 1);
            levels_for_inv = std::move(levels_list);
        }
        std::vector<std::vector<LegRef>> which_as_refs;
        which_as_refs.reserve(which_legs_v.size());
        for (auto const& g : which_legs_v) {
            std::vector<LegRef> refs;
            for (auto i : g) {
                refs.push_back(i);
            }
            which_as_refs.push_back(std::move(refs));
        }
        auto inv_part = combine_legs(charged->invariant_part,
                                     std::move(which_as_refs),
                                     std::move(pipe_dualities),
                                     std::move(pipes),
                                     std::move(levels_for_inv));
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("combine_legs: expected SymmetricTensor invariant_part");
        }
        return make_charged_native(std::move(inv_sym), charged->charged_state);
    }

    if (as_Hidden(tensor)) {
        auto labs = tensor->labels.as_list();
        for (auto const& group : which_legs_v) {
            for (auto idx : group) {
                if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(idx)])) {
                    throw std::invalid_argument(
                      "combine_legs: cannot combine hidden legs (they must not appear inside "
                      "pipes). Hide a pipe after combining public legs instead.");
                }
            }
        }
    }

    int64 N = tensor->num_legs;
    int64 J = tensor->num_codomain_legs();
    std::vector<int64> to_combine;
    for (auto const& group : which_legs_v) {
        to_combine.insert(to_combine.end(), group.begin(), group.end());
    }
    if (!duplicate_entries(to_combine).empty()) {
        throw std::invalid_argument("Groups may not contain duplicates.");
    }

    std::map<int64, std::vector<int64>> codomain_groups;
    std::map<int64, std::vector<int64>> domain_groups;
    for (auto const& group : which_legs_v) {
        if (group[0] < J) {
            codomain_groups[group[0]] = group;
        } else {
            domain_groups[group[0]] = group;
        }
    }
    std::vector<int64> codomain_idcs;
    std::vector<int64> domain_idcs_reversed;
    for (int64 n = 0; n < N; ++n) {
        if (codomain_groups.contains(n)) {
            auto const& g = codomain_groups[n];
            codomain_idcs.insert(codomain_idcs.end(), g.begin(), g.end());
        } else if (domain_groups.contains(n)) {
            auto const& g = domain_groups[n];
            domain_idcs_reversed.insert(domain_idcs_reversed.end(), g.begin(), g.end());
        } else if (contains_int(to_combine, n)) {
        } else if (n < J) {
            codomain_idcs.push_back(n);
        } else {
            domain_idcs_reversed.push_back(n);
        }
    }

    std::vector<int64> domain_idcs = domain_idcs_reversed;
    std::reverse(domain_idcs.begin(), domain_idcs.end());
    {
        std::vector<LegRef> cod_refs(codomain_idcs.begin(), codomain_idcs.end());
        std::vector<LegRef> dom_refs(domain_idcs.begin(), domain_idcs.end());
        tensor = permute_legs(std::move(tensor),
                              std::move(cod_refs),
                              std::move(dom_refs),
                              std::move(levels),
                              std::nullopt);
    }

    std::vector<int64> full_perm = codomain_idcs;
    full_perm.insert(full_perm.end(), domain_idcs_reversed.begin(), domain_idcs_reversed.end());
    auto inv_perm = inverse_permutation(full_perm);
    for (auto& group : which_legs_v) {
        for (auto& l : group) {
            l = inv_perm[static_cast<std::size_t>(l)];
        }
    }
    to_combine.clear();
    for (auto const& group : which_legs_v) {
        to_combine.insert(to_combine.end(), group.begin(), group.end());
    }
    J = tensor->num_codomain_legs();
    codomain_groups.clear();
    domain_groups.clear();
    for (auto const& group : which_legs_v) {
        if (group[0] < J) {
            codomain_groups[group[0]] = group;
        } else {
            domain_groups[group[0]] = group;
        }
    }

    std::vector<Leg::Ptr> pipes_v(which_legs_v.size(), nullptr);
    if (pipes.has_value()) {
        if (pipes->size() != which_legs_v.size()) {
            throw std::invalid_argument(
              std::format("expected {} pipes, got {}", which_legs_v.size(), pipes->size()));
        }
        pipes_v = *pipes;
    }
    auto pipe_dualities_v = normalize_pipe_dualities(pipe_dualities, which_legs_v.size());

    auto backend = tensor->backend;
    std::vector<Leg::Ptr> codomain_spaces;
    std::vector<std::string> codomain_labels;
    std::vector<std::string> domain_labels_reversed;
    std::vector<Leg::Ptr> domain_spaces_reversed;
    std::size_t i = 0;
    int64 label_offset = 0;
    LegLabelList all_labels = tensor->labels.as_list();

    for (int64 n = 0; n < N; ++n) {
        if (codomain_groups.contains(n)) {
            auto const& group = codomain_groups[n];
            std::vector<Leg::Ptr> spaces_to_combine;
            for (int64 g = group.front(); g <= group.back(); ++g) {
                spaces_to_combine.push_back((*tensor->codomain)[g]);
            }
            auto combined =
              backend->make_pipe(std::move(spaces_to_combine), pipe_dualities_v[i], pipes_v[i]);
            pipes_v[i] = combined;
            codomain_spaces.push_back(combined);
            LegLabelList group_labels(all_labels.begin() + group.front(),
                                      all_labels.begin() + group.back() + 1);
            codomain_labels.push_back(_combine_leg_labels(group_labels, label_offset));
            ++i;
            int64 none_count = 0;
            for (auto l : group) {
                if (all_labels[static_cast<std::size_t>(l)].empty()) {
                    ++none_count;
                }
            }
            label_offset += none_count;
        } else if (domain_groups.contains(n)) {
            auto const& group = domain_groups[n];
            int64 domain_idx1 = N - 1 - group.front();
            int64 codomain_idx2 = N - 1 - group.back();
            std::vector<Leg::Ptr> spaces_to_combine;
            for (int64 g = codomain_idx2; g <= domain_idx1; ++g) {
                spaces_to_combine.push_back((*tensor->domain)[g]);
            }
            auto combined =
              backend->make_pipe(std::move(spaces_to_combine), !pipe_dualities_v[i], pipes_v[i]);
            pipes_v[i] = combined;
            domain_spaces_reversed.push_back(combined);
            LegLabelList group_labels(all_labels.begin() + group.front(),
                                      all_labels.begin() + group.back() + 1);
            domain_labels_reversed.push_back(_combine_leg_labels(group_labels, label_offset));
            ++i;
            int64 none_count = 0;
            for (auto l : group) {
                if (all_labels[static_cast<std::size_t>(l)].empty()) {
                    ++none_count;
                }
            }
            label_offset += none_count;
        } else if (contains_int(to_combine, n)) {
        } else if (n < J) {
            codomain_spaces.push_back((*tensor->codomain)[n]);
            codomain_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
        } else {
            domain_spaces_reversed.push_back((*tensor->domain)[N - 1 - n]);
            domain_labels_reversed.push_back(all_labels[static_cast<std::size_t>(n)]);
        }
    }

    std::vector<Leg::Ptr> domain_spaces(domain_spaces_reversed.rbegin(),
                                        domain_spaces_reversed.rend());
    auto codomain_tp = make_tensor_product(std::move(codomain_spaces), tensor->symmetry);
    auto domain_tp = make_tensor_product(std::move(domain_spaces), tensor->symmetry);

    std::sort(which_legs_v.begin(), which_legs_v.end());
    for (std::size_t k = 0; k < which_legs_v.size(); ++k) {
        if (!pipes_v[k] || !pipes_v[k].is_pipe()) {
            throw std::runtime_error("combine_legs: expected a pipe (LegPipe or AbelianLegPipe)");
        }
    }
    auto data = backend->combine_legs(tensor, which_legs_v, pipes_v, codomain_tp, domain_tp);

    LegLabelList res_labels = codomain_labels;
    res_labels.insert(
      res_labels.end(), domain_labels_reversed.begin(), domain_labels_reversed.end());
    auto res = make_symmetric_native(
      std::move(data), std::move(codomain_tp), std::move(domain_tp), backend, res_labels);
    bool wrap = HiddenLegTensor::has_hidden_leg_labels(res_labels) && as_Hidden(tensor);
    return maybe_wrap_hidden(std::move(res), wrap);
}

TensorPtr
combine_to_matrix(TensorCPtr tensor,
                  std::optional<std::vector<LegRef>> codomain,
                  std::optional<std::vector<LegRef>> domain,
                  std::optional<LevelsSpec> levels)
{
    auto res = permute_legs(
      std::move(tensor), std::move(codomain), std::move(domain), std::move(levels), std::nullopt);
    int64 n_cod = res->num_codomain_legs();
    int64 n_legs = res->num_legs;
    std::vector<LegRef> cod_range;
    for (int64 i = 0; i < n_cod; ++i) {
        cod_range.push_back(i);
    }
    std::vector<LegRef> dom_range;
    for (int64 i = n_cod; i < n_legs; ++i) {
        dom_range.push_back(i);
    }
    return combine_legs(std::move(res), { std::move(cod_range), std::move(dom_range) });
}

TensorPtr
split_legs(TensorCPtr tensor, std::optional<std::vector<LegRef>> legs)
{
    if (as_Diagonal(tensor) || as_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for split_legs. Use as_SymmetricTensor() "
                          "explicitly to suppress the warning.";
        tensor = as_tensor_ptr(tensor)->as_SymmetricTensor(false, std::string(msg));
    }
    if (auto charged = as_Charged(tensor)) {
        std::optional<std::vector<LegRef>> legs_for_inv = legs;
        if (legs.has_value()) {
            auto idcs = resolve_leg_refs(tensor, *legs);
            std::vector<LegRef> refs(idcs.begin(), idcs.end());
            legs_for_inv = std::move(refs);
        }
        auto inv_part = split_legs(charged->invariant_part, std::move(legs_for_inv));
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("split_legs: expected SymmetricTensor invariant_part");
        }
        return make_charged_native(std::move(inv_sym), charged->charged_state);
    }

    std::vector<int64> leg_idcs;
    std::vector<int64> codomain_split;
    std::vector<int64> domain_split;
    int64 num_legs = tensor->num_legs;

    if (!legs.has_value()) {
        auto labs = tensor->labels.as_list();
        for (int64 n = 0; n < tensor->codomain->num_factors; ++n) {
            auto l = (*tensor->codomain)[n];
            if (l && l.is_pipe()) {
                if (!(as_Hidden(tensor) &&
                      HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(n)]))) {
                    codomain_split.push_back(n);
                }
            }
        }
        for (int64 n = 0; n < tensor->domain->num_factors; ++n) {
            auto l = (*tensor->domain)[n];
            if (l && l.is_pipe()) {
                int64 leg_idx = num_legs - 1 - n;
                if (!(as_Hidden(tensor) && HiddenLegTensor::is_hidden_leg_label(
                                             labs[static_cast<std::size_t>(leg_idx)]))) {
                    domain_split.push_back(n);
                }
            }
        }
        leg_idcs = codomain_split;
        for (auto it = domain_split.rbegin(); it != domain_split.rend(); ++it) {
            leg_idcs.push_back(num_legs - 1 - *it);
        }
    } else {
        if (as_Hidden(tensor)) {
            auto labs = tensor->labels.as_list();
            auto sorted = resolve_leg_refs(tensor, *legs);
            for (auto idx : sorted) {
                if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(idx)])) {
                    throw std::invalid_argument(
                      "split_legs: cannot specify hidden legs. Omit legs= to split public pipes "
                      "only.");
                }
            }
        }
        auto sorted = resolve_leg_refs(tensor, *legs);
        std::sort(sorted.begin(), sorted.end());
        for (auto l : sorted) {
            auto const [in_domain, co_domain_idx, leg_idx] = tensor->_parse_leg_idx(l);
            leg_idcs.push_back(leg_idx);
            if (in_domain) {
                domain_split.push_back(co_domain_idx);
            } else {
                codomain_split.push_back(co_domain_idx);
            }
            auto leg = tensor->get_leg_co_domain(leg_idx);
            if (!leg || !leg.is_pipe()) {
                throw std::invalid_argument("Not a LegPipe.");
            }
        }
    }

    std::vector<Leg::Ptr> codomain_spaces;
    for (int64 n = 0; n < tensor->codomain->num_factors; ++n) {
        auto lo = (*tensor->codomain)[n];
        if (contains_int(codomain_split, n)) {
            for (auto const& sub : pipe_legs(lo)) {
                codomain_spaces.push_back(sub);
            }
        } else {
            codomain_spaces.push_back(lo);
        }
    }
    std::vector<Leg::Ptr> domain_spaces;
    for (int64 n = 0; n < tensor->domain->num_factors; ++n) {
        auto lo = (*tensor->domain)[n];
        if (contains_int(domain_split, n)) {
            for (auto const& sub : pipe_legs(lo)) {
                domain_spaces.push_back(sub);
            }
        } else {
            domain_spaces.push_back(lo);
        }
    }

    auto codomain_tp = make_tensor_product(std::move(codomain_spaces),
                                           tensor->symmetry,
                                           tensor->codomain->sector_decomposition,
                                           tensor->codomain->multiplicities);
    auto domain_tp = make_tensor_product(std::move(domain_spaces),
                                         tensor->symmetry,
                                         tensor->domain->sector_decomposition,
                                         tensor->domain->multiplicities);

    LegLabelList all_labels = tensor->labels.as_list();
    LegLabelList labels;
    std::unordered_set<int64> leg_idcs_set(leg_idcs.begin(), leg_idcs.end());
    for (int64 idx = 0; idx < static_cast<int64>(all_labels.size()); ++idx) {
        if (leg_idcs_set.contains(idx)) {
            auto pipe = tensor->get_leg_co_domain(idx);
            auto split =
              _split_leg_label(all_labels[static_cast<std::size_t>(idx)], pipe_num_legs(pipe));
            labels.insert(labels.end(), split.begin(), split.end());
        } else {
            labels.push_back(all_labels[static_cast<std::size_t>(idx)]);
        }
    }

    std::sort(leg_idcs.begin(), leg_idcs.end());
    auto backend = tensor->backend;
    auto data = backend->split_legs(tensor, leg_idcs, codomain_tp, domain_tp);
    auto res = make_symmetric_native(
      std::move(data), std::move(codomain_tp), std::move(domain_tp), backend, labels);
    bool wrap = HiddenLegTensor::has_hidden_leg_labels(labels) && as_Hidden(tensor);
    return maybe_wrap_hidden(std::move(res), wrap);
}

TensorPtr
squeeze_legs(TensorCPtr tensor, std::optional<std::vector<LegRef>> legs)
{
    std::vector<int64> legs_v;
    if (!legs.has_value()) {
        int64 n = 0;
        for (auto l : conventional_leg_order(tensor)) {
            if (l.is_trivial()) {
                legs_v.push_back(n);
            }
            ++n;
        }
    } else {
        legs_v = resolve_leg_refs(tensor, *legs);
        for (auto n : legs_v) {
            if (!tensor->get_leg_co_domain(n).is_trivial()) {
                throw std::invalid_argument("Can only squeeze trivial legs");
            }
        }
    }
    if (legs_v.empty()) {
        return as_tensor_ptr(tensor);
    }
    if (as_Diagonal(tensor) || as_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for squeeze_legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = as_tensor_ptr(tensor)->as_SymmetricTensor(false, std::string(msg));
    }
    if (auto charged = as_Charged(tensor)) {
        std::vector<LegRef> refs(legs_v.begin(), legs_v.end());
        auto inv_part = squeeze_legs(charged->invariant_part, std::move(refs));
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("squeeze_legs: expected SymmetricTensor invariant_part");
        }
        return make_charged_native(std::move(inv_sym), charged->charged_state);
    }

    int64 num_legs = tensor->num_legs;
    int64 num_codomain_legs = tensor->num_codomain_legs();
    int64 num_domain_legs = tensor->num_domain_legs();
    std::unordered_set<int64> legs_set(legs_v.begin(), legs_v.end());
    std::vector<int64> remaining;
    for (int64 n = 0; n < num_legs; ++n) {
        if (!legs_set.contains(n)) {
            remaining.push_back(n);
        }
    }

    auto backend = tensor->backend;
    auto data = backend->squeeze_legs(tensor, legs_v);

    std::vector<Leg::Ptr> cod_spaces;
    for (int64 n = 0; n < num_codomain_legs; ++n) {
        if (!legs_set.contains(n)) {
            cod_spaces.push_back((*tensor->codomain)[n]);
        }
    }
    std::vector<Leg::Ptr> dom_spaces;
    for (int64 n = 0; n < num_domain_legs; ++n) {
        if (!legs_set.contains(num_legs - 1 - n)) {
            dom_spaces.push_back((*tensor->domain)[n]);
        }
    }
    auto codomain_tp = make_tensor_product(std::move(cod_spaces),
                                           tensor->symmetry,
                                           tensor->codomain->sector_decomposition,
                                           tensor->codomain->multiplicities);
    auto domain_tp = make_tensor_product(std::move(dom_spaces),
                                         tensor->symmetry,
                                         tensor->domain->sector_decomposition,
                                         tensor->domain->multiplicities);

    LegLabelList all_labels = tensor->labels.as_list();
    LegLabelList labels;
    for (auto n : remaining) {
        labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    // a HiddenLegTensor keeps the hidden prefix in its labels -> wrap the result again
    return HiddenLegTensor::maybe_wrap(make_symmetric_native(
      std::move(data), std::move(codomain_tp), std::move(domain_tp), backend, std::move(labels)));
}

namespace {

[[nodiscard]] int64
leg_idx(TensorCPtr const& tensor, LegRef const& which)
{
    return std::visit([&](auto const& key) { return tensor->labels.index(key); }, which);
}

[[nodiscard]] std::string
unique_temp_label(TensorCPtr const& tensor, std::string base)
{
    if (!tensor->labels.has(base)) {
        return base;
    }
    for (int n = 1;; ++n) {
        auto cand = base + std::to_string(n);
        if (!tensor->labels.has(cand)) {
            return cand;
        }
    }
}

[[nodiscard]] LevelsSpec
levels_with_hidden(TensorCPtr const& tensor, int64 hidden_idx)
{
    auto labs = tensor->labels.as_list();
    LevelsSpec levels(static_cast<std::size_t>(tensor->num_legs), std::optional<int64>{ 0 });
    for (int64 i = 0; i < tensor->num_legs; ++i) {
        if (HiddenLegTensor::is_hidden_leg_label(labs.at(static_cast<std::size_t>(i)))) {
            levels[static_cast<std::size_t>(i)] = -1;
        }
    }
    if (hidden_idx >= 0 && hidden_idx < tensor->num_legs) {
        levels[static_cast<std::size_t>(hidden_idx)] = -1;
    }
    return levels;
}

/// Target of a leg move: position `pos` of the resulting codomain or domain, as in `move_leg`.
struct LegTarget
{
    bool to_domain;
    int64 pos;
};

/// One `move_leg` call of a `MoveLegRoute`.
struct MoveLegStep
{
    LegTarget target;
    bool bend_right;
};

/// A way to move a single leg to its target, together with the legs that it crosses.
struct MoveLegRoute
{
    std::vector<MoveLegStep> steps;
    std::vector<int64> crossed; ///< Indices of the legs crossed, before the move.
};

/// The two routes to move leg `moving` of a tensor to `target`, see `move_leg_avoiding`.
///
/// The routes go around the tensor in opposite directions and cross complementary sets of legs.
/// If the leg stays in the codomain (domain), the first route is the direct one and the second
/// one goes around the tensor, i.e., bends once to the left and once to the right. Otherwise, the
/// first route bends to the right and the second one to the left.
[[nodiscard]] std::array<MoveLegRoute, 2>
move_leg_routes(int64 num_legs, int64 num_codomain_legs, int64 moving, LegTarget target)
{
    // the other legs, by (co)domain index, i.e. from left to right in both codomain and domain
    std::vector<int64> codomain;
    std::vector<int64> domain;
    for (int64 n = 0; n < num_codomain_legs; ++n) {
        if (n != moving) {
            codomain.push_back(n);
        }
    }
    for (int64 n = num_legs - 1; n >= num_codomain_legs; --n) {
        if (n != moving) {
            domain.push_back(n);
        }
    }
    bool const from_domain = moving >= num_codomain_legs;
    int64 const p = from_domain ? num_legs - 1 - moving : moving;
    auto const& src = from_domain ? domain : codomain;
    auto const& other = from_domain ? codomain : domain;
    auto const& tgt = target.to_domain ? domain : codomain;
    int64 const q = to_valid_idx(target.pos, static_cast<int64>(tgt.size()) + 1);
    auto const slice = [](std::vector<int64> const& v, int64 start, int64 stop) {
        return std::vector<int64>(v.begin() + start, v.begin() + stop);
    };
    auto const join = [](std::vector<std::vector<int64>> parts) {
        std::vector<int64> res;
        for (auto const& part : parts) {
            res.insert(res.end(), part.begin(), part.end());
        }
        return res;
    };
    auto const src_size = static_cast<int64>(src.size());
    auto const other_size = static_cast<int64>(other.size());
    auto const tgt_size = static_cast<int64>(tgt.size());
    LegTarget const to_other_start{ !target.to_domain, 0 };
    LegTarget const to_other_end{ !target.to_domain, other_size };
    if (from_domain == target.to_domain) {
        MoveLegRoute direct{ { { target, true } }, slice(src, std::min(p, q), std::max(p, q)) };
        if (q >= p) {
            // around the left edge to the other side, then around the right edge to the target
            return { direct,
                     MoveLegRoute{ { { to_other_start, false }, { target, true } },
                                   join({ slice(src, 0, p), other, slice(src, q, src_size) }) } };
        }
        return { direct,
                 MoveLegRoute{ { { to_other_end, true }, { target, false } },
                               join({ slice(src, p, src_size), other, slice(src, 0, q) }) } };
    }
    return { MoveLegRoute{ { { target, true } },
                           join({ slice(src, p, src_size), slice(tgt, q, tgt_size) }) },
             MoveLegRoute{ { { target, false } }, join({ slice(src, 0, p), slice(tgt, 0, q) }) } };
}

/// The labels of a tensor's codomain and domain, each from left to right.
struct LegLayout
{
    std::vector<std::string> codomain;
    std::vector<std::string> domain;

    [[nodiscard]] static LegLayout of(TensorCPtr const& tensor)
    {
        auto labs = tensor->labels.as_list();
        auto const nc = static_cast<std::ptrdiff_t>(tensor->num_codomain_legs());
        return { std::vector<std::string>(labs.begin(), labs.begin() + nc),
                 std::vector<std::string>(labs.rbegin(), labs.rend() - nc) };
    }

    /// Labels in the order of ``Tensor.labels``, i.e. clockwise around the tensor.
    [[nodiscard]] std::vector<std::string> labels() const
    {
        std::vector<std::string> res = codomain;
        res.insert(res.end(), domain.rbegin(), domain.rend());
        return res;
    }

    [[nodiscard]] int64 index(std::string const& label) const
    {
        auto labs = labels();
        return static_cast<int64>(std::find(labs.begin(), labs.end(), label) - labs.begin());
    }

    /// Where to move `moving` to be in the same (co)domain as `target`, immediately before
    /// (`after=false`) or after (`after=true`) `target`.
    [[nodiscard]] LegTarget next_to(std::string const& moving,
                                    std::string const& target,
                                    bool after) const
    {
        auto const find = [](std::vector<std::string> const& v, std::string const& lab) {
            return static_cast<int64>(std::find(v.begin(), v.end(), lab) - v.begin());
        };
        bool const t_in_dom = find(codomain, target) == static_cast<int64>(codomain.size());
        auto const& side = t_in_dom ? domain : codomain;
        int64 tpos = find(side, target);
        int64 const mpos = find(side, moving);
        if (mpos < tpos) {
            // Removing `moving` from the same side shifts the target left.
            tpos -= 1;
        }
        return { t_in_dom, after ? tpos + 1 : tpos };
    }

    /// The layout after moving `moving` to `target`, as in `move_leg`.
    [[nodiscard]] LegLayout moved(std::string const& moving, LegTarget target) const
    {
        LegLayout res = *this;
        for (auto* side : { &res.codomain, &res.domain }) {
            side->erase(std::remove(side->begin(), side->end(), moving), side->end());
        }
        auto& side = target.to_domain ? res.domain : res.codomain;
        int64 const pos = to_valid_idx(target.pos, static_cast<int64>(side.size()) + 1);
        side.insert(side.begin() + pos, moving);
        return res;
    }

    [[nodiscard]] LegLayout without(std::string const& label) const
    {
        LegLayout res = *this;
        for (auto* side : { &res.codomain, &res.domain }) {
            side->erase(std::remove(side->begin(), side->end(), label), side->end());
        }
        return res;
    }
};

/// The route chosen by `move_leg_avoiding`.
struct ChosenRoute
{
    MoveLegRoute route;
    bool crosses_hidden;
    bool crosses_never_cross;
};

/// Choose the route for `move_leg_avoiding`, see there.
[[nodiscard]] ChosenRoute
choose_route(LegLayout const& layout,
             std::string const& moving,
             LegTarget target,
             std::vector<std::string> const& never_cross,
             std::vector<std::string> const& must_cross = {})
{
    auto const labs = layout.labels();
    auto const is_hidden = [&](int64 i) {
        return HiddenLegTensor::is_hidden_leg_label(labs.at(static_cast<std::size_t>(i)));
    };
    auto const is_never_cross = [&](int64 i) {
        auto const& lab = labs.at(static_cast<std::size_t>(i));
        return std::find(never_cross.begin(), never_cross.end(), lab) != never_cross.end();
    };
    auto const crosses = [](MoveLegRoute const& route, auto const& pred) {
        return std::any_of(route.crossed.begin(), route.crossed.end(), pred);
    };
    auto const acceptable = [&](MoveLegRoute const& route) {
        for (auto const& lab : must_cross) {
            auto const is_lab = [&](int64 i) {
                return labs.at(static_cast<std::size_t>(i)) == lab;
            };
            if (!crosses(route, is_lab)) {
                return false;
            }
        }
        return !crosses(route, is_never_cross);
    };
    auto const routes = move_leg_routes(static_cast<int64>(labs.size()),
                                        static_cast<int64>(layout.codomain.size()),
                                        layout.index(moving),
                                        target);
    for (auto const& route : routes) {
        if (acceptable(route) && !crosses(route, is_hidden)) {
            return { route, false, false };
        }
    }
    for (auto const& route : routes) {
        if (acceptable(route)) {
            return { route, true, false };
        }
    }
    return { routes[0], crosses(routes[0], is_hidden), true };
}

/// Move a single leg like `move_leg`, but choose the route around the tensor to avoid braids.
///
/// The moving leg and all hidden legs have level -1, all public legs level 0 (see
/// `levels_with_hidden`), i.e. the moving leg passes behind the public legs. Crossing another
/// hidden leg is a genuine braid, whose chirality is unspecified for non-symmetric braids.
/// The two routes around the tensor (see `move_leg_routes`) are *not* equivalent for
/// non-symmetric braids, they differ by the moving leg encircling the tensor. The caller selects
/// the route by the legs (labels) it must cross (`must_cross`) or not cross (`never_cross`).
/// We use the first such route that crosses no hidden legs, otherwise the first such route,
/// otherwise the first route.
[[nodiscard]] TensorPtr
move_leg_avoiding(TensorPtr tensor,
                  std::string const& moving,
                  LegTarget target,
                  std::vector<std::string> const& never_cross = {},
                  std::vector<std::string> const& must_cross = {})
{
    auto const chosen =
      choose_route(LegLayout::of(tensor), moving, target, never_cross, must_cross);
    for (auto const& step : chosen.route.steps) {
        auto levels = levels_with_hidden(tensor, tensor->labels.index(moving));
        std::optional<int64> codomain_pos;
        std::optional<int64> domain_pos;
        (step.target.to_domain ? domain_pos : codomain_pos) = step.target.pos;
        tensor = move_leg(tensor,
                          LegRef{ moving },
                          codomain_pos,
                          domain_pos,
                          std::move(levels),
                          BendRight{ step.bend_right });
    }
    return tensor;
}

/// The labels of `layout` clockwise after `label` (excluding it), i.e. in the order in which
/// they appear on the outer boundary when contracting the leg `label` with another tensor.
[[nodiscard]] std::vector<std::string>
clockwise_after(LegLayout const& layout, std::string const& label)
{
    auto labs = layout.labels();
    auto const pos = std::find(labs.begin(), labs.end(), label);
    std::vector<std::string> res(pos + 1, labs.end());
    res.insert(res.end(), labs.begin(), pos);
    return res;
}

/// Where the identity wire of `move_hidden_leg` starts, and in which order its ends are moved.
struct IdentityWirePlan
{
    bool right_edge;      ///< If the identity is placed right of `B` (``outer(B, I)``), else left.
    bool int_in_codomain; ///< If the end `int` starts in the codomain, else `open` does.
    bool open_first;      ///< If `open` is moved next to `axis_B` before `int`.
};

/// The initial layout of `B_ext` for an `IdentityWirePlan`.
[[nodiscard]] LegLayout
identity_wire_layout(LegLayout layout,
                     IdentityWirePlan plan,
                     std::string const& int_lab,
                     std::string const& open_lab)
{
    auto const& top = plan.int_in_codomain ? int_lab : open_lab;
    auto const& bottom = plan.int_in_codomain ? open_lab : int_lab;
    if (plan.right_edge) {
        layout.codomain.push_back(top);
        layout.domain.push_back(bottom);
    } else {
        layout.codomain.insert(layout.codomain.begin(), top);
        layout.domain.insert(layout.domain.begin(), bottom);
    }
    return layout;
}

/// A move of an `IdentityWirePlan`: `moving` right before or after the leg `reference`.
struct WireMove
{
    std::string moving;
    std::string reference;
    bool after;
};

/// The two moves of an `IdentityWirePlan`: `int` right after `axis`, and `open` either right
/// before `axis` (`behind_axis`, the wire passes behind `axis`) or right after `int`.
[[nodiscard]] std::array<WireMove, 2>
identity_wire_moves(IdentityWirePlan plan,
                    bool behind_axis,
                    std::string const& axis,
                    std::string const& int_lab,
                    std::string const& open_lab)
{
    WireMove const int_move{ int_lab, axis, /*after=*/true };
    if (plan.open_first) {
        return { WireMove{ open_lab, axis, /*after=*/!behind_axis }, int_move };
    }
    return { int_move,
             behind_axis ? WireMove{ open_lab, axis, /*after=*/false }
                         : WireMove{ open_lab, int_lab, /*after=*/true } };
}

#if LASYTEN_HAS_PYTHON // FusionTreeBackend is only built with Python
/// Permutation of the block rows (codomain) / columns (domain) of a `FusionTreeBackend` tensor,
/// if the pipe ``old_prod.factors[0]`` is replaced by the `ElementarySpace`
/// ``new_prod.factors[0]``.
///
/// The `FusionTreeBackend` treats pipes as transparent, i.e. the blocks are indexed by the
/// uncoupled sectors of the *flat* legs (the pipe components), the fusion trees and the
/// multiplicities. Since the pipe is the first factor, the first vertices of each fusion tree
/// fuse the pipe components to a sector ``e`` (the "pipe tree"), and the remaining tree fuses
/// ``e`` with the other legs. For the flat leg, the pipe tree and the multiplicities of the
/// components instead enumerate the multiplicity of ``e``, in the order of the rows of the block
/// ``e`` of `pipe_prod`, the `TensorProduct` of the pipe components, which defines
/// `LegPipe::as_ElementarySpace`. No F-moves are required, so this is a pure permutation.
///
/// @returns `perm` such that ``new_block[i] = old_block[perm[i]]``.
[[nodiscard]] std::vector<int64>
flatten_pipe_block_perm(TensorProduct const& old_prod,
                        TensorProduct const& new_prod,
                        TensorProduct const& pipe_prod,
                        Sector const& coupled)
{
    auto const& symmetry = old_prod.symmetry;
    auto const ind_len = symmetry->sector_ind_len;
    auto const n_pipe = static_cast<std::size_t>(pipe_prod.num_flat_legs());
    std::vector<int64> perm(static_cast<std::size_t>(old_prod.block_size(coupled)), -1);
    int64 old_start = 0;
    for (auto const& item : old_prod.iter_uncoupled()) {
        auto const old_trees = fusion_trees(symmetry, item.uncoupled, coupled).all_trees();
        if (old_trees.empty()) {
            continue;
        }
        auto const n_flat = item.uncoupled.size();
        SectorArray pipe_uncoupled(n_pipe, ind_len);
        SectorArray rest_uncoupled(n_flat - n_pipe + 1, ind_len);
        for (std::size_t k = 0; k < n_flat; ++k) {
            (k < n_pipe ? pipe_uncoupled[k] : rest_uncoupled[k - n_pipe + 1]) = item.uncoupled[k];
        }
        int64 pipe_size = 1; // multiplicities of the pipe components
        int64 rest_size = 1; // multiplicities of the other flat legs
        for (std::size_t k = 0; k < n_flat; ++k) {
            (k < n_pipe ? pipe_size : rest_size) *= item.multiplicities[k];
        }
        for (std::size_t alpha = 0; alpha < old_trees.size(); ++alpha) {
            auto const& tree = old_trees[alpha];
            // split into the pipe tree (vertices 0 .. n_pipe - 2) and the rest tree
            Sector const e = n_pipe == n_flat ? coupled
                             : n_pipe == 1    ? item.uncoupled[0]
                                              : tree.inner_sectors[n_pipe - 2];
            rest_uncoupled[0] = e;
            SectorArray pipe_inner(n_pipe >= 2 ? n_pipe - 2 : 0, ind_len);
            std::vector<int64> pipe_mults;
            for (std::size_t k = 0; k + 2 < n_pipe; ++k) {
                pipe_inner[k] = tree.inner_sectors[k];
            }
            SectorArray rest_inner(n_flat >= n_pipe + 1 ? n_flat - n_pipe - 1 : 0, ind_len);
            for (std::size_t k = 0; k < rest_inner.size(); ++k) {
                rest_inner[k] = tree.inner_sectors[n_pipe - 1 + k];
            }
            std::vector<int64> rest_mults;
            for (std::size_t k = 0; k < tree.multiplicities.size(); ++k) {
                (k + 1 < n_pipe ? pipe_mults : rest_mults).push_back(tree.multiplicities[k]);
            }
            int64 alpha_pipe = 0;
            if (n_pipe >= 2) {
                FusionTree pipe_tree(symmetry,
                                     pipe_uncoupled,
                                     e,
                                     std::vector<std::uint8_t>(n_pipe, 0),
                                     pipe_inner,
                                     tree.multiplicities.empty() ? std::nullopt
                                                                 : std::optional(pipe_mults));
                alpha_pipe =
                  static_cast<int64>(fusion_trees(symmetry, pipe_uncoupled, e).index(pipe_tree));
            }
            int64 beta = 0;
            if (rest_uncoupled.size() >= 2) {
                FusionTree rest_tree(symmetry,
                                     rest_uncoupled,
                                     coupled,
                                     std::vector<std::uint8_t>(rest_uncoupled.size(), 0),
                                     rest_inner,
                                     tree.multiplicities.empty() ? std::nullopt
                                                                 : std::optional(rest_mults));
                beta = static_cast<int64>(
                  fusion_trees(symmetry, rest_uncoupled, coupled).index(rest_tree));
            }
            // multiplicity index on the flat leg, i.e. the row in the block `e` of `pipe_prod`
            int64 const e_offset =
              pipe_prod.forest_block_slice(pipe_uncoupled, e).start + alpha_pipe * pipe_size;
            int64 const e_mult = pipe_prod.block_size(e);
            int64 const new_start = new_prod.forest_block_slice(rest_uncoupled, coupled).start +
                                    beta * e_mult * rest_size;
            int64 const old_tree_start =
              old_start + static_cast<int64>(alpha) * pipe_size * rest_size;
            for (int64 mp = 0; mp < pipe_size; ++mp) {
                for (int64 mr = 0; mr < rest_size; ++mr) {
                    perm[static_cast<std::size_t>(new_start + (e_offset + mp) * rest_size + mr)] =
                      old_tree_start + mp * rest_size + mr;
                }
            }
        }
        old_start += static_cast<int64>(old_trees.size()) * pipe_size * rest_size;
    }
    return perm;
}

/// Data of a `FusionTreeBackend` tensor, where the pipe ``old_prod.factors[0]`` in the codomain
/// (or domain, if `in_domain`) is replaced by the `ElementarySpace` ``new_prod.factors[0]``.
[[nodiscard]] TensorBackend::DataPtr
flatten_first_pipe_data_fusion_tree(SymmetricTensor const& tensor,
                                    bool in_domain,
                                    TensorProduct const& old_prod,
                                    TensorProduct const& new_prod,
                                    LegPipe& pipe)
{
    if (pipe.is_dual) {
        // dual pipes are bent to the other side first, see `flatten_pipe_leg_at`
        throw std::logic_error("flatten_first_pipe_data_fusion_tree requires a non-dual pipe");
    }
    auto const pipe_legs = pipe.flat_legs();
    if (static_cast<int64>(pipe_legs.size()) != pipe.num_legs) {
        throw NotImplemented("flatten_pipe_leg for nested pipes with the FusionTreeBackend");
    }
    TensorProduct pipe_prod(pipe_legs, tensor.symmetry);
    auto const flat = as_space(new_prod.factors[0]);
    if (pipe_prod.sector_decomposition != flat->sector_decomposition ||
        pipe_prod.multiplicities != flat->multiplicities) {
        throw NotImplemented("flatten_pipe_leg: the pipe components do not match the flat leg");
    }
    auto const data = FusionTreeBackend::unwrap(tensor.data);
    auto const& other_prod = in_domain ? *tensor.codomain : *tensor.domain;
    auto& block_backend = *tensor.backend->block_backend;
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(data->blocks.size());
    for (std::size_t i = 0; i < data->blocks.size(); ++i) {
        auto const coupled = old_prod.sector_decomposition[static_cast<std::size_t>(
          data->block_inds(i, in_domain ? 1 : 0))];
        auto perm = flatten_pipe_block_perm(old_prod, new_prod, pipe_prod, coupled);
        std::vector<int64> other(static_cast<std::size_t>(other_prod.block_size(coupled)));
        std::iota(other.begin(), other.end(), int64{ 0 });
        std::vector<std::vector<int64>> perms{ std::vector<int64>(perm.begin(), perm.end()),
                                               std::move(other) };
        if (in_domain) {
            std::swap(perms[0], perms[1]);
        }
        blocks.push_back(block_backend.apply_axis_permutations(data->blocks[i], perms));
    }
    return FusionTreeBackend::wrap(std::make_shared<FusionTreeData>(
      data->block_inds, std::move(blocks), data->dtype, data->device, true));
}
#endif // LASYTEN_HAS_PYTHON

/// The unitary ``U: pipe -> flat`` used by `flatten_pipe_leg` for the `FusionTreeBackend`, as a
/// tensor with codomain ``[flat]`` and domain ``[pipe]``.
///
/// For a non-dual pipe, all blocks of ``U`` are identities, i.e. ``U`` uses the same order of
/// the states as `flatten_pipe_block_perm`. A dual pipe is ``P.dual()`` for a non-dual pipe
/// ``P``, i.e. the pipe as seen from the other side of a bond. To be consistent with flattening
/// ``P`` on the other side, we use ``(U_P^dagger)^T``. This bends only the small tensor ``U_P``.
[[nodiscard]] TensorPtr
flatten_pipe_unitary(LegPipe::Ptr const& pipe, Leg::Ptr const& flat, SymmetricTensor const& like)
{
    auto const& symmetry = like.symmetry;
    if (pipe->is_dual) {
        auto dual_leg = pipe->dual();
        auto P = dual_leg.as_leg_pipe();
        if (!P) {
            throw std::logic_error("flatten_pipe_leg: dual of LegPipe must be a LegPipe");
        }
        Leg::Ptr flat_P = dual_leg.as_ElementarySpace(false);
        auto U = transpose(dagger(flatten_pipe_unitary(P, flat_P, like)));
        auto U_sym = std::dynamic_pointer_cast<SymmetricTensor>(U);
        if (!U_sym || !(U_sym->domain->factors.at(0) == Leg{ pipe })) {
            throw std::logic_error(
              "flatten_pipe_leg: the dual unitary does not match the dual pipe");
        }
        // The flat leg is ``flat_P.dual``. It has the same sectors as `flat`, but may differ in
        // the `basis_perm`; it is the one consistent with flattening ``P`` on the other side.
        return U;
    }
    std::vector<Leg::Ptr> pipe_factors{ pipe };
    auto pipe_prod = std::make_shared<TensorProduct>(std::move(pipe_factors), symmetry);
    std::vector<Leg::Ptr> flat_factors{ flat };
    auto flat_prod = std::make_shared<TensorProduct>(std::move(flat_factors),
                                                     symmetry,
                                                     pipe_prod->sector_decomposition,
                                                     pipe_prod->multiplicities);
    auto data = like.backend->eye_data(pipe_prod, like.dtype, like.device);
    return std::make_shared<SymmetricTensor>(std::move(data),
                                             std::move(flat_prod),
                                             std::move(pipe_prod),
                                             like.backend,
                                             symmetry,
                                             LegLabelList{ "?flat", "?pipe" });
}

/// `flatten_pipe_leg` for the `FusionTreeBackend` by contracting the pipe at ``legs_idx`` with
/// the unitary `flatten_pipe_unitary`. In contrast to permuting the pipe to the first position,
/// this does not bend any legs of `tensor`, but only requires (one-sided) F-moves.
[[nodiscard]] TensorPtr
flatten_pipe_leg_by_unitary(SymmetricTensorCPtr const& tensor,
                            int64 legs_idx,
                            bool in_domain,
                            TensorProduct const& product,
                            LegPipe::Ptr const& pipe,
                            Leg::Ptr const& flat)
{
    // act on the unhidden tensor, such that the leg indices include hidden legs
    TensorPtr plain = std::const_pointer_cast<SymmetricTensor>(tensor)->as_SymmetricTensor();
    auto const label = plain->labels.as_list().at(static_cast<std::size_t>(legs_idx));
    auto U = flatten_pipe_unitary(pipe, flat, *tensor); // [flat] <- [pipe]
    if (in_domain) {
        U = dagger(U); // [pipe] <- [flat]
    }
    // the label of the contracted leg of U replaces the pipe label
    U->labels.set(in_domain ? LegLabelList{ "?pipe", label } : LegLabelList{ label, "?pipe" });
    TensorPtr res;
    if (product.num_factors == 1) {
        // partial_compose requires a remaining leg; contract the full (co)domain instead
        res = std::get<TensorPtr>(in_domain ? compose(plain, U) : compose(U, plain));
    } else {
        res = partial_compose(plain, U, LegRef{ legs_idx }, std::nullopt, std::nullopt);
    }
    auto res_sym = std::dynamic_pointer_cast<SymmetricTensor>(res);
    // restore the original labels, including the ``!`` of hidden legs
    return HiddenLegTensor::maybe_wrap(std::make_shared<SymmetricTensor>(res_sym->data,
                                                                         res_sym->codomain,
                                                                         res_sym->domain,
                                                                         res_sym->backend,
                                                                         res_sym->symmetry,
                                                                         tensor->labels.as_list(),
                                                                         false));
}

[[nodiscard]] TensorPtr
flatten_pipe_leg_at(TensorCPtr tensor, int64 legs_idx)
{
    auto sym = std::dynamic_pointer_cast<const SymmetricTensor>(tensor);
    if (!sym) {
        throw std::invalid_argument("flatten_pipe_leg expects a SymmetricTensor");
    }
    auto [in_domain, co_idx, unused] = tensor->_parse_leg_idx(legs_idx);
    (void)unused;
    auto product = in_domain ? tensor->domain : tensor->codomain;
    auto factor = product->factors.at(static_cast<std::size_t>(co_idx));
    if (!factor || !factor.is_pipe()) {
        return std::const_pointer_cast<Tensor>(tensor);
    }
#if LASYTEN_HAS_PYTHON
    bool const fusion_tree =
      static_cast<bool>(std::dynamic_pointer_cast<FusionTreeBackend const>(tensor->backend));
#else
    constexpr bool fusion_tree = false; // FusionTreeBackend is only built with Python
#endif
    ElementarySpace::Ptr es;
    LegPipe::Ptr plain_pipe;
    if (factor.is_abelian_leg_pipe()) {
        // AbelianLegPipe is a pipe composed on ElementarySpace; drop the pipe
        // metadata so the bond is a plain charge-shifted ElementarySpace.
        auto alp = std::dynamic_pointer_cast<AbelianLegPipe>(factor.as_elementary_space());
        es = strip_abelian_leg_pipe(alp);
    } else {
        plain_pipe = factor.as_leg_pipe();
        es = factor.as_ElementarySpace(plain_pipe->is_dual);
    }
    if (fusion_tree && plain_pipe && (co_idx != 0 || plain_pipe->is_dual)) {
        // The data can only be permuted for a non-dual pipe as the first factor of the
        // (co)domain. Otherwise, contract the pipe with a unitary ``pipe -> flat leg``, which
        // requires only one-sided F-moves (no bending of `tensor`).
        return flatten_pipe_leg_by_unitary(sym, legs_idx, in_domain, *product, plain_pipe, es);
    }
    auto new_factors = product->factors;
    new_factors.at(static_cast<std::size_t>(co_idx)) = es;
    auto new_product = std::make_shared<TensorProduct>(std::move(new_factors),
                                                       tensor->symmetry,
                                                       product->sector_decomposition,
                                                       product->multiplicities);
    auto new_cod = in_domain ? tensor->codomain : new_product;
    auto new_dom = in_domain ? new_product : tensor->domain;
#if LASYTEN_HAS_PYTHON
    auto data =
      fusion_tree && plain_pipe
        ? flatten_first_pipe_data_fusion_tree(*sym, in_domain, *product, *new_product, *plain_pipe)
        : sym->data;
#else
    auto data = sym->data;
#endif
    auto out = std::make_shared<SymmetricTensor>(std::move(data),
                                                 std::move(new_cod),
                                                 std::move(new_dom),
                                                 tensor->backend,
                                                 tensor->symmetry,
                                                 tensor->labels.as_list(),
                                                 /*check_complex_dtype=*/false);
    if (HiddenLegTensor::has_hidden_leg_labels(out->labels.as_list())) {
        return std::make_shared<HiddenLegTensor>(std::move(out));
    }
    return out;
}

[[nodiscard]] TensorPtr
replace_legs_space(TensorPtr tensor, int64 legs_idx, Leg::Ptr new_legs_space)
{
    auto sym = std::dynamic_pointer_cast<SymmetricTensor>(tensor);
    if (!sym) {
        throw std::invalid_argument("replace_legs_space expects a SymmetricTensor");
    }
    auto [in_domain, co_idx, unused] = tensor->_parse_leg_idx(legs_idx);
    (void)unused;
    auto product = in_domain ? tensor->domain : tensor->codomain;
    Leg::Ptr factor = in_domain ? new_legs_space.dual() : new_legs_space;
    auto new_factors = product->factors;
    new_factors.at(static_cast<std::size_t>(co_idx)) = std::move(factor);
    auto new_product = std::make_shared<TensorProduct>(std::move(new_factors),
                                                       tensor->symmetry,
                                                       product->sector_decomposition,
                                                       product->multiplicities);
    auto new_cod = in_domain ? tensor->codomain : new_product;
    auto new_dom = in_domain ? new_product : tensor->domain;
    auto out = std::make_shared<SymmetricTensor>(sym->data,
                                                 std::move(new_cod),
                                                 std::move(new_dom),
                                                 tensor->backend,
                                                 tensor->symmetry,
                                                 tensor->labels.as_list(),
                                                 /*check_complex_dtype=*/false);
    if (HiddenLegTensor::has_hidden_leg_labels(out->labels.as_list())) {
        return std::make_shared<HiddenLegTensor>(std::move(out));
    }
    return out;
}

[[nodiscard]] std::vector<std::string>
stripped_hidden_labels_except(HiddenLegTensorCPtr const& tensor,
                              std::optional<std::string> const& skip_hidden)
{
    std::vector<std::string> out;
    auto labs = tensor->labels.as_list();
    for (auto idx : tensor->hidden_leg_idcs()) {
        auto const& lab = labs.at(static_cast<std::size_t>(idx));
        if (lab.empty()) {
            continue;
        }
        if (skip_hidden.has_value() && lab == *skip_hidden) {
            continue;
        }
        auto stripped = HiddenLegTensor::strip_hidden_prefix(lab);
        if (!stripped.empty()) {
            out.push_back(stripped);
        }
    }
    return out;
}

[[nodiscard]] TensorPtr
rehide_labels(TensorPtr tensor, std::vector<std::string> const& stripped)
{
    if (stripped.empty()) {
        return tensor;
    }
    std::vector<std::variant<int64, std::string>> which;
    which.reserve(stripped.size());
    for (auto const& lab : stripped) {
        which.emplace_back(lab);
    }
    return HiddenLegTensor::from_tensor(std::move(tensor), std::move(which));
}

} // namespace

TensorPtr
flatten_pipe_leg(TensorCPtr tensor, LegRef which_leg)
{
    int64 idx = leg_idx(tensor, which_leg);
    return flatten_pipe_leg_at(std::move(tensor), idx);
}

std::pair<TensorPtr, HiddenLegTensorPtr>
move_hidden_leg(HiddenLegTensorCPtr A,
                TensorCPtr B,
                LegRef axis_A,
                LegRef axis_B,
                std::string hidden_leg_label,
                std::optional<int64> target_codomain_pos,
                std::optional<int64> target_domain_pos)
{
    if (!A) {
        throw std::invalid_argument("move_hidden_leg() requires non-null Tensor arguments");
    }
    if (!B) {
        throw std::invalid_argument("move_hidden_leg: B must not be null");
    }
    if (std::dynamic_pointer_cast<ChargedTensor const>(B)) {
        throw std::invalid_argument("move_hidden_leg is not supported for ChargedTensor B.");
    }
    if (std::dynamic_pointer_cast<DiagonalTensor const>(B) ||
        std::dynamic_pointer_cast<Mask const>(B)) {
        throw std::invalid_argument(
          "move_hidden_leg requires B to be a SymmetricTensor (including HiddenLegTensor).");
    }
    auto B_sym = std::dynamic_pointer_cast<SymmetricTensor const>(B);
    if (!B_sym) {
        throw std::invalid_argument(
          "move_hidden_leg requires B to be a SymmetricTensor (including HiddenLegTensor).");
    }
    if (target_codomain_pos.has_value() == target_domain_pos.has_value()) {
        throw std::invalid_argument(
          "move_hidden_leg: specify exactly one of target_codomain_pos and target_domain_pos.");
    }
    if (!HiddenLegTensor::is_hidden_leg_label(hidden_leg_label)) {
        throw std::invalid_argument(std::format(
          "move_hidden_leg: hidden_leg_label '{}' must be a hidden label (including '!').",
          hidden_leg_label));
    }

    int64 hidden_idx = -1;
    try {
        hidden_idx = A->labels.index(hidden_leg_label);
    } catch (std::invalid_argument const&) {
        throw std::invalid_argument(
          std::format("move_hidden_leg: A has no hidden leg '{}'. Labels are not matching.",
                      hidden_leg_label));
    }
    auto A_labs = A->labels.as_list();
    if (!HiddenLegTensor::is_hidden_leg_label(A_labs.at(static_cast<std::size_t>(hidden_idx)))) {
        throw std::invalid_argument(
          std::format("move_hidden_leg: '{}' is not a hidden leg of A.", hidden_leg_label));
    }

    int64 axis_A_idx = leg_idx(A, axis_A);
    int64 axis_B_idx = leg_idx(B, axis_B);
    if (HiddenLegTensor::is_hidden_leg_label(A_labs.at(static_cast<std::size_t>(axis_A_idx)))) {
        throw std::invalid_argument("move_hidden_leg: axis_A must be a public leg.");
    }
    auto B_labs = B->labels.as_list();
    if (HiddenLegTensor::is_hidden_leg_label(B_labs.at(static_cast<std::size_t>(axis_B_idx)))) {
        throw std::invalid_argument("move_hidden_leg: axis_B must be a public leg.");
    }

    auto a_axis_leg = A->get_leg(axis_A_idx);
    auto b_axis_leg = B->get_leg(axis_B_idx);
    if (!(a_axis_leg == b_axis_leg.dual())) {
        throw std::invalid_argument(
          "move_hidden_leg: axis_A and axis_B are not contractible (must be duals).");
    }

    if (B->labels.has(hidden_leg_label)) {
        throw std::invalid_argument(std::format(
          "move_hidden_leg: B already has label '{}'. Relabel one of the hidden legs first.",
          hidden_leg_label));
    }
    auto dual_hidden = _dual_leg_label(hidden_leg_label);
    if (!dual_hidden.empty() && B->labels.has(dual_hidden)) {
        throw std::invalid_argument(std::format(
          "move_hidden_leg: B already has dual hidden label '{}'. Dual hidden pairs on one "
          "tensor are not allowed.",
          dual_hidden));
    }

    auto stripped_h = HiddenLegTensor::strip_hidden_prefix(hidden_leg_label);
    if (stripped_h.empty()) {
        throw std::invalid_argument("move_hidden_leg: hidden_leg_label has no name after '!'.");
    }
    auto A_other_hidden = stripped_hidden_labels_except(A, hidden_leg_label);
    std::vector<std::string> B_other_hidden;
    if (auto B_hid = std::dynamic_pointer_cast<HiddenLegTensor const>(B)) {
        B_other_hidden = stripped_hidden_labels_except(B_hid, std::nullopt);
    }

    auto original_axis_A_label = A_labs.at(static_cast<std::size_t>(axis_A_idx));
    auto original_axis_B_label = B_labs.at(static_cast<std::size_t>(axis_B_idx));

    auto A_work = A->unhide_legs();
    A_work = std::dynamic_pointer_cast<SymmetricTensor>(rehide_labels(A_work, A_other_hidden));
    SymmetricTensorPtr B_work;
    if (auto B_hid = std::dynamic_pointer_cast<HiddenLegTensor const>(B)) {
        B_work = B_hid->unhide_legs();
    } else {
        B_work = std::dynamic_pointer_cast<SymmetricTensor>(
          std::const_pointer_cast<SymmetricTensor>(B_sym)->copy(/*deep=*/false));
        if (!B_work) {
            throw std::runtime_error("move_hidden_leg: expected SymmetricTensor copy of B");
        }
    }
    B_work = std::dynamic_pointer_cast<SymmetricTensor>(rehide_labels(B_work, B_other_hidden));

    if (original_axis_A_label.empty()) {
        original_axis_A_label = unique_temp_label(A_work, "_mhl_axisA");
        A_work->labels[axis_A_idx] = original_axis_A_label;
    }
    if (original_axis_B_label.empty()) {
        original_axis_B_label = unique_temp_label(B_work, "_mhl_axisB");
        B_work->labels[axis_B_idx] = original_axis_B_label;
    }

    auto h_leg = A->get_leg(hidden_idx);
    // For anyons, `dim` is weighted by the quantum dimensions and not exactly an integer.
    bool const flatten_1d = std::abs(h_leg.dim() - 1.) < 1e-8;

    auto open_lab = unique_temp_label(B_work, "_mhl_open");
    auto int_lab = unique_temp_label(B_work, "_mhl_int");
    if (open_lab == int_lab) {
        int_lab = unique_temp_label(B_work, "_mhl_intX");
    }
    LegLayout B_layout = LegLayout::of(B_work);
    for (auto* side : { &B_layout.codomain, &B_layout.domain }) {
        for (auto& lab : *side) {
            if (lab.empty()) {
                // unique placeholders, to keep track of the crossings of unlabelled legs
                lab = unique_temp_label(B_work,
                                        "_mhl_unlabelled_" + std::to_string(&lab - side->data()) +
                                          (side == &B_layout.domain ? "d" : "c"));
            }
        }
    }
    LegTarget const open_target = target_codomain_pos.has_value()
                                    ? LegTarget{ false, *target_codomain_pos }
                                    : LegTarget{ true, *target_domain_pos };

    // Conventions. Like all hidden legs, the moved leg passes behind the public legs and must not
    // braid with other hidden legs. Contracting `axis_A` with `axis_B`, the open legs of the
    // network are, clockwise, the legs of `A` after `axis_A`, then the legs of `B` after `axis_B`.
    // The result is such that the contraction is preserved if we slide the hidden leg along this
    // boundary, from its position on `A` to its target position on `B` (*not* around a tensor,
    // which differs by a braid for non-symmetric braids). The slide passes one of the two gaps
    // next to the bond: "cw", between the last leg of `A` and the first leg of `B`, or "ccw",
    // between the last leg of `B` and the first leg of `A`. We choose the one which does not
    // cross other hidden legs.
    // `h` needs to be right before `axis_A`, i.e. clockwise before it (codomain) or after it
    // (domain), next to the cw or ccw gap.
    bool const A_slot_cw =
      A_work->labels.index(original_axis_A_label) < A_work->num_codomain_legs();
    bool slide_cw;
    {
        auto const A_after = clockwise_after(LegLayout::of(A_work), original_axis_A_label);
        auto const B_after =
          clockwise_after(B_layout.moved(open_lab, open_target), original_axis_B_label);
        auto const h_pos = std::find(A_after.begin(), A_after.end(), stripped_h);
        auto const t_pos = std::find(B_after.begin(), B_after.end(), open_lab);
        auto const hidden_in = [](auto first, auto last) {
            return std::any_of(first, last, [](std::string const& lab) {
                return HiddenLegTensor::is_hidden_leg_label(lab);
            });
        };
        bool const cw_hidden =
          hidden_in(h_pos + 1, A_after.end()) || hidden_in(B_after.begin(), t_pos);
        bool const ccw_hidden =
          hidden_in(A_after.begin(), h_pos) || hidden_in(t_pos + 1, B_after.end());
        // If both or none are possible, use the gap on the side of the slot of `h`.
        slide_cw = (cw_hidden == ccw_hidden) ? A_slot_cw : !cw_hidden;
    }

    // Move `h` on `A` right before `axis_A`. If the slide uses the gap on the other side of the
    // bond, `h` passes behind `axis_A`.
    {
        auto const target =
          LegLayout::of(A_work).next_to(stripped_h, original_axis_A_label, /*after=*/false);
        std::vector<std::string> const axis_A_only{ original_axis_A_label };
        A_work = std::dynamic_pointer_cast<SymmetricTensor>(
          slide_cw == A_slot_cw ? move_leg_avoiding(A_work, stripped_h, target, axis_A_only)
                                : move_leg_avoiding(A_work, stripped_h, target, {}, axis_A_only));
    }
    if (!A_work) {
        throw std::runtime_error(
          "move_hidden_leg: expected SymmetricTensor after moving A's hidden");
    }
    auto h_idx_A = A_work->labels.index(stripped_h);
    A_work = std::dynamic_pointer_cast<SymmetricTensor>(
      combine_legs(A_work,
                   std::vector<std::vector<LegRef>>{
                     { LegRef{ stripped_h }, LegRef{ original_axis_A_label } } },
                   std::nullopt,
                   std::nullopt,
                   levels_with_hidden(A_work, h_idx_A)));
    if (!A_work) {
        throw std::runtime_error("move_hidden_leg: expected SymmetricTensor after combining A");
    }
    auto pipe_A_label = _combine_leg_labels({ stripped_h, original_axis_A_label });
    int64 pipe_A_idx = A_work->labels.index(pipe_A_label);
    A_work->labels[pipe_A_idx] = original_axis_A_label;
    auto pipe_A = A_work->get_leg(A_work->labels.index(original_axis_A_label));
    auto desired_B_legs = pipe_A.dual();

    // We insert an identity wire on the hidden space next to `B`, with ends `int` and `open`,
    // ``B_ext.get_leg(open) == h_leg``. `int` is moved right after `axis_B` and combined with it,
    // and `open` becomes the hidden leg of `B2`. The wire leaves the pipe towards the gap of the
    // slide, i.e. it passes behind `axis_B` if `int` is on the other side, and `open` ends up
    // right before `axis_B` or right after `int`, respectively. The ends of the wire must not
    // cross each other, which would twist it, and should not cross the hidden legs of `B`.
    // We simulate the possible `IdentityWirePlan`s on the labels and use the cheapest valid one.
    // If all of them cross hidden legs, the moves raise for non-symmetric braids.
    bool const int_cw_after_axis =
      B_work->labels.index(original_axis_B_label) < B_work->num_codomain_legs();
    // `int` is next to the cw gap if it is clockwise after `axis_B`
    bool const behind_axis = int_cw_after_axis != slide_cw;
    std::optional<IdentityWirePlan> best_plan;
    bool best_crosses_hidden = true;
    std::size_t best_cost = 0;
    for (bool right_edge : { false, true }) {
        for (bool int_in_codomain : { true, false }) {
            for (bool open_first : { true, false }) {
                IdentityWirePlan const plan{ right_edge, int_in_codomain, open_first };
                auto layout = identity_wire_layout(B_layout, plan, int_lab, open_lab);
                bool crosses_hidden = false;
                bool valid = true;
                std::size_t cost = 0;
                std::map<std::string, int> num_crossings; // of the wire, i.e. by both ends
                auto const moves = identity_wire_moves(
                  plan, behind_axis, original_axis_B_label, int_lab, open_lab);
                for (auto const& move : moves) {
                    auto const& moving = move.moving;
                    auto const& other = moving == int_lab ? open_lab : int_lab;
                    auto const target = layout.next_to(moving, move.reference, move.after);
                    auto const chosen = choose_route(layout, moving, target, { other });
                    auto const labs = layout.labels();
                    for (auto i : chosen.route.crossed) {
                        num_crossings[labs.at(static_cast<std::size_t>(i))] += 1;
                    }
                    valid = valid && !chosen.crosses_never_cross;
                    crosses_hidden = crosses_hidden || chosen.crosses_hidden;
                    cost += chosen.route.crossed.size();
                    layout = layout.moved(moving, target);
                }
                for (auto const& [lab, num] : num_crossings) {
                    if ((num % 2 == 1) != (behind_axis && lab == original_axis_B_label)) {
                        valid = false;
                    }
                }
                if (!valid) {
                    continue;
                }
                if (!best_plan.has_value() || (best_crosses_hidden && !crosses_hidden) ||
                    (best_crosses_hidden == crosses_hidden && cost < best_cost)) {
                    best_plan = plan;
                    best_crosses_hidden = crosses_hidden;
                    best_cost = cost;
                }
            }
        }
    }
    if (!best_plan.has_value()) {
        throw std::logic_error("move_hidden_leg: found no way to insert the identity wire");
    }
    auto const plan = *best_plan;
    auto I = plan.int_in_codomain ? eye(as_space(h_leg.dual()),
                                        A->backend,
                                        LegLabelList{ int_lab, open_lab },
                                        A->dtype,
                                        A->device,
                                        /*diagonal=*/false)
                                  : eye(as_space(h_leg),
                                        A->backend,
                                        LegLabelList{ open_lab, int_lab },
                                        A->dtype,
                                        A->device,
                                        /*diagonal=*/false);
    auto B_ext = plan.right_edge ? outer(B_work, I) : outer(I, B_work);
    for (auto const& move :
         identity_wire_moves(plan, behind_axis, original_axis_B_label, int_lab, open_lab)) {
        auto const& other = move.moving == int_lab ? open_lab : int_lab;
        auto const target = LegLayout::of(B_ext).next_to(move.moving, move.reference, move.after);
        B_ext = move_leg_avoiding(std::move(B_ext), move.moving, target, { other });
    }
    auto int_idx = B_ext->labels.index(int_lab);
    auto [b_in_dom, b_co, b_legs] = B_ext->_parse_leg_idx(int_lab);
    (void)b_co;
    (void)b_legs;
    // Domain combine stores the dual of `result.legs`; codomain stores `result.legs`.
    std::vector<Leg::Ptr> b_pipes{ b_in_dom ? pipe_A : desired_B_legs };
    B_ext = combine_legs(
      B_ext,
      std::vector<std::vector<LegRef>>{ { LegRef{ original_axis_B_label }, LegRef{ int_lab } } },
      PipeDualities{ desired_B_legs.is_dual() },
      b_pipes,
      levels_with_hidden(B_ext, int_idx));
    auto pipe_B_label = _combine_leg_labels({ original_axis_B_label, int_lab });
    int64 pipe_B_idx = B_ext->labels.index(pipe_B_label);
    B_ext->labels[pipe_B_idx] = original_axis_B_label;
    if (flatten_1d) {
        A_work = std::dynamic_pointer_cast<SymmetricTensor>(
          flatten_pipe_leg_at(A_work, A_work->labels.index(original_axis_A_label)));
        if (!A_work) {
            throw std::runtime_error(
              "move_hidden_leg: expected SymmetricTensor after flattening A");
        }
        auto b_axis_idx = B_ext->labels.index(original_axis_B_label);
        B_ext = flatten_pipe_leg_at(B_ext, b_axis_idx);
        auto a_flat = A_work->get_leg(A_work->labels.index(original_axis_A_label));
        if (!(B_ext->get_leg(b_axis_idx) == a_flat.dual())) {
            throw std::logic_error("move_hidden_leg: flattened bond legs do not match");
        }
    }
    // Slide `open` from the gap to its target, along the legs of `B`, i.e. not behind the pipe.
    B_ext = move_leg_avoiding(std::move(B_ext), open_lab, open_target, { original_axis_B_label });
    auto open_idx = B_ext->labels.index(open_lab);
    B_ext->labels[open_idx] = stripped_h;

    auto A_out = rehide_labels(A_work, A_other_hidden);
    std::vector<std::string> B_hide = B_other_hidden;
    B_hide.insert(B_hide.begin(), stripped_h);
    auto B_out_tensor = rehide_labels(std::move(B_ext), B_hide);
    auto B_out = std::dynamic_pointer_cast<HiddenLegTensor>(B_out_tensor);
    if (!B_out) {
        throw std::runtime_error("move_hidden_leg: expected HiddenLegTensor result for B");
    }
    return { std::move(A_out), std::move(B_out) };
}

TensorPtr
flip_leg_duality(TensorCPtr tensor, LegRef which_leg)
{
    if (!tensor) {
        throw std::invalid_argument("flip_leg_duality() requires a non-null Tensor");
    }
    auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(tensor);
    if (!sym) {
        throw std::invalid_argument("flip_leg_duality requires a SymmetricTensor.");
    }
    if (HiddenLegTensor::has_hidden_leg_labels(tensor->labels.as_list())) {
        throw NotImplemented("flip_leg_duality for tensors with hidden legs");
    }
    int64 const idx = leg_idx(tensor, which_leg);
    auto const labels = tensor->labels.as_list();
    auto const orig_leg = tensor->get_leg(idx);
    // a pipe leg is replaced by its flat `ElementarySpace` first (no-op for other legs)
    TensorPtr cur = flatten_pipe_leg_at(tensor, idx);
    bool const was_dual = cur->get_leg(idx).is_dual();
    // a one-leg group keeps its position, so no bending is needed. Combining it into a pipe with
    // the opposite duality and flattening that pipe gives the isomorphic space with opposite
    // `is_dual`; `flatten_pipe_leg_at` takes care of the (Frobenius-Schur) signs.
    auto combined = planar_combine_legs(
      cur, std::vector<std::vector<LegRef>>{ { LegRef{ idx } } }, PipeDualities{ !was_dual });
    TensorPtr res = flatten_pipe_leg_at(combined, idx);
    res->labels = labels;
    auto dss = orig_leg.is_pipe()
                 ? nullptr
                 : std::dynamic_pointer_cast<DirectSumSpace>(orig_leg.as_elementary_space());
    if (dss) {
        // Keep the direct-sum structure (e.g. the summand labels of MPO bonds). The flipped
        // `DirectSumSpace` has the same internal structure as the flat leg that we got from the
        // pipe, so the data of the tensor is valid for it and only the leg object is replaced.
        auto target = dss->with_opposite_duality();
        auto const flat_leg = res->get_leg(idx);
        [[maybe_unused]] auto flat = flat_leg.is_pipe() ? nullptr : flat_leg.as_elementary_space();
        assert(target && flat && target->is_dual == flat->is_dual &&
               target->sector_decomposition == flat->sector_decomposition &&
               target->multiplicities == flat->multiplicities &&
               "flip_leg_duality: the flipped direct sum space must match the flipped leg");
        res = replace_legs_space(std::move(res), idx, Leg{ target });
    }
    return res;
}

std::pair<TensorPtr, TensorPtr>
flip_bond_duality(TensorCPtr A, TensorCPtr B, LegRef leg_A, LegRef leg_B)
{
    if (!A || !B) {
        throw std::invalid_argument("flip_bond_duality() requires non-null Tensor arguments");
    }
    int64 const idx_A = leg_idx(A, leg_A);
    int64 const idx_B = leg_idx(B, leg_B);
    bool const same_tensor = A.get() == B.get();
    if (same_tensor && idx_A == idx_B) {
        throw std::invalid_argument("flip_bond_duality: if A and B are the same tensor, leg_A and "
                                    "leg_B must be different legs.");
    }
    if (!(A->get_leg(idx_A) == B->get_leg(idx_B).dual())) {
        throw std::invalid_argument(
          "flip_bond_duality: leg_A and leg_B are not contractible (must be duals).");
    }
    if (same_tensor) {
        // Both ends of the bond are legs of one tensor (e.g. the wrap bond of a one-site unit
        // cell). Flip both of them there; the position of the legs does not change.
        auto T = flip_leg_duality(flip_leg_duality(A, LegRef{ idx_A }), LegRef{ idx_B });
        assert(T->get_leg(idx_A) == T->get_leg(idx_B).dual() &&
               "flip_bond_duality: the flipped legs must be contractible");
        return { T, T };
    }
    auto A2 = flip_leg_duality(A, LegRef{ idx_A });
    auto B2 = flip_leg_duality(B, LegRef{ idx_B });
    assert(A2->get_leg(idx_A) == B2->get_leg(idx_B).dual() &&
           "flip_bond_duality: the flipped legs must be contractible");
    return { std::move(A2), std::move(B2) };
}

namespace {

ElementarySpace::Ptr
as_elementary_leg(Leg::Ptr const& leg_obj)
{
    if (!leg_obj || leg_obj.is_pipe()) {
        throw std::invalid_argument("slice_leg is not supported on LegPipes.");
    }
    auto es = leg_obj.as_elementary_space();
    if (!es) {
        throw std::invalid_argument("slice_leg requires an ElementarySpace leg.");
    }
    return es;
}

void
check_slice_leg_tensor(TensorCPtr const& tensor)
{
    if (std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        throw std::invalid_argument("slice_leg is not supported for ChargedTensor.");
    }
    if (std::dynamic_pointer_cast<HiddenLegTensor const>(tensor)) {
        throw std::invalid_argument("slice_leg is not supported for HiddenLegTensor.");
    }
}

} // namespace

HiddenLegTensorPtr
slice_leg(TensorCPtr tensor, LegRef leg, int64 idx)
{
    check_slice_leg_tensor(tensor);
    if (!tensor->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("slice_leg with a public-basis index requires a droppable symmetry. "
                      "Got {}.",
                      tensor->symmetry->str()));
    }
    auto co_space = as_elementary_leg(tensor->get_leg_co_domain(leg));
    auto const [sector_idx, multiplicity_idx] = co_space->parse_index(idx);
    Sector sector = co_space->sector_decomposition[static_cast<std::size_t>(sector_idx)];
    // Fusion-tree internal slices are (sector_dim, multiplicity) in C-order, matching
    // outer(ones(dim), diag_block). The reduced copy index is therefore
    // multiplicity_idx % multiplicity (abelian: sector_dim == 1, same as / dim).
    int64 const m = multiplicity_idx % co_space->sector_multiplicity(sector);
    return slice_leg(std::move(tensor), std::move(leg), sector, m);
}

HiddenLegTensorPtr
slice_leg(TensorCPtr tensor, LegRef leg, Sector const& sector, int64 multiplicity)
{
    check_slice_leg_tensor(tensor);
    auto const [in_domain, unused_co, unused_legs] = tensor->_parse_leg_idx(leg);
    (void)unused_co;
    (void)unused_legs;
    auto co_space = as_elementary_leg(tensor->get_leg_co_domain(leg));
    if (!co_space->sector_decomposition_where(sector)) {
        throw std::invalid_argument("slice_leg: sector does not appear on the chosen leg.");
    }
    int64 const mult = co_space->sector_multiplicity(sector);
    int64 const m = to_valid_idx(multiplicity, mult);

    // apply_mask matches the legs[] view; domain factors are dual to that view.
    auto mask_space = as_elementary_leg(tensor->get_leg(leg));
    Sector mask_sector = in_domain ? tensor->symmetry->dual_sector(sector) : sector;

    Sector sector_cap = mask_sector;
    auto bb = tensor->backend->block_backend;
    auto device = tensor->device;
    SectorBlockFactoryFn func = [sector_cap, m, bb, device](std::vector<int64> const& shape,
                                                            Sector const& coupled) {
        auto block = bb->zeros(shape, Dtype::Bool, device);
        if (coupled == sector_cap) {
            block->set_item(m, bb->as_scalar(true));
        }
        return block;
    };

    auto diag = DiagonalTensor::from_sector_block_func(
      std::move(func), mask_space, tensor->backend, std::nullopt, Dtype::Bool, tensor->device);
    auto mask = Mask::from_DiagonalTensor(diag);
    auto masked = apply_mask(tensor, mask, leg);
    auto moved = move_leg(masked,
                          leg,
                          /*codomain_pos=*/std::nullopt,
                          /*domain_pos=*/0);
    moved->labels[-1] = std::string("slice");
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(moved);
    if (!inv) {
        inv = moved->as_SymmetricTensor();
    }
    return HiddenLegTensor::from_tensor(std::move(inv), { LegRef{ int64(-1) } });
}

} // namespace lasyten
