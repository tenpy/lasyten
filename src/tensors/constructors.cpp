#include <lasyten/tensors/constructors.h>

#include <lasyten/backends/backend_factory.h>
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

bool
legs_equal(Leg::Ptr const& a, Leg::Ptr const& b)
{
    return a && b && (a == b);
}

bool
products_equal(TensorProduct::Ptr const& a, TensorProduct::Ptr const& b)
{
    return a && b && (a == b);
}

std::vector<Leg::Ptr>
factors_slice(TensorProduct::Ptr const& tp, int64 start, int64 stop)
{
    auto const n = static_cast<int64>(tp->factors.size());
    if (start < 0) {
        start += n;
    }
    if (stop < 0) {
        stop += n;
    }
    std::vector<Leg::Ptr> out;
    for (int64 i = start; i < stop; ++i) {
        out.push_back(tp->factors[static_cast<std::size_t>(i)]);
    }
    return out;
}

bool
factor_slices_equal(std::vector<Leg::Ptr> const& a, std::vector<Leg::Ptr> const& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!legs_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

TensorPtr
eye(Space::Ptr leg,
    TensorBackend::Ptr backend,
    std::optional<LegLabelList> labels,
    Dtype dtype,
    std::optional<std::string> device,
    bool diagonal)
{
    auto res = DiagonalTensor::from_eye(
      std::move(leg), std::move(backend), std::move(labels), dtype, std::move(device));
    if (diagonal) {
        return res;
    }
    return res->as_SymmetricTensor();
}

TensorPtr
tensor(TensorCPtr obj,
       TensorProduct::Ptr codomain,
       TensorProduct::Ptr domain,
       TensorBackend::Ptr backend,
       std::optional<LegLabelList> labels,
       std::optional<Dtype> dtype,
       std::optional<std::string> device)
{
    if (!products_equal(codomain, obj->codomain)) {
        throw std::invalid_argument("Mismatching codomain");
    }
    if (domain && !products_equal(domain, obj->domain)) {
        throw std::invalid_argument("Mismatching domain");
    }
    if (backend && (!obj->backend || !(*backend == *obj->backend))) {
        throw std::invalid_argument("Mismatching backend");
    }
    TensorPtr out;
    if (labels.has_value() && *labels != obj->labels.as_list()) {
        out = std::const_pointer_cast<Tensor>(obj)->copy();
        out->labels.set(*labels);
    } else {
        out = std::const_pointer_cast<Tensor>(obj);
    }
    if (dtype.has_value()) {
        throw std::invalid_argument("Mismatching dtype");
    }
    if (device.has_value()) {
        throw std::invalid_argument("Mismatching device");
    }
    return out->as_SymmetricTensor();
}

SymmetricTensorPtr
tensor(BlockBackend::BlockPtr obj,
       TensorProduct::Ptr codomain,
       TensorProduct::Ptr domain,
       TensorBackend::Ptr backend,
       std::optional<LegLabelList> labels,
       std::optional<Dtype> dtype,
       std::optional<std::string> device,
       bool understood_braiding)
{
    return SymmetricTensor::from_dense_block(std::move(obj),
                                             std::move(codomain),
                                             std::move(domain),
                                             std::move(backend),
                                             std::move(labels),
                                             dtype,
                                             std::move(device),
                                             1e-6,
                                             understood_braiding);
}

TensorPtr
add_trivial_leg(TensorCPtr tens,
                std::optional<int64> legs_pos_opt,
                std::optional<int64> codomain_pos_opt,
                std::optional<int64> domain_pos_opt,
                std::string label,
                bool is_dual)
{
    if (!label.empty() && tens->labels.has(label)) {
        throw std::invalid_argument(std::format("Tensor already has label '{}'", label));
    }

    // --- hints from Python add_trivial_leg ---
    // parse position to format:
    // - leg_pos: int,  0 <= leg_pos < res_num_legs
    // - add_to_domain: bool
    // - co_domain_pos: int, 0 <= co_domain_pos < num_[co]domain_legs
    // - is_dual: bool, if the leg in the [co]domain should be dual
    // domain[0] is the charge leg, so we need to add 1
    // ---
    int64 res_num_legs = tens->num_legs + 1;
    int64 legs_pos = 0;
    int64 co_domain_pos = 0;
    bool add_to_domain = false;

    if (legs_pos_opt.has_value()) {
        if (codomain_pos_opt.has_value() || domain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        legs_pos = to_valid_idx(*legs_pos_opt, res_num_legs);
        add_to_domain = legs_pos > tens->num_codomain_legs();
        if (add_to_domain) {
            co_domain_pos = res_num_legs - 1 - legs_pos;
        } else {
            co_domain_pos = legs_pos;
        }
    } else if (codomain_pos_opt.has_value()) {
        if (legs_pos_opt.has_value() || domain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        int64 res_codomain_legs = tens->num_codomain_legs() + 1;
        int64 codomain_pos = to_valid_idx(*codomain_pos_opt, res_codomain_legs);
        add_to_domain = false;
        co_domain_pos = codomain_pos;
        legs_pos = codomain_pos;
    } else if (domain_pos_opt.has_value()) {
        if (legs_pos_opt.has_value() || codomain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        int64 res_domain_legs = tens->num_domain_legs() + 1;
        int64 domain_pos = to_valid_idx(*domain_pos_opt, res_domain_legs);
        add_to_domain = true;
        co_domain_pos = domain_pos;
        legs_pos = res_num_legs - 1 - domain_pos;
    } else {
        add_to_domain = false;
        co_domain_pos = 0;
        legs_pos = 0;
    }

    if (std::dynamic_pointer_cast<DiagonalTensor const>(tens) ||
        std::dynamic_pointer_cast<Mask const>(tens) ||
        std::dynamic_pointer_cast<Identity const>(tens)) {
        std::string msg = "Converting to SymmetricTensor for add_trivial_leg. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tens = std::const_pointer_cast<Tensor>(tens)->as_SymmetricTensor(false, std::move(msg));
    }
    if (auto charged = std::dynamic_pointer_cast<ChargedTensor const>(tens)) {
        TensorPtr inv_part;
        if (add_to_domain) {
            // domain[0] is the charge leg, so we need to add 1
            inv_part = add_trivial_leg(charged->invariant_part,
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/std::nullopt,
                                       /*domain_pos=*/co_domain_pos + 1,
                                       label,
                                       is_dual);
        } else {
            inv_part = add_trivial_leg(charged->invariant_part,
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/co_domain_pos,
                                       /*domain_pos=*/std::nullopt,
                                       label,
                                       is_dual);
        }
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("add_trivial_leg expected SymmetricTensor invariant_part");
        }
        return std::make_shared<ChargedTensor>(std::move(inv_sym), charged->charged_state);
    }
    auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(tens);
    if (!sym) {
        throw py::type_error("Invalid type for tens. Expected a Tensor subtype");
    }

    auto new_leg = ElementarySpace::from_trivial_sector(1, tens->symmetry, is_dual);
    TensorProduct::Ptr domain;
    TensorProduct::Ptr codomain;
    if (add_to_domain) {
        domain = tens->domain->insert_multiply(new_leg, co_domain_pos);
        codomain = tens->codomain;
    } else {
        domain = tens->domain;
        codomain = tens->codomain->insert_multiply(new_leg, co_domain_pos);
    }
    auto backend = tens->backend;
    auto data =
      backend->add_trivial_leg(tens, legs_pos, add_to_domain, co_domain_pos, codomain, domain);

    LegLabelList labels = tens->labels.as_list();
    LegLabelList new_labels;
    new_labels.reserve(labels.size() + 1);
    new_labels.insert(new_labels.end(), labels.begin(), labels.begin() + legs_pos);
    new_labels.push_back(label);
    new_labels.insert(new_labels.end(), labels.begin() + legs_pos, labels.end());

    // a HiddenLegTensor keeps the hidden prefix in its labels -> wrap the result again
    return HiddenLegTensor::maybe_wrap(std::make_shared<SymmetricTensor>(std::move(data),
                                                                         std::move(codomain),
                                                                         std::move(domain),
                                                                         backend,
                                                                         tens->symmetry,
                                                                         new_labels));
}

TensorPtr
zero_like(TensorCPtr tensor)
{
    if (auto mask = std::dynamic_pointer_cast<Mask const>(tensor)) {
        return Mask::from_zero(
          mask->large_leg(), mask->backend, mask->labels.as_list(), mask->device);
    }
    if (auto diag = std::dynamic_pointer_cast<DiagonalTensor const>(tensor)) {
        return DiagonalTensor::from_zero(
          diag->leg(), diag->backend, diag->labels.as_list(), diag->dtype, diag->device);
    }
    if (auto charged = std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        auto charge = charged->charge_leg.as_elementary_space();
        if (!charge) {
            throw std::invalid_argument("zero_like: charge_leg must be an ElementarySpace");
        }
        return ChargedTensor::from_zero(charged->codomain,
                                        charged->domain,
                                        charge,
                                        charged->charged_state,
                                        charged->backend,
                                        charged->labels.as_list(),
                                        charged->dtype,
                                        charged->device);
    }
    if (auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(tensor)) {
        // a HiddenLegTensor keeps the hidden prefix in its labels -> wrap the result again
        return HiddenLegTensor::maybe_wrap(SymmetricTensor::from_zero(sym->codomain,
                                                                      sym->domain,
                                                                      sym->backend,
                                                                      sym->labels.as_list(),
                                                                      sym->dtype,
                                                                      sym->device));
    }
    throw py::type_error("Invalid type for tensor.");
}

TensorPtr
tensor_from_grid(std::vector<std::vector<TensorPtr>> grid,
                 std::vector<LegRef> stack_legs,
                 std::optional<Dtype> dtype_opt,
                 std::vector<std::optional<LegLabelList>> stack_labels,
                 std::vector<std::optional<std::vector<ElementarySpace::Ptr>>> stack_summands)
{
    if (stack_legs.empty() || stack_legs.size() > 2) {
        throw std::invalid_argument("stack_legs must contain one or two leg references");
    }
    auto const n_dimensions = static_cast<int64>(stack_legs.size());
    if (!stack_labels.empty() && static_cast<int64>(stack_labels.size()) != n_dimensions) {
        throw std::invalid_argument("stack_labels must have one entry per stacking dimension");
    }
    if (!stack_summands.empty() && static_cast<int64>(stack_summands.size()) != n_dimensions) {
        throw std::invalid_argument("stack_summands must have one entry per stacking dimension");
    }

    auto n_rows = static_cast<int64>(grid.size());
    int64 n_cols = n_rows > 0 ? static_cast<int64>(grid[0].size()) : 0;
    for (auto const& row : grid) {
        if (static_cast<int64>(row.size()) != n_cols) {
            throw std::invalid_argument("grid rows must have equal length");
        }
    }
    if (stack_legs.size() == 2 && n_rows == 0) {
        throw std::invalid_argument("a two-dimensional grid must have at least one row");
    }
    if (stack_legs.size() == 1 && n_rows != 1) {
        throw std::invalid_argument("a one-dimensional grid must be passed as a single row");
    }

    std::vector<TensorCPtr> ops;
    for (auto const& row : grid) {
        for (auto const& op : row) {
            if (op) {
                ops.push_back(op);
            }
        }
    }
    if (ops.empty()) {
        throw std::invalid_argument("grid must contain at least one tensor");
    }

    auto backend = get_same_backend(ops);
    std::string device = ops[0]->device;
    for (auto const& op : ops) {
        if (op->device != device) {
            throw std::invalid_argument("Incompatible devices.");
        }
    }

    Dtype dtype;
    if (dtype_opt.has_value()) {
        dtype = *dtype_opt;
    } else {
        std::vector<Dtype> dtypes;
        dtypes.reserve(ops.size());
        for (auto const& op : ops) {
            dtypes.push_back(op->dtype);
        }
        dtype = dtype::common(dtypes);
    }

    auto const& ref = ops[0];
    int64 n_cod = ref->num_codomain_legs();
    int64 n_dom = ref->num_domain_legs();
    for (auto const& op : ops) {
        if (op->num_codomain_legs() != n_cod || op->num_domain_legs() != n_dom) {
            throw std::runtime_error("inconsistent number of legs in grid");
        }
    }

    auto stack_indices = ref->labels.indices(stack_legs);
    if (stack_indices.size() != stack_legs.size()) {
        throw std::runtime_error("tensor_from_grid: failed to resolve stacking leg references");
    }
    if (stack_indices.size() == 2 && stack_indices[0] == stack_indices[1]) {
        throw std::invalid_argument("stack_legs must refer to distinct legs");
    }
    auto n_legs = ref->num_legs;
    auto is_codomain_leg = [n_cod](int64 i) { return i >= 0 && i < n_cod; };

    std::vector<int64> non_stacked;
    for (int64 i = 0; i < n_legs; ++i) {
        if (std::find(stack_indices.begin(), stack_indices.end(), i) == stack_indices.end()) {
            non_stacked.push_back(i);
        }
    }
    std::vector<Leg::Ptr> ref_non_stacked;
    for (auto i : non_stacked) {
        ref_non_stacked.push_back(ref->get_leg(i));
    }
    for (auto const& op : ops) {
        std::vector<Leg::Ptr> op_non_stacked;
        for (auto i : non_stacked) {
            op_non_stacked.push_back(op->get_leg(i));
        }
        if (!factor_slices_equal(ref_non_stacked, op_non_stacked)) {
            throw std::runtime_error("inconsistent non-stacking legs in grid");
        }
        for (auto idx : stack_indices) {
            if (!op->get_leg(idx).as_elementary_space()) {
                throw std::runtime_error("stacking legs must be ElementarySpace");
            }
        }
    }

    std::vector<int64> normalized_cod;
    std::vector<int64> normalized_dom;
    if (stack_indices.size() == 1) {
        auto selected = stack_indices[0];
        for (int64 i = 0; i < n_cod; ++i) {
            if (i != selected) {
                normalized_cod.push_back(i);
            }
        }
        std::vector<int64> original_domain;
        for (int64 i = n_legs - 1; i >= n_cod; --i) {
            original_domain.push_back(i);
        }
        if (is_codomain_leg(selected)) {
            normalized_cod.insert(normalized_cod.begin(), selected);
            normalized_dom = original_domain;
        } else {
            normalized_dom = original_domain;
            normalized_dom.erase(
              std::find(normalized_dom.begin(), normalized_dom.end(), selected));
            normalized_dom.push_back(selected);
        }
    } else {
        auto row_leg = stack_indices[0];
        auto col_leg = stack_indices[1];
        normalized_cod.push_back(row_leg);
        for (int64 i = 0; i < n_cod; ++i) {
            if (i != row_leg && i != col_leg) {
                normalized_cod.push_back(i);
            }
        }
        for (int64 i = n_legs - 1; i >= n_cod; --i) {
            if (i != row_leg && i != col_leg) {
                normalized_dom.push_back(i);
            }
        }
        normalized_dom.push_back(col_leg);
    }

    auto permute_for_grid = [&](TensorCPtr tensor,
                                std::vector<int64> const& cod,
                                std::vector<int64> const& dom) -> TensorPtr {
        std::vector<LegRef> cod_refs;
        std::vector<LegRef> dom_refs;
        cod_refs.reserve(cod.size());
        dom_refs.reserve(dom.size());
        for (auto i : cod) {
            cod_refs.emplace_back(i);
        }
        for (auto i : dom) {
            dom_refs.emplace_back(i);
        }
        LevelsSpec levels;
        levels.reserve(static_cast<std::size_t>(tensor->num_legs));
        for (int64 i = 0; i < tensor->num_legs; ++i) {
            levels.emplace_back(i);
        }
        return permute_legs(std::move(tensor),
                            std::move(cod_refs),
                            std::move(dom_refs),
                            std::move(levels),
                            BendRight{ true });
    };

    std::vector<std::vector<TensorPtr>> normalized_grid = grid;
    for (auto& row : normalized_grid) {
        for (auto& op : row) {
            if (op) {
                op = permute_for_grid(op, normalized_cod, normalized_dom);
            }
        }
    }

    int64 row_leg_idx = 0;
    int64 col_leg_idx = static_cast<int64>(normalized_cod.size());
    auto row_space_for = [&](int64 i, int64 j) -> ElementarySpace::Ptr {
        auto const& op = normalized_grid[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        return op ? op->get_leg_co_domain(row_leg_idx).as_elementary_space() : nullptr;
    };
    auto col_space_for = [&](int64 i, int64 j) -> ElementarySpace::Ptr {
        auto const& op = normalized_grid[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        return op ? op->get_leg_co_domain(col_leg_idx).as_elementary_space() : nullptr;
    };

    bool const is_one_dimensional = stack_indices.size() == 1;
    bool const selected_is_codomain = is_codomain_leg(stack_indices[0]);
    std::string dummy_label;
    if (is_one_dimensional) {
        auto contains_label = [&](std::string const& label) {
            return std::any_of(
              ops.begin(), ops.end(), [&](TensorCPtr const& op) { return op->labels.has(label); });
        };
        dummy_label = "__tensor_from_grid_dummy__";
        for (int64 suffix = 0; contains_label(dummy_label); ++suffix) {
            dummy_label = std::format("__tensor_from_grid_dummy_{}__", suffix);
        }
        for (auto& row : normalized_grid) {
            for (auto& op : row) {
                if (op) {
                    op =
                      selected_is_codomain
                        ? add_trivial_leg(op, std::nullopt, std::nullopt, int64{ -1 }, dummy_label)
                        : add_trivial_leg(op, std::nullopt, int64{ 0 }, std::nullopt, dummy_label);
                }
            }
        }
        if (selected_is_codomain) {
            normalized_dom.push_back(-1);
            std::vector<std::vector<TensorPtr>> column_grid;
            column_grid.reserve(normalized_grid.front().size());
            for (auto& op : normalized_grid.front()) {
                column_grid.push_back({ std::move(op) });
            }
            normalized_grid = std::move(column_grid);
            row_leg_idx = 0;
        } else {
            normalized_cod.insert(normalized_cod.begin(), -1);
            row_leg_idx = 0;
        }
        col_leg_idx = static_cast<int64>(normalized_cod.size());
    }

    auto n_stacked_rows = static_cast<int64>(normalized_grid.size());
    auto n_stacked_cols =
      n_stacked_rows > 0 ? static_cast<int64>(normalized_grid[0].size()) : int64{ 0 };
    for (auto const& row : normalized_grid) {
        if (static_cast<int64>(row.size()) != n_stacked_cols) {
            throw std::invalid_argument("grid rows must have equal length");
        }
    }

    auto labels_for_dimension = [&](std::size_t dim) -> std::optional<LegLabelList> {
        return stack_labels.empty() ? std::nullopt : stack_labels[dim];
    };
    auto summands_for_dimension =
      [&](std::size_t dim) -> std::optional<std::vector<ElementarySpace::Ptr>> {
        return stack_summands.empty() ? std::nullopt : stack_summands[dim];
    };

    std::optional<LegLabelList> row_labels;
    std::optional<LegLabelList> col_labels;
    std::optional<std::vector<ElementarySpace::Ptr>> row_summands;
    std::optional<std::vector<ElementarySpace::Ptr>> col_summands;
    if (is_one_dimensional) {
        if (selected_is_codomain) {
            row_labels = labels_for_dimension(0);
            row_summands = summands_for_dimension(0);
        } else {
            col_labels = labels_for_dimension(0);
            col_summands = summands_for_dimension(0);
        }
    } else {
        row_labels = labels_for_dimension(0);
        col_labels = labels_for_dimension(1);
        row_summands = summands_for_dimension(0);
        col_summands = summands_for_dimension(1);
    }
    if (row_labels && static_cast<int64>(row_labels->size()) != n_stacked_rows) {
        throw std::invalid_argument("stack_labels entry must match its grid dimension");
    }
    if (col_labels && static_cast<int64>(col_labels->size()) != n_stacked_cols) {
        throw std::invalid_argument("stack_labels entry must match its grid dimension");
    }
    if (row_summands && static_cast<int64>(row_summands->size()) != n_stacked_rows) {
        throw std::invalid_argument("stack_summands entry must match its grid dimension");
    }
    if (col_summands && static_cast<int64>(col_summands->size()) != n_stacked_cols) {
        throw std::invalid_argument("stack_summands entry must match its grid dimension");
    }

    auto infer_summands = [&](bool rows_axis,
                              std::optional<std::vector<ElementarySpace::Ptr>> const& specified,
                              int64 count) {
        std::vector<ElementarySpace::Ptr> spaces =
          specified.value_or(std::vector<ElementarySpace::Ptr>(static_cast<std::size_t>(count)));
        for (int64 i = 0; i < count; ++i) {
            auto& current = spaces[static_cast<std::size_t>(i)];
            if (specified && !current) {
                throw std::invalid_argument("stack_summands must not contain null spaces");
            }
            for (int64 j = 0; j < (rows_axis ? n_stacked_cols : n_stacked_rows); ++j) {
                auto cell_space = rows_axis ? row_space_for(i, j) : col_space_for(j, i);
                if (!cell_space) {
                    continue;
                }
                if (!current) {
                    current = cell_space;
                } else if (!current->equals_es(*cell_space)) {
                    throw std::invalid_argument(
                      "stack_summands must agree with every tensor along its grid slice");
                }
            }
            if (!current) {
                throw std::invalid_argument(
                  "Must specify stack_summands for each entirely empty grid slice.");
            }
        }
        return spaces;
    };
    auto row_spaces = infer_summands(true, row_summands, n_stacked_rows);
    auto col_spaces = infer_summands(false, col_summands, n_stacked_cols);

    auto make_stacked_space = [](std::vector<ElementarySpace::Ptr> const& spaces,
                                 std::optional<LegLabelList> const& summand_labels) {
        std::vector<ElementarySpace::Ptr> rest(spaces.begin() + 1, spaces.end());
        return spaces[0]->direct_sum(rest, summand_labels);
    };
    auto left_space = make_stacked_space(row_spaces, row_labels);
    auto right_space = make_stacked_space(col_spaces, col_labels);

    auto left_dss = std::dynamic_pointer_cast<DirectSumSpace>(left_space);
    auto right_dss = std::dynamic_pointer_cast<DirectSumSpace>(right_space);
    if (!left_dss || !right_dss) {
        throw std::runtime_error("tensor_from_grid: direct_sum did not return a DirectSumSpace");
    }
    // Slices per grid cell, not per flattened summand (cells may themselves be DirectSumSpaces).
    auto cell_mult_slices = [](DirectSumSpace const& dss,
                               std::vector<ElementarySpace::Ptr> const& cells) {
        std::vector<std::vector<int64>> out;
        out.reserve(static_cast<std::size_t>(dss.num_sectors));
        for (auto const& sector : dss.sector_decomposition) {
            std::vector<int64> slices{ 0 };
            for (auto const& cell : cells) {
                auto idx = cell->sector_decomposition_where(sector);
                slices.push_back(
                  slices.back() +
                  (idx.has_value() ? cell->multiplicities[static_cast<std::size_t>(*idx)] : 0));
            }
            out.push_back(std::move(slices));
        }
        return out;
    };
    auto left_mult_slices = cell_mult_slices(*left_dss, row_spaces);
    auto right_mult_slices = cell_mult_slices(*right_dss, col_spaces);

    std::vector<Leg::Ptr> cod_legs;
    TensorCPtr normalized_ref;
    std::vector<TensorCPtr> normalized_ops;
    for (auto const& row : normalized_grid) {
        for (auto const& op : row) {
            if (op) {
                if (!normalized_ref) {
                    normalized_ref = op;
                }
                normalized_ops.push_back(op);
            }
        }
    }
    if (!normalized_ref) {
        throw std::runtime_error("tensor_from_grid: normalized grid contains no tensors");
    }
    cod_legs.push_back(left_space);
    auto ref_cod_rest = factors_slice(
      normalized_ref->codomain, 1, static_cast<int64>(normalized_ref->codomain->factors.size()));
    cod_legs.insert(cod_legs.end(), ref_cod_rest.begin(), ref_cod_rest.end());
    std::vector<Leg::Ptr> dom_legs = factors_slice(normalized_ref->domain, 0, -1);
    dom_legs.push_back(right_space);

    auto codomain = std::make_shared<TensorProduct>(std::move(cod_legs));
    auto domain = std::make_shared<TensorProduct>(std::move(dom_legs));

    std::vector<std::vector<TensorCPtr>> cgrid;
    cgrid.reserve(normalized_grid.size());
    for (auto const& row : normalized_grid) {
        cgrid.emplace_back(row.begin(), row.end());
    }

    auto data = backend->from_grid(std::move(cgrid),
                                   codomain,
                                   domain,
                                   std::move(left_mult_slices),
                                   std::move(right_mult_slices),
                                   dtype,
                                   device);
    LegLabelList labs;
    auto const& normalized_labels = normalized_ref->labels.as_list();
    labs.reserve(normalized_labels.size());
    for (std::size_t i = 0; i < normalized_labels.size(); ++i) {
        auto const& candidate = normalized_labels[i];
        bool matches = true;
        for (auto const& op : normalized_ops) {
            auto const& op_labels = op->labels.as_list();
            if (i >= op_labels.size() || op_labels[i] != candidate) {
                matches = false;
                break;
            }
        }
        labs.push_back(matches ? candidate : std::string{});
    }
    auto result = std::make_shared<SymmetricTensor>(std::move(data),
                                                    std::move(codomain),
                                                    std::move(domain),
                                                    backend,
                                                    ref->symmetry,
                                                    std::move(labs));
    TensorPtr final_result = HiddenLegTensor::maybe_wrap(std::move(result));

    if (is_one_dimensional) {
        final_result = squeeze_legs(final_result, std::vector<LegRef>{ dummy_label });
        normalized_dom.erase(std::remove(normalized_dom.begin(), normalized_dom.end(), -1),
                             normalized_dom.end());
        normalized_cod.erase(std::remove(normalized_cod.begin(), normalized_cod.end(), -1),
                             normalized_cod.end());
    }

    int64 const result_num_cod = final_result->num_codomain_legs();
    int64 const result_num_legs = final_result->num_legs;
    std::vector<int64> result_cod_ids = normalized_cod;
    std::vector<int64> result_dom_ids = normalized_dom;
    auto result_leg_indices = [&](std::vector<int64> const& cod_ids,
                                  std::vector<int64> const& dom_ids) {
        std::vector<int64> by_original(static_cast<std::size_t>(n_legs));
        for (std::size_t i = 0; i < cod_ids.size(); ++i) {
            by_original[static_cast<std::size_t>(cod_ids[i])] = static_cast<int64>(i);
        }
        for (std::size_t i = 0; i < dom_ids.size(); ++i) {
            by_original[static_cast<std::size_t>(dom_ids[i])] =
              result_num_cod + static_cast<int64>(dom_ids.size() - 1 - i);
        }
        return by_original;
    };
    auto output_indices = result_leg_indices(result_cod_ids, result_dom_ids);
    std::vector<LegRef> original_cod_refs;
    std::vector<LegRef> original_dom_refs;
    original_cod_refs.reserve(static_cast<std::size_t>(n_cod));
    original_dom_refs.reserve(static_cast<std::size_t>(n_dom));
    for (int64 i = 0; i < n_cod; ++i) {
        original_cod_refs.emplace_back(output_indices[static_cast<std::size_t>(i)]);
    }
    for (int64 i = n_legs - 1; i >= n_cod; --i) {
        original_dom_refs.emplace_back(output_indices[static_cast<std::size_t>(i)]);
    }
    LevelsSpec restore_levels;
    restore_levels.resize(static_cast<std::size_t>(result_num_legs));
    for (int64 i = 0; i < n_legs; ++i) {
        restore_levels[static_cast<std::size_t>(output_indices[static_cast<std::size_t>(i)])] = i;
    }
    return permute_legs(final_result,
                        std::move(original_cod_refs),
                        std::move(original_dom_refs),
                        std::move(restore_levels),
                        BendRight{ true });
}

namespace {

[[nodiscard]] int64
normalize_summand_index(DirectSumSpace const& space, int64 i)
{
    auto const n = static_cast<int64>(space.spaces.size());
    if (i < 0) {
        i += n;
    }
    if (i < 0 || i >= n) {
        throw std::invalid_argument(
          std::format("summand index {} out of range for DirectSumSpace with {} summands", i, n));
    }
    return i;
}

[[nodiscard]] std::vector<int64>
normalize_summand_indices(DirectSumSpace const& space,
                          std::vector<DirectSumSpace::SummandRef> indices,
                          bool require_nonempty)
{
    if (require_nonempty && indices.empty()) {
        throw std::invalid_argument("keep_indices requires at least one summand reference");
    }
    std::vector<int64> normalized;
    normalized.reserve(indices.size());
    for (auto& index : indices) {
        normalized.push_back(space.get_summand_idx(std::move(index)));
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    return normalized;
}

[[nodiscard]] std::vector<int64>
select_summand_indices(DirectSumSpace const& space,
                       std::optional<std::vector<DirectSumSpace::SummandRef>> keep_indices,
                       std::optional<std::vector<DirectSumSpace::SummandRef>> discard_indices)
{
    if (keep_indices.has_value() == discard_indices.has_value()) {
        throw std::invalid_argument(
          "exactly one of keep_indices or discard_indices must be provided");
    }
    if (keep_indices) {
        return normalize_summand_indices(space, std::move(*keep_indices), true);
    }

    auto discarded = normalize_summand_indices(space, std::move(*discard_indices), false);
    std::vector<int64> kept;
    kept.reserve(space.spaces.size() - discarded.size());
    for (int64 i = 0; i < static_cast<int64>(space.spaces.size()); ++i) {
        if (!std::binary_search(discarded.begin(), discarded.end(), i)) {
            kept.push_back(i);
        }
    }
    return kept;
}

[[nodiscard]] TensorBackend::Ptr
resolve_backend_for_space(TensorBackend::Ptr backend, Space::Ptr const& space)
{
    if (!backend) {
        return get_backend(space->symmetry);
    }
    return backend;
}

} // namespace

MaskPtr
DirectSumSpace::projection_onto_summands(std::optional<std::vector<SummandRef>> keep_indices,
                                         std::optional<std::vector<SummandRef>> discard_indices,
                                         std::shared_ptr<TensorBackend> backend,
                                         std::optional<LegLabelList> labels,
                                         std::optional<std::string> device) const
{
    auto kept = select_summand_indices(*this, std::move(keep_indices), std::move(discard_indices));
    backend = resolve_backend_for_space(std::move(backend), shared_es());

    auto const slices = mult_slices();
    auto space_cap = shared_dss();
    auto kept_cap = kept;
    auto bb = backend->block_backend;

    SectorBlockFactoryFn func = [space_cap, slices, kept_cap, bb, device](
                                  std::vector<int64> const& shape, Sector const& coupled) {
        auto sector_idx = space_cap->sector_decomposition_where(coupled);
        if (!sector_idx.has_value()) {
            throw std::runtime_error(
              "DirectSumSpace::projection_onto_summands: sector missing from DirectSumSpace");
        }
        auto const& slc = slices[static_cast<std::size_t>(*sector_idx)];
        if (shape.empty() || shape[0] != slc.back()) {
            throw std::runtime_error(
              "DirectSumSpace::projection_onto_summands: unexpected diagonal block shape");
        }
        std::vector<bool> mask(static_cast<std::size_t>(shape[0]), false);
        for (auto i_cap : kept_cap) {
            int64 const start = slc[static_cast<std::size_t>(i_cap)];
            int64 const stop = slc[static_cast<std::size_t>(i_cap) + 1];
            for (int64 i = start; i < stop; ++i) {
                mask[static_cast<std::size_t>(i)] = true;
            }
        }
        return bb->block_from_bool_vector(mask, device);
    };

    auto diag = DiagonalTensor::from_sector_block_func(
      std::move(func), shared_es(), backend, labels, Dtype::Bool, device);
    return Mask::from_DiagonalTensor(diag);
}

MaskPtr
DirectSumSpace::inclusion_of_summands(std::optional<std::vector<SummandRef>> keep_indices,
                                      std::optional<std::vector<SummandRef>> discard_indices,
                                      std::shared_ptr<TensorBackend> backend,
                                      std::optional<LegLabelList> labels,
                                      std::optional<std::string> device) const
{
    auto proj = projection_onto_summands(std::move(keep_indices),
                                         std::move(discard_indices),
                                         std::move(backend),
                                         std::move(labels),
                                         std::move(device));
    auto incl = std::dynamic_pointer_cast<Mask>(proj->dagger());
    if (!incl) {
        throw std::runtime_error("Mask::dagger did not return a Mask");
    }
    return incl;
}

MaskPtr
DirectSumSpace::projection_onto_summand(SummandRef which,
                                        std::shared_ptr<TensorBackend> backend,
                                        std::optional<LegLabelList> labels,
                                        std::optional<std::string> device) const
{
    return projection_onto_summands(std::vector<SummandRef>{ std::move(which) },
                                    std::nullopt,
                                    std::move(backend),
                                    std::move(labels),
                                    std::move(device));
}

MaskPtr
DirectSumSpace::inclusion_of_summand(SummandRef which,
                                     std::shared_ptr<TensorBackend> backend,
                                     std::optional<LegLabelList> labels,
                                     std::optional<std::string> device) const
{
    return inclusion_of_summands(std::vector<SummandRef>{ std::move(which) },
                                 std::nullopt,
                                 std::move(backend),
                                 std::move(labels),
                                 std::move(device));
}

SymmetricTensorPtr
DirectSumSpace::unit_vector_of_summand(int64 i,
                                       std::shared_ptr<TensorBackend> backend,
                                       std::optional<LegLabelList> labels,
                                       std::optional<Dtype> dtype,
                                       std::optional<std::string> device) const
{
    i = normalize_summand_index(*this, i);
    auto const& summand = spaces[static_cast<std::size_t>(i)];
    // Must be the one-dimensional trivial sector.
    if (summand->num_sectors != 1 || summand->multiplicities[0] != 1 ||
        !(summand->defining_sectors[0] == Space::symmetry->trivial_sector)) {
        throw std::invalid_argument(
          "DirectSumSpace::unit_vector_of_summand requires the summand to be the "
          "1-dimensional trivial sector");
    }

    auto incl = inclusion_of_summand(i, std::move(backend), std::nullopt, device);
    Dtype out_dtype = dtype.value_or(Dtype::Complex128);
    auto tens = incl->as_SymmetricTensor(/*guarantee_copy=*/false, std::nullopt, out_dtype);
    // Inclusion is trivial → fused; squeeze the trivial domain to get a pure vector on `this`.
    auto squeezed = squeeze_legs(tens, std::vector<LegRef>{ LegRef{ int64{ -1 } } });
    auto out = std::dynamic_pointer_cast<SymmetricTensor>(squeezed);
    if (!out) {
        throw std::runtime_error(
          "DirectSumSpace::unit_vector_of_summand: squeeze_legs did not return SymmetricTensor");
    }
    if (labels.has_value()) {
        if (labels->size() != 1) {
            throw std::invalid_argument(
              "DirectSumSpace::unit_vector_of_summand labels must have length 1 "
              "(the fused DirectSumSpace leg)");
        }
        out->labels.set(*labels);
    }
    return out;
}

TensorPtr
tensor_grid_cell(TensorCPtr tensor,
                 DirectSumSpace::SummandRef row,
                 DirectSumSpace::SummandRef col,
                 LegRef row_leg,
                 LegRef col_leg)
{
    if (!tensor) {
        throw std::invalid_argument("tensor_grid_cell: tensor must be non-null");
    }

    // Use get_leg (not get_leg_co_domain): Mask.large_leg must match the
    // codomain-side view of the stacking leg so apply_mask works for domain legs.
    auto row_space = tensor->get_leg(row_leg);
    auto col_space = tensor->get_leg(col_leg);
    auto row_dss = std::dynamic_pointer_cast<DirectSumSpace>(row_space.as_elementary_space());
    auto col_dss = std::dynamic_pointer_cast<DirectSumSpace>(col_space.as_elementary_space());

    TensorPtr cell = std::const_pointer_cast<Tensor>(tensor);
    if (row_dss) {
        auto proj =
          row_dss->projection_onto_summand(row_dss->get_summand_idx(row), tensor->backend);
        cell = apply_mask(cell, proj, row_leg);
    } else if (std::holds_alternative<std::string>(row)) {
        throw std::invalid_argument(
          "tensor_grid_cell: row stacking leg is not a DirectSumSpace; labels are not available");
    } else if (auto const i = std::get<int64>(row); i != 0 && i != -1) {
        throw std::invalid_argument(
          "tensor_grid_cell: row stacking leg is not a DirectSumSpace; only row 0/-1 is valid");
    }
    if (col_dss) {
        auto proj = col_dss->projection_onto_summand(col_dss->get_summand_idx(col), cell->backend);
        cell = apply_mask(cell, proj, col_leg);
    } else if (std::holds_alternative<std::string>(col)) {
        throw std::invalid_argument("tensor_grid_cell: column stacking leg is not a "
                                    "DirectSumSpace; labels are not available");
    } else if (auto const i = std::get<int64>(col); i != 0 && i != -1) {
        throw std::invalid_argument(
          "tensor_grid_cell: column stacking leg is not a DirectSumSpace; only col 0/-1 is valid");
    }
    return cell;
}

TensorPtr
tensor_subgrid(TensorCPtr tensor,
               std::vector<DirectSumSpace::SummandRef> rows,
               std::vector<DirectSumSpace::SummandRef> cols,
               LegRef row_leg,
               LegRef col_leg)
{
    if (!tensor) {
        throw std::invalid_argument("tensor_subgrid: tensor must be non-null");
    }
    if (rows.empty() || cols.empty()) {
        throw std::invalid_argument("tensor_subgrid: rows and cols must be non-empty");
    }

    // Summand labels of the selected rows / columns, if the stacking legs are direct sums.
    auto select_labels =
      [&](LegRef leg,
          std::vector<DirectSumSpace::SummandRef> const& refs) -> std::optional<LegLabelList> {
        auto dss =
          std::dynamic_pointer_cast<DirectSumSpace>(tensor->get_leg(leg).as_elementary_space());
        if (!dss) {
            return std::nullopt;
        }
        LegLabelList out;
        out.reserve(refs.size());
        for (auto const& ref : refs) {
            out.push_back(
              dss->summand_labels.at(static_cast<std::size_t>(dss->get_summand_idx(ref))));
        }
        return out;
    };

    std::vector<std::vector<TensorPtr>> grid;
    grid.reserve(rows.size());
    for (auto const& r : rows) {
        auto& grid_row = grid.emplace_back();
        grid_row.reserve(cols.size());
        for (auto const& c : cols) {
            grid_row.push_back(tensor_grid_cell(tensor, r, c, row_leg, col_leg));
        }
    }
    return tensor_from_grid(std::move(grid),
                            std::vector<LegRef>{ row_leg, col_leg },
                            std::nullopt,
                            std::vector<std::optional<LegLabelList>>{
                              select_labels(row_leg, rows), select_labels(col_leg, cols) });
}

TensorPtr
grid_project(TensorCPtr tensor,
             std::vector<LegRef> legs,
             std::vector<DirectSumSpace::SummandRef> cells,
             bool squeeze)
{
    if (!tensor) {
        throw std::invalid_argument("grid_project: tensor must be non-null");
    }
    if (legs.size() != cells.size()) {
        throw std::invalid_argument("grid_project: legs and cells must have equal length");
    }

    TensorPtr result = std::const_pointer_cast<Tensor>(tensor);
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto dss = std::dynamic_pointer_cast<DirectSumSpace>(
          result->get_leg(legs[i]).as_elementary_space());
        if (!dss) {
            throw std::invalid_argument("grid_project: selected leg is not a DirectSumSpace");
        }
        auto const summand = dss->get_summand_idx(cells[i]);
        auto projector = dss->projection_onto_summand(summand, result->backend);
        result = apply_mask(result, projector, legs[i]);
    }
    if (squeeze && !legs.empty()) {
        result = squeeze_legs(result, legs);
    }
    return result;
}

} // namespace lasyten
