#include <lasyten/tensors/helpers.h>

#include <lasyten/backends/no_symmetry.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tools.h>

#include <cassert>
#include <format>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <variant>
#include <vector>

namespace lasyten {

namespace {

LegLabelList
apply_relabel(LegLabelList labels,
              std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (!relabel.has_value()) {
        return labels;
    }
    for (auto& lab : labels) {
        if (lab.empty()) {
            continue;
        }
        auto it = relabel->find(lab);
        if (it != relabel->end()) {
            lab = it->second;
        }
    }
    return labels;
}

std::unordered_set<std::string>
duplicate_label_entries(LegLabelList const& labels)
{
    std::unordered_set<std::string> seen;
    std::unordered_set<std::string> dups;
    for (auto const& lab : labels) {
        if (lab.empty()) {
            continue;
        }
        if (!seen.insert(lab).second) {
            dups.insert(lab);
        }
    }
    return dups;
}

int64
prod_i64(std::vector<int64> const& vals)
{
    return std::accumulate(vals.begin(), vals.end(), int64{ 1 }, std::multiplies<int64>{});
}

/// Odometer-style cartesian product over ``ranges[i] == [0, ranges[i])``.
class SectorIndexProduct
{
  public:
    explicit SectorIndexProduct(std::vector<int64> ranges)
      : ranges_(std::move(ranges))
      , cur_(ranges_.size(), 0)
      , done_(false)
    {
        for (auto r : ranges_) {
            if (r <= 0) {
                done_ = true;
                return;
            }
        }
        if (ranges_.empty()) {
            // one empty combination
            done_ = false;
        }
    }

    [[nodiscard]] bool done() const { return done_; }

    [[nodiscard]] std::vector<int64> const& current() const { return cur_; }

    void next()
    {
        if (done_ || ranges_.empty()) {
            done_ = true;
            return;
        }
        for (std::size_t i = ranges_.size(); i-- > 0;) {
            ++cur_[i];
            if (cur_[i] < ranges_[i]) {
                return;
            }
            cur_[i] = 0;
        }
        done_ = true;
    }

  private:
    std::vector<int64> ranges_;
    std::vector<int64> cur_;
    bool done_;
};

std::vector<Space::Ptr>
spaces_of_product(TensorProduct::Ptr const& tp)
{
    std::vector<Space::Ptr> out;
    out.reserve(static_cast<std::size_t>(tp->num_factors));
    for (auto const& f : tp->factors) {
        out.push_back(as_space(f));
    }
    return out;
}

std::vector<int64>
num_sectors_per_leg(std::vector<Space::Ptr> const& legs)
{
    std::vector<int64> out;
    out.reserve(legs.size());
    for (auto const& leg : legs) {
        out.push_back(leg->num_sectors);
    }
    return out;
}

std::vector<Sector>
sectors_at(std::vector<Space::Ptr> const& legs, std::vector<int64> const& idcs)
{
    std::vector<Sector> out;
    out.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        out.push_back(legs[i]->sector_decomposition[static_cast<std::size_t>(idcs[i])]);
    }
    return out;
}

std::vector<int64>
mults_at(std::vector<Space::Ptr> const& legs, std::vector<int64> const& idcs)
{
    std::vector<int64> out;
    out.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        out.push_back(legs[i]->multiplicities[static_cast<std::size_t>(idcs[i])]);
    }
    return out;
}

using BlockIndex = BlockBackend::BlockIndex;
using AxisSlice = BlockBackend::AxisSlice;

void
assign_block_slice(BlockBackend::BlockPtr& dest,
                   std::initializer_list<BlockIndex> key,
                   BlockBackend::BlockPtr const& value)
{
    dest->set_item(std::span<const BlockIndex>(key.begin(), key.size()), *value);
}

BlockBackend::BlockPtr
get_block_slice(BlockBackend::BlockPtr const& src, std::initializer_list<BlockIndex> key)
{
    return (*src)[std::span<const BlockIndex>(key.begin(), key.size())];
}

template<class T>
void
check_compatible_impl(std::vector<std::shared_ptr<T>> const& legs1,
                      std::vector<std::shared_ptr<T>> const& legs2,
                      bool expect_equal)
{
    if (legs1.size() != legs2.size()) {
        throw std::invalid_argument("Different number of legs");
    }
    for (std::size_t i = 0; i < legs1.size(); ++i) {
        auto const& l1 = legs1[i];
        auto const& l2 = legs2[i];
        if (!l1->symmetry->is_equivalent_to(*l2->symmetry)) {
            throw std::invalid_argument("Different symmetries");
        }
        auto rhs = expect_equal ? l2 : l2->dual();
        // Explicit ``operator==``: C++20 reversed candidates make ``*a == *b`` ambiguous
        // for TensorProduct / Space.
        if (!l1->operator==(*rhs)) {
            throw std::invalid_argument("Incompatible legs.");
        }
    }
}

} // namespace

void
_check_compatible_legs(std::vector<Leg::Ptr> const& legs1,
                       std::vector<Leg::Ptr> const& legs2,
                       bool expect_equal)
{
    check_compatible_impl(legs1, legs2, expect_equal);
}

void
_check_compatible_legs(std::vector<Space::Ptr> const& legs1,
                       std::vector<Space::Ptr> const& legs2,
                       bool expect_equal)
{
    check_compatible_impl(legs1, legs2, expect_equal);
}

TensorPtr
_compose_with_Mask(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    // --- hints from Python _compose_with_Mask ---
    // deal with other tensor types
    // ---
    auto [in_domain, co_domain_idx, parsed_leg_idx] = tensor->_parse_leg_idx(leg_idx);
    leg_idx = parsed_leg_idx;

    if (in_domain) {
        _check_compatible_legs(std::vector<Leg::Ptr>{ (*tensor->domain)[co_domain_idx] },
                               std::vector<Leg::Ptr>{ (*mask->codomain)[0] });
    } else {
        _check_compatible_legs(std::vector<Leg::Ptr>{ (*tensor->codomain)[co_domain_idx] },
                               std::vector<Leg::Ptr>{ (*mask->domain)[0] });
    }

    if (auto charged = std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        auto invariant_part = _compose_with_Mask(charged->invariant_part, mask, leg_idx);
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(invariant_part);
        if (!inv_sym) {
            throw std::runtime_error("_compose_with_Mask expected SymmetricTensor invariant_part");
        }
        return std::make_shared<ChargedTensor>(std::move(inv_sym), charged->charged_state);
    }
    if (std::dynamic_pointer_cast<Mask const>(tensor)) {
        throw NotImplemented("tensors._compose_with_Mask not implemented for Mask");
    }
    // hidden legs are restored below, so the conversion is lossless for a HiddenLegTensor
    auto tens = std::dynamic_pointer_cast<HiddenLegTensor const>(tensor)
                  ? std::const_pointer_cast<Tensor>(tensor)->as_SymmetricTensor()
                  : std::const_pointer_cast<Tensor>(tensor)->as_SymmetricTensor(
                      false, std::string("Converting to SymmetricTensor."));

    auto backend = get_same_backend(std::vector<TensorCPtr>{ tens, mask });
    std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> contracted;
    if (in_domain == mask->is_projection) {
        contracted = backend->mask_contract_small_leg(tens, mask, leg_idx);
    } else {
        contracted = backend->mask_contract_large_leg(tens, mask, leg_idx);
    }
    auto& [data, codomain, domain] = contracted;
    // use the labels of `tensor`, which keep the ``!`` of hidden legs -> HiddenLegTensor result
    return HiddenLegTensor::maybe_wrap(
      std::make_shared<SymmetricTensor>(std::move(data),
                                        std::move(codomain),
                                        std::move(domain),
                                        backend,
                                        tens->symmetry,
                                        tensor->labels.as_list()));
}

std::variant<SymmetricTensorPtr, BlockBackend::Scalar>
_compose_SymmetricTensors(SymmetricTensorCPtr tensor1,
                          SymmetricTensorCPtr tensor2,
                          std::optional<std::map<std::string, std::string>> relabel1,
                          std::optional<std::map<std::string, std::string>> relabel2)
{
    // --- hints from Python _compose_SymmetricTensors ---
    // no remaining open legs
    // drop duplicate labels
    // ---
    auto backend = get_same_backend(std::vector<TensorCPtr>{ tensor1, tensor2 });
    if (tensor1->num_codomain_legs() == 0 && tensor2->num_domain_legs() == 0) {
        return backend->inner(tensor1, tensor2, /*do_dagger=*/false);
    }

    LegLabelList labels_codomain = apply_relabel(tensor1->codomain_labels(), relabel1);
    LegLabelList labels_domain = apply_relabel(tensor2->domain_labels(), relabel2);

    LegLabelList labels = labels_codomain;
    for (auto it = labels_domain.rbegin(); it != labels_domain.rend(); ++it) {
        labels.push_back(*it);
    }

    auto duplicates = duplicate_label_entries(labels);
    int64 dup_counter = 0;
    for (auto& lab : labels) {
        if (!lab.empty() && duplicates.contains(lab)) {
            lab = std::format("?{}", dup_counter);
            ++dup_counter;
        }
    }

    auto data = backend->compose(tensor1, tensor2);
    return std::make_shared<SymmetricTensor>(
      std::move(data), tensor1->codomain, tensor2->domain, backend, tensor1->symmetry, labels);
}

FusionTreeData::Ptr
_convert_abelian_to_FT(TensorCPtr tensor,
                       FusionTreeBackend::Ptr backend,
                       Dtype dtype,
                       std::string device)
{
    auto const& codomain = tensor->codomain;
    auto const& domain = tensor->domain;
    auto const& symmetry = tensor->symmetry;
    auto ab_data = AbelianBackend::data_from_tensor(tensor);
    auto old_bb = tensor->backend->block_backend;

    int64 num_codomain_legs = tensor->num_codomain_legs();
    int64 num_domain_legs = tensor->num_domain_legs();
    int64 num_legs = tensor->num_legs;

    // Start with all allowed blocks initialized with zeros
    // OPTIMIZE create the blocks on-demand instead?
    auto res = FusionTreeBackend::unwrap(
      backend->zero_data(codomain, domain, dtype, device, /*all_blocks=*/true));
    std::vector<bool> blocks_touched(res->blocks.size(), false);

    auto cod_legs = spaces_of_product(codomain);
    auto dom_legs = spaces_of_product(domain);

    if (num_codomain_legs == 0) {
        int64 i2 = 0;
        for (SectorIndexProduct prod(num_sectors_per_leg(dom_legs)); !prod.done(); prod.next()) {
            auto const& dom_sector_idcs = prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            if (c != symmetry->trivial_sector) {
                continue;
            }
            int64 tree_block_width = prod_i64(b_mults);
            std::vector<int64> ab_block_inds(dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
            auto ab_i = ab_data->get_block_num(BlockInds::from_row(ab_block_inds));
            if (!ab_i.has_value()) {
                // block is missing (zero) -> nothing to do
                i2 += tree_block_width;
                continue;
            }
            auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
            std::vector<int64> all_axes(static_cast<std::size_t>(num_legs));
            std::iota(all_axes.begin(), all_axes.end(), int64{ 0 });
            auto tree_block =
              old_bb->combine_legs(ab_block, std::vector<std::vector<int64>>{ all_axes }, false);
            tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
            assign_block_slice(res->blocks[0],
                               { int64{ 0 }, AxisSlice{ i2, i2 + tree_block_width, {} } },
                               tree_block);
            blocks_touched[0] = true;
            i2 += tree_block_width;
        }
    } else if (num_domain_legs == 0) {
        int64 i1 = 0;
        for (SectorIndexProduct prod(num_sectors_per_leg(cod_legs)); !prod.done(); prod.next()) {
            auto const& cod_sector_idcs = prod.current();
            auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
            auto a_mults = mults_at(cod_legs, cod_sector_idcs);
            Sector c = symmetry->multiple_fusion(a_sectors);
            if (c != symmetry->trivial_sector) {
                continue;
            }
            int64 tree_block_height = prod_i64(a_mults);
            auto ab_i = ab_data->get_block_num(BlockInds::from_row(cod_sector_idcs));
            if (!ab_i.has_value()) {
                // block is missing (zero) -> nothing to do
                i1 += tree_block_height;
                continue;
            }
            auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
            std::vector<int64> all_axes(static_cast<std::size_t>(num_legs));
            std::iota(all_axes.begin(), all_axes.end(), int64{ 0 });
            auto tree_block =
              old_bb->combine_legs(ab_block, std::vector<std::vector<int64>>{ all_axes }, true);
            tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
            assign_block_slice(res->blocks[0],
                               { AxisSlice{ i1, i1 + tree_block_height, {} }, int64{ 0 } },
                               tree_block);
            blocks_touched[0] = true;
            i1 += tree_block_height;
        }
    } else {
        int64 i1 = 0;
        std::vector<int64> i2_per_coupled(res->blocks.size(), 0);
        std::vector<std::vector<int64>> combine{
            [&] {
                std::vector<int64> v(static_cast<std::size_t>(num_codomain_legs));
                std::iota(v.begin(), v.end(), int64{ 0 });
                return v;
            }(),
            [&] {
                std::vector<int64> v(static_cast<std::size_t>(num_domain_legs));
                std::iota(v.begin(), v.end(), num_codomain_legs);
                return v;
            }(),
        };
        std::vector<bool> cstyles{ true, false };

        for (SectorIndexProduct dom_prod(num_sectors_per_leg(dom_legs)); !dom_prod.done();
             dom_prod.next()) {
            auto const& dom_sector_idcs = dom_prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            auto ft_bi = res->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // this can happen if c does not appear in the codomain at all -> no
                          // block
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];

            for (SectorIndexProduct cod_prod(num_sectors_per_leg(cod_legs)); !cod_prod.done();
                 cod_prod.next()) {
                auto const& cod_sector_idcs = cod_prod.current();
                auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
                auto a_mults = mults_at(cod_legs, cod_sector_idcs);
                Sector c2 = symmetry->multiple_fusion(a_sectors);
                int64 tree_block_height = prod_i64(a_mults);
                if (c2 != c) {
                    continue; // sector combination violates fusion rules -> no contributions
                }

                std::vector<int64> ab_block_inds = cod_sector_idcs;
                ab_block_inds.insert(
                  ab_block_inds.end(), dom_sector_idcs.rbegin(), dom_sector_idcs.rend());

                // OPTIMIZE use that the data.block_inds are lexsorted for this lookup (also above)
                auto ab_i = ab_data->get_block_num(BlockInds::from_row(ab_block_inds));
                if (!ab_i.has_value()) {
                    // block is missing (zero) -> nothing to do
                    i1 += tree_block_height;
                    continue;
                }

                auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
                // cstyle combine in the codomain, Fstyle in the domain
                auto tree_block = old_bb->combine_legs(ab_block, combine, cstyles);
                tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
                assign_block_slice(res->blocks[static_cast<std::size_t>(*ft_bi)],
                                   { AxisSlice{ i1, i1 + tree_block_height, {} },
                                     AxisSlice{ i2, i2 + tree_block_width, {} } },
                                   tree_block);
                blocks_touched[static_cast<std::size_t>(*ft_bi)] = true;

                i1 += tree_block_height; // move down by one tree-block
            }

            // reset to the top
            i1 = 0;
            // move to the right by one tree-block, for the next time we visit this block
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    }

    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t n = 0; n < res->blocks.size(); ++n) {
        if (blocks_touched[n]) {
            blocks.push_back(res->blocks[n]);
        }
    }
    BlockInds block_inds = res->block_inds.take_mask(blocks_touched);
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), /*is_sorted=*/true);
}

AbelianBackendData::Ptr
_convert_FT_to_abelian(TensorCPtr tensor,
                       AbelianBackend::Ptr backend,
                       Dtype dtype,
                       std::string device)
{
    auto const& domain = tensor->domain;
    auto const& codomain = tensor->codomain;
    auto const& symmetry = tensor->symmetry;
    auto ft_data = FusionTreeBackend::data_from_tensor(tensor);
    auto old_bb = tensor->backend->block_backend;

    int64 num_codomain_legs = tensor->num_codomain_legs();
    int64 num_domain_legs = tensor->num_domain_legs();
    int64 num_legs = tensor->num_legs;

    auto cod_legs = spaces_of_product(codomain);
    auto dom_legs = spaces_of_product(domain);

    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_block_inds;

    if (num_codomain_legs == 0) {
        std::vector<int64> i2_per_coupled(ft_data->blocks.size(), 0);
        for (SectorIndexProduct prod(num_sectors_per_leg(dom_legs)); !prod.done(); prod.next()) {
            auto const& dom_sector_idcs = prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            if (c != symmetry->trivial_sector) {
                continue; // fusion rule violated
            }
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block
                          // either
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];
            auto tree_block =
              get_block_slice(ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
                              { int64{ 0 }, AxisSlice{ i2, i2 + tree_block_width, {} } });
            auto ab_block = old_bb->split_legs(tree_block, { 0 }, { b_mults }, false);
            // convert to new block_backend
            ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
            res_blocks.push_back(ab_block);
            res_block_inds.emplace_back(dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    } else if (num_domain_legs == 0) {
        std::vector<int64> i1_per_coupled(ft_data->blocks.size(), 0);
        for (SectorIndexProduct prod(num_sectors_per_leg(cod_legs)); !prod.done(); prod.next()) {
            auto const& cod_sector_idcs = prod.current();
            auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
            auto a_mults = mults_at(cod_legs, cod_sector_idcs);
            Sector c = symmetry->multiple_fusion(a_sectors);
            if (c != symmetry->trivial_sector) {
                continue; // fusion rule violated
            }
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block
                          // either
            }
            int64 tree_block_height = prod_i64(a_mults);
            int64 i1 = i1_per_coupled[static_cast<std::size_t>(*ft_bi)];
            auto tree_block =
              get_block_slice(ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
                              { AxisSlice{ i1, i1 + tree_block_height, {} }, int64{ 0 } });
            auto ab_block = old_bb->split_legs(tree_block, { 0 }, { a_mults }, true);
            // convert to new block_backend
            ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
            res_blocks.push_back(ab_block);
            res_block_inds.push_back(cod_sector_idcs);
            i1_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_height;
        }
    } else {
        std::vector<int64> i2_per_coupled(ft_data->blocks.size(), 0);
        int64 i1 = 0;
        for (SectorIndexProduct dom_prod(num_sectors_per_leg(dom_legs)); !dom_prod.done();
             dom_prod.next()) {
            auto const& dom_sector_idcs = dom_prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block
                          // either
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];
            for (SectorIndexProduct cod_prod(num_sectors_per_leg(cod_legs)); !cod_prod.done();
                 cod_prod.next()) {
                auto const& cod_sector_idcs = cod_prod.current();
                auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
                auto a_mults = mults_at(cod_legs, cod_sector_idcs);
                Sector c2 = symmetry->multiple_fusion(a_sectors);
                int64 tree_block_height = prod_i64(a_mults);
                if (c2 != c) {
                    continue; // sector combination violates fusion rules -> no contributions
                }
                auto tree_block =
                  get_block_slice(ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
                                  { AxisSlice{ i1, i1 + tree_block_height, {} },
                                    AxisSlice{ i2, i2 + tree_block_width, {} } });
                auto ab_block = old_bb->split_legs(
                  tree_block, { 0, 1 }, { a_mults, b_mults }, std::vector<bool>{ true, false });
                // convert to new block_backend
                ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
                res_blocks.push_back(ab_block);
                std::vector<int64> row = cod_sector_idcs;
                row.insert(row.end(), dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
                res_block_inds.push_back(std::move(row));
                i1 += tree_block_height; // move down by one tree-block
            }
            // reset to the top
            i1 = 0;
            // move to the right by one tree-block, for the next time we visit this block
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    }

    BlockInds block_inds = res_block_inds.empty()
                             ? BlockInds::zeros(0, static_cast<std::size_t>(num_legs))
                             : BlockInds::from_rows(res_block_inds);
    return std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(res_blocks), std::move(block_inds));
}

std::tuple<SymmetricTensorPtr, TensorProduct::Ptr, bool, bool>
_decomposition_prepare(TensorCPtr tensor, bool new_leg_dual)
{
    // --- hints from Python _decomposition_prepare ---
    // do not define decompositions for ChargedTensors.
    // ---
    if (tensor->num_codomain_legs() <= 0) {
        throw std::runtime_error("empty codomain");
    }
    if (tensor->num_domain_legs() <= 0) {
        throw std::runtime_error("empty domain");
    }

    if (std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        // do not define decompositions for ChargedTensors.
        throw NotImplemented("_decomposition_prepare for ChargedTensor");
    }
    auto tens = std::const_pointer_cast<Tensor>(tensor)->as_SymmetricTensor();

    auto new_leg = ElementarySpace::from_largest_common_subspace(
      std::vector<Space::Ptr>{ tens->codomain, tens->domain }, new_leg_dual);
    auto new_co_domain = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ new_leg });

    bool combine_codomain = false;
    bool combine_domain = false;
    auto backend = tens->backend;
    if (!backend->can_decompose_tensors()) {
        combine_codomain = tens->num_codomain_legs() > 1;
        combine_domain = tens->num_domain_legs() > 1;
        int64 n_cod = tens->num_codomain_legs();
        int64 n_legs = tens->num_legs;
        std::vector<LegRef> cod_idcs;
        std::vector<LegRef> dom_idcs;
        for (int64 i = 0; i < n_cod; ++i) {
            cod_idcs.emplace_back(i);
        }
        for (int64 i = n_cod; i < n_legs; ++i) {
            dom_idcs.emplace_back(i);
        }
        TensorPtr combined = tens;
        if (combine_codomain && combine_domain) {
            combined = combine_legs(tens, { std::move(cod_idcs), std::move(dom_idcs) });
        } else if (combine_codomain) {
            combined = combine_legs(tens, { std::move(cod_idcs) });
        } else if (combine_domain) {
            combined = combine_legs(tens, { std::move(dom_idcs) });
        }
        tens = std::dynamic_pointer_cast<SymmetricTensor>(combined);
    }
    return { tens, new_co_domain, combine_codomain, combine_domain };
}

std::pair<std::string, std::string>
_decomposition_labels(LegLabelList const& new_labels)
{
    if (new_labels.size() == 1) {
        std::string a = new_labels[0];
        return { a, _dual_leg_label(a) };
    }
    if (new_labels.size() == 2) {
        return { new_labels[0], new_labels[1] };
    }
    throw std::invalid_argument(std::format("Expected 1 or 2 labels. Got {}", new_labels.size()));
}

std::tuple<std::string, std::string, std::string, std::string>
_svd_new_labels(std::optional<LegLabelList> new_labels)
{
    if (!new_labels.has_value()) {
        return { "", "", "", "" };
    }
    LegLabelList const& labels = *new_labels;
    std::string a, b, c, d;
    if (labels.size() == 1) {
        a = c = labels[0];
        b = d = _dual_leg_label(labels[0]);
    } else if (labels.size() == 2) {
        a = c = labels[0];
        b = d = labels[1];
    } else if (labels.size() == 4) {
        a = labels[0];
        b = labels[1];
        c = labels[2];
        d = labels[3];
    } else {
        throw std::invalid_argument(
          std::format("Expected 1, 2 or 4 new_labels. Got {}", labels.size()));
    }
    if (!b.empty() && !c.empty() && b == c) {
        throw std::invalid_argument(
          "new_labels for SVD must not reuse the same label on both sides");
    }
    return { a, b, c, d };
}

} // namespace lasyten
