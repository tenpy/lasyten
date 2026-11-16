#!/usr/bin/env python3
"""Second pass: fix remaining py:: patterns in fusion_tree_backend.cpp."""
from __future__ import annotations

import re
from pathlib import Path

CPP = Path(__file__).resolve().parents[1] / "src/backends/fusion_tree_backend.cpp"
text = CPP.read_text()

# parse_leg_index + zero_scalar_of after namespace lasyten {
insert = r'''
namespace {

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

'''
if "parse_leg_index(ElementarySpace" not in text:
    text = text.replace("namespace lasyten {\n\nFusionTreeData::FusionTreeData", insert + "FusionTreeData::FusionTreeData")

helper_add = r'''
BlockBackend::Scalar
zero_scalar_of(BlockBackend& bb, Dtype dtype)
{
    return bb.as_scalar(complex128{ 0. }, dtype);
}

[[nodiscard]] std::vector<int64>
indices_from_basis_perm(Space const& space, AxisSlice const& slice, std::vector<int64> const& mask_ranked)
{
    if (!space.basis_perm) {
        return mask_ranked;
    }
    std::vector<int64> out;
    out.reserve(mask_ranked.size());
  for (std::size_t k = 0; k < mask_ranked.size(); ++k)
        out.push_back((*space.basis_perm)[static_cast<std::size_t>(slice.start) + k]);
    return rank_data(out);
}

'''
if "zero_scalar_of" not in text:
    text = text.replace("is_zero_scalar(BlockBackend::Scalar const& a)", helper_add + "is_zero_scalar(BlockBackend::Scalar const& a)")

text = text.replace(
    "    auto instructions_vec = instructions_from_python(instructions);\n\n    auto const key =\n      plan_key(instructions,",
    "    auto const key =\n      plan_key(instructions,",
)

text = text.replace("dtype::zero_scalar(", "zero_scalar_of(*block_backend, ")

# permute_legs rewrite
permute_old = text[text.find("FusionTreeBackend::permute_legs"):text.find("FusionTreeBackend::_get_forest_block_contribution")]
permute_new = '''FusionTreeBackend::permute_legs(TensorCPtr a,
                                std::vector<int64> codomain_idcs,
                                std::vector<int64> domain_idcs,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain,
                                bool mixes_codomain_domain,
                                std::vector<std::optional<int64>> levels,
                                std::vector<std::optional<bool>> bend_right)
{
    auto [new_codomain_idcs, new_domain_idcs, flat_levels, flat_bend_right] =
      flat_leg_permutation_inputs(a, codomain_idcs, domain_idcs, levels, bend_right);
    int64 const num_codomain_flat_legs = a->num_codomain_flat_legs();
    int64 const num_domain_flat_legs = a->num_domain_flat_legs();
    bool const has_symmetric_braid = a->symmetry->has_symmetric_braid();
    PermuteLegsInstructionEngine engine(num_codomain_flat_legs,
                                        num_domain_flat_legs,
                                        new_codomain_idcs,
                                        new_domain_idcs,
                                        flat_levels,
                                        flat_bend_right,
                                        has_symmetric_braid);
    auto instructions = engine.evaluate_instructions();
    return apply_instructions(a,
                              instructions,
                              new_codomain_idcs,
                              new_domain_idcs,
                              new_codomain,
                              new_domain,
                              mixes_codomain_domain);
}

'''
text = text.replace(permute_old, permute_new)

# diagonal_from_block
text = re.sub(
    r"TensorBackend::DataPtr\nFusionTreeBackend::diagonal_from_block\(BlockBackend::BlockPtr a,[\s\S]*?return wrap\(make_data\(dt, block_backend->get_device\(a\), std::move\(blocks\), block_inds\)\);\n\}",
    '''TensorBackend::DataPtr
FusionTreeBackend::diagonal_from_block(BlockBackend::BlockPtr a,
                                       TensorProduct::Ptr co_domain,
                                       float64 tol)
{
    auto leg = co_domain->factors[0];
    auto const& space = *as_space(leg);
    Dtype dt = block_backend->get_dtype(a);
    BlockInds block_inds =
      BlockInds::arange_diag(static_cast<std::size_t>(co_domain->num_sectors));
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t idx = 0; idx < co_domain->sector_decomposition.size(); ++idx) {
        Sector coupled = co_domain->sector_decomposition[idx];
        int64 dim_c = co_domain->symmetry->sector_dim(coupled);
        auto j = space.sector_decomposition_where(coupled);
        if (!j)
            throw std::runtime_error("diagonal_from_block: coupled sector not in leg");
        auto slc = leg_slice(space, *j);
        auto entries = block_backend->reshape(b_get(a, BlockIndexVec{ slc }),
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
}''',
    text,
    count=1,
)

text = text.replace(
    "    auto qdims = py::cast<std::vector<float64>>(py::cast(a->domain).attr(\"sector_qdims\"));",
    "    auto qdims = a->domain->sector_qdims;",
)

# diagonal_tensor_to_block simplified native port
text = re.sub(
    r"BlockBackend::BlockPtr\nFusionTreeBackend::diagonal_tensor_to_block\(DiagonalTensorCPtr a\)[\s\S]*?return res;\n\}",
    '''BlockBackend::BlockPtr
FusionTreeBackend::diagonal_tensor_to_block(DiagonalTensorCPtr a)
{
    if (!a->symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "diagonal_tensor_to_block requires a droppable symmetry. Got {}.", a->symmetry->str()));
    }
    auto a_data = data_from_tensor(a);
    auto leg_es = a->leg();
    auto const& leg_space = *leg_es;
    auto res = block_backend->zeros({ static_cast<int64>(std::llround(leg_space.dim)) }, a->dtype);
    auto const& bi = a_data->block_inds;
    auto codomain = a->codomain;
    for (std::size_t n = 0; n < bi.nrows(); ++n) {
        int64 bi_cod = bi(n, 0);
        Sector c = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        int64 dim_c = static_cast<int64>(codomain->sector_dims[static_cast<std::size_t>(bi_cod)]);
        auto entries = block_backend->reshape(
          block_backend->outer(block_backend->ones_block({ dim_c }, a->dtype), a_data->blocks[n]),
          std::vector<int64>{ -1 });
        auto j = leg_space.sector_decomposition_where(c);
        if (!j)
            throw std::runtime_error("diagonal_tensor_to_block: sector not in leg");
        b_set(res, BlockIndexVec{ leg_slice(leg_space, *j) }, entries);
    }
    return res;
}''',
    text,
    count=1,
)

more_re = [
    (r"py::cast<std::vector<float64>>\(([^)]+)\.attr\(\"sector_qdims\"\)\)", r"\1->sector_qdims"),
    (r"py::cast<int64>\(py::cast\(([^)]+)\)\.attr\(\"dim\"\)\)", r"static_cast<int64>(std::llround(as_space(\1)->dim))"),
    (r"py::cast\(a->domain\)", "a->domain"),
    (r"py::cast\(a->codomain\)", "a->codomain"),
    (r"py::cast\(tensor->codomain\)", "tensor->codomain"),
    (r"py::cast\(tensor->domain\)", "tensor->domain"),
    (r"py::cast\(tens->codomain\)", "tens->codomain"),
    (r"py::cast\(mask1->symmetry\)", "mask1->symmetry"),
    (r"py::cast\(S->symmetry\)", "S->symmetry"),
    (r"py::cast\(S->domain\)", "S->domain"),
    (r"py::cast\(S->leg\(\)\)", "S->leg()"),
    (r"mults_of\(py::cast\(([^)]+)\)\)", r"mults_of(\1)"),
    (r"nsec\(py::cast\(([^)]+)\)\)", r"nsec(\1)"),
    (r"SectorArray a_sectors_arr = py::cast<SectorArray>\(a_sectors\);", ""),
    (r"SectorArray b_sectors_arr = py::cast<SectorArray>\(b_sectors\);", ""),
    (r"a_sectors_arr", "a_sectors"),
    (r"b_sectors_arr", "b_sectors"),
]
for pat, rep in more_re:
    text = re.sub(pat, rep, text)

# b_get/b_set with py::make_tuple -> BlockIndexVec
text = re.sub(
    r"b_get\(([^,]+),\s*py::make_tuple\(([^)]+)\)\)",
    r"b_get(\1, BlockIndexVec{ \2 })",
    text,
)
text = re.sub(
    r"b_set\(([^,]+),\s*py::make_tuple\(([^)]+)\)",
    r"b_set(\1, BlockIndexVec{ \2 }",
    text,
)
text = re.sub(
    r"b_set_add\(([^,]+),\s*concat_slice_lists\(([^)]+)\)",
    r"b_set_add(\1, concat_slice_axes(\2)",
    text,
)
text = text.replace("concat_slice_lists", "concat_slice_axes")
text = re.sub(
    r"j1\.append\(slice_from_index_slice\(slc\)\)",
    "j1.push_back(slice_from_index_slice(slc))",
    text,
)
text = re.sub(
    r"j2\.append\(slice_from_index_slice\(slc\)\)",
    "j2.push_back(slice_from_index_slice(slc))",
    text,
)

CPP.write_text(text)
print("pass2 done")
