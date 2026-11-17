#!/usr/bin/env python3
"""Mechanical de-Python pass on fusion_tree_backend.cpp (see convert_nanobind leftover #1)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "src/backends/fusion_tree_backend.cpp"

HELPERS = r'''
namespace {

BlockInds
zeros_i64(std::size_t rows, std::size_t cols)
{
    return BlockInds::zeros(rows, cols);
}

using AxisSlice = BlockBackend::AxisSlice;
using BlockIndexVec = std::vector<BlockBackend::BlockIndex>;

[[nodiscard]] AxisSlice
slice_of(int64 start, int64 stop)
{
    return AxisSlice{ start, stop, std::nullopt };
}

[[nodiscard]] AxisSlice
slice_from_index_slice(IndexSlice slc)
{
    return AxisSlice{ slc.start, slc.stop, std::nullopt };
}

[[nodiscard]] AxisSlice
axis_all()
{
    return BlockBackend::AxisSlice::all();
}

[[nodiscard]] AxisSlice
leg_slice(Space const& space, int64 idx)
{
    if (!space.slices) {
        throw std::invalid_argument("FusionTreeBackend: space has no slices");
    }
    auto const& s = space.slices->at(static_cast<std::size_t>(idx));
    return slice_of(s[0], s[1]);
}

[[nodiscard]] BlockIndexVec
concat_slice_axes(std::vector<AxisSlice> const& j1, std::vector<AxisSlice> const& j2)
{
    BlockIndexVec out;
    out.reserve(j1.size() + j2.size());
    for (auto const& s : j1)
        out.push_back(s);
    for (auto const& s : j2)
        out.push_back(s);
    return out;
}

BlockBackend::BlockPtr
b_get(BlockBackend::BlockPtr const& b, BlockIndexVec const& key)
{
    return b->get_item(std::span<BlockBackend::BlockIndex const>(key));
}

void
b_set(BlockBackend::BlockPtr const& b, BlockIndexVec const& key, BlockBackend::BlockPtr const& v)
{
    b->set_item(std::span<BlockBackend::BlockIndex const>(key), *v);
}

void
b_set_add(BlockBackend::BlockPtr const& b, BlockIndexVec const& key, BlockBackend::BlockPtr const& v)
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
mults_of(Leg::Ptr const& leg)
{
    return as_space(leg)->multiplicities;
}

int64
nsec(Leg::Ptr const& leg)
{
    return as_space(leg)->num_sectors;
}

std::vector<int64>
legs_flat_leg_permutation(std::vector<Leg::Ptr> const& legs)
{
    std::vector<int64> out;
    int64 offset = 0;
    for (auto const& leg : legs) {
        auto part_perm = leg._flat_leg_permutation(offset);
        out.insert(out.end(), part_perm.begin(), part_perm.end());
        offset += leg.num_flat_legs();
    }
    return out;
}

std::vector<BlockBackend::BlockPtr>
permute_blocks(std::vector<BlockBackend::BlockPtr> const& blocks,
               std::span<std::size_t const> perm)
{
    std::vector<BlockBackend::BlockPtr> out;
    out.reserve(perm.size());
    for (std::size_t i : perm)
        out.push_back(blocks[i]);
    return out;
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

std::vector<std::uint8_t>
flat_are_dual(TensorProduct::Ptr tp)
{
    std::vector<std::uint8_t> out;
    for (auto const& leg : tp->flat_legs())
        out.push_back(static_cast<std::uint8_t>(leg.is_dual()));
    return out;
}

/// Build flat-leg target indices from leg permutation idcs (permute_legs).
std::tuple<std::vector<int64>, std::vector<int64>, std::vector<std::optional<int64>>,
           std::vector<std::optional<bool>>>
flat_leg_permutation_inputs(TensorCPtr const& a,
                            std::vector<int64> const& codomain_idcs,
                            std::vector<int64> const& domain_idcs,
                            std::vector<std::optional<int64>> const& levels,
                            std::vector<std::optional<bool>> const& bend_right)
{
    std::vector<std::vector<int64>> leg_comb;
    std::vector<std::optional<int64>> flat_levels;
    std::vector<std::optional<bool>> flat_bend_right;
    int64 flat_index = 0;
    auto const legs = a->legs();
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto const& leg = legs[i];
        bool const is_codomain = static_cast<int64>(i) < a->num_codomain_legs();
        std::vector<int64> indices;
        if (leg.is_pipe()) {
            int64 const num = leg.num_flat_legs();
            indices.reserve(static_cast<std::size_t>(num));
            for (int64 k = 0; k < num; ++k)
                indices.push_back(flat_index + k);
            flat_index += num;
            for (int64 k = 0; k < num; ++k) {
                flat_levels.push_back(levels[i]);
                flat_bend_right.push_back(bend_right[i]);
            }
        } else {
            indices.push_back(flat_index);
            ++flat_index;
            flat_levels.push_back(levels[i]);
            flat_bend_right.push_back(bend_right[i]);
        }
        leg_comb.push_back(std::move(indices));
    }
    std::vector<int64> new_domain_idcs;
  for (int64 idx : domain_idcs) {
        auto const& group = leg_comb[static_cast<std::size_t>(idx)];
        for (auto it = group.rbegin(); it != group.rend(); ++it)
            new_domain_idcs.push_back(*it);
    }
    std::vector<int64> new_codomain_idcs;
    for (int64 idx : codomain_idcs) {
        for (int64 k : leg_comb[static_cast<std::size_t>(idx)])
            new_codomain_idcs.push_back(k);
    }
    return { new_codomain_idcs, new_domain_idcs, std::move(flat_levels), std::move(flat_bend_right) };
}

} // namespace
'''

REPLACEMENTS: list[tuple[str, str]] = [
    (r"#include <lasyten/backends/block_inds_numpy.h>\n", ""),
    (r"#include <lasyten/symmetries/sector_numpy.h>\n", ""),
    (r"py::cast<TensorProduct::Ptr>\(py::cast\(([^)]+)\)\)", r"\1"),
    (r"py::cast<Symmetry::Ptr>\(py::cast\(([^)]+)\)\)", r"\1"),
    (r"py::cast<int64>\(py::cast\(([^)]+)\)\.attr\(\"num_sectors\"\)\)", r"\1->num_sectors"),
    (r"tp_mults\(py::cast\(([^)]+)\)\)", r"\1->multiplicities"),
    (r"mults_of\(py::cast\(([^)]+)\)\)", r"\1->multiplicities"),
    (r"nsec\(py::cast\(([^)]+)\)\)", r"\1->num_sectors"),
    (
        r"py::cast<std::vector<int64>>\(([^)]+)\.attr\(\"multiplicities\"\)\)",
        r"\1->multiplicities",
    ),
    (
        r"py::cast<std::vector<float64>>\(([^)]+)\.attr\(\"sector_qdims\"\)\)",
        r"\1->sector_qdims",
    ),
    (
        r"dtype::common\(\{ ([^}]+), py::cast<Dtype>\(py::cast\(([^)]+)\)\.attr\(\"fusion_tensor_dtype\"\)\) \}\)",
        r"dtype::common({ \1, \2->fusion_tensor_dtype.value_or(Dtype::Complex128) })",
    ),
    (r"py::cast\(a_item\.uncoupled\)", "a_item.uncoupled"),
    (r"py::cast\(b_item\.uncoupled\)", "b_item.uncoupled"),
    (
        r"py::to_bool\(py::cast\(([^)]+)\)\.attr\(\"has_symmetric_braid\"\)\)",
        r"\1->has_symmetric_braid()",
    ),
    (r"py::cast<Dtype>\(py::cast\(([^)]+)\)\.attr\(\"fusion_tensor_dtype\"\)\)", r"\1->fusion_tensor_dtype.value_or(Dtype::Complex128)"),
]


def main() -> None:
    text = CPP.read_text()
    text = text.replace(
        "#include <lasyten/backends/fusion_tree_backend.h>\n",
        "#include <lasyten/backends/fusion_tree_backend.h>\n#include <lasyten/symmetries/sector.h>\n",
    )
    # Replace helper namespace (first one after FusionTreeData::from_hdf5)
    start = text.find("// FusionTreeBackend method implementations")
    end = text.find("TensorBackend::DataPtr\nFusionTreeBackend::wrap")
    if start == -1 or end == -1:
        raise SystemExit("could not locate helper namespace")
    text = text[:start] + "// FusionTreeBackend method implementations (append after FusionTreeData::from_hdf5)\n\n" + HELPERS + "\n" + text[end:]

    # Remove py_bool_dtype and old anonymous helpers at file top (lines 47-51)
    text = re.sub(
        r"namespace \{\n\n\[\[nodiscard\]\] py::object\npy_bool_dtype\(\)[\s\S]*?\n\} // namespace\n\n",
        "",
        text,
        count=1,
    )

    for pat, rep in REPLACEMENTS:
        text = re.sub(pat, rep, text)

    # apply_instructions signature + body
    text = text.replace(
        "FusionTreeBackend::apply_instructions(TensorCPtr tensor,\n                                      py::object instructions,",
        "FusionTreeBackend::apply_instructions(TensorCPtr tensor,\n                                      std::vector<Instruction> const& instructions,",
    )
    text = text.replace(
        "    auto codomain = py::cast<TensorProduct::Ptr>(py::cast(tensor->codomain));\n    auto domain = py::cast<TensorProduct::Ptr>(py::cast(tensor->domain));\n    auto instructions_vec = instructions_from_python(instructions);\n\n    auto const key =\n      plan_key(instructions_vec, codomain, domain, t_data->block_inds, mixes_codomain_domain);",
        "    auto const& codomain = tensor->codomain;\n    auto const& domain = tensor->domain;\n\n    auto const key =\n      plan_key(instructions, codomain, domain, t_data->block_inds, mixes_codomain_domain);",
    )
    text = text.replace("instructions_vec,", "instructions,")

    # Forest block helper signatures
    text = text.replace("py::object a_sectors,\n                                                  py::object b_sectors,", "SectorArray const& a_sectors,\n                                                  SectorArray const& b_sectors,")
    text = text.replace("py::object a_sectors,\n                                             py::object b_sectors,", "SectorArray const& a_sectors,\n                                             SectorArray const& b_sectors,")

    # test_tensor_sanity / test_mask_sanity data access
    text = re.sub(
        r"    py::object raw = py::cast\(a\)\.attr\(\"data\"\);\n    FusionTreeData::Ptr data;\n    try \{\n        data = unwrap\(py::cast<DataPtr>\(raw\)\);\n    \} catch \(\.\.\.\) \{\n        return;\n    \}",
        "    FusionTreeData::Ptr data;\n    try {\n        data = data_from_tensor(a);\n    } catch (std::invalid_argument const&) {\n        return;\n    }",
        text,
    )
    text = re.sub(
        r"    py::object raw = py::cast\(a\)\.attr\(\"data\"\);\n    FusionTreeData::Ptr data;\n    try \{\n        data = unwrap\(py::cast<DataPtr>\(raw\)\);\n    \} catch \(\.\.\.\) \{\n        return;\n    \}",
        "    FusionTreeData::Ptr data;\n    try {\n        data = data_from_tensor(a);\n    } catch (std::invalid_argument const&) {\n        return;\n    }",
        text,
        count=1,
    )

    CPP.write_text(text)
    print(f"Wrote {CPP} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
