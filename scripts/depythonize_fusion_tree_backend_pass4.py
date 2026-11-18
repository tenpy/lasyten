#!/usr/bin/env python3
import re
from pathlib import Path

CPP = Path(__file__).resolve().parents[1] / "src/backends/fusion_tree_backend.cpp"
text = CPP.read_text()

text = text.replace("block_backend->as_block(py::cast(b), std::nullopt, device)", "block_backend->convert_block(b, std::nullopt, device)")
text = text.replace("bb->as_block(py::cast(b), dt, dev)", "bb->convert_block(b, dt, dev)")

# slice / index helpers
text = text.replace("py::slice slice_from_index_slice", "AxisSlice slice_from_index_slice")  # already AxisSlice return
text = re.sub(r"py::slice idx1\(i1, i1 \+ tree_block_height, 1\)", "AxisSlice{ i1, i1 + tree_block_height, std::nullopt }", text)
text = re.sub(r"py::slice idx2\(i2, i2 \+ tree_block_width, 1\)", "AxisSlice{ i2, i2 + tree_block_width, std::nullopt }", text)
text = text.replace("b_get(block, py::make_tuple(idx1, idx2))", "b_get(block, BlockIndexVec{ idx1, idx2 })")
text = text.replace("b_set(block, py::make_tuple(idx1, idx2), tree_block)", "b_set(block, BlockIndexVec{ idx1, idx2 }, tree_block)")

# test sanity
text = re.sub(
    r"    py::object raw = py::cast\(a\)\.attr\(\"data\"\);\n    FusionTreeData::Ptr data;\n    try \{\n        data = unwrap\(py::cast<DataPtr>\(raw\)\);\n    \} catch \(\.\.\.\) \{\n        return;\n    \}",
    "    FusionTreeData::Ptr data;\n    try {\n        data = data_from_tensor(a);\n    } catch (std::invalid_argument const&) {\n        return;\n    }",
    text,
)

# lists for block_inds rows in from_dense_block
text = text.replace("py::list block_inds_rows;", "std::vector<std::vector<int64>> block_inds_rows;")
text = re.sub(r"block_inds_rows\.append\(py::make_tuple\(i, j\)\)", "block_inds_rows.push_back({ i, j })", text)
text = re.sub(
    r"    if \(block_inds_rows\.size\(\) > 0\) \{\n        std::vector<std::vector<int64>> rows;\n        rows\.reserve\(static_cast<std::size_t>\(block_inds_rows\.size\(\)\)\);\n        for \(py::handle h : block_inds_rows\) \{\n            auto t = py::to_tuple\(h\);\n            rows\.push_back\(\{ py::cast<int64>\(t\[0\]\), py::cast<int64>\(t\[1\]\) \}\);\n        \}\n        block_inds = BlockInds::from_rows\(rows\);\n    \}",
    "    if (!block_inds_rows.empty())\n        block_inds = BlockInds::from_rows(block_inds_rows);",
    text,
)

# j1/j2 in from_dense_block
text = text.replace("py::list j1;", "std::vector<AxisSlice> j1;")
text = text.replace("py::list j2;", "std::vector<AxisSlice> j2;")
text = text.replace("j1.append(slice_from_index_slice(slc))", "j1.push_back(slice_from_index_slice(slc))")
text = text.replace("j2.append(slice_from_index_slice(slc))", "j2.push_back(slice_from_index_slice(slc))")
text = text.replace("concat_slice_lists(j1, j2)", "concat_slice_axes(j1, j2)")

# to_dense block j1/j2
text = text.replace("py::list legs = py::cast(a->legs());", "auto legs = a->legs();")
text = text.replace("for (py::handle g : py::cast(a->codomain).attr(\"flat_legs_nesting\")())", "for (auto const& g : a->codomain->flat_legs_nesting())")
text = text.replace("combine_axes.push_back(py::cast<std::vector<int64>>(g));", "combine_axes.push_back(g);")
text = text.replace("py::list dom_nesting = py::cast(a->domain).attr(\"flat_legs_nesting\")();", "auto dom_nesting = a->domain->flat_legs_nesting();")
text = re.sub(r"for \(py::ssize_t gi = dom_nesting\.size\(\) - 1; gi >= 0; --gi\) \{\n            auto group = py::cast<std::vector<int64>>\(dom_nesting\[gi\]\);", "for (int64 gi = static_cast<int64>(dom_nesting.size()) - 1; gi >= 0; --gi) {\n            auto const& group = dom_nesting[static_cast<std::size_t>(gi)];", text)

# dual_sectors diagonal_transpose
text = re.sub(
    r"    auto perm_py = py::cast\(tens->symmetry\)\n                     \.attr\(\"dual_sectors\"\)\(py::cast\(tens->domain\)\.attr\(\"sector_decomposition\"\)\)\n                     \.attr\(\"lexsort_indices\"\)\(\);\n    auto inv = inverse_permutation\(py::cast<std::vector<int64>>\(perm_py\)\);",
    "    auto perm = tens->symmetry->dual_sectors(tens->domain->sector_decomposition).lexsort_indices();\n    std::vector<int64> perm_i64(perm.size());\n    for (std::size_t i = 0; i < perm.size(); ++i)\n        perm_i64[i] = static_cast<int64>(perm[i]);\n    auto inv = inverse_permutation(perm_i64);",
    text,
)
text = text.replace("{ py::cast<Space::Ptr>(py::cast(tens->leg()).attr(\"dual\")), wrap(data) }", "{ tens->leg()->dual_space(), wrap(data) }")

# split legs from_dense pipes section - stub native using legs vector
text = re.sub(
    r"if \(codomain->has_pipes\(\) \|\| domain->has_pipes\(\)\) \{[\s\S]*?a = block_backend->permute_axes\(a, axes_perm\);\n    \}",
    '''if (codomain->has_pipes() || domain->has_pipes()) {
        std::vector<Leg::Ptr> legs;
        for (auto const& f : codomain->factors)
            legs.push_back(f);
        for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it)
            legs.push_back(it->dual());
        auto axes_perm = legs_flat_leg_permutation(legs);
        std::vector<int64> split_idcs;
        std::vector<std::vector<int64>> split_dims;
        int64 perm_idx = 0;
        for (std::size_t n = 0; n < legs.size(); ++n) {
            auto const& leg = legs[n];
            if (leg.is_pipe()) {
                split_idcs.push_back(static_cast<int64>(n));
                std::vector<int64> dims;
                for (auto const& fl : leg.as_leg_pipe()->flat_legs())
                    dims.push_back(fl.dim());
                auto const nf = leg.num_flat_legs();
                std::vector<int64> perm(axes_perm.begin() + perm_idx, axes_perm.begin() + perm_idx + nf);
                int64 perm_min = *std::min_element(perm.begin(), perm.end());
                for (auto& p : perm)
                    p -= perm_min;
                auto inv = inverse_permutation(perm);
                std::vector<int64> sd;
                for (int64 ih : inv)
                    sd.push_back(dims[static_cast<std::size_t>(ih)]);
                split_dims.push_back(std::move(sd));
                perm_idx += nf;
            } else {
                perm_idx += 1;
            }
        }
        a = block_backend->split_legs(a, split_idcs, split_dims);
        a = block_backend->permute_axes(a, axes_perm);
    }''',
    text,
    count=1,
)

CPP.write_text(text)
print("pass4")
