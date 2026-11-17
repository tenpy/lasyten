#!/usr/bin/env python3
"""Surgical fixes on fusion_tree_backend.cpp after pass1."""
from __future__ import annotations

import re
from pathlib import Path

CPP = Path(__file__).resolve().parents[1] / "src/backends/fusion_tree_backend.cpp"
text = CPP.read_text()

if "zero_scalar_of" not in text:
    text = text.replace(
        "bool\nis_zero_scalar(BlockBackend::Scalar const& a)",
        """BlockBackend::Scalar
zero_scalar_of(BlockBackend& bb, Dtype dtype)
{
    return bb.as_scalar(complex128{ 0. }, dtype);
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

bool
is_zero_scalar(BlockBackend::Scalar const& a)""",
    )

text = text.replace("dtype::zero_scalar(", "zero_scalar_of(*block_backend, ")
text = text.replace(
    "    auto instructions_vec = instructions_from_python(instructions);\n\n    auto const key =\n      plan_key(instructions,",
    "    auto const key =\n      plan_key(instructions,",
)

# permute_legs
start = text.find("FusionTreeBackend::permute_legs")
end = text.find("FusionTreeBackend::_get_forest_block_contribution")
if start != -1 and end != -1:
    text = (
        text[:start]
        + '''FusionTreeBackend::permute_legs(TensorCPtr a,
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
    PermuteLegsInstructionEngine engine(a->num_codomain_flat_legs(),
                                        a->num_domain_flat_legs(),
                                        new_codomain_idcs,
                                        new_domain_idcs,
                                        flat_levels,
                                        flat_bend_right,
                                        a->symmetry->has_symmetric_braid());
    return apply_instructions(a,
                              engine.evaluate_instructions(),
                              new_codomain_idcs,
                              new_domain_idcs,
                              new_codomain,
                              new_domain,
                              mixes_codomain_domain);
}

'''
        + text[end:]
    )

repls = [
    (r"py::cast<std::vector<float64>>\(([^)]+)\.attr\(\"sector_qdims\"\)\)", r"\1->sector_qdims"),
    (r"SectorArray a_sectors_arr = py::cast<SectorArray>\(a_sectors\);\n\s*", ""),
    (r"SectorArray b_sectors_arr = py::cast<SectorArray>\(b_sectors\);\n\s*", ""),
    (r"\ba_sectors_arr\b", "a_sectors"),
    (r"\bb_sectors_arr\b", "b_sectors"),
    (r"py::cast\(a->domain\)", "a->domain"),
    (r"py::cast\(a->codomain\)", "a->codomain"),
]
for p, r in repls:
    text = re.sub(p, r, text)

CPP.write_text(text)
print("pass3 ok")
