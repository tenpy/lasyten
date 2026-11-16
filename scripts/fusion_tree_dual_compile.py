#!/usr/bin/env python3
"""Wrap selected methods: py body from HEAD when LASYTEN_HAS_PYTHON, else NotImplemented stub."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "src/backends/fusion_tree_backend.cpp"
orig = subprocess.check_output(
    ["git", "show", "HEAD:src/backends/fusion_tree_backend.cpp"], cwd=ROOT, text=True
)

NAMES = [
    "truncate_singular_values",
    "_mask_contract",
    "diagonal_to_mask",
    "mask_binary_operand",
    "mask_from_block",
    "mask_to_block",
    "mask_transpose",
    "mask_unary_operand",
    "partial_trace",
    "get_element",
    "from_grid",
    "partial_compose",
    "outer",
    "scale_axis",
    "get_element_mask",
    "diagonal_from_block",
    "diagonal_tensor_to_block",
    "from_tree_pairs",
    "compose",
    "qr",
    "lq",
    "svd",
    "combine_legs",
    "linear_combination",
]


def extract_function(src: str, name: str) -> str | None:
    pat = re.compile(rf"([\w:<>\s,\*&]+\n)?{re.escape(name)}\([^;{{]*\)\s*\{{", re.MULTILINE)
    m = pat.search(src)
    if not m:
        # try with FusionTreeBackend:: prefix only
        sig = f"FusionTreeBackend::{name}"
        idx = src.find(sig)
        if idx < 0:
            return None
        brace = src.find("{", idx)
    else:
        idx = m.start()
        brace = src.find("{", idx)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[idx : i + 1]
    return None


text = CPP.read_text()
for name in NAMES:
    cur = extract_function(text, f"FusionTreeBackend::{name}")
    pyf = extract_function(orig, f"FusionTreeBackend::{name}")
    if not cur or not pyf:
        print("skip", name)
        continue
    # split py function into signature + body
    brace = pyf.find("{")
    sig = pyf[:brace].rstrip()
    body = pyf[brace + 1 : -1]  # inside braces
    replacement = (
        sig
        + " {\n#if LASYTEN_HAS_PYTHON\n"
        + body
        + "#else\n    throw NotImplemented(\""
        + name
        + "\");\n#endif\n}\n"
    )
    text = text.replace(cur, replacement)
    print("wrapped", name)

# fix accidental double as_scalar on zero_scalar
text = text.replace(
    "block_backend->as_scalar(zero_scalar_of(*block_backend, data->dtype), data->dtype)",
    "zero_scalar_of(*block_backend, data->dtype)",
)
text = text.replace(
    "block_backend->as_scalar(zero_scalar_of(*block_backend, a_data->dtype), a_data->dtype)",
    "zero_scalar_of(*block_backend, a_data->dtype)",
)
text = text.replace(
    "block_backend->as_scalar(zero_scalar_of(*block_backend, a->dtype), a->dtype)",
    "zero_scalar_of(*block_backend, a->dtype)",
)

CPP.write_text(text)
print("done")
