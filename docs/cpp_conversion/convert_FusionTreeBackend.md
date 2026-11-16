# Conversion of FusionTreeBackend

## Status

**Done / monkey-patched** on branch `convert_backends`. `lasyten.backends.fusion_tree_backend` re-exports C++ types from `lasyten._core` and keeps `_tree_block_iter` for tests.

### Python leftovers

- `_tree_block_iter` — Python helper used by tests / cross-checks (not part of the backend method surface).

With `LASYTEN_WITH_PYTHON=ON`, the FusionTreeBackend method surface is native C++ (no Python
delegation). `partial_trace` / `permute_legs` / `apply_instructions` are implemented in C++, but
several heavy paths still reach into Space / Leg / NumPy via the `lasyten::py` layer and are
compiled out when `LASYTEN_WITH_PYTHON=OFF` (see below).

`tests/python_tests/backends/test_fusion_tree_backend.py`: **21 passed**. Fixed `partial_trace_helper` multiplicity check for 2-leg trees (must match numpy broadcast of `multiplicities[:2] == [0, 0]`).

## `LASYTEN_WITH_PYTHON=OFF` gaps (issue #326 item 3)

`FusionTreeBackend` lives in `LASYTEN_PYTHON_FREE_SOURCES` and links in OFF builds. Core paths
used by `ctest -R no_python` (factory, tree mapping, permute / ``apply_instructions``, dense block
I/O helpers) stay native.

### Native without libpython (``LASYTEN_HAS_PYTHON=OFF``)

| Group | Methods |
| --- | --- |
| Legs / algebra | `combine_legs`, `compose`, `linear_combination` |
| Decompositions | `svd`, `qr`, `lq`, `truncate_singular_values` (without custom `basis_perm`) |
| Trace / elements | `get_element_mask` |
| Diagonal | `diagonal_from_block`, `diagonal_tensor_to_block` |

Shared mask helpers (`MaskPieces`, `finish_mask`, …) are in the anonymous namespace for the next
OFF ports.

### Still stubbed OFF (`throw NotImplemented` in the ``#else`` of ``#if LASYTEN_HAS_PYTHON``)

| Group | Methods |
| --- | --- |
| Legs / algebra | `partial_compose`, `outer`, `scale_axis` |
| Trace / elements | `partial_trace`, `get_element`, `from_tree_pairs`, `from_grid` |
| Diagonal | `diagonal_to_mask` |
| Masks | `_mask_contract`, `mask_binary_operand`, `mask_unary_operand`, `mask_from_block`, `mask_to_block`, `mask_transpose` |

Always stubbed (even with Python ON; match historical Python API):
`from_dense_block_trivial_sector`, `inv_part_*`, `state_tensor_product`,
`to_dense_block_trivial_sector`. Conditional inside the Python path:
`truncate_singular_values` with `basis_perm`; `_mask_contract` on a `LegPipe`.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `FusionTreeBackend` (+ Instruction / Engine / TensorMapping helpers) |
| original python file | `lasyten/backends/fusion_tree_backend.py` |
| original python module | `lasyten.backends` |
| declaration | `include/lasyten/backends/fusion_tree_backend.h` (+ optional `fusion_tree_permute.h`, `fusion_tree_mapping.h`) |
| definition | `src/backends/fusion_tree_backend.cpp` (may split) |
| pybind11 binding | `pybind/backends/py_fusion_tree_backend.cpp` |
| trampoline | no (no Python subclasses) |
| first line of docstring | A backend based on fusion trees. |

## Design notes

- Inherit `TensorBackend`; ctor `(block_backend, eps=5e-14)`; `can_decompose_tensors = true`.
- Helpers: `wrap` / `unwrap` / `data_from_tensor` for `FusionTreeData::Ptr`.
- No `make_pipe` / HDF5 / `is_real` overrides (inherit base).
- Tensor args interim `py::object`; callables → `py::function`.
- Stubs matching Python: `from_dense_block_trivial_sector`, `inv_part_*`, `state_tensor_product`, `to_dense_block_trivial_sector` → `NotImplemented`.
- **Permute stack** (required by `permute_legs` / `partial_trace`):
  1. Instruction POD (`Braid` / `Bend` / `Twist`)
  2. `PermuteLegsInstructionEngine`
  3. `TensorMapping` / `TreePairMapping` / `FactorizedTreeMapping` — **done** ([convert_TreePairMapping.md](convert_TreePairMapping.md))
- Forest helpers `_add_forest_block_entries` / `_get_forest_block_contribution` for dense I/O.
- Prefer calling `lasyten.tools.misc` / mappings via pybind for iterators when clearer than a full port.
- **SparseMapping** / **TreePairMapping** / **FactorizedTreeMapping** / **PermuteLegsInstructionEngine** / **`_partial_trace_helper`** are native C++.
- `partial_trace` full-trace path: binding returns `(scalar, None, None)`; scalar data uses `block_inds` shape `(1, 2)`.
- QR / LQ / SVD new-leg multiplicities come from `new_co_domain->multiplicities` (TensorProduct mults), not `factors[0]`.

## Suggested implementation order

1. Header: `FusionTreeBackend` + all TensorBackend overrides + `eps`
2. Thin native methods (compose, dagger, eye, zeros, QR/SVD/LQ, …)
3. Instruction + Engine (+ bind for existing unit tests)
4. TensorMapping hierarchy
5. Forest dense↔sparse helpers
6. Remaining complex methods; replace any Python delegation — **done**
7. Bindings + wire `get_backend('fusion_tree', …)` + monkey-patch — **done**

## TODO checklist

- [x] planning (this file)
- [x] declaration
- [x] definitions + compile
- [x] bindings + factory
- [x] monkey-patch via `fusion_tree_backend.py` (keeps `_tree_block_iter`)
- [x] pytest
