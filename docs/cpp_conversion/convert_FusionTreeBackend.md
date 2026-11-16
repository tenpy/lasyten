# Conversion of FusionTreeBackend

## Status

**Done / monkey-patched** on branch `convert_backends`. `lasyten.backends.fusion_tree_backend` re-exports C++ types from `lasyten._core` and keeps `_tree_block_iter` for tests.

### Python leftovers

- `_tree_block_iter` — Python helper used by tests / cross-checks (not part of the backend method surface).

With `LASYTEN_WITH_PYTHON=ON`, the FusionTreeBackend method surface is native C++ (no Python
delegation). Heavy tree / mask paths (`partial_compose`, `outer`, `from_grid`, `mask_*`, …) are
implemented in C++ for both ON and OFF builds.

`tests/python_tests/backends/test_fusion_tree_backend.py`: **21 passed**. Fixed `partial_trace_helper` multiplicity check for 2-leg trees (must match numpy broadcast of `multiplicities[:2] == [0, 0]`).

## `LASYTEN_WITH_PYTHON=OFF` gaps (issue #326 item 3)

`FusionTreeBackend` lives in `LASYTEN_PYTHON_FREE_SOURCES` and links in OFF builds. Core paths
used by `ctest -R no_python` (factory, tree mapping, permute / ``apply_instructions``, dense block
I/O helpers) stay native.

### Native without libpython (``LASYTEN_HAS_PYTHON=OFF``)

| Group | Methods |
| --- | --- |
| Legs / algebra | `combine_legs`, `compose`, `linear_combination`, `partial_compose`, `outer`, `scale_axis` |
| Decompositions | `svd`, `qr`, `lq`, `truncate_singular_values` (without custom `basis_perm`) |
| Trace / elements | `partial_trace`, `get_element`, `get_element_mask`, `from_tree_pairs`, `from_grid` |
| Diagonal / masks | `diagonal_from_block`, `diagonal_tensor_to_block`, `diagonal_to_mask`, `_mask_contract` (single flat leg), `mask_binary_operand`, `mask_unary_operand`, `mask_from_block`, `mask_to_block`, `mask_transpose` |

Shared mask helpers (`MaskPieces`, `finish_mask`, …) live in the anonymous namespace.

### Still stubbed OFF (intentional ``NotImplemented``)

| Reason | Methods / conditions |
| --- | --- |
| Always stubbed (historical Python API) | `from_dense_block_trivial_sector`, `inv_part_*`, `state_tensor_product`, `to_dense_block_trivial_sector` |
| Conditional | `truncate_singular_values` when a leg has custom `basis_perm`; `_mask_contract` when masking a `LegPipe` |

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
