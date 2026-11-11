# C++ conversion tracking

This folder stores per-object conversion plans and notes for the Python-to-C++ migration of the lasyten core library.

Each conversion follows the workflow in [.cursor/skills/pybind11-codegen/SKILL.md](../../.cursor/skills/pybind11-codegen/SKILL.md) and is documented in a file `convert_<pyname>.md` (e.g. `convert_format_like_list.md`).

Conversion order follows the plan: Layer 0 (tools) → Layer 1 (block_backends) → Layer 2 (symmetries) → Layer 3 (backends) → Layer 4 (tensors) → Layer 5 (models).


## Conversion workflow (per object)

The project uses a **one-class-or-function-at-a-time** workflow with the pybind11 codegen in [.cursor/skills/pybind11-codegen/](.cursor/skills/pybind11-codegen/) (see [SKILL.md](.cursor/skills/pybind11-codegen/SKILL.md)):

1. **Setup** — Clean git, branch `convert_<pyname>`, run pytest for the module’s tests.
2. **Plan** — Create `docs/cpp_conversion/convert_<pyname>.md` from the template (e.g. in `.cursor/skills/pybind11-codegen/assets/convert_pyobj.md`), list dependencies and TODOs.
3. **Declaration** — `gen_cpp_declaration --py-name <pyname> --header-file <header>`, then fix types/namespaces (C++23, use `pybind11_codegen.toml` type mappings).
4. **Definitions** — `gen_cpp_definition` into a `.cpp`, fix CHECKME/FIXME, get clang-tidy clean and CTest passing.
5. **Bindings** — `gen_pyb11_binding` (and trampoline for virtual classes), adjust if C++ API diverged.
6. **Monkey-patch** — In the original Python file, right below the (removed) definition, add `from .._core import <pyname>`.
7. **Tests** — Run pytest for the converted object, then full suite; remove original Python only after all tests pass.

Script entry point: run the codegen from the repo root; the script lives under `.cursor/skills/pybind11-codegen/` (e.g. `python .cursor/skills/pybind11-codegen/pybind11_codegen.py list_python_names`). Config: [pybind11_codegen.toml](pybind11_codegen.toml).

## Dependency order and module breakdown

Convert in **dependency order** so that C++ types exist before use. Suggested layers:

```mermaid
flowchart LR
  subgraph layer0 [Layer 0 - Tools]
    mappings
    misc
    string
    dtypes
    cost_polynomials
    math
  end
  subgraph layer1 [Layer 1 - Block backends]
    block_backend
    numpy
    array_api
  end
  subgraph layer2 [Layer 2 - Symmetries]
    symmetries
    su2data
    trees
    spaces
  end
  subgraph layer3 [Layer 3 - Backends]
    backend
    no_symmetry
    abelian
    fusion_tree_backend
    backend_factory
  end
  subgraph layer4 [Layer 4 - Tensors]
    tensors
    planar
    sparse
    krylov_based
  end
  subgraph layer5 [Layer 5 - Models]
    dof
    sites
    couplings
    tenpy_models
  end
  layer0 --> layer1
  layer1 --> layer2
  layer2 --> layer3
  layer3 --> layer4
  layer4 --> layer5
```



### Layer 0 — Minimal tools

- **SparseMapping / IdentityMapping** — **C++ templates** in `_core` as `SparseMappingFusionTree` / `…Pair` (+ Identity aliases); Python `lasyten/tools/mappings.py` removed — [convert_SparseMapping.md](convert_SparseMapping.md).
- **lasyten/tools/misc.py** — `to_iterable`, `rank_data`, `argsort`, `combine_constraints`, `as_immutable_array`, `duplicate_entries`, `inverse_permutation`, `iter_common_sorted_arrays`, `to_valid_idx`, `is_iterable` (used everywhere).
- **lasyten/tools/string.py** — `format_like_list` (tiny).
- **lasyten/block_backends/dtypes.py** — `Dtype`, numpy/lasyten dtype mapping (needed by block_backends).
- **lasyten/tools/cost_polynomials.py** — `BigOMonomial`, `BigOPolynomial` — **done in C++** (see [convert_cost_polynomials.md](convert_cost_polynomials.md)).
- **lasyten/tools/math.py** — `speigs`, `speigsh` (used by sparse/tensors).

Convert **functions and classes** in each file in dependency order within the file (e.g. helpers before public API). Use `list_python_names` to get the exact list of convertible names per module.

### Layer 1 — Block backends (no torch)

- **lasyten/block_backends/_block_backend.py** — `BlockBackend` (abstract) — **done in C++** (see [convert_BlockBackend.md](convert_BlockBackend.md)). Nested C++ types `BlockBackend::Block` and `BlockBackend::Scalar` were added as part of this conversion. Python leftovers remain for ArrayApi/Torch subclasses.
- **lasyten/block_backends/numpy.py** — `NumpyBlockBackend` — **done in C++** (same doc). Exported from `lasyten._core` via `__init__.py`.
- **lasyten/block_backends/array_api.py** — `ArrayApiBlockBackend` (still Python).

Skip `block_backends/torch.py` per scope.

### Layer 2 — Symmetries

- **Sector / SectorArray** — C++ foundation (see [convert_Sector.md](convert_Sector.md)): owning `Sector` (`int16_t[7]` + length, 16 bytes), `SectorArray` as `std::vector<Sector>` with stored `sector_ind_len` (including when empty), ops as methods, Python classes + `to_numpy()` / load casters. Do **not** template `Symmetry` on sector length.
- **lasyten/symmetries/_symmetries.py** — Large (~2.6k lines). Convert in order: exceptions → enums → `BaseSymmetry` (virtual API on `Sector` / `SectorArray`) → `SymmetryFactor` / concrete factors → product `Symmetry` → free functions.
- **lasyten/symmetries/_su2data.py** — SU2 data (small).
- **lasyten/symmetries/trees.py** — `FusionTree`, `fusion_trees` (~1k lines).
- **lasyten/symmetries/spaces.py** — `ElementarySpace`, `Leg`, `LegPipe`, `Space`, `TensorProduct`, `AbelianLegPipe` (~2.2k lines). Depends on symmetries and trees.

### Layer 3 — Backends

Overview: [convert_backends.md](convert_backends.md). Branch: `convert_backends`.

- **lasyten/backends/_backend.py** — `TensorBackend` (abstract), helpers. **Monkey-patched** from `_core` — [convert_TensorBackend.md](convert_TensorBackend.md).
- **lasyten/backends/no_symmetry.py** — `NoSymmetryBackend`. **Monkey-patched** — [convert_NoSymmetryBackend.md](convert_NoSymmetryBackend.md).
- **lasyten/backends/abelian.py** — `AbelianBackendData` + `AbelianBackend`. **Monkey-patched** — [convert_AbelianBackendData.md](convert_AbelianBackendData.md), [convert_AbelianBackend.md](convert_AbelianBackend.md).
- **lasyten/backends/fusion_tree_backend.py** — FT stack **monkey-patched** from `_core` (keeps `_tree_block_iter`). See [convert_FusionTreeBackend.md](convert_FusionTreeBackend.md).
- **lasyten/backends/backend_factory.py** — `get_backend` **monkey-patched** from `_core` — [convert_get_backend.md](convert_get_backend.md).

### Layer 4 — Tensors

- **lasyten/tensors/_tensors.py** — **C++ + bindings + monkey-patched**; Python file is a `_core` re-export. Batches 1–17: [convert_tensors.md](convert_tensors.md). Typed C++ API (no leftover tensor/space/leg `py::object`): [convert_tensor_typed_api.md](convert_tensor_typed_api.md).
- **lasyten/tensors/planar.py** — Planar diagram utilities. **C++ + monkey-patched**; thin Python `PlanarLinearOperator` wrapper kept — [convert_planar.md](convert_planar.md).
- **lasyten/tensors/sparse.py** — `LinearOperator`, `ProjectedLinearOperator`, etc. **C++ + monkey-patched** (numpy/scipy helpers remain Python) — [convert_sparse.md](convert_sparse.md).
- **lasyten/tensors/krylov_based.py** — `KrylovBased`, `Arnoldi`, `Lanczos*`, `GMRES`, `lanczos` **C++ + monkey-patched**; `lanczos_arpack` remains Python — [convert_krylov_based.md](convert_krylov_based.md).

### Layer 5 — Models

Branch: `convert_models`. Overview: [convert_models.md](convert_models.md).

- **lasyten/models/degrees_of_freedom.py** — `Site`, `DegreeOfFreedom`, DOF classes (`SpinDOF`, `ClockDOF`, `AnyonDOF`, `OccupationDOF`, `BosonicDOF`, `FermionicDOF`), `ALL_SPECIES`. **C++ + monkey-patched** — [convert_Site.md](convert_Site.md), DOF composition — [convert_SiteDOF.md](convert_SiteDOF.md).
- **lasyten/models/sites.py** — Concrete sites (`SpinSite`, `SpinlessBosonSite`, `SpinlessFermionSite`, `SpinHalfFermionSite`, anyon sites, …). **C++ + monkey-patched**.
- **lasyten/models/couplings.py** — `Coupling`, factory functions (`spin_spin_coupling`, `hopping`, …). **C++ + monkey-patched**; `tenpy_models.py` remains Python.
- **lasyten/models/tenpy_models.py** — TenPy model helpers (~130 lines). **Still Python** (out of scope).

## C++ layout

- **Headers:** [include/lasyten/](include/lasyten/) — add e.g. `tools.hpp`, `block_backend.hpp`, `symmetries.hpp`, `spaces.hpp`, `trees.hpp`, `backend.hpp`, `tensors.hpp`, `models.hpp` (or split further as needed).
- **Sources:** Add under a `src/` (or equivalent) and register in CMake.
- **Bindings:** [pybind/](pybind/) — extend `_core.cpp` or add separate `.cpp` files and include them in the same module.

Keep namespaces and include structure consistent with existing headers.


## Practical recommendations

1. Use the script from [.cursor/skills/pybind11-codegen/SKILL.md](.cursor/skills/pybind11-codegen/SKILL.md) to provide drafts. This will ensure that the docstrings and names are carried over correctly and provide hints for things to check.
2. **Create `docs/cpp_conversion/`** — Store one `convert_<pyname>.md` per conversion (and optionally high-level `convert_tensors.md`, `convert_symmetries.md`) to track progress and decisions.
3. **Extend type mappings** — Update [pybind11_codegen.toml](pybind11_codegen.toml) `[types_py_to_cpp]` as you find recurring types (e.g. `npt.NDArray` → appropriate C++ container or pybind11 buffer).
2. **Holder type** — Use `py::smart_holder` (pybind11 v3 built-in) as the holder for `py::class_` bindings of C++ types that use `std::shared_ptr` in the API (e.g. `Block`, `NumpyBlock`, `BlockBackend`, `NumpyBlockBackend`). Prefer `py::class_<T, py::smart_holder>` over `py::class_<T, std::shared_ptr<T>>` so Python/C++ conversions work with both `std::unique_ptr` and `std::shared_ptr` and trampolines are handled safely. No extra include is required.
4. **Trampolines** — Use `gen_pyb11_trampoline` only for classes that are subclassed in Python (e.g. `TensorBackend`, `BlockBackend`, `Symmetry`); for internal-only bases, bindings may not need trampolines.
5. **Tests** — For each converted object, run the relevant pytest under `tests/`; keep the full suite passing before removing the corresponding Python implementation.
6. **Incremental commits** — Per the skill: WIP commits with `--no-verify` (e.g. "WIP: convert : ") so pre-commit doesn’t block mid-conversion.
7. **Docstrings** — See [docstring_porting.md](docstring_porting.md) (dual-language policy from [tenpy/lasyten#242](https://github.com/tenpy/lasyten/issues/242); `///` + per-header `pybind/docstrings/` + `DOC` / `R"pydoc"`).
