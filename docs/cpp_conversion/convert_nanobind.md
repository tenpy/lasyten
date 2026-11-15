# Port from pybind11 to nanobind (#255)

Master note for the binding-layer migration and the Eigen/LAPACK BlockBackend
work that unlocks zero-copy `nb::ndarray` exchange. Related: issue
[#255](https://github.com/tenpy/lasyten/issues/255), MI cleanup
[#254](https://github.com/tenpy/lasyten/issues/254) (done).

Eigen backend details: [convert_EigenBlockBackend.md](convert_EigenBlockBackend.md).
BlockBackend status: [convert_BlockBackend.md](convert_BlockBackend.md).
TeNPy downstream: [convert_nanobind_tenpy.md](convert_nanobind_tenpy.md).

## Status

**In progress** on branch `nanobind_refactor`.

| Milestone | State |
| --- | --- |
| #254 domain MI removed | Done |
| Conversion docs for #255 | Done (this file + siblings) |
| nanobind in CMake / pyproject | Done (`NB_MODULE`, nanobind in env/pyproject) |
| Eigen/LAPACK `BlockBackend` | Done (factory `"eigen"`; `"cpu"` still Numpy while Numpy is built) |
| Full pybind → nanobind `_core` | Done (LO trampolines only; DummyTensor tests skipped) |
| Phase 4 cleanup (gating, `nb::ndarray` boundary, `py::` inventory) | Done for first port (Numpy gate + Eigen `to_ndarray` / `block_from_ndarray`; default `"cpu"` switch is leftover) |
| `LASYTEN_WITH_PYTHON=OFF` Python-free | Scaffolding done: configures without Python, builds Eigen subset, Eigen ctest passes; full library still has Python-only TUs (leftover #1) |

## Motivations

- `nb::ndarray`: zero-copy NumPy / Torch / JAX / … via buffer + DLPack
- Stable ABI (abi3) for easier wheels — **follow-up**, not first merge
- Lower call overhead, faster compiles, smaller binaries
- Porting guide: https://nanobind.readthedocs.io/en/latest/porting.html

**Important:** `nb::ndarray` is an **interchange** type only (shape / dtype /
device / `data()`). It does **not** implement SVD, QR, matmul, tensordot, etc.
Compute lives in a real backend → native Eigen/LAPACK ([convert_EigenBlockBackend.md](convert_EigenBlockBackend.md)).

## #254 MI (cleared)

Nanobind does not support binding C++ multiple inheritance. The blockers from
#254 are gone:

| Old | New |
| --- | --- |
| `Tensor : LabelledLegs, VectorLike, enable_shared_from_this` | `Tensor : VectorLike` + composed `LegLabels` |
| Site DOF diamonds (`SpinHalfFermionSite : SpinDOF, FermionicDOF`) | `Site` + composed DOFs |
| Spaces diamonds (`ElementarySpace` / `AbelianLegPipe`) | Linear hierarchy; value-type `Leg` — [convert_spaces_MI.md](convert_spaces_MI.md) |

Remaining `X : Y, enable_shared_from_this<X>` is fine (only `Y` is the Python base).

## Removed / changed pybind features that still matter

| Feature | Impact | Policy |
| --- | --- | --- |
| Bound C++ MI | Was the #255 hard blocker | Cleared by #254 |
| Holders (`smart_holder`) | Widespread in `pybind/` | Drop holders; exchange `std::shared_ptr<T>` |
| `trampoline_self_life_support` | All polymorphic trampolines | Nanobind `NB_TRAMPOLINE` / `NB_OVERRIDE*` |
| `py::array` | Deep in `include/` + Numpy backend | Replace boundary with `nb::ndarray` / Eigen storage |
| Custom type casters | Sector, BlockInds, Leg, FusionSymbol | Rewrite to nanobind caster API |
| Implicit conversions | Sector / BlockInds from arrays | Prefer explicit overloads / `init_implicit` |
| Custom ctors returning values | Many factory-as-init patterns | Placement-new `__init__` |
| `None` arguments | Default-allowed in pybind | Opt-in `nb::arg("x").none()` |
| `native_enum` + monkey-patch (`Dtype`) | `py_dtypes.cpp` | Redesign under nanobind |
| Module-local / metaclasses / embed | Unused | N/A |

## Ownership: `shared_ptr` (no holders)

- Bind with `nb::class_<T>` (no holder type).
- APIs that need shared ownership take/return `std::shared_ptr<T>`
  (`#include <nanobind/stl/shared_ptr.h>`).
- Keep `enable_shared_from_this`; always cross the boundary as `shared_ptr`
  when `shared_from_this()` may be used.
- Broad `unique_ptr` migration is a **leftover** (see below).

## Trampolines (minimal set)

TeNPy `v2_main` subclasses only Krylov-related C++ types (see
[convert_nanobind_tenpy.md](convert_nanobind_tenpy.md)).

| Keep | Drop initially |
| --- | --- |
| `PyLinearOperator`, `PyPlanarLinearOperator` | `PyTensor*`, `PyBlockBackend*`, `PySymmetry*`, `PySite*`, `PySpace*`, `PyTensorBackend*`, … |

Python subclassing of dropped types is **unsupported** until trampolines are
restored (leftover #3).

## Target architecture

```text
NumPy / Torch / JAX  <--+-- nb::ndarray (bindings only)
                        |
                   EigenBlockBackend (CPU, native LAPACK)
                   TorchBlockBackend (GPU / ATen; optional)
                        |
                   Tensor / Symmetry / Models (liblasyten)
                        |
                   Lanczos / Arnoldi  <-- trampoline matvec -- tenpy LinearOperator
```

## Long-term: `LASYTEN_WITH_PYTHON=OFF` without Python

**Not part of the first port merge.** Design toward it.

Status: **scaffolding is in place**, the end state is not reached. All Python / pybind11 /
nanobind use in liblasyten goes through one header, `include/lasyten/detail/python_api.h`:
with `LASYTEN_HAS_PYTHON=1` it includes pybind11 (+ the nanobind bridge) as before; with `0` it
provides *opaque, member-less* `py::` stand-ins (so declarations compile, but any real use, e.g.
`.cast<T>()`, `py::array`, `py::cast`, fails at compile time), plus real `std::runtime_error`
subclasses for `py::value_error` / `type_error` / `key_error` / `index_error`. Headers that are
pure Python glue (`numpy.h`, `array_api.h`, `torch.h`, `*_numpy.h`) `#error` out when OFF.

End state:

1. No `py::` / `nb::` in `include/lasyten/**` for the OFF build
2. Default CPU backend = Eigen/LAPACK (no interpreter); `"cpu"` is chosen in exactly one place,
   `default_cpu_block_backend_shared()` (`block_backend.h`): Numpy while built, else Eigen
3. Torch optional via `LASYTEN_WITH_TORCH` (libtorch), independent of Python
4. Numpy/ArrayApi backends gated behind `LASYTEN_WITH_NUMPY_BACKEND` (**done**, default ON with
   Python; forced OFF without) and eventually removed (leftover #5)
5. CMake OFF → no `find_package(Python)`, no nanobind/pybind (**done**)
6. CTest suite that runs without embedding Python (**started**: `test_eigen_block_backend_no_python`)

### Trying `-DLASYTEN_WITH_PYTHON=OFF`

```bash
cmake -S . -B /tmp/lasyten_off -G Ninja -DLASYTEN_WITH_PYTHON=OFF   # prints the blocker summary
cmake --build /tmp/lasyten_off
ctest --test-dir /tmp/lasyten_off --output-on-failure
```

Needs only a C++ compiler, Eigen3, LAPACK/LAPACKE and HDF5 (no Python Development.Module/Embed,
pybind11, nanobind or Torch). Result: `liblasyten` is built from `LASYTEN_PYTHON_FREE_SOURCES`
(`src/CMakeLists.txt`; core `BlockBackend`, `EigenBlockBackend`, dtypes, config, tools, HDF5 I/O
of blocks, `FusionSymbol`, symmetry classes, fusion trees, tensor ops that do not touch
`Tensor`/`Space` objects, ...). `LASYTEN_PYTHON_API_SOURCES` (spaces, Tensor classes,
TensorBackends, `krylov_based`, `backend_factory`, sector/block-inds numpy glue, `hdf5_export.cpp`,
models, `su2.cpp`, torch/numpy/array_api backends) are only built with Python. The ctest links the
Eigen backend, HDF5 I/O and `apply_axis_permutations` against this subset and `ldd` shows no
libpython / torch. Note it is a static library: unresolved symbols are only reported for
objects that something pulls in, so *linking an Eigen-only program* working does not mean the
library is complete.

Remaining blockers (for the real end state):

- `py::object` / `py::array` in the public API of `Tensor`, `SymmetricTensor`, `DiagonalTensor`,
  `Mask`, `TensorBackend`s, `Space` / `Leg` (spaces.cpp: `apply_basis_perm(py::object)`,
  `take_slice(py::array)`, `__repr__` of children via Python, `slices_to_py`), `Site` / `Coupling` /
  DOFs (models), `krylov_based` (`py::object` matvec; should become `std::function`).
- `BlockBackend` virtuals taking/returning `py::array` / `py::object` (`block_from_numpy`,
  `to_numpy`, `apply_leg_permutations`, ...) are only *guarded*, not replaced.
- `TorchBlockBackend` (`torch.cpp`) uses `py::array` for its boundary although libtorch itself
  is Python-free.
- `tests/ctest` programs other than the Eigen one embed Python (`pybind11::embed`).
- `OptionValue` / config are Python-free, but `lasyten.config` Python-side options and the
  backend factory overload taking `py::object` still need the interpreter.

Tracked in leftover #1.

### `py::` inventory (`include/lasyten/`, `rg -c 'py::'`, ~261 mentions excluding the stubs)

| Group | Headers (count) |
| --- | --- |
| Models | `models/degrees_of_freedom.h` (46), `models/sites.h` (29), `models/couplings.h` (14) |
| Python-dispatch / glue backends (guarded; `#error` when OFF) | `block_backend/array_api.h` (24), `numpy.h` (16), `torch.h` (10), `symmetries/sector_numpy.h` (4), `backends/block_inds_numpy.h` (2) |
| `BlockBackend` interface | `block_backend/block_backend.h` (21, guarded), `eigen.h` (11, guarded), `dtypes.h` (7, guarded) |
| Tensors / Krylov | `tensors/krylov_based.h` (13), `tensors/tensor.h` (8), `tensors/mask.h` (5), `diagonal_tensor.h` (2), `symmetric_tensor.h` (1), `sparse.h` (1) |
| TensorBackends | `backends/tensor_backend.h` (9), `fusion_tree_backend.h` (7), `fusion_tree_mapping.h` (5), `no_symmetry.h` (2), `abelian.h` (2), `backend_factory.h` (1, guarded) |
| Config / tools / symmetries | `config.h` (7, mostly guarded), `symmetries/spaces.h` (6), `tools.h` (3, guarded), `tools/hdf5_export.h` (2, guarded), `symmetries/fusion_symbol.h` (2, guarded) |
| Infrastructure (not counted) | `detail/python_api.h` (8), `py_nanobind_bridge.h` (7) |

(Counts are mentions, "guarded" = already behind `#if LASYTEN_HAS_PYTHON`, so they do not
block the OFF build.)

## Phased checklist

- [x] Docs: this file, Eigen note, tenpy downstream, BlockBackend update, README links
- [x] Phase 0: nanobind in build (`environment.yml`, `pyproject.toml`, `find_package(nanobind)`, `nanobind_add_module(_core)`); the whole `_core` is `NB_MODULE` (Option A). `nb::ndarray` smoke is not done: `py::array` is bridged instead (see status below)
- [x] Phase 1: Eigen/LAPACK BlockBackend + contract tests (`"eigen"` factory; CTest + pytest)
- [x] Phase 2: core types with `shared_ptr` (no holders); only `LinearOperator` / `PlanarLinearOperator` trampolines (`NB_TRAMPOLINE`); all other trampoline classes deleted (headers are empty stubs)
- [x] Phase 3: remaining `_core` bindings, casters (`Leg`, `FusionSymbol`, pybind object types), `Dtype` / styles as `nb::enum_`, stubs (`nanobind.stubgen` + `scripts/generate_core_stubs.py` understands `nb::`); `py::array` -> `nb::ndarray` conversion of the binding signatures is **not** done (leftover #10)
- [x] Phase 4 (first-port scope): gate Python-dispatch backends; Eigen `nb::ndarray` helpers; inventory remaining `py::`
  - [x] `py::` inventory by header (above) and leftover #1 list
  - [x] Numpy / ArrayApi marked **transitional** (`backend_factory.cpp`, docs); Eigen (`"eigen"`) preferred CPU, Torch for GPU
  - [x] CMake `LASYTEN_WITH_NUMPY_BACKEND` (default ON with Python): OFF drops `numpy.cpp` / `array_api.cpp`, their bindings and the Python classes (`lasyten.block_backends.WITH_NUMPY_BACKEND`), keeps Eigen
  - [x] `nb::ndarray` boundary for `EigenBlockBackend` (`pybind/block_backend/eigen_ndarray.h`: `block_from_ndarray`, zero-copy `to_ndarray`)
  - [ ] `py::array` in the *remaining* binding signatures / `BlockBackend::to_numpy` -> `nb::ndarray` (leftover #10)
  - [ ] Default `"cpu"` -> Eigen (deliberately not switched; test-suite parity first)
  - [ ] Remove pybind11 from `include/` / `src/` (dual-framework bridge still needed)
- [x] Phase 5 scaffolding (first-port scope): OFF configures/builds Eigen subset without Python
  - [x] `include/lasyten/detail/python_api.h` as the single pybind11 entry point + OFF stubs
  - [x] Root CMake OFF: no Python Development.Embed / pybind11 / nanobind / Torch, Eigen + LAPACK + HDF5, clear STATUS of the blockers
  - [x] Python-free pieces: Eigen block HDF5 save/load, `apply_axis_permutations`, `block_from_fusion_symbol` / `fusion_symbol_from_block` for Eigen, native repr, `warn`, `OptionValue`
  - [x] Python-free ctest (`test_eigen_block_backend_no_python`, `hdf5_io_core_roundtrip`)
  - [ ] Tensors / spaces / TensorBackends / models without `py::` (blockers above — leftover #1)
  - [ ] Full ctest suite without embedded Python

## Status of the nanobind port

`pip install -e .` (scikit-build) builds the nanobind `_core`; `import lasyten` works. Python
test-suite (`tests/python_tests`): DummyTensor-based tests are **skipped** (Tensor trampoline
removed — leftover #3); Eigen BlockBackend pytest + no-Python ctest pass. With
`LASYTEN_WITH_NUMPY_BACKEND=OFF`, `"cpu"` resolves to Eigen. Embedding programs call
`lasyten::init_embedded_nanobind()`.

### Design decisions / deviations from the plan above

- **Dual-framework bridge (Phase 4 not done).** `include/lasyten/` and `src/` still use `py::`.
  `include/lasyten/py_nanobind_bridge.h` (+ `py_nanobind_bridge_types.inc`, the list of all bound
  classes) specializes the *pybind11* `type_caster` of every nanobind-bound class (`T`,
  `shared_ptr<T>`, `shared_ptr<const T>`, enums) so liblasyten's `.cast<T>()`, `py::cast(...)`
  keep working; use `lasyten::py_isinstance<T>(obj)` / `lasyten::py_type_of<T>()` instead of
  `py::isinstance<T>` / `py::type::of<T>` (those only know pybind11 types). Add new bound classes
  (also nested ones) to the `.inc` / `LASYTEN_PY_BRIDGE_CLASS(...)`.
  `pybind/py_pybind11_bridge.h` is the opposite direction: nanobind casters for `py::object`,
  `py::array`, ... so binding lambdas can keep using them (reinterpretation of `PyObject*`).
  Exceptions are translated both ways (`py::value_error` -> `ValueError` etc.).
- **Constructors.** Classes deriving from `std::enable_shared_from_this` (Tensors, Space, Site,
  Symmetry factors, backends, ...) are created with `nb::new_` returning `std::shared_ptr<T>`
  (helpers `lasyten::ctor<decltype(cls), Args...>()` / `ctor_for<T, Args...>()` in
  `pybind/py_lasyten_nanobind.h`), *not* with placement-new `__init__`: an in-place instance has no
  owning `shared_ptr`, so `shared_from_this()` would throw `bad_weak_ptr`. Consequence: `__new__`
  receives the arguments of a Python subclass' `__init__`. Abstract classes (no trampoline) get a
  `__new__` that raises `TypeError`.
- **`None`.** nanobind rejects `None` for every parameter without `.none()`. Instead of annotating
  all of them, `has_arg_defaults<...>` is specialized (in `pybind/py_pybind11_bridge.h`) for the
  pybind object types, `std::shared_ptr<T>` and `Leg`, which restores the old behavior for them.
- **`*args` / `**kwargs`** are `nb::args` / `nb::kwargs` in the lambda signatures (they need an
  `nb::arg(...)` annotation each if any annotation is given) and are converted to `py::args` /
  `py::kwargs` at the top of the body (`pyargs()`, `pykwargs()`).
- **Casters** that must be visible in *every* binding TU (`Leg`, `FusionSymbol`) are included via
  `py_lasyten_nanobind.h`; otherwise nanobind silently uses the generic caster and fails at runtime.
- `Sector` / `SectorArray` / `BlockInds`: `nb::implicitly_convertible<py::array, ...>` (calls the
  Python constructor, so lists / tuples work too).

### Known gaps / behavior changes

- `Tensor` (abstract) can no longer be subclassed from Python (no trampoline, as planned):
  `DummyTensor` in `tests/python_tests/test_tensors.py` (used by `test_base_Tensor`,
  `test_Tensor_ascii_diagram`, `test_LegLabels_api`, `test_is_scalar`) fails with a `TypeError`.
  Restore a `Tensor` trampoline (leftover #3) or rewrite those tests.
- nanobind's `bool` caster only accepts `True` / `False`, **not `numpy.bool_`** (pybind11 did).
  Callers must use `bool(x)`; a few call sites in the tests were adjusted.
- `py::array` is still used in the binding signatures (stub parity / typing shows `typing.Any`).
- The pybind11 headers are still a build requirement when `LASYTEN_WITH_PYTHON=ON` (leftover #1); with OFF they are not.

## Open points / leftover todos

1. **`LASYTEN_WITH_PYTHON=OFF` truly Python-free** — CMake, stub header, INSTALL notes and a first
   Python-free ctest are done; what is left is the header decoupling of the inventory above
   (models, Tensor / Space / TensorBackend APIs, `krylov_based`, `torch.cpp`) and moving
   `LASYTEN_PYTHON_API_SOURCES` into `LASYTEN_PYTHON_FREE_SOURCES` one by one.
2. **Stable ABI / abi3 wheels** — once nanobind port is green.
3. **Restore trampolines** (optional) for `Tensor`, `BlockBackend`, `TensorBackend`, `Symmetry`, `Site`, … if downstream needs Python subclassing again.
4. **`unique_ptr` where ownership is unique** — audit `shared_ptr`-everywhere; migrate carefully without breaking `shared_from_this`.
5. **Fate of NumpyBlockBackend / ArrayApiBlockBackend** — gated behind `LASYTEN_WITH_NUMPY_BACKEND` and marked transitional; still to decide: remove for good (and switch `"cpu"` to Eigen).
6. **Eigen numerical parity** — SVD/QR/eigh vs SciPy `gesdd`/`gesvd`; document defaults and tolerances for TeNPy.
7. **GPU strategy** — Torch-only vs future CUDA Eigen; `nb::ndarray` device tags vs factory strings.
8. **Implicit conversions / caster ergonomics** — Sector / BlockInds / Leg may need explicit Python helpers.
9. **Stubgen / typing** — nanobind stub generation vs `scripts/check_core_stub_parity.py`.
10. **Dual-framework zero-copy edge cases / `nb::ndarray` everywhere** — done for `EigenBlockBackend` only (copying import, zero-copy writable export; unsupported dtypes raise `TypeError`). Open: other binding signatures still use `py::array`, read-only DLPack exports, non-CPU devices, JAX/TF return paths, `to_numpy()` of the other backends.
11. **TeNPy `MPOTransferMatrix`** — still `NotImplementedError`; when implemented, needs LinearOperator trampoline.
12. **Docs drift** — refresh notes that still describe pybind `smart_holder` / old MI.
