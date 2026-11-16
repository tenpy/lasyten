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
| Eigen/LAPACK `BlockBackend` | Done (factory `"eigen"`; `"cpu"` stays Numpy for now — switch deferred) |
| Full pybind → nanobind `_core` | Done (LO trampolines only; DummyTensor tests skipped) |
| Phase 4 cleanup (gating, `nb::ndarray` boundary, `py::` inventory) | Done for first port (Numpy gate + Eigen `to_ndarray` / `block_from_ndarray`; default `"cpu"` → Eigen **deferred**, leftover #5) |
| pybind11 removed from liblasyten | Done: `include/` / `src/` use nanobind only, through the thin `lasyten::py` layer (`detail/python_api.h`, `detail/py_array.h`); no `find_package(pybind11)` |
| Stable ABI (abi3) | Done: `LASYTEN_STABLE_ABI` (default ON, Python >= 3.12), `_core.abi3.so`, `wheel.py-api = "cp312"`; CI in `.github/workflows/wheels.yml` (ubuntu + macOS micromamba abi3 wheels + `scripts/check_abi3_wheel.py`; optional Linux `cibuildwheel`) |
| `LASYTEN_WITH_PYTHON=OFF` Python-free | Partial: Eigen + Space/Leg/Tensors, `NoSymmetryBackend`, **`AbelianBackend`**, **`FusionTreeBackend`** (core paths green; OFF stub inventory in [convert_FusionTreeBackend.md](convert_FusionTreeBackend.md) / leftover #1), `fusion_tree_mapping`, krylov, and native `svd`/`qr`/`lq`/`eigh`/`eig*`/`entropy` (incl. Charged/HiddenLeg factors) build without libpython (`ctest -R no_python`); models and Numpy/ArrayApi/Torch still Python-gated |

## Motivations

- `nb::ndarray`: zero-copy NumPy / Torch / JAX / … via buffer + DLPack
- Stable ABI (abi3) for easier wheels — **done** (see "Stable ABI" below)
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

Status: **partially done**. All Python / nanobind use in liblasyten goes through
`include/lasyten/detail/python_api.h` (+ `detail/py_array.h`): with `LASYTEN_HAS_PYTHON=1` it
includes nanobind (with the stl casters) and defines the namespace `lasyten::py`, a thin
pybind11-flavoured layer (`py::object`, `py::tuple`, `py::list`, `py::dict`, `py::slice`,
`py::array`, `py::array_t<T>`, `py::cast`, `py::module_::import`, `py::value_error`, ...); with `0`
it provides *opaque, member-less* `py::` stand-ins (so declarations compile, but any real use, e.g.
`py::cast`, `py::array`, fails at compile time), plus real `std::runtime_error` subclasses for
`py::value_error` / `type_error` / `key_error` / `index_error`. Headers that are pure Python glue
(`numpy.h`, `array_api.h`, `torch.h`, `*_numpy.h`) `#error` out when OFF.

End state:

1. No `py::` / `nb::` in `include/lasyten/**` for the OFF build
2. Default CPU backend long-term = Eigen/LAPACK (no interpreter); `"cpu"` is chosen in exactly one
   place, `default_cpu_block_backend_shared()` (`block_backend.h`): **Numpy while built**, else Eigen.
   Do **not** flip `"cpu"` → Eigen yet (leftover #5): on the same BLAS, the pytest suite is ~6×
   slower under Eigen because it is tiny-block / densify-check dominated; NumPy’s small-array
   kernels still win. Keep `"eigen"` opt-in via `--block-backends eigen` / factory until Eigen’s
   small-block path (or the default workload) catches up.
3. Torch optional via `LASYTEN_WITH_TORCH` (libtorch), independent of Python
4. Numpy/ArrayApi backends gated behind `LASYTEN_WITH_NUMPY_BACKEND` (**done**, default ON with
   Python; forced OFF without). Removal + `"cpu"` → Eigen remain deferred (leftover #5)
5. CMake OFF → no `find_package(Python)`, no nanobind/pybind (**done**)
6. CTest suite that runs without embedding Python (**started**: `test_eigen_block_backend_no_python`, `test_symmetric_blocks_no_python`)

### Trying `-DLASYTEN_WITH_PYTHON=OFF`

```bash
cmake -S . -B /tmp/lasyten_off -G Ninja -DLASYTEN_WITH_PYTHON=OFF   # prints what is built / skipped
cmake --build /tmp/lasyten_off
ctest --test-dir /tmp/lasyten_off --output-on-failure
```

Needs only a C++ compiler, Eigen3, LAPACK/LAPACKE and HDF5 (no Python Development.Module/Embed,
nanobind or Torch). Result: `liblasyten` is built from `LASYTEN_PYTHON_FREE_SOURCES`
(`src/CMakeLists.txt`; core `BlockBackend`, `EigenBlockBackend`, dtypes, config, tools, HDF5 I/O
of blocks / sectors / symmetries, `FusionSymbol`, the whole symmetry stack (`Sector`, `Symmetry`,
all factors), fusion trees, tensor ops that do not touch `Tensor`/`Space` objects, ...; numpy
conversion helpers in e.g. `sector_numpy.cpp` and the `Space` / `Leg` / `Tensor` / backend parts of
`hdf5_export.cpp` are under `#if LASYTEN_HAS_PYTHON`). `LASYTEN_PYTHON_API_SOURCES` (spaces,
Tensor classes, TensorBackends, `krylov_based`, `backend_factory`, block-inds numpy glue, models,
`su2.cpp`, torch/numpy/array_api backends) are only built with Python.

Two ctests run without Python and `ldd` shows no libpython / torch:

- `test_eigen_block_backend_no_python`: Eigen block ops (elementwise, `tdot` contractions, SVD, QR,
  eigh), HDF5 round trip, `apply_axis_permutations`.
- `test_symmetric_blocks_no_python`: `U(1) x Z_3` `Symmetry` (validity, fusion, duals, sector
  dims) and a sector-wise block contraction + SVD on top of `EigenBlockBackend` (compared with the
  dense product of the assembled block-diagonal matrices).

(With Python ON the same sources run in `CommonCppTests` with an embedded interpreter, since
`Symmetry` reads factor attributes through Python to support Python subclasses.) Note liblasyten
is a static library: unresolved symbols are only reported for objects that something pulls in, so
*linking an Eigen-only program* working does not mean the library is complete.

Remaining blockers (for the real end state):

- `py::object` / `py::array` in the public API of `Tensor`, `SymmetricTensor`, `DiagonalTensor`,
  `Mask`, `TensorBackend`s, `Space` / `Leg` (spaces.cpp: `apply_basis_perm(py::object)`,
  `take_slice(py::array)`, `__repr__` of children via Python, `slices_to_py`), `Site` / `Coupling` /
  DOFs (models), `krylov_based` (`py::object` matvec; should become `std::function`).
- `BlockBackend` virtuals taking/returning `py::array` / `py::object` (`block_from_numpy`,
  `to_numpy`, `apply_leg_permutations`, ...) are only *guarded*, not replaced.
- `TorchBlockBackend` (`torch.cpp`) uses `py::array` for its boundary although libtorch itself
  is Python-free.
- `tests/ctest` programs other than the two Python-free ones embed Python
  (`<lasyten/embed.h>`: `ScopedInterpreter` + `init_embedded_nanobind()`, linking `Python::Python`).
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
| Infrastructure (not counted) | `detail/python_api.h`, `detail/py_array.h` (the `lasyten::py` layer over nanobind) |

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
  - [x] Numpy / ArrayApi marked **transitional** (`backend_factory.cpp`, docs); Eigen available as
        `"eigen"`, Torch for GPU; keep Numpy as default `"cpu"` for now (see leftover #5)
  - [x] CMake `LASYTEN_WITH_NUMPY_BACKEND` (default ON with Python): OFF drops `numpy.cpp` / `array_api.cpp`, their bindings and the Python classes (`lasyten.block_backends.WITH_NUMPY_BACKEND`), keeps Eigen
  - [x] `nb::ndarray` boundary for `EigenBlockBackend` (`pybind/block_backend/eigen_ndarray.h`: `block_from_ndarray`, zero-copy `to_ndarray`)
  - [x] `BlockBackend` / `Block` `block_from_ndarray` + `to_ndarray(writable=…)` dispatch (Eigen zero-copy; others via `to_numpy` copy); read-only export; Torch CPU import smoke test (leftover #10 progress)
  - [x] Native Numpy / Torch `block_from_ndarray` / `to_ndarray` (zero-copy NumPy view / pytorch-tagged DLPack; CUDA export for Torch)
  - [x] Non-CPU devices: Torch CUDA import/export (device-preserving); Eigen/Numpy reject non-CPU with `TypeError`
  - [x] JAX/TF-tagged returns via `to_ndarray(..., framework='jax'|'tensorflow'|'torch'|'numpy')`
  - [x] Models DOF public surfaces: ctors / properties / `*_numpy` factories use `nb::ndarray`
        interchange (zero-copy read-only views) vs owning `to_numpy` / `*_numpy` copies; C++
        storage stays `py::array` (Python-gated)
  - [ ] Default `"cpu"` → Eigen (**deferred**): suite is correct under Eigen, but ~6× slower on
        tiny-block / `to_numpy` check-heavy paths vs Numpy on the same BLAS; revisit after small-op
        tuning or when default workloads are large-block dominated (leftover #5)
  - [x] Remove pybind11 from `include/` / `src/` / CMake (no `find_package(pybind11)`, no `pybind11::` link, not in `environment.yml` / `pyproject.toml` build requires); `py::` is now `lasyten::py` over nanobind, the two bridges (`py_nanobind_bridge.h`, `py_pybind11_bridge.h`) are deleted
- [x] Stable ABI: `LASYTEN_STABLE_ABI`, `Development.SABIModule`, `nanobind-static-abi3` linked into liblasyten and `_core`, `nanobind_add_module(_core STABLE_ABI ...)`, `wheel.py-api = "cp312"`
- [x] Phase 5 scaffolding (first-port scope): OFF configures/builds Eigen subset without Python
  - [x] `include/lasyten/detail/python_api.h` as the single Python (nanobind) entry point + OFF stubs
  - [x] Root CMake OFF: no Python Development.Embed / pybind11 / nanobind / Torch, Eigen + LAPACK + HDF5, clear STATUS of the blockers
  - [x] Python-free pieces: Eigen block HDF5 save/load, `apply_axis_permutations`, `block_from_fusion_symbol` / `fusion_symbol_from_block` for Eigen, native repr, `warn`, `OptionValue`
  - [x] Python-free ctests (`test_eigen_block_backend_no_python`, `test_symmetric_blocks_no_python`, `hdf5_io_core_roundtrip`); Symmetry / Sector stack and their HDF5 I/O compile without Python
  - [x] Abelian + FusionTree backends without `py::` in the OFF lib (models / Numpy / Torch still gated; leftover #1 remainder)
  - [ ] Full ctest suite without embedded Python

## Status of the nanobind port

`pip install -e .` (scikit-build) builds the nanobind `_core`; `import lasyten` works. Python
test-suite (`tests/python_tests`): DummyTensor-based tests are **skipped** (Tensor trampoline
removed — leftover #3); Eigen BlockBackend pytest + no-Python ctest pass. With
`LASYTEN_WITH_NUMPY_BACKEND=OFF`, `"cpu"` resolves to Eigen. Embedding programs call
`lasyten::init_embedded_nanobind()`.

### Design decisions / deviations from the plan above

- **One framework: nanobind (Phase 4 done).** pybind11 cannot be compiled with `Py_LIMITED_API`,
  so it had to go from liblasyten to enable the Stable ABI. `include/lasyten/` and `src/` still
  *spell* `py::object`, `py::tuple`, `py::cast<T>(obj)`, ..., but `lasyten::py`
  (`detail/python_api.h`) is a thin shim over nanobind: `using` re-exports of nanobind types plus
  wrapper classes with the pybind11 semantics the code relies on (`py::tuple(n)` with item assignment,
  implicit conversions of `py::dict` / `py::list` / `py::str`, `py::slice` with `compute`, `py::module_::import`,
  `std::string` exception messages, ...). Differences to know about:
  `.cast<T>()` members became `py::cast<T>(obj)`; `py::cast<py::tuple|list|dict>` no longer converts
  (use `py::to_tuple` / `to_list` / `to_dict`); `cast<bool>` of `numpy.bool_` no longer works (use
  `py::to_bool`); `py::cast(lvalue)` of a bound class first looks up the existing Python instance
  (as pybind11 did; nanobind copies) via an extra `py::cast` overload; `py::object` / `py::handle`
  arguments accept `None` (`has_arg_defaults` specializations). `detail/py_array.h` implements
  `py::array`, `py::array_t<T>`, `py::dtype`, `py::buffer_info` on top of `nb::ndarray` and the
  Stable-ABI buffer protocol (it only uses the limited C API).
- **Constructors.** Classes deriving from `std::enable_shared_from_this` (Tensors, Space, Site,
  Symmetry factors, backends, ...) are created with `nb::new_` returning `std::shared_ptr<T>`
  (helpers `lasyten::ctor<decltype(cls), Args...>()` / `ctor_for<T, Args...>()` in
  `pybind/py_lasyten_nanobind.h`), *not* with placement-new `__init__`: an in-place instance has no
  owning `shared_ptr`, so `shared_from_this()` would throw `bad_weak_ptr`. Consequence: `__new__`
  receives the arguments of a Python subclass' `__init__`. Abstract classes (no trampoline) get a
  `__new__` that raises `TypeError`.
- **`None`.** nanobind rejects `None` for every parameter without `.none()`. Instead of annotating
  all of them, `has_arg_defaults<...>` is specialized (in `detail/python_api.h`, `detail/py_array.h`
  and `symmetries/spaces.h`) for `py::object` / `py::handle` / `py::array` / `py::dtype`,
  `std::shared_ptr<T>` and `Leg`, which restores the old behavior for them.
- **`*args` / `**kwargs`** are `nb::args` / `nb::kwargs` in the lambda signatures (they need an
  `nb::arg(...)` annotation each if any annotation is given); `pyargs()` / `pykwargs()` are now
  identity helpers kept for the existing call sites.
- **Casters** for `Leg` / `FusionSymbol` / the numpy adapter types live next to the types (`symmetries/spaces.h`,
  `symmetries/fusion_symbol.h`, `detail/py_array.h`) so that liblasyten and the bindings convert
  them identically; they must be visible in *every* TU (otherwise nanobind silently uses the
  generic caster and fails at runtime).
- **Iterators** from `nb::make_iterator` are registered as private `_<Name>Iterator` classes in the scope of the
  container class (they appear lazily on first use; the underscore keeps the stub parity test happy).
- **Stable ABI.** `LASYTEN_STABLE_ABI` (default ON; needs `LASYTEN_WITH_PYTHON` and CPython >= 3.12, else it is
  silently off): CMake finds `Python ... Development.SABIModule`, builds `nanobind-static-abi3`
  (`Py_LIMITED_API=0x030C0000`, PUBLIC), links it into `lasyten` and `_core`, and calls
  `nanobind_add_module(_core STABLE_ABI ...)`; the result is `lasyten/_core.abi3.so`. `pyproject.toml`
  sets `wheel.py-api = "cp312"` (tag `cp312-abi3`). All Python C-API calls in liblasyten (`warn.h`,
  `py_array.h`) are in the limited API. Embedding programs link `Python::Python` (Development.Embed) and use
  `<lasyten/embed.h>` (`ScopedInterpreter`, `init_embedded_nanobind()`).
- `Sector` / `SectorArray` / `BlockInds`: `nb::implicitly_convertible<py::array, ...>` (calls the
  Python constructor, so lists / tuples work too).

### Known gaps / behavior changes

- `Tensor` (abstract) can no longer be subclassed from Python (no trampoline, as planned):
  `DummyTensor` in `tests/python_tests/test_tensors.py` (used by `test_base_Tensor`,
  `test_Tensor_ascii_diagram`, `test_LegLabels_api`, `test_is_scalar`) fails with a `TypeError`.
  Restore a `Tensor` trampoline (leftover #3) or rewrite those tests.
- nanobind's `bool` caster only accepts `True` / `False`, **not `numpy.bool_`** (pybind11 did).
  Python callers must use `bool(x)`; inside liblasyten `py::to_bool(obj)` is used for results of numpy calls.
- `py::array` is still used in many binding signatures (IDE `.pyi` types it as
  `numpy.ndarray` once stubgen loads the correct `_core`; true `py::object`
  boundaries remain `object` / `Any` until leftover #10).
- pybind11 is no longer a build requirement (neither ON nor OFF).

## Open points / leftover todos

1. **`LASYTEN_WITH_PYTHON=OFF` truly Python-free** — **NoSymmetry + Abelian + FusionTree (partial) on Eigen work without libpython.**
   With OFF, liblasyten includes Space / Leg, the Tensor classes (`Symmetric` / `Diagonal` / `Mask` /
   `Charged`), `NoSymmetryBackend`, **`AbelianBackend`**, **`FusionTreeBackend`** (in
   `LASYTEN_PYTHON_FREE_SOURCES`; native helpers + `apply_instructions` / `permute_legs` / dense
   block I/O), **`fusion_tree_mapping`**, **`krylov_based`** (typed `KrylovOptions`),
   `decompositions` (native `svd` / `qr` / `lq` / `eigh` / `eig*` / `entropy`, including
   Charged/HiddenLeg factors for svd/qr/lq), and `backend_factory` (`"no_symmetry"`, `"abelian"`,
   and `"fusion_tree"` + Eigen). `ctest -R no_python` passes in an OFF build
   (`cmake -S . -B build_off -G Ninja -DLASYTEN_WITH_PYTHON=OFF && ninja -C build_off &&
   ctest --test-dir build_off -R no_python`).
   **FusionTree OFF** (issue #326 item 3; table in
   [convert_FusionTreeBackend.md](convert_FusionTreeBackend.md)): OFF-native —
   `combine_legs`, `compose`, `linear_combination`, `svd` / `qr` / `lq`,
   `truncate_singular_values` (no custom `basis_perm`), `get_element_mask`,
   `diagonal_from_block` / `diagonal_tensor_to_block`. Still OFF stubs —
   `partial_compose`, `outer`, `scale_axis`, `partial_trace`, `get_element`,
   `from_tree_pairs` / `from_grid`, `diagonal_to_mask`, `_mask_contract`, `mask_*`
   (Python Space/NumPy glue when ON).
   **Still in `LASYTEN_PYTHON_API_SOURCES`:** `block_inds_numpy.cpp` only (NumPy casting for
   `BlockInds`). Models / `su2.cpp` and Numpy / ArrayApi / Torch stay in separate Python-gated
   `target_sources` (not shrunk further while models keep `py::array` storage).
2. **Stable ABI / abi3 wheels** — **done** (`LASYTEN_STABLE_ABI`, `_core.abi3.so`, `wheel.py-api = "cp312"`).
   CI (issue #326 item 4): `.github/workflows/wheels.yml` builds Release `cp312-abi3` wheels on
   **ubuntu-latest** and **macos-latest** via micromamba (`python -m build --no-isolation`),
   checks the tag / `_core.abi3.*` with `scripts/check_abi3_wheel.py`, and smoke-imports.
   `[tool.cibuildwheel]` in `pyproject.toml` drives an optional Linux `cibuildwheel` job
   (`continue-on-error` until manylinux Torch/HDF5 bundling is fully hardened).
3. **Restore trampolines** (optional) for `Tensor`, `BlockBackend`, `TensorBackend`, `Symmetry`, `Site`, … if downstream needs Python subclassing again.
4. **`unique_ptr` where ownership is unique** — audit `shared_ptr`-everywhere; migrate carefully without breaking `shared_from_this`.
5. **Fate of NumpyBlockBackend / ArrayApiBlockBackend + default `"cpu"`** — **deferred.** Both stay:
   Numpy remains default `"cpu"` while `LASYTEN_WITH_NUMPY_BACKEND` is ON; Eigen is opt-in
   (`"eigen"`). Reason: full pytest under Eigen is ~6× slower than Numpy despite sharing conda
   BLAS/LAPACK (tiny-block allocate/dispatch + densify checks dominate; NumPy C ufuncs win).
   Revisit switching `"cpu"` / removing Numpy only after Eigen small-block performance is
   competitive (or workloads shift to large blocks / Python-free embeds). Gate already exists;
   do not flip the factory default yet.
6. **Eigen numerical parity** — SVD/QR/eigh vs SciPy `gesdd`/`gesvd`; document defaults and tolerances for TeNPy.
7. **GPU strategy** — Torch-only vs future CUDA Eigen; `nb::ndarray` device tags vs factory strings.
8. **Implicit conversions / caster ergonomics** — Sector / BlockInds / Leg may need explicit Python helpers.
9. **Stubgen / typing** — **done.** Two pipelines:
    - **Sphinx bag-of-names stub:** `scripts/generate_core_stubs.py` (parses `nb::`
      bindings) checked by `scripts/check_core_stub_parity.py` /
      `test_core_stub_parity` against the live extension (prefers `_core.abi3.so`).
    - **IDE `.pyi`:** `scripts/generate_core_pyi.py` + optional
      `scripts/stubgen_patterns.txt`, wired from CMake with
      `--so $<TARGET_FILE:_core>` (do **not** use `PYTHONPATH=build` + `-m _core`:
      a leftover `_core.cpython-*.so` shadows abi3 and types `py::array` as
      `typing.Any`). Regression: `test_core_pyi_array_typing`. Remaining
      `object`/`Any` on true `py::object` APIs is leftover #10 (`nb::ndarray`).
10. **Dual-framework zero-copy / `nb::ndarray` everywhere** — **Eigen + Numpy + Torch paths done**, plus abstract
    `BlockBackend` dispatch:
    - Import: `block_from_ndarray` on `EigenBlockBackend` and `BlockBackend` (Eigen copies from any
      CPU DLPack/buffer array; Numpy/Torch copy into their storage; others fall back to
      `block_from_numpy`). Unsupported dtypes raise `TypeError`. Torch CPU smoke test in
      `test_eigen_block_backend.py`; native Numpy/Torch roundtrips in `test_ndarray_backends.py`.
    - Export: `to_ndarray(writable=True|False)` on Eigen / Numpy / Torch backend+block and on abstract
      `BlockBackend` / `Block` (writable = zero-copy alias; `writable=False` = read-only view;
      other backends = owning NumPy copy). Torch export is pytorch-tagged (CPU or CUDA DLPack).
      Documented dual path vs `to_numpy()` (always owning NumPy copy).
    - Stubgen patterns for the new methods in `scripts/stubgen_patterns.txt`.
    - `Sector` / `SectorArray` / `BlockInds` `.to_numpy()` (and related index helpers) now return
      owning `nb::ndarray` via `pybind/ndarray_utils.h`.
    - Space / Leg property helpers (`basis_perm`, multiplicities, slices, …) export via
      `nb::ndarray`. `Tensor` / `Mask` / `DiagonalTensor.to_numpy` / `diagonal_as_numpy` stay
      owning NumPy copies (do not route through `block_to_ndarray`, which is torch-tagged for
      Torch and broke `np.*` callers).
    - Non-CPU: Torch `block_from_ndarray` accepts CUDA DLPack sources without a host bounce;
      `to_ndarray` returns pytorch-tagged CUDA views. Eigen/Numpy stay host-only and raise
      `TypeError` on non-CPU inputs (`require_cpu_device`).
    - Framework tags: `to_ndarray(..., framework='numpy'|'torch'|'jax'|'tensorflow')` retags the
      same buffer for the chosen DLPack consumer (defaults: numpy for Eigen/Numpy, torch for Torch).
    - Models DOF: binding ctors accept CPU `nb::ndarray`; properties export zero-copy read-only
      views; `to_numpy` / `get_*_numpy` / creation factories return owning NumPy copies. C++
      members remain `py::array` (models stay Python-gated).
    TeNPy follow-up: `~/code/tenpy_v2` sets EffectiveH `dtype` after LO init and treats dense
    `to_numpy` as read-only; MPS still uses `block_from_numpy` (Numpy `block_from_ndarray`
    not yet stable there) — see [convert_nanobind_tenpy.md](convert_nanobind_tenpy.md).
11. **TeNPy `MPOTransferMatrix`** — still `NotImplementedError` on the TeNPy side. lasyten
    trampolines for `LinearOperator` (`matvec` / `to_tensor` / `adjoint`, `nb::dynamic_attr`)
    and `PlanarLinearOperator` (same three virtuals; `adjoint` must live on the *derived*
    trampoline or C++ wrappers hit `LinearOperator::adjoint` and throw) are in place for
    `TransferMatrix` / `EffectiveH` / future `MPOTransferMatrix`.
12. **Docs drift** — refresh notes that still describe pybind `smart_holder` / old MI.
