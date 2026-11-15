# Conversion of BlockBackend / NumpyBlockBackend (and Block, Scalar)

## Status

**Done (pybind11 era).** The abstract `BlockBackend`, concrete `NumpyBlockBackend`,
and the supporting `Block` / `Scalar` types live in C++. Python imports them from
`lasyten._core` via `lasyten/block_backends/__init__.py`.

**Follow-up (#255):** migrate bindings to nanobind, add native
[Eigen/LAPACK BlockBackend](convert_EigenBlockBackend.md), replace `py::array`
boundaries with `nb::ndarray` / owning Eigen storage, and make
`LASYTEN_WITH_PYTHON=OFF` viable. See [convert_nanobind.md](convert_nanobind.md).

| Backend | Storage / compute | #255 direction |
| --- | --- | --- |
| `NumpyBlockBackend` | `py::array` + Python NumPy/SciPy | **Transitional**; CMake option `LASYTEN_WITH_NUMPY_BACKEND` (default ON with Python) |
| `TorchBlockBackend` | `torch::Tensor` + ATen | Keep for GPU |
| `ArrayApiBlockBackend` | `py::object` + Array API | **Transitional**; same CMake option as Numpy |
| `EigenBlockBackend` | Owning buffer + Eigen/LAPACK | **Preferred CPU backend** (`"eigen"`); `"cpu"` is still Numpy while that is built |

`LASYTEN_WITH_NUMPY_BACKEND=OFF` drops `numpy.cpp` / `array_api.cpp` (and their bindings and the
Python classes `NumpyBlockBackend` / `ArrayApiBlockBackend`); `"cpu"` then resolves to Eigen
(`default_cpu_block_backend_shared()` in `block_backend.h` is the single switch point). With
`LASYTEN_WITH_PYTHON=OFF` the option is forced OFF. The `py::`-typed virtuals of `BlockBackend`
(`block_from_numpy`, `to_numpy`, `apply_leg_permutations`, ...) are compiled out in that
configuration; the Python-free replacement for the latter is `apply_axis_permutations`.

Related notes: [convert_Spaces.md](convert_Spaces.md) — `apply_basis_perm` now uses C++ `Leg`.

## Metadata

| Object | Original Python | C++ header | C++ source | Bindings |
| --- | --- | --- | --- | --- |
| `BlockBackend` | `lasyten/block_backends/_block_backend.py` | `include/lasyten/block_backend/block_backend.h` | `src/block_backend/block_backend.cpp` | `pybind/block_backend/py_block_backend.cpp` |
| `Block` | TypeVar / `np.ndarray` (numpy) | nested `BlockBackend::Block` | same + `numpy.cpp` for numpy impl | nested as `BlockBackend.BlockCls` |
| `Scalar` | *(new on C++ side)* | nested `BlockBackend::Scalar` | `block_backend.cpp` | nested as `BlockBackend.Scalar` |
| `NumpyBlockBackend` | `lasyten/block_backends/numpy.py` | `include/lasyten/block_backend/numpy.h` | `src/block_backend/numpy.cpp` | `pybind/block_backend/py_numpy.cpp` |
| `Dtype` | `lasyten/block_backends/dtypes.py` | `include/lasyten/block_backend/dtypes.h` | `src/block_backend/dtypes.cpp` | `pybind/block_backend/py_dtypes.cpp` |

Trampolines (for Python subclasses of the ABCs): `pybind/block_backend/py_trampolines.hpp` (`PyBlockBackend`, `PyBlock`).

Python wiring:

```python
# lasyten/block_backends/__init__.py
from .._core import BlockBackend, NumpyBlockBackend

Block = BlockBackend.BlockCls
Scalar = BlockBackend.Scalar
```

## Why Block and Scalar were added in C++

In Python, a “block” for the numpy backend was simply an `np.ndarray`, and scalar-valued operations returned plain `float` / `complex` / `bool`. That does not translate cleanly to a typed C++ API shared across backends:

1. **`BlockBackend::Block`** — Abstract dense-array handle. Ownership is `std::shared_ptr` (`BlockPtr` / `BlockCPtr`), with `enable_shared_from_this`. Elementwise arithmetic, comparisons, indexing, HDF5, and `to_numpy()` live on the block (or on the backend operating on blocks). Concrete storage is backend-specific; for numpy that is `NumpyBlockBackend::Block` wrapping a `py::array`.

2. **`BlockBackend::Scalar`** — Typed scalar with a `Dtype`, implemented as a wrapper around a **0-d `Block`**. Arithmetic, comparisons, `real`/`imag`/`abs`/`sqrt`/`exp`/`log`/`pow`, and HDF5 go through the backend via that block. Accessors (`as_float64`, `as_complex128`, `as_bool`, …) cast to C++ primitives when needed. This replaces ad-hoc `float | complex` return types in the C++ interface and keeps dtype information.

Python still accepts `numbers.Number` in many tensor APIs; those call sites were extended to also accept `Scalar` (e.g. `isinstance(other, (Number, Scalar))`).

## Design decisions worth remembering

- **Nested types.** `Block` and `Scalar` are nested under `BlockBackend` (exposed in Python as `BlockBackend.BlockCls` and `BlockBackend.Scalar`). `NumpyBlockBackend::Block` subclasses `BlockBackend::Block`.
- **`py::smart_holder`.** Bindings use `py::class_<T, …, py::smart_holder>` so `shared_ptr` / trampolines work with pybind11 v3.
- **Factory, not free construction.** `NumpyBlockBackend::from_factory(device)` (and `from_factory_shared`) nearly-singletons backends per device; public ctor is protected. Base `BlockBackend::from_factory` dispatches by device string.
- **Scalar-valued backend methods return `Scalar`.** Examples: `item`, `max`, `min`, `max_abs`, `norm`, `sum_all`, `trace_full`, `inner`, `get_block_element`, …
- **`Dtype::Int64`.** Added so index-producing ops (notably `argsort` / `_argsort`) can return a `Block` of indices, not only float/complex/bool.
- **Renames to match Python public API.** `block_all` / `block_any` → `all` / `any`.
- **Elementwise comparisons on blocks.** `Block` supports `<`, `<=`, `>`, `>=`, `==`, `!=` (and mixed Block/Scalar/float overloads) so mask-style logic stays on the C++ side.
- **Block × Scalar.** `Block * Scalar`, `Scalar * Block`, `Block / Scalar`, etc., are first-class; Python bindings also overload with bare `float64` / `complex128` via `as_scalar`.
- **`apply_basis_perm`.** Uses C++ `Leg::basis_perm` / `inverse_basis_perm` (see [convert_Spaces.md](convert_Spaces.md)).
- **Default implementations on the base.** Combinators such as `combine_legs` / `split_legs`, `dagger`, `linear_combination`, `mul`, `permute_combined_*`, `eye_block`, `argsort` (wrapper around `_argsort`), etc., live on `BlockBackend` and call pure-virtual primitives implemented by `NumpyBlockBackend`.
- **HDF5.** `Block`, `Scalar`, and backends implement `save_hdf5` / `from_hdf5` and round-trip through the existing Python HDF5 helpers (covered by pytest).
- **Native C++ indexing.** `BlockBackend::AxisSlice` and `BlockIndex` (`variant` of `int64` / `AxisSlice` / `vector<int64>` / `BlockCPtr`) provide `get_item` / `set_item` / `operator[]` without constructing `py::slice`. Torch implements this with libtorch; numpy/Array API convert `BlockIndex` to Python keys at the backend boundary.

## What stayed in Python

- `ArrayApiBlockBackend`, `TorchBlockBackend` (out of current C++ scope).
- The Python `_block_backend.py` / `numpy.py` modules as the base for those backends (not used for the default numpy path exported from `__init__.py`).
- Higher layers (`TensorBackend`, tensors, symmetries) still call into the C++ block backend from Python.

## Lessons / tips for later conversions

- Prefer a real C++ value type (here: `Scalar`) over returning `py::object` or bare floating types when the Python API mixed numbers and array scalars.
- Nesting backend-specific `Block` under the concrete backend keeps the abstract `Block` API small while still allowing `dynamic_cast` / `NumpyBlockBackend::ptr(...)` in the numpy implementation.
- Keep trampolines for ABCs that remaining Python backends (or user code) may subclass; make trampolines non-template (see existing `PyBlockBackend` / `PyBlock`).
- Expect substantial binding surface area for arithmetic dunders; mirror both Block–Block and Block–Scalar (and float/complex) overloads or Python will silently fall back to `NotImplemented`.
- After swapping the public import to `_core`, run the full tensor/backend pytest suite early: most breakages showed up as missing Scalar/Block operator overloads rather than missing named methods.

## TODO checklist (completed for original C++ conversion)

- [x] initial setup / planning
- [x] C++ declaration + definition for `BlockBackend`, `Block`, `Scalar`, `NumpyBlockBackend`
- [x] pybind11 bindings + trampolines
- [x] monkey-patch via `lasyten/block_backends/__init__.py`
- [x] adjust tensor/backend Python for `Scalar` where needed
- [x] pytest green with C++ BlockBackend
- [x] tighten `apply_basis_perm` when `Space`/`Leg` exists in C++ ([convert_Spaces.md](convert_Spaces.md))

## TODO checklist (#255 follow-up)

- [ ] Eigen/LAPACK backend — [convert_EigenBlockBackend.md](convert_EigenBlockBackend.md)
- [ ] Decouple `py::array` / `py::object` from `block_backend.h` public API
- [x] Nanobind bindings
- [x] `nb::ndarray` I/O for the Eigen backend (`pybind/block_backend/eigen_ndarray.h`); `to_numpy` / `block_from_numpy` (`py::array`) remain on the C++ API (leftover #10)
- [ ] Drop `PyBlockBackend` trampoline initially (tenpy does not subclass it)
- [ ] Default `"cpu"` → Eigen once parity is good
- [x] Gate Numpy/ArrayApi Python-dispatch backends (`LASYTEN_WITH_NUMPY_BACKEND`, marked transitional in `backend_factory.cpp`); removal is still open (leftover #5)
- [ ] `LASYTEN_WITH_PYTHON=OFF` builds without Python (Phase 5)
