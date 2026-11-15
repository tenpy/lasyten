# Conversion note: Eigen/LAPACK BlockBackend (#255)

Native CPU `BlockBackend` that stores dense buffers in C++ and uses Eigen +
LAPACK for linear algebra. Python exchange uses `nb::ndarray` (zero-copy via
buffer/DLPack). Master port note: [convert_nanobind.md](convert_nanobind.md).

## Status

**Implemented (v1); `"cpu"` still points to `NumpyBlockBackend` (while that is built).** Eigen is the preferred CPU backend (`"eigen"`).

| Piece | State |
| --- | --- |
| Design | Done (this doc) |
| `include/lasyten/block_backend/eigen.h` | Done |
| `src/block_backend/eigen.cpp` | Done (all pure virtuals; no NumPy calls for compute) |
| CMake: Eigen + LAPACK(E) | Done |
| Python I/O (`py::array`) | Copy-based (`to_numpy` / `block_from_numpy`), kept for liblasyten-internal callers; compiled out without Python |
| Python I/O (`nb::ndarray`) | Done: `block_from_ndarray` (copying import, any CPU DLPack / buffer-protocol array, dtypes bool / int64 / float32/64 / complex64/128), `to_ndarray` (zero-copy writable NumPy view that keeps the block alive) on backend and on `BlockCls`; `pybind/block_backend/eigen_ndarray.h` |
| Python-free operation | Done for `LASYTEN_WITH_PYTHON=OFF`: native HDF5 save/load (`hdf5_io::Hdf5Buffer`), `apply_axis_permutations`, repr; ctest `test_eigen_block_backend_no_python` |
| Bindings (`pybind/block_backend/py_eigen.cpp`) | Done |
| BlockBackend-contract tests | Basic ops + SVD/QR/eigh/tdot (`tests/ctest/test_eigen_block_backend.cpp`, `tests/python_tests/backends/test_eigen_block_backend.py`) |
| Default `"cpu"` factory | Leftover until parity |

### Implementation notes

- Storage: `EigenBlockBackend::Block` owns a C-contiguous `std::vector<std::byte>` plus
  `std::vector<int64>` shape. No `py::array` is stored.
- Elementwise ops, reductions, permute, tile, indexing: type-dispatched loops over the
  contiguous buffer with NumPy broadcasting. Indexing returns copies (not views).
- `matrix_dot`, `tdot`, `outer`: `Eigen::Map` gemm (BLAS-backed, `EIGEN_USE_BLAS` for
  `eigen.cpp` only).
- `matrix_svd`, `matrix_qr`, `eigh`, `eig*`: LAPACKE (row-major) with `std::complex` as complex type.
  `matrix_exp` uses `unsupported/Eigen/MatrixFunctions`.
- Factory: `EigenBlockBackend::from_factory("eigen")` (also `"cpu"` on the class itself).
  The central `get_backend(..., block_backend="eigen")` selects Eigen. `"cpu"` / `"numpy"` still
  select `NumpyBlockBackend`.
- `possible_svd_algorithms()`: `"gesdd"` (default, divide and conquer), `"gesvd"` (QR iteration,
  slower but more robust), `"robust"` (gesdd, falling back to gesvd with a warning if it fails)
  and `"robust_silent"` (same, without the warning).
- Int64 inputs of `as_block` are converted to float64, like the other backends.

## Why not `nb::ndarray` alone?

`nb::ndarray` only wraps metadata + a data pointer. It has **no** matmul, SVD,
QR, eigh, tensordot, etc. Nanobind’s Eigen casters are a separate convenience
for Eigen↔Python exchange, not a BlockBackend.

## Design

### Storage

- Owning contiguous host buffer + shape (row-major / C-order by default).
- Prefer a simple owning layout (`std::vector` of bytes or typed storage +
  `std::vector<int64>` shape) mapped to Eigen views for ops, or
  `Eigen::Tensor` where ergonomic.
- Device: **CPU only** for v1 (`device()` → `"cpu"` / `"eigen"`).

### Compute

| Op family | Implementation |
| --- | --- |
| Elementwise, reductions, reshape, permute, tile | Eigen / hand loops on contiguous data |
| `matrix_dot`, `tdot`, `kron`, `outer`, traces | Eigen contractions / gemm |
| `matrix_svd`, `matrix_qr`, `eigh`, `eig*` | LAPACK/LAPACKE (or Eigen with `EIGEN_USE_LAPACKE`) aiming for SciPy-comparable quality |
| Indexing (`BlockIndex`) | Native C++ path (no `py::object`) |

### Python boundary

- Ingress: accept `nb::ndarray<>` (any CPU framework via buffer/DLPack); copy
  into owning storage when needed for layout/dtype, else share when safe.
- Egress: return `nb::ndarray` (typically numpy-tagged) for `to_numpy`-style
  APIs; zero-copy view when the buffer is stable and contiguous.
- Implemented in `pybind/block_backend/eigen_ndarray.h` (`block_from_ndarray`, `to_ndarray`;
  Python: `bb.block_from_ndarray(arr, dtype=None)`, `bb.to_ndarray(block)`, `block.to_ndarray()`).
  Import always copies (the backend owns its memory); export is a zero-copy view aliasing the
  block, so mutate it only if you want to mutate the block. Unsupported dtypes (e.g. int32,
  float16) raise `TypeError`. `nb::ndarray` stays a bindings-only type: liblasyten itself never
  includes it.
- With `LASYTEN_WITH_PYTHON=OFF` the `py::` members of `eigen.h` are compiled out; use
  `Block::shape()` / `dtype()` / `data<T>()` / `bytes()` and the native HDF5 I/O instead.
- Long-term: no `py::array` in `include/lasyten/block_backend/` public API
  ([convert_nanobind.md](convert_nanobind.md) Phase 5, leftover #10).

### Factory

- Name: `"eigen"` initially; become default for `"cpu"` once parity + tests are
  good enough.
- Keep `TorchBlockBackend` for CUDA / libtorch.
- `NumpyBlockBackend` / `ArrayApiBlockBackend`: transitional; gate or remove
  under leftover #5.

## Dependencies

- **Eigen3** (header-only) — via conda-forge `eigen` / `find_package(Eigen3)`.
- **LAPACK** — conda already provides OpenBLAS/MKL (`liblapack`); link LAPACKE
  where needed for SVD/QR/eigh.

## Tests

- CTest / pytest BlockBackend contract: elementwise, `matrix_dot`,
  `matrix_svd` / `matrix_qr`, `eigh`, `tdot`, indexing, dtype conversion,
  `zeros` / `eye` / random, HDF5 if applicable.
- Compare a sample of SVD/eigh results against NumPy/SciPy within documented
  tolerances (leftover #6).

## Metadata (target)

| Object | Header | Source | Bindings |
| --- | --- | --- | --- |
| `EigenBlockBackend` | `include/lasyten/block_backend/eigen.h` | `src/block_backend/eigen.cpp` | nanobind block_backend module |
| nested `Block` | same | same | nested under backend |

## TODO checklist

- [x] CMake `find_package(Eigen3)` + LAPACK/LAPACKE link
- [x] Header + implementation of pure-virtual BlockBackend surface
- [x] Python I/O helpers (copy-based `to_numpy` / `block_from_numpy`)
- [x] `nb::ndarray` helpers (`block_from_ndarray` / `to_ndarray`, zero-copy export)
- [x] Python-free compile + ctest with `LASYTEN_WITH_PYTHON=OFF` (native HDF5, `apply_axis_permutations`)
- [x] Register in the backend factory as `"eigen"` (`"cpu"` deliberately unchanged)
- [x] Basic contract tests green (zeros, ones, eye, reshape, permute, matrix_dot, SVD, QR, eigh, tdot, abs/add, numpy round trip)
- [x] Document numerical defaults (LAPACK `gesdd` default; `gesvd` / `robust` options; Eigen Jacobi not used)
- [ ] Run the full BlockBackend/tensor test-suite with `default_block_backend="eigen"`
- [ ] Switch default `"cpu"` (after parity)
