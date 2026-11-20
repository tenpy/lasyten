# I/O: HDF5 and pickle

Two serialization routes exist for the nanobind-bound C++ types. They share the same *logical
state* but target different workflows.

| Route | Use case | Implementation |
| --- | --- | --- |
| HDF5 (`save_hdf5` / `from_hdf5`) | Long-lived files, interchange with TeNPy | C++ `save_hdf5` / `from_hdf5`, bound via `pybind/tools/hdf5_bind.h`; Python driver is the external `tenpy-hdf5` package (`lasyten.tools.hdf5_io`) |
| pickle (`__getstate__` / `__reduce__`) | In-Python workflows: `multiprocessing`, `copy.deepcopy`, caching, `dask`/`joblib` | `pybind/tools/pickle_bind.h` (+ `pybind/tensors/tensor_pickle.h`) |

## Decisions

- **Keep `tenpy-hdf5` / `tenpy_hdf5_core`.** HDF5 stays the on-disk format. Pickle does not replace it
  and is not a stable file format: do not store pickles long-term or load untrusted ones.
- **Pickle and HDF5 share logical state.** A type's pickle state is the set of fields that its
  `save_hdf5` persists, and it is rebuilt equivalently to `from_hdf5` (same constructor, same
  derived fields restored afterwards). For composite types the state is a `dict` whose keys are the
  HDF5 sub-group names (e.g. `ElementarySpace`: `symmetry`, `defining_sectors`, `multiplicities`,
  `_basis_perm`, `is_dual`; `SymmetricTensor`: `codomain`, `domain`, `backend`, `data`, `symmetry`,
  `dtype`, `device`, `shape`, `labels`).
- **No HDF5 / h5py in the pickle path.** State consists of plain Python objects (numpy arrays of
  `int64`, `dict`, bound lasyten objects, ...), nested objects are pickled recursively, and the pickle
  memo preserves shared references (e.g. a space shared by two tensors).
- **Pickle is for in-Python workflows**: it needs the same lasyten build (the same `_core`) on both
  sides.

## Mechanisms (`pickle_bind.h`)

nanobind types do not pickle automatically, and nanobind functions (`nb_func`, i.e. also static
methods) cannot be pickled *by reference*. Types can, so reconstructors are either a class (constructor)
or `operator.methodcaller(name, *args)` applied to a class or instance.

| Helper | Types | Behaviour |
| --- | --- | --- |
| `def_value_pickle` | value types (`Sector`, `SectorArray`, `LasytenConfig`) | `__getstate__` + placement-new `__setstate__` |
| `def_reduce_ctor` | types with a Python constructor (symmetry factors, `U1`, `NoSymmetry`, ...) | `__reduce__ -> (type(self), ctor_args)`; args match the HDF5 fields |
| `def_state_pickle` | `shared_ptr`-held / factory types (`Symmetry`, spaces, tensors, backend data, ...) | `__getstate__ -> dict`, static `_from_pickle_state(dict)`, `__reduce__ -> (methodcaller('_from_pickle_state', state), (type,))` |
| `reduce_via_method` | custom cases | `__reduce__ -> (methodcaller(name, *args), (target,))` |

`__setstate__` is deliberately not used for `shared_ptr`-held types: nanobind's `__new__` creates an
uninitialised instance without an owning `shared_ptr`, and `__setstate__` is treated as a
constructor-like method.

## Covered types

- Value types: `Dtype` (nanobind enum), `Sector`, `SectorArray` (int64, empty arrays keep
  `sector_ind_len`), `LasytenConfig`.
- Symmetry factors: `NoSymmetry`, `U1`, `ZN`, `FermionNumber`, `FermionParity`, `SU2`, `SUN`, the anyon
  categories (Fibonacci, Ising, SU(2)_k, SU(3)_3, ZN, toric code, quantum double), and `Symmetry`
  (product of factors).
- Spaces: `ElementarySpace` (incl. `basis_perm`), `TensorProduct`, `LegPipe` / `AbelianLegPipe` /
  `DirectSumSpace`.
- Backends: block backends (singletons per device, rebuilt through `from_factory`), tensor backends
  (`NoSymmetryBackend`, `AbelianBackend`, `FusionTreeBackend`; constructor-based, equality rather than
  identity is what matters), `Block` and `Scalar` (numpy-based via
  `backend.block_from_numpy(array, dtype, device)`), backend `Data` objects, `BlockInds`, `LegLabels`.
- Tensors: `SymmetricTensor`, `HiddenLegTensor`, `DiagonalTensor`, `Identity` (data is regenerated, as in
  `from_hdf5`), `Mask`, `ChargedTensor`.
- Models: `Site` (and subclasses) via the same `init_kwargs` as HDF5.

Pure-Python types are not touched; they pickle through the default mechanism.

## Known gaps

- Blocks go through numpy (`block_from_numpy`), so Torch autograd state is dropped; Array-API blocks
  only pickle if their array namespace does.
- Types without `save_hdf5` schemas (`NoSymmetryBackendBlockData`) get pickle hooks anyway, with
  state derived from their constructors.
- `Hdf5Exportable`-style Python objects and other types outside `pybind/**` are not covered here.

## Tests

`tests/python_tests/tools/test_pickle.py` roundtrips Dtype, Sector, SectorArray, symmetries
(including all `any_symmetry` fixtures), spaces, backends, Block / Scalar over all block backends,
nested containers, `copy.deepcopy`, and tensors (`SymmetricTensor`, `DiagonalTensor`, `Identity`,
`Mask`, `ChargedTensor`) over all compatible backend/symmetry pairs, plus config and a `Site`.
It mirrors the structure of `tests/python_tests/tools/test_hdf5.py`.
