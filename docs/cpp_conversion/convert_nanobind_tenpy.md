# Downstream: TeNPy `v2_main` after nanobind / Eigen (#255)

Required TeNPy changes when bumping the lasyten submodule after the nanobind
port and Eigen BlockBackend. Checked against `~/code/tenpy_v2` (tree near
`v2_main`). Master note: [convert_nanobind.md](convert_nanobind.md).

## Must keep working (trampoline-dependent)

TeNPy **does** subclass C++-bound lasyten types — but only Krylov operators:

| Tenpy type | Base | Files |
| --- | --- | --- |
| `TransferMatrix` | `lasy.tensors.sparse.LinearOperator` | `tenpy/networks/mps.py` |
| `EffectiveH`, `OneSiteH`, `TwoSiteH`, `ZeroSiteH`, `DummyTwoSiteH`, … | `PlanarLinearOperator` | `tenpy/algorithms/mps_common.py` (+ purification / dmrg_parallel) |
| `MPOTransferMatrix` | `lasy.sparse.LinearOperator` | `tenpy/networks/mpo.py` (stub `NotImplementedError`, still a subclass) |

These override `matvec` / `adjoint` and are passed into C++ `LanczosGroundState`
/ `Arnoldi`. lasyten **must** keep `PyLinearOperator` /
`PyPlanarLinearOperator` trampolines across the nanobind port.

Status (nanobind port): both trampolines are bound. `PyPlanarLinearOperator`
overrides `matvec`, `to_tensor`, **and** `adjoint` (the base trampoline alone is
not enough — the dynamic type is the derived trampoline). `LinearOperator` is
bound with `nb::dynamic_attr` for TeNPy instance attributes. Regression:
`test_LinearOperator_trampoline_virtual_dispatch`,
`test_PlanarLinearOperator` (adjoint via `ShiftedLinearOperator`).

After bumping lasyten: no TeNPy API change expected for these if trampolines
are preserved — confirm Krylov still dispatches into Python `matvec`.

**nanobind pitfall:** C++ `LinearOperator.dtype` is a property. TeNPy subclasses
must assign `self.dtype = …` only *after* `PlanarLinearOperator.__init__` /
`LinearOperator.__init__` (uninitialized trampoline instances raise
`TypeError` / warn). Prefer `env.H.dtype` (already a lasyten `Dtype`) over
`Dtype.from_numpy_dtype(...)`.

## Likely API / behavior changes

| Area | Current tenpy usage | Expected lasyten change | Downstream action |
| --- | --- | --- | --- |
| Default CPU backend | Implicit numpy / `block_backend` | Eigen becomes default `"cpu"` | Smoke MPS/MPO/DMRG; pin backend only if numerical diffs appear |
| `to_numpy()` / `block_from_numpy` | Widespread (mps, mpo, algorithms) | Tensor/`Mask`/`to_numpy` export read-only `nb::ndarray`; prefer `block_from_ndarray` | TeNPy: use `block_from_ndarray` for imports; treat `to_numpy()` as read-only (no in-place writes) |
| `Dtype.to_numpy_dtype()` | e.g. MPS init | Keep or re-export; binding style may change | Re-run type-sensitive paths after bump |
| Import paths | `from lasyten.tensors.planar import PlanarLinearOperator` | Still exported from nanobind `_core` | Bump + full test; fix only if export layout changes |
| `Scalar(...).to_numpy()` | norms, traces, residuals | Same idea; Eigen scalar blocks underneath | Compare float tolerances if Eigen vs SciPy SVD/eigh differ |
| Subclassing non-LO C++ types | Not used in tenpy | Other trampolines dropped | None for tenpy; document for other downstreams |

## Does not need trampolines (composition-only)

Sites (`lasy.sites.*`), symmetries, `SymmetricTensor` construction,
`get_same_backend`, planar helpers — TeNPy uses factories/instances, not
Python subclasses of those C++ ABCs.

## Call-site map (focus)

### Trampoline subclasses

```text
tenpy/networks/mps.py
  class TransferMatrix(lasy.tensors.sparse.LinearOperator)
    def matvec(...)

tenpy/algorithms/mps_common.py
  class EffectiveH(PlanarLinearOperator)
  class OneSiteH / TwoSiteH / ZeroSiteH / DummyTwoSiteH(EffectiveH)
    def matvec(...); def adjoint(...)

tenpy/networks/mpo.py
  class MPOTransferMatrix(lasy.sparse.LinearOperator)  # NotImplementedError
```

### Array bridge hotspots (review after Eigen default)

- `tenpy/networks/mps.py` — **`block_from_ndarray`** (was `block_from_numpy`), `to_numpy` for dtype
  conversion only, `permute_axes`, `reshape`
- `tenpy/networks/mpo.py` — `to_numpy` on rearranged tensors / traces (read-only under Eigen)
- Algorithm residuals / norms calling `.to_numpy()` on scalars (owning 0-d copies; unchanged)

## Smoke checklist after bumping lasyten

- [ ] DMRG one-site / two-site (`EffectiveH.matvec` → Lanczos)
- [ ] MPS transfer-matrix / `Arnoldi` (`TransferMatrix.matvec`)
- [x] `to_numpy` / `block_from_ndarray` round-trips on MPS tensors (TeNPy call sites updated)
- [ ] Abelian + no-symmetry small models
- [ ] Optional Torch device path unchanged
- [ ] Compare a few SVD/truncation energies vs pre-Eigen baseline (tolerances)

## Related prior downstream notes

- Spaces MI: [convert_spaces_MI.md](convert_spaces_MI.md) (`is_leg_pipe`, etc.)
- Site DOF composition: [convert_SiteDOF.md](convert_SiteDOF.md)
