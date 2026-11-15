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

After bumping lasyten: no TeNPy API change expected for these if trampolines
are preserved — confirm Krylov still dispatches into Python `matvec`.

## Likely API / behavior changes

| Area | Current tenpy usage | Expected lasyten change | Downstream action |
| --- | --- | --- | --- |
| Default CPU backend | Implicit numpy / `block_backend` | Eigen becomes default `"cpu"` | Smoke MPS/MPO/DMRG; pin backend only if numerical diffs appear |
| `to_numpy()` / `block_from_numpy` | Widespread (mps, mpo, algorithms) | May become zero-copy views via `nb::ndarray` | Treat returns as read-only unless documented writable; avoid in-place mutation of `to_numpy()` buffers |
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

- `tenpy/networks/mps.py` — `block_from_numpy`, `to_numpy`, `permute_axes`, `reshape`
- `tenpy/networks/mpo.py` — `to_numpy` on rearranged tensors / traces
- Algorithm residuals / norms calling `.to_numpy()` on scalars

## Smoke checklist after bumping lasyten

- [ ] DMRG one-site / two-site (`EffectiveH.matvec` → Lanczos)
- [ ] MPS transfer-matrix / `Arnoldi` (`TransferMatrix.matvec`)
- [ ] `to_numpy` / `block_from_numpy` round-trips on MPS tensors
- [ ] Abelian + no-symmetry small models
- [ ] Optional Torch device path unchanged
- [ ] Compare a few SVD/truncation energies vs pre-Eigen baseline (tolerances)

## Related prior downstream notes

- Spaces MI: [convert_spaces_MI.md](convert_spaces_MI.md) (`is_leg_pipe`, etc.)
- Site DOF composition: [convert_SiteDOF.md](convert_SiteDOF.md)
