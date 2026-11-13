# Conversion note: spaces MI removal (#254 item 3)

Removes the `ElementarySpace` / `AbelianLegPipe` diamonds so the hierarchy is nanobind-ready (#255).

## Hierarchy

| Old | New |
| --- | --- |
| `LegOrSpace` shared virtual root | **deleted** |
| Abstract polymorphic `Leg` base | Value-type `Leg` handle over `ElementarySpace \| LegPipe` |
| `ElementarySpace : Space, virtual Leg` | `ElementarySpace : Space` only |
| `LegPipe : virtual Leg` | `LegPipe` independent (`enable_shared_from_this<LegPipe>`) |
| `AbelianLegPipe : LegPipe, ElementarySpace` | `AbelianLegPipe : ElementarySpace` only; pipe fields composed |

```text
Space
├── ElementarySpace
│   ├── DirectSumSpace
│   └── AbelianLegPipe   # composed legs / combine_cstyle / fusion maps
└── TensorProduct

LegPipe                  # plain nesting pipe (FusionTree); not a Space
```

## isinstance / pipe detection

| Old | New |
| --- | --- |
| `isinstance(x, Leg)` (C++ base) | Python `Leg` ABC registers `ElementarySpace` and `LegPipe` (not `TensorProduct`) |
| `isinstance(alp, LegPipe)` | **False** — use `x.is_pipe` / `lasy.is_leg_pipe(x)` / `isinstance(x, AbelianLegPipe)` |
| `isinstance(x, LegPipe) and not isinstance(x, ElementarySpace)` | plain pipe: `is_pipe and not is_abelian_leg_pipe` |
| `dynamic_pointer_cast<LegPipe>(leg)` for “any pipe” | `leg.is_pipe()` / `leg.as_leg_pipe()` vs `as_elementary_space()` |

## C++ `Leg::Ptr`

| Old | New |
| --- | --- |
| `shared_ptr<Leg>` polymorphic | Value handle `Leg` (`using Ptr = Leg`) |
| `leg->symmetry` / `leg->is_dual` | `leg.symmetry()` / `leg.is_dual()` |
| `dynamic_pointer_cast<ElementarySpace>(leg)` | `leg.as_elementary_space()` |
| `dynamic_pointer_cast<LegPipe>(leg)` | `leg.as_leg_pipe()` |
| `TensorProduct::factors` | `std::vector<Leg>` |

## Backend APIs

| Old | New |
| --- | --- |
| `make_pipe` → `LegPipe::Ptr` | `make_pipe` → `Leg::Ptr` (abelian returns `AbelianLegPipe`) |
| `combine_legs(..., vector<LegPipe::Ptr>)` | `vector<Leg::Ptr>` pipes |

## Bindings

- `class_<ElementarySpace, Space, …>` (no `Leg` base)
- `class_<AbelianLegPipe, ElementarySpace, …>`
- `class_<LegPipe, …>` top-level (no `Leg` / `LegOrSpace` base)
- No bound C++ `Leg` class; Python ABC in `lasyten/symmetries/spaces.py`

## Behavior preserved

- AbelianBackend combined legs remain `ElementarySpace` (`Site.leg`, nested pipes, HDF5 ES path)
- FusionTree `make_pipe` still yields plain `LegPipe`
- `flat_spaces` keeps abelian pipes nested; flatten strips ALP to plain ES

## Downstream: TeNPy (`v2_main`)

Follow-up for the spaces MI refactor (issue #254 item 3). After bumping lasyten, TeNPy
code that assumes **`AbelianLegPipe` is-a `LegPipe`** breaks. Most MPS/MPO paths that only
use `ElementarySpace`, `combine_legs` / `split_legs` by name, or construct `AbelianLegPipe`
explicitly are unchanged.

### Call-site map (Python)

| Old | New |
| --- | --- |
| `isinstance(x, LegPipe)` for any combined leg (incl. abelian) | `lasy.is_leg_pipe(x)` / `x.is_pipe` |
| `isinstance(x, LegPipe)` meaning plain FusionTree nesting only | keep `isinstance(x, LegPipe)` (ALP is no longer a subclass) |
| `isinstance(x, LegPipe) and not isinstance(x, ElementarySpace)` | plain pipe: `is_leg_pipe(x) and not isinstance(x, ElementarySpace)` |
| `isinstance(x, Leg)` | unchanged (`Leg` ABC still registers ES + `LegPipe`) |
| `isinstance(x, ElementarySpace)` / `AbelianLegPipe(...)` | unchanged |

### Known hit on `v2_main` (checked against current tree)

**`tenpy/networks/mps.py` — `_flatten_domain_pipe`**

- `assert isinstance(pipe, lasy.LegPipe) and pipe.dim == 1`
- → `assert lasy.is_leg_pipe(pipe) and pipe.dim == 1`
  (AbelianBackend pipes are `AbelianLegPipe`, not `LegPipe`)

### Does **not** need changes (if used as today)

- Explicit `AbelianLegPipe([...])` construction (e.g. `exact_diag.py`, tests)
- `Site.leg` / `ElementarySpace` factories / `as_ElementarySpace`
- `combine_legs` / `split_legs` / `flatten_pipe_leg` call sites that do not type-check `LegPipe`
- Site DOF composition (already covered by [convert_SiteDOF.md](convert_SiteDOF.md) / #317)

### Suggested smoke checklist after bumping lasyten

- [ ] `_flatten_domain_pipe` / hidden-leg MPS paths with AbelianBackend
- [ ] `combine_legs` / `split_legs` on MPS/MPO bonds (abelian)
- [ ] ExactDiag grouping via `AbelianLegPipe`
- [ ] Optional: FusionTree backend still sees plain `LegPipe` from `make_pipe`
