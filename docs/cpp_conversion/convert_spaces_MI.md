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
