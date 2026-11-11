# Conversion note: Site DOF composition (issues #164 / #254)

Mechanical replacement summary after removing DOF mixin inheritance from sites.
Degrees of freedom are composed into `Site` via a named `dofs` registry.

## Type / MRO

- Introduce `DegreeOfFreedom` base (not a `Site`).
- `SpinDOF`, `ClockDOF`, `AnyonDOF`, `OccupationDOF` / `BosonicDOF` / `FermionicDOF`
  inherit `DegreeOfFreedom` only (no `Site`, no virtual inheritance).
- Concrete sites inherit **only** `Site`.
- `Site` owns `std::map<std::string, DegreeOfFreedom::Ptr> dofs`.
- Conventional keys: `"spin"`, `"occupation"`, `"clock"`, `"anyon"`.

## Call-site map (Python)

| Old | New |
| --- | --- |
| `class SpinSite(SpinDOF)` | `class SpinSite(Site)` + `site.spin` / `site.dofs["spin"]` |
| `class SpinHalfFermionSite(SpinDOF, FermionicDOF)` | `class SpinHalfFermionSite(Site)` + `spin` + `occupation` |
| `site.spin_vector` | `site.spin.spin_vector` |
| `site.number_operators` / `n_tot` / `JW` / `Nmax` | `site.occupation.<attr>` |
| `site.clock_operators` | `site.clock.clock_operators` |
| `site.sector_names` | `site.anyon.sector_names` |
| `isinstance(site, SpinDOF)` | `site.spin is not None` / `"spin" in site.dofs` |
| `isinstance(site, FermionicDOF)` | `isinstance(site.occupation, FermionicDOF)` |
| `isinstance(site, BosonicDOF)` | `isinstance(site.occupation, BosonicDOF)` |
| `isinstance(site, ClockDOF)` | `site.clock is not None` |
| `isinstance(site, AnyonDOF)` | `site.anyon is not None` |

## Call-site map (C++)

| Old | New |
| --- | --- |
| `dynamic_pointer_cast<SpinDOF>(site)` | `site->get_spin()` / `site->require_spin()` |
| `dynamic_pointer_cast<OccupationDOF>(site)` | `site->get_occupation()` / `site->require_occupation()` |
| `dynamic_pointer_cast<BosonicDOF>(site)` | `dynamic_pointer_cast<BosonicDOF>(site->get_occupation())` |
| `dynamic_pointer_cast<ClockDOF>(site)` | `site->get_clock()` / `site->require_clock()` |
| `OccupationDOF::add_*_ops()` on `*this` | `occ->add_*_ops(site)` |

## Multi-DOF / grouping (#152)

- C++ `GroupedSite` / `group_sites` (abelian only via `AbelianLegPipe`): physical leg is a
  pipe; onsite ops are Kronecker-embedded as `opname + label` (e.g. `Sz0`); child DOFs are
  registered under `"{label}.{kind}"` with dense arrays embedded into the grouped dim.
- `symmetry_combine`: `'same'` | `'drop'` | `'independent'`. For mismatched site types, call
  C++ `set_common_symmetry` first, then `GroupedSite(..., symmetry_combine='same')`.
- Spin coupling factories take optional `dof=` (default `"spin"` / per-site `"spin"`) so
  grouped multi-spin sites can target `"0.spin"` / `"1.spin"`.

## Common symmetry (#153)

C++ `set_common_symmetry` (pybind-exported; re-exported from `lasyten.models.sites`) remaps a
list of sites onto one shared product symmetry (`by_name` / `same`, `independent`, `drop`,
callable, or explicit linear-combo list), via `Site.change_leg`.

Non-abelian grouping (general `LegPipe` as `Site.leg`) remains deferred.
