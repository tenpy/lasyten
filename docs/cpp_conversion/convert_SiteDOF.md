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

## Multi-DOF / grouping

Multiple DOFs of the same kind are allowed under different names
(e.g. `"spin"` + `"valley"`, or `"0.spin"` / `"1.spin"` after grouping).
Coupling factories currently use the default conventional names; an explicit
`dof=` argument is deferred to issue #152.
