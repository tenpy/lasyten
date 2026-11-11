"""Defines classes that describe the sites of a lattice."""

# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from .._core import (  # noqa: F401
    AnyonSite,
    ClockSite,
    FibonacciAnyonSite,
    GoldenSite,
    IsingAnyonSite,
    NoSymmetry,
    SectorArray,
    SpinHalfFermionSite,
    SpinlessBosonSite,
    SpinlessFermionSite,
    SpinSite,
    SU2kSpin1Site,
    Symmetry,
)
from .degrees_of_freedom import Site

if TYPE_CHECKING:
    from ..symmetries import Symmetry as SymmetryType

_NO_SYMMETRY_FACTOR = NoSymmetry()


def set_common_symmetry(
    sites: Sequence[Site],
    symmetry_combine: str | Callable | list = 'by_name',
    new_symmetry: SymmetryType | None = None,
) -> None:
    """Adjust the symmetries of the given sites *in place* so they can be used together.

    Before contracting operators (and tensors) from different :class:`~lasyten.models.degrees_of_freedom.Site`
    instances, their physical legs need a single shared :class:`~lasyten.symmetries.Symmetry`, with
    sectors remapped accordingly. Call this typically from a model's ``init_sites``.

    This is the lasyten counterpart of TeNPy v1 ``set_common_charges``. Issue #152
    (:class:`GroupedSite`) should reuse the same combine modes before piping legs.

    Parameters
    ----------
    sites :
        Sites to combine. Modified **in place**; the same object must not appear twice
        (make copies first).
    symmetry_combine : ``'by_name'`` | ``'same'`` | ``'drop'`` | ``'independent'`` | callable | list
        How the common symmetry is built from the individual ones.

        ``'by_name'`` / ``'same'``
            Default. Split each product symmetry into factors. Equal factors (same mathematical
            group *and* ``descriptive_name``) are merged (charges add); unequal factors stay
            independent.
        ``'independent'``
            Treat every non-:class:`~lasyten.symmetries.NoSymmetry` factor as independent.
        ``'drop'``
            Drop all symmetries.
        callable
            ``new_sectors = symmetry_combine(site_idx, old_sectors)``. Requires `new_symmetry`.
        list of list of tuple
            Linear combinations of old sector columns (TeNPy v1 style). Each outer entry is one
            new column; inner entries are ``(prefactor, site_idx, old_col_idx)``. Requires
            `new_symmetry`. Prefactors may be non-integer if the result is integer.
    new_symmetry :
        Required for callable / list modes; ignored for the string presets.

    """
    for i, site in enumerate(sites):
        for site2 in sites[i + 1 :]:
            if site2 is site:
                raise ValueError('`sites` contains the same object multiple times. Make copies!')

    if symmetry_combine in ('by_name', 'same'):
        _set_common_symmetry_by_name(sites)
        return

    if symmetry_combine == 'drop':
        if new_symmetry is not None:
            raise ValueError("new_symmetry must be None when symmetry_combine='drop'")
        for site in sites:
            site.change_leg(site.leg.drop_symmetry())
        return

    if symmetry_combine == 'independent':
        _set_common_symmetry_independent(sites)
        return

    if isinstance(symmetry_combine, str):
        raise ValueError(f'Unknown symmetry_combine keyword: {symmetry_combine!r}')

    if isinstance(symmetry_combine, list):
        input_symmetry_combine = [list(col) for col in symmetry_combine]

        def symmetry_combine(site_idx: int, old_sectors: SectorArray) -> SectorArray:
            old = old_sectors.to_numpy()
            cols = []
            for col_spec in input_symmetry_combine:
                col = np.zeros((len(old_sectors),), dtype=np.int64)
                for factor, i, col_idx in col_spec:
                    if not isinstance(col_idx, (int, np.integer)):
                        raise TypeError(f'old_col_idx must be int, got {type(col_idx).__name__}: {col_idx!r}')
                    if i != site_idx:
                        continue
                    col = col + factor * old[:, int(col_idx)]
                rounded = np.rint(col)
                if not np.allclose(col, rounded):
                    raise ValueError(f'Sectors must have integer entries. Got {col}')
                cols.append(rounded.astype(np.int64, copy=False))
            if not cols:
                raise ValueError('symmetry_combine list must be non-empty')
            return SectorArray(np.stack(cols, axis=1))

    if new_symmetry is None:
        raise ValueError('Need to specify new_symmetry')
    if not callable(symmetry_combine):
        raise TypeError(f'symmetry_combine must be str, callable, or list; got {type(symmetry_combine).__name__}')

    for i, site in enumerate(sites):
        new_leg = site.leg.change_symmetry(symmetry=new_symmetry, sector_map=partial(symmetry_combine, i))
        site.change_leg(new_leg)


def _site_factors_and_slices(site: Site):
    """Return ``(factors, sector_slices)`` for ``site.leg.symmetry``."""
    sym = site.leg.symmetry
    factors = list(sym.factors)
    slices = [slice(int(sym.sector_slices[n]), int(sym.sector_slices[n + 1])) for n in range(len(factors))]
    return factors, slices


def _symmetry_from_factors(factors: list) -> Symmetry:
    if len(factors) == 0:
        return _NO_SYMMETRY_FACTOR.as_Symmetry()
    if len(factors) == 1:
        return factors[0].as_Symmetry()
    return Symmetry(factors)


def _set_common_symmetry_by_name(sites: Sequence[Site]) -> None:
    factors = []
    # For every merged factor: list of (site_idx, old_sector_slice)
    sites_and_slices: list[list[tuple[int, slice]]] = []
    for i, site in enumerate(sites):
        site_factors, site_slices = _site_factors_and_slices(site)
        for f, slc in zip(site_factors, site_slices, strict=True):
            # SymmetryFactor.__eq__ checks mathematical group *and* descriptive_name.
            if f in factors:
                n = factors.index(f)
                sites_and_slices[n].append((i, slc))
            else:
                factors.append(f)
                sites_and_slices.append([(i, slc)])

    new_symmetry = _symmetry_from_factors(factors)
    new_symm_slices = [
        slice(int(new_symmetry.sector_slices[n]), int(new_symmetry.sector_slices[n + 1])) for n in range(len(factors))
    ]

    for i, site in enumerate(sites):
        slice_tuples = []
        for n in range(len(factors)):
            for j, slc in sites_and_slices[n]:
                if i == j:
                    slice_tuples.append((new_symm_slices[n], slc))

        trivial = np.asarray(new_symmetry.trivial_sector.to_numpy(), dtype=np.int64)

        def sector_map(s, _slice_tuples=slice_tuples, _trivial=trivial):
            old = s.to_numpy()
            mapped = np.tile(_trivial[None, :], (len(s), 1))
            for mapped_slice, old_slice in _slice_tuples:
                mapped[:, mapped_slice] = old[:, old_slice]
            return SectorArray(mapped)

        site.change_leg(site.leg.change_symmetry(symmetry=new_symmetry, sector_map=sector_map))


def _set_common_symmetry_independent(sites: Sequence[Site]) -> None:
    factors = []
    # Per site: slices of old sector columns that belong to non-NoSymmetry factors.
    kept_slices_per_site: list[list[slice]] = []
    for site in sites:
        site_factors, site_slices = _site_factors_and_slices(site)
        kept_slices: list[slice] = []
        for f, slc in zip(site_factors, site_slices, strict=True):
            if f != _NO_SYMMETRY_FACTOR:
                factors.append(f)
                kept_slices.append(slc)
        kept_slices_per_site.append(kept_slices)

    new_symmetry = _symmetry_from_factors(factors)
    trivial = np.asarray(new_symmetry.trivial_sector.to_numpy(), dtype=np.int64)

    start = 0
    for site, kept_slices in zip(sites, kept_slices_per_site, strict=True):
        ind_len = sum(slc.stop - slc.start for slc in kept_slices)

        # Default-arg capture avoids the late-binding bug of the old cyten draft.
        def sector_map(s, _start=start, _kept=kept_slices, _trivial=trivial):
            old = s.to_numpy()
            mapped = np.tile(_trivial[None, :], (len(s), 1))
            col = _start
            for slc in _kept:
                width = slc.stop - slc.start
                mapped[:, col : col + width] = old[:, slc]
                col += width
            return SectorArray(mapped)

        site.change_leg(site.leg.change_symmetry(symmetry=new_symmetry, sector_map=sector_map))
        start += ind_len
