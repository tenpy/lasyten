"""Defines classes describing the local physical Hilbert spaces.

:class:`Site` owns the local Hilbert space and onsite operators.
Degrees of freedom (:class:`SpinDOF`, :class:`OccupationDOF`, …) are composed into a
site via the named ``dofs`` registry (see also issues #164 / #254), not inherited as
mixins.
"""

# Copyright (C) TeNPy Developers, Apache license

from .._core import (  # noqa: F401
    ALL_SPECIES,
    AnyonDOF,
    BosonicDOF,
    ClockDOF,
    DegreeOfFreedom,
    FermionicDOF,
    OccupationDOF,
    Site,
    SpinDOF,
)
